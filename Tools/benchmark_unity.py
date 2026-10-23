#!/usr/bin/env python3
"""Capture a fixed-camera performance baseline from the running Unity scene.

Start Play mode first. Usage: python3 Tools/benchmark_unity.py OUTPUT_DIRECTORY
The measurements include Unity Editor overhead and are not standalone player FPS.
"""
import csv
import hashlib
import json
from pathlib import Path
import select
import statistics
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class Unity:
    def __init__(self):
        self.process = subprocess.Popen(
            ["unity", "mcp", "--project-path", str(ROOT / "UnityRayTracer")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True,
        )
        self.sequence = 0
        self.call("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "performance-baseline", "version": "1.0"},
        })
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.process.stdin.flush()

    def call(self, method, params):
        self.sequence += 1
        self.process.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": self.sequence, "method": method, "params": params,
        }) + "\n")
        self.process.stdin.flush()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if not select.select([self.process.stdout], [], [], max(0, deadline - time.monotonic()))[0]:
                break
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Unity MCP connection closed")
            response = json.loads(line)
            if response.get("id") == self.sequence:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response["result"]
        raise TimeoutError(method)

    def evaluate(self, code):
        response = self.call("tools/call", {"name": "eval", "arguments": {"code": code, "timeout": 150}})
        for content in response.get("content", []):
            if content.get("type") == "text":
                try:
                    result = json.loads(content["text"])
                except ValueError as error:
                    raise RuntimeError(content["text"]) from error
                if not isinstance(result, dict) or not result.get("success"):
                    raise RuntimeError(result)
                return result.get("result")
        raise RuntimeError(response)

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("frames.csv", "done", "error.txt", "results.json"):
        if (output / name).exists():
            raise FileExistsError(f"Use a fresh output directory: {output}")
    unity = Unity()
    original_input = None
    original_background = None
    original_denoise = None
    try:
        metadata = unity.evaluate('''
if (!UnityEditor.EditorApplication.isPlaying) throw new System.Exception("Start Play mode first");
if (CudaScenePreview.Enabled) throw new System.Exception("Turn off CUDA Scene preview before measuring FPS");
var r = UnityEngine.Object.FindFirstObjectByType<CudaRenderer>();
bool inputWasEnabled=r.InputEnabled; r.InputEnabled=false;
bool denoiseWasEnabled=r.denoise; r.denoise=false;
bool backgroundWasEnabled=UnityEngine.Application.runInBackground;
UnityEngine.Application.runInBackground = true;
var game = UnityEditor.EditorWindow.GetWindow(typeof(UnityEditor.Editor).Assembly.GetType("UnityEditor.GameView"));
game.Show(); game.Focus();
r.backend = CudaRenderer.RenderBackend.Cuda;
r.RenderFrame();
return new {
    width=r.width, height=r.height, triangles=r.TriangleCount, inputWasEnabled=inputWasEnabled, backgroundWasEnabled=backgroundWasEnabled, denoiseWasEnabled=denoiseWasEnabled, denoisingEnabled=false, sampling="Scrambled Sobol",
    bvhNodes=r.BvhNodeCount, bvhDepth=r.BvhDepth, bvhBuildMs=r.BvhBuildMilliseconds,
    scene=UnityEngine.SceneManagement.SceneManager.GetActiveScene().path,
    cameraX=r.transform.position.x, cameraY=r.transform.position.y, cameraZ=r.transform.position.z,
    cameraPitch=r.transform.eulerAngles.x, cameraYaw=r.transform.eulerAngles.y, cameraRoll=r.transform.eulerAngles.z,
    fov=r.GetComponent<UnityEngine.Camera>().fieldOfView,
    renderer="Progressive path tracer", raysPerPixelPerFrame=r.raysPerPixel, maxBounces=r.maxBounces,
    exposure=r.exposure, aperture=r.aperture, focusDistance=r.focusDistance, skyStrength=r.skyStrength, seed=r.Settings.seed,
    gpu=UnityEngine.SystemInfo.graphicsDeviceName,
    graphicsApi=UnityEngine.SystemInfo.graphicsDeviceType.ToString(),
    cpu=UnityEngine.SystemInfo.processorType, logicalProcessors=UnityEngine.SystemInfo.processorCount,
    unity=UnityEngine.Application.unityVersion, vsync=UnityEngine.QualitySettings.vSyncCount,
    targetFrameRate=UnityEngine.Application.targetFrameRate,
    screenWidth=UnityEngine.Screen.width, screenHeight=UnityEngine.Screen.height,
    qualityCheck=r.CompareBackends()
};''')
        print(json.dumps(metadata, indent=2), flush=True)
        if not isinstance(metadata, dict):
            raise RuntimeError("Unity did not return structured scene metadata")
        original_input = metadata["inputWasEnabled"]
        original_background = metadata["backgroundWasEnabled"]
        original_denoise = metadata["denoiseWasEnabled"]
        code = r'''
var r = UnityEngine.Object.FindFirstObjectByType<CudaRenderer>();
string directory = __OUTPUT__;
int mode = 0, warmup = 3, count = 0, lastFrame = r.RenderedFrames;
double previous = UnityEngine.Time.realtimeSinceStartupAsDouble;
var csv = new System.Text.StringBuilder("backend,sample,frames,frame_interval_ms,compute_ms,native_call_ms,cpu_threads,samples_per_pixel\n");
var culture = System.Globalization.CultureInfo.InvariantCulture;
r.backend = CudaRenderer.RenderBackend.CpuSingle;
r.ResetAccumulation();
UnityEditor.EditorApplication.CallbackFunction tick = null;
tick = () => {
    try {
        if (!UnityEditor.EditorApplication.isPlaying) throw new System.Exception("Play mode stopped during capture");
        if (r.RenderedFrames == lastFrame) return;
        double now = UnityEngine.Time.realtimeSinceStartupAsDouble;
        int frames = r.RenderedFrames - lastFrame;
        double interval = (now - previous) * 1000.0;
        lastFrame = r.RenderedFrames; previous = now;
        if (warmup > 0) { warmup--; return; }
        if ((int)r.backend != mode) throw new System.Exception("Backend changed during capture");
        csv.Append(mode).Append(',').Append(count).Append(',').Append(frames).Append(',')
           .Append(interval.ToString("R", culture)).Append(',')
           .Append(r.ComputeMilliseconds.ToString("R", culture)).Append(',')
           .Append(r.RenderCallMilliseconds.ToString("R", culture)).Append(',')
           .Append(r.CpuThreadCount).Append(',').Append(r.SampleCount).Append('\n');
        count++;
        if (count < 30) return;
        System.IO.File.WriteAllBytes(System.IO.Path.Combine(directory, "backend-" + mode + ".png"), r.Image.EncodeToPNG());
        System.IO.File.WriteAllText(System.IO.Path.Combine(directory, "frames.csv"), csv.ToString());
        mode++; count = 0; warmup = 3;
        if (mode < 4) { r.backend = (CudaRenderer.RenderBackend)mode; r.ResetAccumulation(); return; }
        UnityEditor.EditorApplication.update -= tick;
        r.backend = CudaRenderer.RenderBackend.Cuda;
        System.IO.File.WriteAllText(System.IO.Path.Combine(directory, "done"), "Capture complete");
    } catch (System.Exception error) {
        UnityEditor.EditorApplication.update -= tick;
        r.backend = CudaRenderer.RenderBackend.Cuda;
        System.IO.File.WriteAllText(System.IO.Path.Combine(directory, "error.txt"), error.ToString());
    }
};
UnityEditor.EditorApplication.update += tick;
return "Capturing 3 warm-up frames then 30 samples per backend";
'''.replace("__OUTPUT__", json.dumps(str(output)))
        print(unity.evaluate(code), flush=True)
        deadline = time.monotonic() + 3600
        completed = 0
        while not (output / "done").exists():
            if (output / "error.txt").exists():
                raise RuntimeError((output / "error.txt").read_text())
            if time.monotonic() > deadline:
                raise TimeoutError("Unity benchmark exceeded one hour")
            if (output / "frames.csv").exists():
                rows = list(csv.DictReader((output / "frames.csv").open()))
                if len(rows) != completed:
                    completed = len(rows)
                    print(f"Saved {completed} frame samples", flush=True)
            time.sleep(1)
        rows = list(csv.DictReader((output / "frames.csv").open()))
        summaries = []
        reference = Image.open(output / "backend-0.png").convert("RGB").tobytes()
        for mode, name in enumerate(("CPU single", "CPU multi", "CUDA GPU", "Embree CPU multi")):
            samples = [row for row in rows if int(row["backend"]) == mode]
            if len(samples) != 30 or any(int(row["frames"]) != 1 for row in samples) or [int(row["samples_per_pixel"]) for row in samples] != [i * metadata["raysPerPixelPerFrame"] for i in range(4, 34)]:
                raise RuntimeError("Frame samples were missed; repeat the capture")
            image = Image.open(output / f"backend-{mode}.png").convert("RGB").tobytes()
            errors = [abs(a-b) for a,b in zip(reference,image)]
            mae = statistics.mean(errors)
            fraction = sum(e>1 for e in errors)/len(errors)
            if mae>.5 or fraction>.01:
                raise RuntimeError(f"Backend image quality differs: {name}, MAE {mae}, fraction {fraction}")
            summaries.append({
                "backend": name, "samples": len(samples),
                "mean_abs_channel_error_vs_cpu":mae,
                "rmse_vs_cpu":statistics.mean(e*e for e in errors)**.5,
                "max_channel_error_vs_cpu":max(errors),
                "channels_over_1_fraction":fraction,
                "cpu_threads": sorted({int(row["cpu_threads"]) for row in samples}),
                "compute_median_ms": statistics.median(float(row["compute_ms"]) for row in samples),
                "native_call_median_ms": statistics.median(float(row["native_call_ms"]) for row in samples),
                "unity_editor_fps": 1000 * len(samples) / sum(float(row["frame_interval_ms"]) for row in samples),
                "frame_interval_median_ms": statistics.median(float(row["frame_interval_ms"]) for row in samples),
                "frame_interval_min_ms": min(float(row["frame_interval_ms"]) for row in samples),
                "frame_interval_max_ms": max(float(row["frame_interval_ms"]) for row in samples),
                "image_sha256": hashlib.sha256((output / f"backend-{mode}.png").read_bytes()).hexdigest(),
            })
        # Compare equal-sample RGB images. Embree uses independent intersections;
        # report numeric differences as well as hashes rather than requiring bit identity.
        result = {
            "captured_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            "metadata": metadata, "warmup_frames_per_backend": 3,
            "order": [row["backend"] for row in summaries],
            "method": "One fixed-camera batch in Unity Editor. 30 consecutive path-tracing passes per backend after 3 warm-up passes. Accumulation resets at every backend switch; measured sample counts are 4 through 33 times rays-per-pixel on all four backends. FPS is completed frames divided by wall time, observed via EditorApplication.update. Compute uses existing CPU wall clock or CUDA events. Native-call time includes CUDA synchronization and readback, but excludes texture upload and display. Scene upload and initialization are excluded. Editor frame pacing, other applications, and power conditions affect results. Embree uses its independent triangle intersections with the shared integrator. RGB image errors are reported at the same accumulated sample count. These Editor measurements are separate from the standalone native benchmark.",
            "results": summaries,
        }
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if original_input is not None:
            try:
                unity.evaluate('var r=UnityEngine.Object.FindFirstObjectByType<CudaRenderer>(); if(r!=null) { r.InputEnabled=' + str(original_input).lower() + '; r.denoise=' + str(original_denoise).lower() + '; } UnityEngine.Application.runInBackground=' + str(original_background).lower() + '; return true;')
            except Exception:
                pass
        unity.close()


if __name__ == "__main__":
    main()
