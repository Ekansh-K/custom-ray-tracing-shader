using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using UnityEngine;

// Shared by the Game view and the editor preview. Each owns its own native buffers.
public sealed class RayTraceSession : IDisposable
{
    [Serializable, StructLayout(LayoutKind.Sequential, Pack = 4)]
    private struct Triangle
    {
        public Vector3 origin, edge1, edge2, color;
        public float emission;
        public int oneSided;
        public Vector3 normal0, normal1, normal2;
        public int mesh;
        public Vector3 specularColor;
        public float smoothness, specularProbability;
    }
    [Serializable, StructLayout(LayoutKind.Sequential, Pack = 4)]
    private struct CameraData
    {
        public Vector3 origin, right, up, forward;
        public float tanHalfFov, aspect, orthographicSize;
        public int orthographic;
    }
    [Serializable, StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct Settings
    {
        public int maxBounces;
        public float exposure, aperture, focusDistance, skyStrength;
        public uint seed;
        public int raysPerPixel;
    }
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr RtCreate();
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void RtDestroy(IntPtr session);
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern int RtInitialize(IntPtr session, [In] Triangle[] triangles, int count, int w, int h);
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern int RtRender(IntPtr session, ref CameraData camera, ref Settings settings,
        [Out] byte[] pixels, int outputBytes, int backend, out float ms);
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern int RtCpuThreadCount(IntPtr session);
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void RtBvhInfo(IntPtr session, out int nodes, out int depth, out float buildMs);
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern void RtReset(IntPtr session);
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint RtSampleCount(IntPtr session);
    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr RtLastError(IntPtr session);

    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern float RtEmbreeBuildMilliseconds(IntPtr session);

    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern int RtDenoise(IntPtr session, [Out] byte[] pixels, int outputBytes, out float milliseconds);

    [DllImport("CudaRayTracer", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint RtAccumulationVersion(IntPtr session);

    private uint accumulationVersion;
    private bool denoisingLastFrame, denoisedImageReady;
    private uint denoisedSamples;
    public uint DisplayedSamples { get; private set; }
    public float DenoiseMilliseconds { get; private set; }
    public string DenoiseError { get; private set; }
    private IntPtr native;
    private byte[] pixels;
    private readonly Stopwatch timer = new Stopwatch();
    public Texture2D Image { get; private set; }
    public int TriangleCount { get; private set; }
    public int CpuThreadCount { get; private set; }
    public int BvhNodeCount { get; private set; }
    public int BvhDepth { get; private set; }
    public float BvhBuildMilliseconds { get; private set; }
    public float EmbreeBuildMilliseconds => native == IntPtr.Zero ? 0 : RtEmbreeBuildMilliseconds(native);
    public float ComputeMilliseconds { get; private set; }
    public float RenderCallMilliseconds { get; private set; }
    public uint SampleCount => native == IntPtr.Zero ? 0 : RtSampleCount(native);
    public void ResetAccumulation() { if (native != IntPtr.Zero) RtReset(native); denoisedImageReady = false; }
    public int Width => Image == null ? 0 : Image.width;
    public int Height => Image == null ? 0 : Image.height;
    public bool Ready { get; private set; }

    private void Check(int success)
    {
        if (success == 0) throw new InvalidOperationException(Marshal.PtrToStringAnsi(RtLastError(native)));
    }

    // Also used by the benchmark exporter, so its scene data matches the renderer.
    private static Triangle[] CollectTriangles()
    {
        var triangles = new List<Triangle>();
        foreach (var item in UnityEngine.Object.FindObjectsByType<RayTraceObject>(FindObjectsSortMode.InstanceID))
        {
            if (!item.isActiveAndEnabled) continue;
            var mesh = item.GetComponent<MeshFilter>().sharedMesh;
            if (mesh == null) continue;
            if (!mesh.isReadable) throw new InvalidOperationException($"Enable Read/Write on the mesh import settings for {item.name}.");
            var vertices = mesh.vertices;
            var indices = mesh.triangles;
            var normals = mesh.normals;
            Matrix4x4 normalMatrix = item.transform.localToWorldMatrix.inverse.transpose;
            for (int i = 0; i < indices.Length; i += 3)
            {
                int ia = indices[i], ib = indices[i + 1], ic = indices[i + 2];
                Vector3 a = item.transform.TransformPoint(vertices[ia]);
                Vector3 b = item.transform.TransformPoint(vertices[ib]);
                Vector3 c = item.transform.TransformPoint(vertices[ic]);
                Vector3 face = Vector3.Cross(b - a, c - a).normalized;
                if (face.sqrMagnitude < .5f) continue;
                bool hasNormals = normals.Length == vertices.Length;
                triangles.Add(new Triangle
                {
                    origin = a, edge1 = b - a, edge2 = c - a,
                    color = new Vector3(item.color.r, item.color.g, item.color.b),
                    emission = item.emissive ? item.emissionStrength : 0,
                    oneSided = item.oneSided ? 1 : 0,
                    mesh = item.GetInstanceID(),
                    specularColor = new Vector3(item.specularColor.r, item.specularColor.g, item.specularColor.b),
                    smoothness = item.smoothness, specularProbability = item.specularProbability,
                    normal0 = hasNormals ? normalMatrix.MultiplyVector(normals[ia]).normalized : face,
                    normal1 = hasNormals ? normalMatrix.MultiplyVector(normals[ib]).normalized : face,
                    normal2 = hasNormals ? normalMatrix.MultiplyVector(normals[ic]).normalized : face
                });
            }
        }
        return triangles.ToArray();
    }

    public void UploadScene(int width, int height)
    {
        if (width < 1 || height < 1 || width > 4096 || height > 4096)
            throw new ArgumentOutOfRangeException("Render size must be between 1 and 4096 per dimension.");
        Triangle[] triangles = CollectTriangles();
        Ready = false;
        DenoiseError = null; DenoiseMilliseconds = 0; denoisedImageReady = false;
        if (triangles.Length == 0) throw new InvalidOperationException("Add a Ray Trace Object component to at least one active mesh.");
        if (native == IntPtr.Zero) native = RtCreate();
        Check(RtInitialize(native, triangles, triangles.Length, width, height));
        RtBvhInfo(native, out int nodes, out int depth, out float buildMs);
        BvhNodeCount = nodes; BvhDepth = depth; BvhBuildMilliseconds = buildMs;
        if (Width != width || Height != height)
        {
            DestroyImage();
            pixels = new byte[checked(width * height * 4)];
            Image = new Texture2D(width, height, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point, hideFlags = HideFlags.HideAndDontSave };
        }
        TriangleCount = triangles.Length;
        Ready = true;
    }

    private static CameraData GetCameraData(Camera camera, int width, int height)
    {
        return new CameraData
        {
            origin = camera.transform.position, right = camera.transform.right,
            up = camera.transform.up, forward = camera.transform.forward,
            tanHalfFov = Mathf.Tan(camera.fieldOfView * Mathf.Deg2Rad * .5f),
            aspect = (float)width / height,
            orthographic = camera.orthographic ? 1 : 0, orthographicSize = camera.orthographicSize
        };
    }

#if UNITY_EDITOR
    [Serializable] private sealed class Snapshot
    {
        public Triangle[] triangles;
        public CameraData camera;
        public Settings settings;
        public int width, height;
    }
    // Export once; standalone benchmarks can then run without Unity overhead.
    public static void SaveSnapshot(string path, Camera camera, Settings settings, int width, int height)
    {
        var snapshot = new Snapshot { triangles = CollectTriangles(),
            camera = GetCameraData(camera, width, height), settings = settings, width = width, height = height };
        System.IO.File.WriteAllText(path, JsonUtility.ToJson(snapshot));
    }
#endif

    public byte[] Calculate(Camera camera, CudaRenderer.RenderBackend backend, Settings settings)
    {
        if (!Ready) throw new InvalidOperationException("Upload the scene before rendering.");
        var data = GetCameraData(camera, Width, Height);
        timer.Restart();
        Check(RtRender(native, ref data, ref settings, pixels, pixels.Length, (int)backend, out float ms));
        timer.Stop();
        uint version = RtAccumulationVersion(native);
        if (version != accumulationVersion) { denoisedImageReady = false; accumulationVersion = version; }
        ComputeMilliseconds = ms;
        RenderCallMilliseconds = (float)timer.Elapsed.TotalMilliseconds;
        CpuThreadCount = RtCpuThreadCount(native);
        return pixels;
    }

    public void Render(Camera camera, CudaRenderer.RenderBackend backend, Settings settings, bool denoise = false)
    {
        Calculate(camera, backend, settings);
        if (denoise && !denoisingLastFrame) { DenoiseError = null; denoisedImageReady = false; }
        if (!denoise) { DenoiseMilliseconds = 0; denoisedImageReady = false; }
        denoisingLastFrame = denoise;
        if (denoise && DenoiseError == null)
        {
            // On this GPU inference costs more than one path pass. Refine the
            // raw sum between AI updates while displaying the last clean image.
            // Slow CPU passes and fresh camera views still update immediately.
            if (denoisedImageReady && SampleCount - denoisedSamples < 8 && RenderCallMilliseconds < DenoiseMilliseconds) return;
            if (RtDenoise(native, pixels, pixels.Length, out float milliseconds) != 0)
            {
                DenoiseMilliseconds = milliseconds; denoisedSamples = SampleCount; denoisedImageReady = true;
            }
            else DenoiseError = Marshal.PtrToStringAnsi(RtLastError(native));
        }
        DisplayedSamples = SampleCount;
        Image.LoadRawTextureData(pixels);
        Image.Apply(false);
    }

    private void DestroyImage()
    {
        if (Image == null) return;
        if (Application.isPlaying) UnityEngine.Object.Destroy(Image);
        else UnityEngine.Object.DestroyImmediate(Image);
        Image = null;
    }

    public void Dispose()
    {
        if (native != IntPtr.Zero) RtDestroy(native);
        native = IntPtr.Zero;
        Ready = false;
        DestroyImage();
    }
}
