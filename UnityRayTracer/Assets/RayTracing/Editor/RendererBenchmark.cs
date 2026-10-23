using System.IO;
using UnityEditor;
using UnityEngine;

public static class RendererBenchmark
{
    [MenuItem("Ray tracer/Export benchmark scene")]
    public static void Export()
    {
        var renderer = Object.FindFirstObjectByType<CudaRenderer>();
        if (renderer == null) throw new System.InvalidOperationException("Open a scene with a Cuda Renderer camera.");
        string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "../../scratch/benchmarks"));
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, "scene.json");
        RayTraceSession.SaveSnapshot(path, renderer.GetComponent<Camera>(), renderer.Settings, renderer.width, renderer.height);
        Debug.Log("Benchmark scene exported to " + path);
    }
}
