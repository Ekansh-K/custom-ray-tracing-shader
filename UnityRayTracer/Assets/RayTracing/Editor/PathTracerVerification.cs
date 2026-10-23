using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class PathTracerVerification
{
    [MenuItem("Ray tracer/Verify progressive renderer")]
    public static void Verify() { Debug.Log(Run()); }

    // Use real scene geometry and the installed CUDA device, independently of Play mode.
    public static string Run()
    {
        var renderer = UnityEngine.Object.FindFirstObjectByType<CudaRenderer>();
        if (renderer == null) throw new InvalidOperationException("No renderer in the scene.");
        var camera = renderer.GetComponent<Camera>();
        var settings = renderer.Settings;
        settings.raysPerPixel = 1;
        string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "../../scratch/progressive"));
        Directory.CreateDirectory(folder);
        using (var session = new RayTraceSession())
        {
            session.UploadScene(80, 45);
            byte[] reference = null;
            int maxDifference = 0;
            foreach (var backend in new[] { CudaRenderer.RenderBackend.CpuSingle, CudaRenderer.RenderBackend.CpuMulti, CudaRenderer.RenderBackend.Cuda })
            {
                session.ResetAccumulation();
                for (int i = 0; i < 8; i++) session.Calculate(camera, backend, settings);
                byte[] image = session.Calculate(camera, backend, settings);
                if (session.SampleCount != 9) throw new InvalidOperationException("Accumulation did not advance.");
                if (reference == null) reference = (byte[])image.Clone();
                else maxDifference = Math.Max(maxDifference, reference.Zip(image, (a,b) => Math.Abs(a-b)).Max());
            }
            if (maxDifference > 1) throw new InvalidOperationException($"Backends differ by {maxDifference}/255.");
            session.UploadScene(384, 216);
            byte[] first = null, sixteen = null;
            for (int i = 1; i <= 256; i++)
            {
                session.Render(camera, CudaRenderer.RenderBackend.Cuda, settings);
                if (i == 1) { first = session.Image.GetRawTextureData<byte>().ToArray(); Save(session, folder, "sample-001"); }
                if (i == 16) { sixteen = session.Image.GetRawTextureData<byte>().ToArray(); Save(session, folder, "sample-016"); }
            }
            byte[] settled = session.Image.GetRawTextureData<byte>().ToArray();
            Save(session, folder, "sample-256");
            double initialError = Rmse(first, settled), refinedError = Rmse(sixteen, settled);
            if (refinedError >= initialError * .65) throw new InvalidOperationException("Progressive samples did not reduce noise.");
            settings.maxBounces = 1;
            session.ResetAccumulation();
            for (int i = 0; i < 256; i++) session.Render(camera, CudaRenderer.RenderBackend.Cuda, settings);
            Save(session, folder, "direct-only-diagnostic");
            byte[] direct = session.Image.GetRawTextureData<byte>().ToArray();
            double directMean = direct.Where((_,i) => i % 4 != 3).Average(b => (double)b);
            double bouncedMean = settled.Where((_,i) => i % 4 != 3).Average(b => (double)b);
            if (bouncedMean <= directMean + 2) throw new InvalidOperationException("Bounced lighting did not illuminate the room.");
            settings = renderer.Settings;
            settings.raysPerPixel = 1;
            session.Render(camera, CudaRenderer.RenderBackend.Cuda, settings);
            if (session.SampleCount != 1) throw new InvalidOperationException("Changing settings did not reset accumulation.");
            session.Render(camera, CudaRenderer.RenderBackend.Cuda, settings);
            if (session.SampleCount != 2) throw new InvalidOperationException("Stationary camera did not accumulate.");
            session.ResetAccumulation();
            session.Render(camera, CudaRenderer.RenderBackend.Cuda, settings);
            if (!first.SequenceEqual(session.Image.GetRawTextureData<byte>().ToArray())) throw new InvalidOperationException("Reset changed the deterministic first sample.");
            using (var preview = new RayTraceSession())
            {
                preview.UploadScene(53, 37);
                preview.Render(camera, CudaRenderer.RenderBackend.Cuda, settings);
            }
            if (session.SampleCount != 1 || session.Width != 384) throw new InvalidOperationException("Preview session disturbed Game accumulation.");
            return $"Actual scene: CPU single/multi/CUDA agree within {maxDifference}/255 after 9 samples; noise RMSE {initialError:F2} -> {refinedError:F2}; direct/bounced mean {directMean:F2}/{bouncedMean:F2}; reset and independent sessions passed. Images: {folder}";
        }
    }
    private static void Save(RayTraceSession session, string folder, string name)
    {
        File.WriteAllBytes(Path.Combine(folder, name + ".png"), session.Image.EncodeToPNG());
    }
    private static double Rmse(byte[] a, byte[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++) if (i % 4 != 3) sum += (a[i]-b[i]) * (a[i]-b[i]);
        return Math.Sqrt(sum / (a.Length / 4 * 3));
    }
}
