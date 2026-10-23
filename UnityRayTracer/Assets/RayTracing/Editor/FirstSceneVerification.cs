using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

public static class FirstSceneVerification
{
    [MenuItem("Ray tracer/Verify first scene in Play mode")]
    public static void Verify() { Debug.Log(Run()); }

    // Run against the actual Unity scene, native library, and CUDA device.
    public static string Run()
    {
        if (!EditorApplication.isPlaying) throw new InvalidOperationException("Enter Play mode first.");
        var gameView = EditorWindow.GetWindow(typeof(Editor).Assembly.GetType("UnityEditor.GameView"));
        gameView.Show(); gameView.Focus();
        var renderer = UnityEngine.Object.FindFirstObjectByType<CudaRenderer>();
        if (renderer == null || !renderer.enabled) throw new InvalidOperationException("Renderer is unavailable.");
        var front = UnityEngine.Object.FindObjectsByType<RayTraceObject>(FindObjectsSortMode.None)
            .Single(o => o.name == "Front wall - one sided");
        Vector3 position = renderer.transform.position;
        Quaternion rotation = renderer.transform.rotation;
        CudaRenderer.RenderBackend mode = renderer.backend;
        bool sided = front.oneSided;
        bool denoise = renderer.denoise;
        renderer.denoise = false;
        var yaw = typeof(CudaRenderer).GetField("yaw", BindingFlags.Instance | BindingFlags.NonPublic);
        var pitch = typeof(CudaRenderer).GetField("pitch", BindingFlags.Instance | BindingFlags.NonPublic);
        object oldYaw = yaw.GetValue(renderer), oldPitch = pitch.GetValue(renderer);
        Mouse mouse = null;
        Keyboard keyboard = null;
        try
        {
            renderer.backend = CudaRenderer.RenderBackend.Cuda;
            renderer.ResetAccumulation();
            renderer.RenderFrame();
            byte[] baseline = renderer.Image.GetRawTextureData<byte>().ToArray();
            string comparison = renderer.CompareBackends();
            if (!comparison.Contains("differing pixels (>1/255): 0;"))
                throw new InvalidOperationException(comparison);

            mouse = InputSystem.AddDevice<Mouse>();
            keyboard = InputSystem.AddDevice<Keyboard>();
            InputSystem.QueueStateEvent(mouse, new MouseState());
            InputSystem.Update(); mouse.MakeCurrent();
            var keys = new[] { Key.Digit1, Key.Digit2, Key.Digit3 };
            for (int i = 0; i < keys.Length; i++)
            {
                InputSystem.QueueStateEvent(keyboard, new KeyboardState(keys[i]));
                InputSystem.Update(); keyboard.MakeCurrent();
                renderer.SendMessage("Update");
                int difference = baseline.Zip(renderer.Image.GetRawTextureData<byte>().ToArray(), (a,b) => Math.Abs(a-b)).Max();
                if ((int)renderer.backend != i || difference > 1)
                    throw new InvalidOperationException($"Backend shortcut {i + 1}: selected {renderer.backend}, maximum pixel difference {difference}.");
                InputSystem.QueueStateEvent(keyboard, new KeyboardState());
                InputSystem.Update();
            }
            InputSystem.QueueStateEvent(keyboard, new KeyboardState(Key.Tab));
            InputSystem.Update(); keyboard.MakeCurrent();
            renderer.SendMessage("Update");
            if (renderer.backend != CudaRenderer.RenderBackend.Embree)
                throw new InvalidOperationException("Tab did not cycle from CUDA to Embree.");
            InputSystem.QueueStateEvent(keyboard, new KeyboardState());
            InputSystem.Update();
            InputSystem.QueueStateEvent(keyboard, new KeyboardState(Key.Tab));
            InputSystem.Update(); keyboard.MakeCurrent();
            renderer.SendMessage("Update");
            if (renderer.backend != CudaRenderer.RenderBackend.CpuSingle)
                throw new InvalidOperationException("Tab did not wrap from Embree to CPU single.");
            InputSystem.QueueStateEvent(keyboard, new KeyboardState(Key.Digit4));
            InputSystem.Update(); keyboard.MakeCurrent();
            renderer.SendMessage("Update");
            if (renderer.backend != CudaRenderer.RenderBackend.Embree)
                throw new InvalidOperationException("Shortcut 4 did not select Embree.");
            InputSystem.QueueStateEvent(keyboard, new KeyboardState());
            InputSystem.Update();
            renderer.backend = CudaRenderer.RenderBackend.Cuda;

            renderer.transform.position += new Vector3(.3f, 0, 0);
            renderer.RenderFrame();
            bool cameraChangedImage = !baseline.SequenceEqual(renderer.Image.GetRawTextureData<byte>().ToArray());
            renderer.transform.position = position;

            front.oneSided = false;
            renderer.UploadScene(); renderer.RenderFrame();
            bool frontOccludes = !baseline.SequenceEqual(renderer.Image.GetRawTextureData<byte>().ToArray());
            front.oneSided = sided;
            renderer.UploadScene();

            // Send actual Input System states through the runtime controller.
            InputSystem.QueueStateEvent(mouse, new MouseState { buttons = 2, delta = new Vector2(20, 10) });
            InputSystem.QueueStateEvent(keyboard, new KeyboardState(Key.W));
            InputSystem.Update();
            mouse.MakeCurrent(); keyboard.MakeCurrent();
            SceneView.lastActiveSceneView?.Focus();
            renderer.SendMessage("Update");
            if (Quaternion.Angle(rotation, renderer.transform.rotation) > .001f || Vector3.Distance(position, renderer.transform.position) > .00001f)
                throw new InvalidOperationException("Scene view navigation also moved the Game camera.");
            gameView.Focus();
            InputSystem.QueueStateEvent(mouse, new MouseState { buttons = 2, delta = new Vector2(20, 10) });
            InputSystem.QueueStateEvent(keyboard, new KeyboardState(Key.W));
            InputSystem.Update();
            mouse.MakeCurrent(); keyboard.MakeCurrent();
            renderer.SendMessage("Update");
            bool mouseRotated = Quaternion.Angle(rotation, renderer.transform.rotation) > .1f;
            bool keyboardMoved = Vector3.Distance(position, renderer.transform.position) > .0001f;
            if (!cameraChangedImage || !frontOccludes || !mouseRotated || !keyboardMoved)
                throw new InvalidOperationException($"camera image={cameraChangedImage}, front={frontOccludes}, mouse={mouseRotated}, keyboard={keyboardMoved}");
            return comparison + "; backend shortcuts, camera image, front-wall culling, mouse look, keyboard movement, and Scene/Game input isolation passed";
        }
        finally
        {
            // Release virtual controls before returning to the physical devices.
            if (mouse != null) InputSystem.QueueStateEvent(mouse, new MouseState());
            if (keyboard != null) InputSystem.QueueStateEvent(keyboard, new KeyboardState());
            InputSystem.Update();
            if (mouse != null) InputSystem.RemoveDevice(mouse);
            if (keyboard != null) InputSystem.RemoveDevice(keyboard);
            renderer.transform.SetPositionAndRotation(position, rotation);
            yaw.SetValue(renderer, oldYaw); pitch.SetValue(renderer, oldPitch);
            front.oneSided = sided;
            renderer.backend = mode;
            renderer.denoise = denoise;
            renderer.UploadScene(); renderer.RenderFrame();
            Cursor.lockState = CursorLockMode.None; Cursor.visible = true;
            string folder = Path.GetFullPath(Path.Combine(Application.dataPath, "../../scratch"));
            Directory.CreateDirectory(folder);
            File.WriteAllBytes(Path.Combine(folder, "first-cuda-render.png"), renderer.Image.EncodeToPNG());
        }
    }
}
