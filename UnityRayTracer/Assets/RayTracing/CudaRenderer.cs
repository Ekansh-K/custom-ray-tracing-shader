using System;
using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(Camera))]
public sealed class CudaRenderer : MonoBehaviour
{
    // Keep these values in sync with Native/renderer.cu.
    public enum RenderBackend
    {
        [InspectorName("CPU single")] CpuSingle = 0,
        [InspectorName("CPU multi")] CpuMulti = 1,
        [InspectorName("CUDA GPU")] Cuda = 2,
        [InspectorName("Embree CPU multi")] Embree = 3
    }
    public int width = 1280;
    public int height = 720;
    public RenderBackend backend = RenderBackend.Cuda;
    public float moveSpeed = 3f;
    public float mouseSensitivity = 0.12f;

    public bool denoise = true;

    [Min(.01f)] public float exposure = 1f;
    [Range(1, 16)] public int raysPerPixel = 1;
    [Range(1, 16)] public int maxBounces = 8;
    [Min(0)] public float aperture = 0f;
    [Min(.01f)] public float focusDistance = 8.8f;
    [Min(0)] public float skyStrength = .15f;

    private Camera viewCamera;
    private int previousCullingMask;
    private CameraClearFlags previousClearFlags;
    private Color previousBackground;
    private RayTraceSession session;
    private float yaw, pitch;
    private bool controlledLastFrame;
    private float fps;
    private double fpsStart;
    private int fpsFrames;
    public bool InputEnabled { get; set; } = true;
    public float FramesPerSecond => fps;
    public float ComputeMilliseconds => session?.ComputeMilliseconds ?? 0;
    public float DenoiseMilliseconds => session?.DenoiseMilliseconds ?? 0;
    public string DenoiseError => session?.DenoiseError;
    public float RenderCallMilliseconds => session?.RenderCallMilliseconds ?? 0;
    public uint DisplayedSamples => session?.DisplayedSamples ?? 0;
    public uint SampleCount => session?.SampleCount ?? 0;
    public void ResetAccumulation() { session?.ResetAccumulation(); }
    public int RenderedFrames { get; private set; }
    public int TriangleCount => session?.TriangleCount ?? 0;
    public Texture2D Image => session?.Image;
    public int CpuThreadCount => session?.CpuThreadCount ?? 0;
    public int BvhNodeCount => session?.BvhNodeCount ?? 0;
    public int BvhDepth => session?.BvhDepth ?? 0;
    public float BvhBuildMilliseconds => session?.BvhBuildMilliseconds ?? 0;
    public float EmbreeBuildMilliseconds => session?.EmbreeBuildMilliseconds ?? 0;
    public RayTraceSession.Settings Settings => new RayTraceSession.Settings
    {
        maxBounces = maxBounces, exposure = exposure, aperture = aperture,
        focusDistance = focusDistance, skyStrength = skyStrength, seed = 42, raysPerPixel = raysPerPixel
    };

    private void OnEnable()
    {
        viewCamera = GetComponent<Camera>();
        previousCullingMask = viewCamera.cullingMask;
        previousClearFlags = viewCamera.clearFlags;
        previousBackground = viewCamera.backgroundColor;
        // Unity only presents the generated image, rather than also drawing
        // its ordinary mesh renderer behind it.
        viewCamera.cullingMask = 0;
        viewCamera.clearFlags = CameraClearFlags.SolidColor;
        viewCamera.backgroundColor = Color.black;
        yaw = transform.eulerAngles.y;
        pitch = Mathf.DeltaAngle(0, transform.eulerAngles.x);
        fpsStart = Time.realtimeSinceStartupAsDouble;
    }

    public void UploadScene()
    {
        if (session == null) session = new RayTraceSession();
        session.UploadScene(width, height);
    }

    private void Update()
    {
        // Scene objects may still be activating during OnEnable. The first
        // Update runs after that activation, so all tagged meshes are available.
        if (session == null || !session.Ready || session.Width != width || session.Height != height)
        {
            try { UploadScene(); }
            catch (Exception error)
            {
                UnityEngine.Debug.LogException(error, this);
                enabled = false;
                return;
            }
        }
        var keyboard = Keyboard.current;
        var mouse = Mouse.current;
        bool handleInput = Application.isFocused;
#if UNITY_EDITOR
        var focused = UnityEditor.EditorWindow.focusedWindow;
        handleInput = focused != null && focused.GetType().Name == "GameView";
#endif
        handleInput &= InputEnabled;
        if (handleInput && keyboard != null)
        {
            if (keyboard.tabKey.wasPressedThisFrame) backend = (RenderBackend)(((int)backend + 1) % 4);
            if (keyboard.digit1Key.wasPressedThisFrame) backend = RenderBackend.CpuSingle;
            if (keyboard.digit2Key.wasPressedThisFrame) backend = RenderBackend.CpuMulti;
            if (keyboard.digit3Key.wasPressedThisFrame) backend = RenderBackend.Cuda;
            if (keyboard.digit4Key.wasPressedThisFrame) backend = RenderBackend.Embree;
        }
        if (handleInput && keyboard != null && keyboard.rKey.wasPressedThisFrame) UploadScene();
        bool looking = handleInput && mouse != null && mouse.rightButton.isPressed;
        if (handleInput || controlledLastFrame)
        {
            Cursor.lockState = looking ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !looking;
        }
        controlledLastFrame = looking;
        if (looking)
        {
            Vector2 delta = mouse.delta.ReadValue();
            yaw += delta.x * mouseSensitivity;
            pitch = Mathf.Clamp(pitch - delta.y * mouseSensitivity, -85, 85);
            transform.rotation = Quaternion.Euler(pitch, yaw, 0);
            if (keyboard != null)
            {
                Vector3 movement = Vector3.zero;
                if (keyboard.wKey.isPressed) movement += transform.forward;
                if (keyboard.sKey.isPressed) movement -= transform.forward;
                if (keyboard.aKey.isPressed) movement -= transform.right;
                if (keyboard.dKey.isPressed) movement += transform.right;
                if (keyboard.eKey.isPressed) movement += Vector3.up;
                if (keyboard.qKey.isPressed) movement -= Vector3.up;
                transform.position += movement * (moveSpeed * Time.unscaledDeltaTime);
            }
        }
        RenderFrame();
        fpsFrames++;
        double elapsed = Time.realtimeSinceStartupAsDouble - fpsStart;
        if (elapsed >= 1) { fps = (float)(fpsFrames / elapsed); fpsStart = Time.realtimeSinceStartupAsDouble; fpsFrames = 0; }
    }

    public void RenderFrame()
    {
        if (session == null || !session.Ready || session.Width != width || session.Height != height) UploadScene();
        session.Render(viewCamera, backend, Settings, denoise);
        RenderedFrames++;
    }

    // Compare our three implementations against the single-core reference image.
    public string CompareBackends()
    {
        if (session == null || !session.Ready) RenderFrame();
        session.ResetAccumulation();
        byte[] cpu = (byte[])session.Calculate(viewCamera, RenderBackend.CpuSingle, Settings).Clone();
        var times = new float[3];
        times[0] = session.ComputeMilliseconds;
        int differingPixels = 0, maxDifference = 0, multiThreads = 0;
        foreach (var mode in new[] { RenderBackend.CpuMulti, RenderBackend.Cuda })
        {
            session.ResetAccumulation();
            byte[] candidate = session.Calculate(viewCamera, mode, Settings);
            times[(int)mode] = session.ComputeMilliseconds;
            if (mode == RenderBackend.CpuMulti) multiThreads = session.CpuThreadCount;
            for (int i = 0; i < cpu.Length; i += 4)
            {
                bool differs = false;
                for (int c = 0; c < 3; c++)
                {
                    int difference = Math.Abs(cpu[i + c] - candidate[i + c]);
                    maxDifference = Math.Max(maxDifference, difference);
                    if (difference > 1) differs = true;
                }
                if (differs) differingPixels++;
            }
        }
        session.ResetAccumulation();
        return $"{TriangleCount} triangles, {width}x{height}; CPU single {times[0]:F2} ms; CPU multi ({multiThreads} threads) {times[1]:F2} ms; CUDA {times[2]:F2} ms; differing pixels (>1/255): {differingPixels}; max channel difference: {maxDifference}";
    }

    private void OnGUI()
    {
        if (Image == null || session == null || !session.Ready) return;
        GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), Image, ScaleMode.ScaleToFit);
    }

    private void OnDisable()
    {
        session?.Dispose();
        session = null;
        if (viewCamera != null)
        {
            viewCamera.cullingMask = previousCullingMask;
            viewCamera.clearFlags = previousClearFlags;
            viewCamera.backgroundColor = previousBackground;
        }
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
    }
}
