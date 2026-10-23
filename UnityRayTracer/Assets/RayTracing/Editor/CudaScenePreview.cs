using System;
using UnityEditor;
using UnityEditor.Overlays;
using UnityEngine;
using UnityEngine.UIElements;

[InitializeOnLoad]
public static class CudaScenePreview
{
    private const string Menu = "Ray tracer/CUDA Scene preview";
    private static RayTraceSession session;
    private static bool dirty = true;
    private static double nextRepaint;
    private static string status = "Off";
    public static string Status => status;
    public static Texture2D Image => session?.Image;
    public static bool Enabled
    {
        get => SessionState.GetBool("CudaScenePreview.Enabled", false);
        set
        {
            SessionState.SetBool("CudaScenePreview.Enabled", value);
            if (!value) Release();
            dirty = true;
            SceneView.RepaintAll();
        }
    }

    static CudaScenePreview()
    {
        SceneView.duringSceneGui += Draw;
        EditorApplication.update += Tick;
        EditorApplication.hierarchyChanged += MarkDirty;
        UnityEditor.SceneManagement.EditorSceneManager.sceneClosing += (scene, removing) => Release();
        UnityEngine.SceneManagement.SceneManager.sceneUnloaded += scene => Release();
        Undo.undoRedoPerformed += MarkDirty;
        ObjectChangeEvents.changesPublished += OnObjectsChanged;
        AssemblyReloadEvents.beforeAssemblyReload += Release;
        EditorApplication.quitting += Release;
        EditorApplication.playModeStateChanged += _ => { Release(); dirty = true; };
    }

    [MenuItem(Menu)] public static void Toggle() { Enabled = !Enabled; }
    [MenuItem(Menu, true)] private static bool Validate() { UnityEditor.Menu.SetChecked(Menu, Enabled); return true; }
    private static void OnObjectsChanged(ref ObjectChangeEventStream changes) { dirty = true; }
    private static void MarkDirty() { dirty = true; }
    public static void Refresh() { dirty = true; SceneView.RepaintAll(); }

    private static void Tick()
    {
        if (!Enabled || EditorApplication.isCompiling || EditorApplication.isPlayingOrWillChangePlaymode != EditorApplication.isPlaying) return;
        var view = SceneView.lastActiveSceneView;
        // Update at most ten times per second and avoid background benchmark work.
        if (view == null || (!view.hasFocus && !(EditorWindow.focusedWindow is RendererSettingsWindow) && !dirty) || EditorApplication.timeSinceStartup < nextRepaint) return;
        nextRepaint = EditorApplication.timeSinceStartup + .1;
        view.Repaint();
    }

    private static void Draw(SceneView view)
    {
        if (!Enabled || view != SceneView.lastActiveSceneView || Event.current.type != EventType.Repaint || EditorApplication.isCompiling) return;
        var renderer = UnityEngine.Object.FindFirstObjectByType<CudaRenderer>();
        if (renderer == null) { status = "Add a Cuda Renderer to a camera"; return; }
        try
        {
            if (session == null) session = new RayTraceSession();
            var camera = view.camera;
            int width = Mathf.Clamp(renderer.width, 64, 1920);
            float aspect = (float)camera.pixelWidth / Mathf.Max(1, camera.pixelHeight);
            // Keep the aspect ratio without exceeding the Game view's pixel budget.
            width = Mathf.Min(width, Mathf.RoundToInt(Mathf.Sqrt(renderer.width * renderer.height * aspect)));
            width = Mathf.Max(1, width);
            int height = Mathf.Clamp(Mathf.RoundToInt(width / aspect), 1, 4096);
            if (dirty || !session.Ready || session.Width != width || session.Height != height)
            {
                session.UploadScene(width, height);
                dirty = false;
            }
            session.Render(camera, CudaRenderer.RenderBackend.Cuda, renderer.Settings, renderer.denoise);
            status = $"{width} x {height} | {session.SampleCount} samples | CUDA {session.ComputeMilliseconds:F2} ms | AI {session.DenoiseMilliseconds:F1} ms | BVH {session.BvhNodeCount} nodes";
            if (session.DenoiseError != null) status += " | " + session.DenoiseError;
            Handles.BeginGUI();
            GUI.DrawTexture(new Rect(0, 0, camera.pixelWidth / EditorGUIUtility.pixelsPerPoint,
                camera.pixelHeight / EditorGUIUtility.pixelsPerPoint), session.Image, ScaleMode.StretchToFill);
            Handles.EndGUI();
        }
        catch (Exception error)
        {
            Enabled = false;
            status = "Error: " + error.Message;
            Debug.LogException(error);
        }
    }

    private static void Release()
    {
        session?.Dispose(); session = null;
        status = "Off";
    }
}

[Overlay(typeof(SceneView), "CUDA preview", true)]
public sealed class CudaPreviewOverlay : Overlay
{
    public override VisualElement CreatePanelContent()
    {
        var root = new VisualElement();
        var toggle = new Toggle("CUDA preview") { value = CudaScenePreview.Enabled };
        toggle.RegisterValueChangedCallback(evt => CudaScenePreview.Enabled = evt.newValue);
        var status = new Label();
        var refresh = new Button(CudaScenePreview.Refresh) { text = "Refresh scene" };
        root.Add(toggle); root.Add(status); root.Add(refresh);
        root.schedule.Execute(() =>
        {
            toggle.SetValueWithoutNotify(CudaScenePreview.Enabled);
            status.text = CudaScenePreview.Status;
        }).Every(250);
        return root;
    }
}
