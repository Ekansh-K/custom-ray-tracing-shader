using UnityEditor;
using UnityEngine;

// A normal Unity window: drag its tab to float it or dock it beside the view.
public sealed class RendererSettingsWindow : EditorWindow
{
    private CudaRenderer renderer;
    private SerializedObject settings;
    private Editor materialEditor;
    private Vector2 scroll;
    private bool showMaterial;
    private static readonly Vector2Int[] Sizes = {
        new Vector2Int(320, 180), new Vector2Int(640, 360),
        new Vector2Int(1280, 720), new Vector2Int(1920, 1080)
    };
    private static readonly string[] SizeNames = { "320 x 180", "640 x 360", "1280 x 720", "1920 x 1080", "Custom" };

    [MenuItem("Ray tracer/Renderer settings _F1")]
    public static void Open()
    {
        var window = GetWindow<RendererSettingsWindow>("Renderer settings");
        window.minSize = new Vector2(360, 620);
        window.Show();
    }
    private void OnInspectorUpdate() { Repaint(); }
    private void OnSelectionChange() { Repaint(); }
    private void OnDisable() { if (materialEditor != null) DestroyImmediate(materialEditor); }

    private void Field(string name, string label, string tooltip = "")
    {
        EditorGUILayout.PropertyField(settings.FindProperty(name), new GUIContent(label, tooltip));
    }

    private void OnGUI()
    {
        EditorGUIUtility.labelWidth = 150;
        EditorGUI.BeginDisabledGroup(DemoScenes.Loading);
        int current = DemoScenes.Current;
        int selected = EditorGUILayout.Popup("Demo scene", current, DemoScenes.Names);
        EditorGUI.EndDisabledGroup();
        EditorGUIUtility.labelWidth = Mathf.Min(225, position.width * .56f);
        if (selected != current && selected >= 0)
        {
            // Run after this GUI event so destroyed scene targets are not reused.
            EditorApplication.delayCall += () => DemoScenes.Load(selected);
            return;
        }
        if (DemoScenes.Loading) EditorGUILayout.LabelField("Loading selected scene...");
        if (renderer == null)
        {
            renderer = Object.FindFirstObjectByType<CudaRenderer>();
            if (renderer == null)
            {
                EditorGUILayout.HelpBox("Open a scene with a Cuda Renderer camera.", MessageType.Info);
                return;
            }
        }
        if (settings == null || settings.targetObject != renderer) settings = new SerializedObject(renderer);
        settings.Update();
        scroll = EditorGUILayout.BeginScrollView(scroll);
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("Rendering", EditorStyles.boldLabel);
        EditorGUI.BeginChangeCheck();
        Field("backend", "Backend");
        EditorGUILayout.LabelField("Sampling", "Scrambled Sobol");
        Field("denoise", "AI denoising", "NVIDIA OptiX filters the displayed image on the GPU for every backend. Raw accumulation is preserved. Disable for raw CPU/GPU comparisons.");
        var width = settings.FindProperty("width");
        var height = settings.FindProperty("height");
        int preset = System.Array.FindIndex(Sizes, s => s.x == width.intValue && s.y == height.intValue);
        int chosen = EditorGUILayout.Popup("Resolution", preset < 0 ? Sizes.Length : preset, SizeNames);
        if (chosen < Sizes.Length && chosen != preset)
        {
            width.intValue = Sizes[chosen].x; height.intValue = Sizes[chosen].y;
        }
        width.intValue = Mathf.Clamp(EditorGUILayout.DelayedIntField("Width", width.intValue), 1, 4096);
        height.intValue = Mathf.Clamp(EditorGUILayout.DelayedIntField("Height", height.intValue), 1, 4096);
        Field("raysPerPixel", "Rays per pixel / frame", "New camera paths for every pixel each frame. More rays take longer to compute.");
        Field("maxBounces", "Maximum bounces", "Maximum surface hits along each path.");
        EditorGUILayout.HelpBox("Samples accumulate while the camera is still. Rendering changes restart the image.", MessageType.None);
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("Camera and lighting", EditorStyles.boldLabel);
        Field("exposure", "Exposure");
        Field("aperture", "Aperture", "Lens diameter in world units. Zero keeps everything sharp.");
        Field("focusDistance", "Focus distance");
        Field("skyStrength", "Sky strength", "Sky illumination for bounced light and reflections. Empty background remains black.");
        Field("moveSpeed", "Camera move speed");
        Field("mouseSensitivity", "Mouse sensitivity");
        bool rendererChanged = EditorGUI.EndChangeCheck();
        settings.ApplyModifiedProperties();

        var camera = renderer.GetComponent<Camera>();
        EditorGUI.BeginChangeCheck();
        float fov = EditorGUILayout.Slider("Field of view", camera.fieldOfView, 15, 100);
        if (EditorGUI.EndChangeCheck())
        {
            Undo.RecordObject(camera, "Change camera field of view");
            camera.fieldOfView = fov;
            EditorUtility.SetDirty(camera);
            rendererChanged = true;
        }
        if (rendererChanged) SceneView.RepaintAll();

        EditorGUILayout.Space(8);
        EditorGUILayout.LabelField("Performance", EditorStyles.boldLabel);
        if (EditorApplication.isPlaying)
        {
            EditorGUILayout.LabelField("Active backend", renderer.backend == CudaRenderer.RenderBackend.Embree ? "Embree CPU multi" : renderer.backend == CudaRenderer.RenderBackend.Cuda ? "CUDA GPU" : renderer.backend == CudaRenderer.RenderBackend.CpuMulti ? "CPU multi" : "CPU single");
            EditorGUILayout.LabelField("Render size", $"{renderer.width} x {renderer.height}");
            EditorGUILayout.LabelField("Accumulated samples / pixel", renderer.SampleCount.ToString());
            if (renderer.denoise) EditorGUILayout.LabelField("Displayed samples / pixel", renderer.DisplayedSamples.ToString());
            EditorGUILayout.LabelField("Frames per second", renderer.FramesPerSecond.ToString("F1"));
            EditorGUILayout.LabelField("Ray tracing compute", $"{renderer.ComputeMilliseconds:F2} ms");
            EditorGUILayout.LabelField("Raw call with readback", $"{renderer.RenderCallMilliseconds:F2} ms");
            if (renderer.denoise)
                EditorGUILayout.LabelField("Last AI pass with transfer", $"{renderer.DenoiseMilliseconds:F2} ms");
            if (!string.IsNullOrEmpty(renderer.DenoiseError))
                EditorGUILayout.HelpBox(renderer.DenoiseError + " Displaying the raw image. Toggle AI denoising to retry.", MessageType.Warning);
            EditorGUILayout.LabelField("Triangles", renderer.TriangleCount.ToString());
            if (renderer.backend == CudaRenderer.RenderBackend.Embree)
                EditorGUILayout.LabelField("Embree build", $"{renderer.EmbreeBuildMilliseconds:F2} ms");
            EditorGUILayout.LabelField("Our BVH nodes / depth", $"{renderer.BvhNodeCount} / {renderer.BvhDepth}");
            EditorGUILayout.LabelField("Our BVH build", $"{renderer.BvhBuildMilliseconds:F2} ms");
            if (renderer.backend != CudaRenderer.RenderBackend.Cuda) EditorGUILayout.LabelField("CPU threads", renderer.CpuThreadCount.ToString());
        }
        else EditorGUILayout.HelpBox("Enter Play mode to see Game rendering performance.", MessageType.None);
        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Reset image"))
        {
            renderer.ResetAccumulation();
            CudaScenePreview.Refresh();
        }
        if (GUILayout.Button("Refresh scene"))
        {
            if (EditorApplication.isPlaying) renderer.UploadScene();
            CudaScenePreview.Refresh();
        }
        EditorGUILayout.EndHorizontal();
        EditorGUILayout.Space(6);
        bool preview = EditorGUILayout.Toggle("CUDA Scene preview", CudaScenePreview.Enabled);
        if (preview != CudaScenePreview.Enabled) CudaScenePreview.Enabled = preview;
        if (preview) EditorGUILayout.LabelField(CudaScenePreview.Status, EditorStyles.wordWrappedMiniLabel);

        EditorGUILayout.Space(8);
        showMaterial = EditorGUILayout.Foldout(showMaterial, "Selected object's material", true);
        if (showMaterial)
        {
            var surface = Selection.activeGameObject == null ? null : Selection.activeGameObject.GetComponent<RayTraceObject>();
            if (surface == null) EditorGUILayout.HelpBox("Select a model, wall or light with a Ray Trace Object component.", MessageType.None);
            else
            {
                Editor.CreateCachedEditor(surface, null, ref materialEditor);
                EditorGUI.BeginChangeCheck();
                materialEditor.OnInspectorGUI();
                if (EditorGUI.EndChangeCheck())
                {
                    if (EditorApplication.isPlaying) renderer.UploadScene();
                    CudaScenePreview.Refresh();
                }
            }
        }
        EditorGUILayout.EndScrollView();
    }
}
