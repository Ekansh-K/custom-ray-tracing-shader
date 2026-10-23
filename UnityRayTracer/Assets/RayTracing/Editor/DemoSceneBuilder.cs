using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class DemoSceneBuilder
{
    private const string Models = "Assets/RayTracing/Models/";
    private const string Materials = "Assets/RayTracing/Demos/Materials/";

    [MenuItem("Ray tracer/Build demo scenes")]
    public static void Build()
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode first.");
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        string original = EditorSceneManager.GetActiveScene().path;
        bool preview = CudaScenePreview.Enabled;
        CudaScenePreview.Enabled = false;
        Directory.CreateDirectory(Materials);
        AssetDatabase.Refresh();
        try
        {
            for (int i = 1; i < DemoScenes.Paths.Length; i++)
            {
                // Re-running the builder never replaces rooms the user has edited.
                if (File.Exists(DemoScenes.Paths[i])) continue;
                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                if (i == 1) Dragon();
                if (i == 2) Gallery();
                if (i == 3) DepthOfField();
                if (i == 4) ManySuzannes();
                EditorSceneManager.SaveScene(scene, DemoScenes.Paths[i]);
            }
            // Keep any unrelated build scenes already present.
            var existing = EditorBuildSettings.scenes.ToList();
            foreach (string path in DemoScenes.Paths)
                if (!existing.Any(s => s.path == path)) existing.Add(new EditorBuildSettingsScene(path, true));
            EditorBuildSettings.scenes = existing.ToArray();
            AssetDatabase.SaveAssets();
        }
        finally
        {
            if (!string.IsNullOrEmpty(original)) EditorSceneManager.OpenScene(original, OpenSceneMode.Single);
            EditorUtility.UnloadUnusedAssetsImmediate();
            CudaScenePreview.Enabled = preview;
        }
    }

    private static RayTraceObject Surface(string name, PrimitiveType type, Vector3 position, Vector3 scale,
        Color color, float specular = 0, float smoothness = .65f)
    {
        var go = GameObject.CreatePrimitive(type);
        UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
        go.name = name; go.transform.position = position; go.transform.localScale = scale;
        return Finish(go, color, specular, smoothness);
    }
    private static RayTraceObject Model(string name, Mesh mesh, Vector3 position, float scale,
        Color color, float specular, float smoothness)
    {
        var go = new GameObject(name);
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        go.AddComponent<MeshRenderer>();
        go.transform.position = position; go.transform.localScale = Vector3.one * scale;
        return Finish(go, color, specular, smoothness);
    }
    private static RayTraceObject Finish(GameObject go, Color color, float specular, float smoothness)
    {
        string path = Materials + ColorUtility.ToHtmlStringRGB(color) + ".mat";
        var material = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (material == null)
        {
            material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
            material.SetColor("_BaseColor", color); AssetDatabase.CreateAsset(material, path);
        }
        go.GetComponent<MeshRenderer>().sharedMaterial = material;
        var surface = go.AddComponent<RayTraceObject>();
        surface.color = color; surface.specularProbability = specular; surface.smoothness = smoothness;
        return surface;
    }
    private static RayTraceObject Quad(string name, Vector3 center, Vector3 normal, Vector2 size, Color color,
        float specular = 0, float smoothness = .65f)
    {
        var result = Surface(name, PrimitiveType.Quad, center, new Vector3(size.x, size.y, 1), color, specular, smoothness);
        result.transform.rotation = Quaternion.FromToRotation(Vector3.back, normal);
        return result;
    }
    private static void Panel(string name, Vector3 center, Vector3 normal, Vector2 size, Color color, float strength)
    {
        var light = Quad(name, center, normal, size, color);
        light.emissive = true; light.emissionStrength = strength; light.oneSided = true;
    }
    private static CudaRenderer Camera(Vector3 position, Vector3 target, float fov = 48, int bounces = 6)
    {
        var go = new GameObject("Main Camera"); go.tag = "MainCamera";
        var camera = go.AddComponent<Camera>(); camera.fieldOfView = fov;
        go.transform.position = position; go.transform.LookAt(target);
        go.AddComponent<AudioListener>();
        var renderer = go.AddComponent<CudaRenderer>();
        renderer.maxBounces = bounces; renderer.skyStrength = 0;
        renderer.focusDistance = Vector3.Distance(position, target);
        return renderer;
    }
    private static void Room(float halfWidth, float height, float halfDepth)
    {
        Color white = new Color(.78f, .78f, .78f);
        Quad("Floor", Vector3.zero, Vector3.up, new Vector2(halfWidth * 2, halfDepth * 2), white, .15f, .85f);
        Quad("Ceiling", new Vector3(0, height, 0), Vector3.down, new Vector2(halfWidth * 2, halfDepth * 2), white);
        Quad("Back wall", new Vector3(0, height / 2, halfDepth), Vector3.back, new Vector2(halfWidth * 2, height), new Color(.18f, .25f, .4f));
        Quad("Left wall", new Vector3(-halfWidth, height / 2, 0), Vector3.right, new Vector2(halfDepth * 2, height), new Color(.75f, .16f, .12f));
        Quad("Right wall", new Vector3(halfWidth, height / 2, 0), Vector3.left, new Vector2(halfDepth * 2, height), new Color(.12f, .6f, .3f));
        var front = Quad("Front wall - one sided", new Vector3(0, height / 2, -halfDepth), Vector3.forward, new Vector2(halfWidth * 2, height), white);
        front.oneSided = true;
        Panel("Ceiling area light", new Vector3(0, height - .02f, -.3f), Vector3.down, new Vector2(halfWidth, halfDepth * .8f), Color.white, 12);
    }

    private static void Dragon()
    {
        // Episode02 DragonBox layout and original 80K mesh. Material values are
        // translated into our diffuse/Phong model, so the two renderers differ.
        var dragonAsset = AssetDatabase.LoadAssetAtPath<GameObject>(Models + "Dragon_80K.obj");
        var mesh = dragonAsset.GetComponentInChildren<MeshFilter>().sharedMesh;
        var dragon = Model("Dragon - Episode02 mesh", mesh, new Vector3(-.04f, 1.43f, -.02f), 5.350002f,
            new Color(.9622642f, .5221968f, .06808472f), .274f, .482f);
        dragon.transform.rotation = new Quaternion(0, .48016095f, 0, .8771804f);
        Quad("Ceiling", new Vector3(0, 4, 0), Vector3.down, new Vector2(6, 4), new Color(.1462264f, .56306875f, 1), .024f, .728f);
        Quad("Left wall", new Vector3(-2.925f, 2, 0), Vector3.right, new Vector2(4, 4), new Color(.8679245f, .2538512f, .16785331f), .765f, .979f);
        Quad("Right wall", new Vector3(2.925f, 2, 0), Vector3.left, new Vector2(4, 4), new Color(.10980392f, .85882354f, .29803923f), .765f, .979f);
        Quad("Back mirror", new Vector3(0, 2, 2), Vector3.back, new Vector2(6, 4), new Color(.009433985f, .009433985f, .009433985f), .765f, .979f);
        var front = Quad("Front wall - one sided", new Vector3(0, 2, -1.999f), Vector3.forward, new Vector2(6.1f, 4.014f), new Color(.2503595f, .2117647f, .7921569f), .774f, 1);
        front.oneSided = true;
        // Actual tiles use ordinary triangles, without a special checker shader.
        for (int z = 0; z < 8; z++) for (int x = 0; x < 12; x++)
            Quad("Floor tile", new Vector3(-2.75f + x * .5f, 0, -1.75f + z * .5f), Vector3.up, Vector2.one * .5f,
                (x + z) % 2 == 0 ? new Color(.03f, .03f, .03f) : new Color(.78f, .78f, .78f), .086f, 1);
        Panel("White rectangular light", new Vector3(0, 3.9428f, -.621f), Vector3.down, new Vector2(2.2916377f, .63445795f), Color.white, 15);
        Camera(new Vector3(0, 1.99f, -5.85f), new Vector3(0, 1.99f, 0), 60, 8);
    }
    private static void Gallery()
    {
        Room(4, 4, 3);
        Quad("Large mirror", new Vector3(0, 2, 2.98f), Vector3.back, new Vector2(5.5f, 3.2f), Color.black, 1, 1);
        var colors = new[] { new Color(.85f, .18f, .08f), new Color(.9f, .65f, .2f), Color.white, Color.white };
        for (int i = 0; i < 4; i++)
        {
            var ball = Surface(new[] { "Matte", "Glossy", "Polished", "Mirror" }[i], PrimitiveType.Sphere,
                new Vector3(-2.55f + i * 1.7f, .85f, 0), Vector3.one * 1.5f, colors[i], i == 0 ? 0 : i == 1 ? .55f : 1, i == 1 ? .75f : i == 2 ? .94f : 1);
            if (i == 1) ball.specularColor = new Color(1, .76f, .35f);
        }
        Panel("Warm side panel", new Vector3(-3.98f, 2, 0), Vector3.right, new Vector2(1.5f, 2), new Color(1, .65f, .3f), 5);
        Camera(new Vector3(0, 2.6f, -8.8f), new Vector3(0, 1.4f, .7f), 48, 8);
    }
    private static void DepthOfField()
    {
        Room(3, 4, 5);
        var monkey = AssetDatabase.LoadAssetAtPath<Mesh>(Models + "Suzanne.asset");
        for (int i = 0; i < 5; i++)
        {
            float z = -2.5f + i * 1.5f;
            float x = (i % 2 == 0 ? -.8f : .8f);
            Model(i == 0 ? "Focus subject" : "Distant Suzanne", monkey, new Vector3(x, .95f, z), .65f,
                Color.HSVToRGB(i * .15f, .7f, .9f), .25f, .85f);
            Surface("Plinth", PrimitiveType.Cube, new Vector3(x, .2f, z), new Vector3(1.2f, .4f, .9f), new Color(.35f, .35f, .4f));
        }
        var renderer = Camera(new Vector3(-.8f, 1.5f, -7), new Vector3(-.8f, .95f, -2.5f), 44);
        renderer.aperture = .22f;
    }
    private static void ManySuzannes()
    {
        Room(7, 4, 7);
        var monkey = AssetDatabase.LoadAssetAtPath<Mesh>(Models + "Suzanne.asset");
        for (int z = 0; z < 7; z++) for (int x = 0; x < 9; x++)
        {
            var item = Model("Suzanne " + (z * 9 + x + 1), monkey, new Vector3((x - 4) * 1.35f, .8f, (z - 3) * 1.65f), .5f,
                Color.HSVToRGB((x + z * 2) / 20f, .6f, .85f), .3f, .82f);
            item.transform.rotation = Quaternion.Euler(0, (x - 4) * 5, 0);
        }
        Camera(new Vector3(0, 3.7f, -13), new Vector3(0, .9f, 0), 58, 4);
    }
}
