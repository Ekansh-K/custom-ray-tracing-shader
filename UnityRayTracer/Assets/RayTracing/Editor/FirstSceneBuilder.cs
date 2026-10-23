using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class FirstSceneBuilder
{
    private const string Root = "Assets/RayTracing";
    [Serializable] private class MeshData { public Vector3[] vertices; public int[] triangles; }

    [MenuItem("Ray tracer/Build first scene")]
    public static void Build()
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode first.");
        const string scenePath = "Assets/Scenes/CudaRoom.unity";
        if (EditorSceneManager.GetActiveScene().isDirty)
            throw new InvalidOperationException("Save the current scene before rebuilding.");
        // Keep the tutorial scene and any user scenes. Replace only this generated scene.
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        Mesh monkey = AssetDatabase.LoadAssetAtPath<Mesh>(Root + "/Models/Suzanne.asset");
        if (monkey == null)
        {
            var data = JsonUtility.FromJson<MeshData>(File.ReadAllText(Root + "/Models/Suzanne.json"));
            monkey = new Mesh { name = "Suzanne", vertices = data.vertices, triangles = data.triangles };
            monkey.RecalculateNormals(); monkey.RecalculateBounds();
            AssetDatabase.CreateAsset(monkey, Root + "/Models/Suzanne.asset");
        }

        var room = new GameObject("Room").transform;
        // A 6 x 4 x 6 box. Every side is an editable two-triangle quad.
        var floor = Quad("Floor", room, new Vector3(0, 0, 0), Vector3.up, new Vector2(6, 6), new Color(.78f,.78f,.72f));
        floor.GetComponent<RayTraceObject>().specularProbability = .15f;
        floor.GetComponent<RayTraceObject>().smoothness = .85f;
        Quad("Ceiling", room, new Vector3(0, 4, 0), Vector3.down, new Vector2(6, 6), new Color(.78f,.78f,.72f));
        Quad("Back wall", room, new Vector3(0, 2, 3), Vector3.back, new Vector2(6, 4), new Color(0f,.25f,1f));
        Quad("Left wall", room, new Vector3(-3, 2, 0), Vector3.right, new Vector2(6, 4), new Color(.72f,.12f,.10f));
        Quad("Right wall", room, new Vector3(3, 2, 0), Vector3.left, new Vector2(6, 4), new Color(.12f,.55f,.20f));
        var front = Quad("Front wall - one sided", room, new Vector3(0, 2, -3), Vector3.forward, new Vector2(6, 4), new Color(.78f,.78f,.72f));
        front.GetComponent<RayTraceObject>().oneSided = true;

        var model = new GameObject("Suzanne");
        model.AddComponent<MeshFilter>().sharedMesh = monkey;
        model.AddComponent<MeshRenderer>().sharedMaterial = Material("Suzanne", new Color(.85f,.64f,.30f));
        model.AddComponent<RayTraceObject>().color = new Color(.85f,.64f,.30f);
        model.GetComponent<RayTraceObject>().specularProbability = .35f;
        model.GetComponent<RayTraceObject>().smoothness = .85f;
        model.GetComponent<RayTraceObject>().specularColor = new Color(.95f, .8f, .5f);
        model.transform.position = new Vector3(0, 1.25f, .3f);
        model.transform.rotation = Quaternion.Euler(0, -15, 0);

        var panel = Quad("Ceiling light - white rectangle", room, new Vector3(0, 3.97f, -.3f), Vector3.down, new Vector2(3.2f, 1.8f), Color.white);
        panel.GetComponent<RayTraceObject>().emissive = true;
        panel.GetComponent<RayTraceObject>().emissionStrength = 10;
        panel.GetComponent<RayTraceObject>().oneSided = true;
        var lightObject = new GameObject("Area light preview");
        lightObject.transform.position = panel.transform.position;
        lightObject.transform.rotation = Quaternion.LookRotation(Vector3.down, Vector3.forward);
        lightObject.transform.SetParent(panel.transform, true);
        var light = lightObject.AddComponent<Light>();
        light.type = LightType.Rectangle; light.color = Color.white;
        light.areaSize = new Vector2(3.2f, 1.8f); light.intensity = 2;
        light.lightmapBakeType = LightmapBakeType.Baked;

        var cameraObject = new GameObject("Main Camera");
        cameraObject.tag = "MainCamera";
        var camera = cameraObject.AddComponent<Camera>();
        camera.fieldOfView = 48;
        cameraObject.transform.position = new Vector3(0, 2.3f, -8.5f);
        cameraObject.transform.LookAt(new Vector3(0, 1.9f, 0));
        cameraObject.AddComponent<CudaRenderer>();
        cameraObject.AddComponent<AudioListener>();

        EditorSceneManager.SaveScene(scene, scenePath);
        EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(scenePath, true) };
        Selection.activeGameObject = model;
        SceneView.lastActiveSceneView?.Frame(new Bounds(new Vector3(0, 2, 0), new Vector3(8, 6, 8)), false);
        AssetDatabase.SaveAssets();
    }

    private static Material Material(string name, Color color)
    {
        string path = Root + "/Models/" + name + ".mat";
        var material = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (material != null) return material;
        material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        material.SetColor("_BaseColor", color);
        AssetDatabase.CreateAsset(material, path);
        return material;
    }

    private static GameObject Quad(string name, Transform parent, Vector3 center, Vector3 normal, Vector2 size, Color color)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = name; go.transform.SetParent(parent);
        go.transform.position = center;
        // Unity's built-in quad faces local -Z.
        go.transform.rotation = Quaternion.FromToRotation(Vector3.back, normal);
        go.transform.localScale = new Vector3(size.x, size.y, 1);
        UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
        go.GetComponent<MeshRenderer>().sharedMaterial = Material(name.Replace(" - ", "_"), color);
        var surface = go.AddComponent<RayTraceObject>();
        surface.color = color; surface.specularProbability = 0;
        return go;
    }
}
