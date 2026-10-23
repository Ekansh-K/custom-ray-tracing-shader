using UnityEngine;
using UnityEngine.SceneManagement;

// Paths only: this catalogue holds no meshes, materials or inactive demo objects.
public static class DemoScenes
{
    public static readonly string[] Names = {
        "Suzanne room", "Dragon room (Sebastian Lague)", "Material gallery",
        "Depth of field", "Many Suzannes (BVH)"
    };
    public static readonly string[] Paths = {
        "Assets/Scenes/CudaRoom.unity", "Assets/Scenes/DragonRoom.unity",
        "Assets/Scenes/MaterialGallery.unity", "Assets/Scenes/DepthOfField.unity",
        "Assets/Scenes/ManySuzannes.unity"
    };
    public static bool Loading { get; private set; }
    public static int Current => System.Array.IndexOf(Paths, SceneManager.GetActiveScene().path);

    public static void Load(int index)
    {
        if (Loading || index < 0 || index >= Paths.Length || index == Current) return;
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            if (!UnityEditor.SceneManagement.EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            UnityEditor.SceneManagement.EditorSceneManager.OpenScene(Paths[index], UnityEditor.SceneManagement.OpenSceneMode.Single);
            UnityEditor.EditorUtility.UnloadUnusedAssetsImmediate();
            return;
        }
#endif
        Loading = true;
        SceneManager.LoadSceneAsync(Paths[index], LoadSceneMode.Single).completed += _ =>
        {
            // Old cameras dispose their native sessions on unload. Release unused
            // Unity mesh/texture assets as well before enabling another selection.
            Resources.UnloadUnusedAssets().completed += unused => Loading = false;
        };
    }
}
