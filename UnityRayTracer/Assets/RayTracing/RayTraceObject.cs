using UnityEngine;

// Only meshes with this component are sent to the native renderer.
[RequireComponent(typeof(MeshFilter))]
public sealed class RayTraceObject : MonoBehaviour
{
    public Color color = Color.white;
    public Color specularColor = Color.white;
    [Range(0, 1)] public float specularProbability = .05f;
    [Range(0, 1)] public float smoothness = .65f;
    public bool emissive;
    [Min(0.01f)] public float emissionStrength = 30f;
    public bool oneSided;
}
