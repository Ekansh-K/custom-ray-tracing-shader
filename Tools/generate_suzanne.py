"""Run with Blender --background --python Tools/generate_suzanne.py."""
import json
from pathlib import Path
import bpy

bpy.ops.mesh.primitive_monkey_add()
mesh = bpy.context.object.data
mesh.calc_loop_triangles()
# Blender is Z-up. Reflect into Unity's Y-up coordinates and reverse winding.
data = {
    "vertices": [{"x": v.co.x, "y": v.co.z, "z": v.co.y} for v in mesh.vertices],
    "triangles": [i for t in mesh.loop_triangles for i in reversed(t.vertices)],
}
target = Path(__file__).resolve().parents[1] / "UnityRayTracer/Assets/RayTracing/Models/Suzanne.json"
target.write_text(json.dumps(data, separators=(",", ":")) + "\n")
print(f"Suzanne: {len(mesh.vertices)} vertices, {len(mesh.loop_triangles)} triangles")
