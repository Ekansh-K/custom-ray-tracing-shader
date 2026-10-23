"""The native renderer's plain C ABI, shared by verification and benchmarks."""
import ctypes as c
import os
from pathlib import Path

class Vector(c.Structure):
    _fields_ = [(axis, c.c_float) for axis in 'xyz']

class Triangle(c.Structure):
    _fields_ = [(name, Vector) for name in ('origin', 'edge1', 'edge2', 'color')]
    _fields_ += [('emission', c.c_float), ('oneSided', c.c_int)]
    _fields_ += [(name, Vector) for name in ('normal0', 'normal1', 'normal2')]
    _fields_ += [('mesh', c.c_int), ('specularColor', Vector), ('smoothness', c.c_float), ('specularProbability', c.c_float)]

class Camera(c.Structure):
    _fields_ = [(name, Vector) for name in ('origin', 'right', 'up', 'forward')]
    _fields_ += [('tanHalfFov', c.c_float), ('aspect', c.c_float), ('orthographicSize', c.c_float), ('orthographic', c.c_int)]

class Settings(c.Structure):
    _fields_ = [('maxBounces', c.c_int)]
    _fields_ += [(name, c.c_float) for name in ('exposure', 'aperture', 'focusDistance', 'skyStrength')]
    _fields_ += [('seed', c.c_uint), ('raysPerPixel', c.c_int)]

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / 'UnityRayTracer/Assets/Plugins/x86_64/libCudaRayTracer.so'
assert (c.sizeof(Triangle), c.sizeof(Camera), c.sizeof(Settings)) == (116, 64, 28)

def load_library(path=None):
    path = path or os.environ.get("RAYTRACER_LIBRARY", LIBRARY)
    lib = c.CDLL(str(Path(path).resolve()))
    lib.RtCreate.restype = c.c_void_p
    lib.RtDestroy.argtypes = lib.RtShutdown.argtypes = lib.RtReset.argtypes = lib.RtSampleCount.argtypes = [c.c_void_p]
    lib.RtInitialize.argtypes = [c.c_void_p, c.POINTER(Triangle), c.c_int, c.c_int, c.c_int]
    lib.RtRender.argtypes = [c.c_void_p, c.POINTER(Camera), c.POINTER(Settings), c.POINTER(c.c_ubyte), c.c_int, c.c_int, c.POINTER(c.c_float)]
    lib.RtLastError.argtypes = [c.c_void_p]
    lib.RtLastError.restype = c.c_char_p
    lib.RtCpuThreadCount.argtypes = [c.c_void_p]
    if hasattr(lib, "RtDenoise"):
        lib.RtDenoise.argtypes = [c.c_void_p, c.POINTER(c.c_ubyte), c.c_int, c.POINTER(c.c_float)]
    if hasattr(lib, "RtAccumulationVersion"):
        lib.RtAccumulationVersion.argtypes = [c.c_void_p]
        lib.RtAccumulationVersion.restype = c.c_uint
    return lib

def unpack(structure, values):
    """Read Unity's JSON snapshot without changing its float inputs."""
    return structure(**{name: Vector(**values[name]) if field is Vector else values[name]
                        for name, field in structure._fields_})
