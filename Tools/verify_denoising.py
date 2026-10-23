#!/usr/bin/env python3
"""Exercise the real OptiX stage on all ray-tracing backends without Unity.

Run with OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE. No scene files are modified.
"""
import ctypes as c
import json
import numpy as np
from native_api import Triangle, Camera, Settings, ROOT, load_library, unpack

lib = load_library()
scene = json.loads((ROOT / 'Benchmarks/embree/scene.json').read_text())
triangles = (Triangle * len(scene['triangles']))(*(unpack(Triangle, t) for t in scene['triangles']))
camera = unpack(Camera, scene['camera'])
settings = unpack(Settings, scene['settings'])
settings.raysPerPixel = 16
ms = c.c_float()
session, other = lib.RtCreate(), lib.RtCreate()

def check(ok, ctx=session):
    assert ok, lib.RtLastError(ctx).decode()

def render(ctx, w, h, backend=2):
    output = (c.c_ubyte * (w*h*4))()
    check(lib.RtRender(ctx, c.byref(camera), c.byref(settings), output, len(output), backend, c.byref(ms)), ctx)
    return output

def denoise(ctx, output):
    before = lib.RtSampleCount(ctx)
    version = lib.RtAccumulationVersion(ctx)
    check(lib.RtDenoise(ctx, output, len(output), c.byref(ms)), ctx)
    assert lib.RtSampleCount(ctx) == before
    assert lib.RtAccumulationVersion(ctx) == version
    assert ms.value > 0
    return bytes(output)

try:
    for w, h in ((97, 55), (37, 25)):
        camera.aspect = w/h
        check(lib.RtInitialize(session, triangles, len(triangles), w, h))
        output = (c.c_ubyte * (w*h*4))()
        assert not lib.RtDenoise(session, output, len(output), c.byref(ms)), 'Must render before denoising'
        reference = None
        for backend in range(4):
            lib.RtReset(session)
            output = render(session, w, h, backend)
            ai = denoise(session, output)
            assert all(a == 255 for a in ai[3::4])
            if reference is None:
                reference = np.frombuffer(ai, dtype=np.uint8).astype(int)
            difference = abs(np.frombuffer(ai, dtype=np.uint8).astype(int)-reference)
            assert difference.max() <= 1, (backend, difference.max())
        assert not lib.RtDenoise(session, output, len(output)-1, c.byref(ms))
        version = lib.RtAccumulationVersion(session)
        camera.origin.x += .01
        output = render(session, w, h)
        assert lib.RtSampleCount(session) == 16
        assert lib.RtAccumulationVersion(session) != version
        version = lib.RtAccumulationVersion(session)
        settings.raysPerPixel = 4
        render(session, w, h)
        assert lib.RtSampleCount(session) == 4
        assert lib.RtAccumulationVersion(session) != version
        camera.origin.x -= .01
        settings.raysPerPixel = 16
        print(f'PASS {w}x{h}: filtered backend agreement, resize, alpha, camera/settings reset and buffer validation')

    # A second AI context must not disturb the first context's raw accumulation.
    camera.aspect = 37/25
    check(lib.RtInitialize(other, triangles, len(triangles), 37, 25), other)
    lib.RtReset(session)
    first = render(session, 37, 25)
    second = render(other, 37, 25)
    assert bytes(first) == bytes(second)
    denoise(session, first)
    denoise(other, second)
    denoise(session, first)
    assert bytes(render(session, 37, 25)) == bytes(render(other, 37, 25))
    lib.RtShutdown(other)
    assert not lib.RtDenoise(other, second, len(second), c.byref(ms))
    # Primary misses must stay exactly black after the neural filter.
    camera.origin.x = 10000
    first = render(session, 37, 25)
    ai = denoise(session, first)
    assert all(value == (255 if i%4 == 3 else 0) for i, value in enumerate(ai))
    print('PASS independent AI contexts, repeated filtering preserves raw accumulation, shutdown and black background')
finally:
    lib.RtDestroy(session)
    lib.RtDestroy(other)
