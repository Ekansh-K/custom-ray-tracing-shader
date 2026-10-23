#!/usr/bin/env python3
"""Compare renderer libraries using an identical Unity scene snapshot, outside Unity.

Example: OMP_NUM_THREADS=12 python3 Tools/benchmark_native.py scene.json results --library renderer.so
Keep Unity closed while measuring. Timings include all ray tracing and accumulation,
with native call/readback reported separately. Texture upload/display are excluded.
"""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from PIL import Image
from native_api import Triangle, Camera, Settings, LIBRARY, load_library, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    parser.add_argument('--backend',type=int,choices=(0,1,2,3),help='measure one backend only')
    parser.add_argument('--sizes', default='320x180,1280x720')
    parser.add_argument('--bounces', default='3,8')
    parser.add_argument('--passes', type=int, default=5, help='measured passes per backend, after one warm-up')
    args = parser.parse_args()
    if (args.output / 'results.json').exists():
        raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    scene_bytes = args.snapshot.read_bytes()
    (args.output / 'scene.json').write_bytes(scene_bytes)
    scene = json.loads(scene_bytes)
    triangles = (Triangle * len(scene['triangles']))(*(unpack(Triangle,t) for t in scene['triangles']))
    camera, settings = unpack(Camera,scene['camera']), unpack(Settings,scene['settings'])
    lib = load_library(args.library)
    session = lib.RtCreate()
    results = []

    def check(ok):
        if not ok: raise RuntimeError(lib.RtLastError(session).decode())
    try:
        for size in args.sizes.split(','):
            width, height = map(int, size.split('x'))
            camera.aspect = width / height
            for bounces in map(int, args.bounces.split(',')):
                settings.maxBounces = bounces
                begin = time.perf_counter()
                check(lib.RtInitialize(session,triangles,len(triangles),width,height))
                initialization = (time.perf_counter()-begin)*1000
                pixels = (c.c_ubyte * (width*height*4))()
                case = {'width':width,'height':height,'bounces':bounces,
                        'rays_per_pixel':settings.raysPerPixel,'initialize_ms':initialization,'backends':[]}
                if hasattr(lib,'RtBvhInfo'):
                    nodes, depth, build = c.c_int(),c.c_int(),c.c_float()
                    lib.RtBvhInfo.argtypes = [c.c_void_p,c.POINTER(c.c_int),c.POINTER(c.c_int),c.POINTER(c.c_float)]
                    lib.RtBvhInfo(session,c.byref(nodes),c.byref(depth),c.byref(build))
                    case['bvh'] = {'nodes':nodes.value,'depth':depth.value,'build_ms':build.value}
                reference = None
                for mode,name in enumerate(('CPU single','CPU multi','CUDA GPU','Embree CPU multi')):
                    if mode == 3 and not hasattr(lib,'RtEmbreeBuildMilliseconds'): continue
                    if args.backend is not None and mode != args.backend: continue
                    if mode == 3:
                        setup_ms = c.c_float()
                        check(lib.RtRender(session,c.byref(camera),c.byref(settings),pixels,len(pixels),mode,c.byref(setup_ms)))
                    lib.RtReset(session)
                    compute, calls = [], []
                    ms = c.c_float()
                    for i in range(args.passes+1):
                        begin = time.perf_counter()
                        check(lib.RtRender(session,c.byref(camera),c.byref(settings),pixels,len(pixels),mode,c.byref(ms)))
                        elapsed = (time.perf_counter()-begin)*1000
                        if i: compute.append(ms.value); calls.append(elapsed)
                    image = bytes(pixels)
                    key = f'{width}x{height}-b{bounces}-backend-{mode}'
                    (args.output / (key+'.rgba')).write_bytes(image)
                    Image.frombytes('RGBA',(width,height),image).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(args.output/(key+'.png'))
                    maximum = max((abs(a-b) for a,b in zip(reference,image)),default=0) if reference else None
                    if mode == 0: reference = image; maximum = 0
                    errors = [abs(a-b) for i,(a,b) in enumerate(zip(reference,image)) if i%4!=3] if reference else []
                    embree_build = None
                    if hasattr(lib,'RtEmbreeBuildMilliseconds'):
                        lib.RtEmbreeBuildMilliseconds.argtypes = [c.c_void_p]
                        lib.RtEmbreeBuildMilliseconds.restype = c.c_float
                        embree_build = lib.RtEmbreeBuildMilliseconds(session) if mode == 3 else None
                    row = {'backend':name,'compute_median_ms':statistics.median(compute),
                           'call_median_ms':statistics.median(calls),'compute_samples_ms':compute,
                           'call_samples_ms':calls,'cpu_threads':lib.RtCpuThreadCount(session),
                           'samples_per_pixel':lib.RtSampleCount(session),'max_channel_error_vs_cpu':maximum,
                           'raw_image_sha256':hashlib.sha256(image).hexdigest(),
                           'mean_abs_channel_error_vs_cpu':statistics.mean(errors) if errors else None,
                           'rmse_vs_cpu':(statistics.mean(e*e for e in errors)**.5) if errors else None,
                           'channels_over_1_fraction':sum(e>1 for e in errors)/len(errors) if errors else None,
                           'embree_build_ms':embree_build}
                    case['backends'].append(row)
                    print(f'{size}, {bounces} hits, {name}: {row["compute_median_ms"]:.3f} ms, max error {maximum}',flush=True)
                results.append(case)
                (args.output/'results.json').write_text(json.dumps({'snapshot_sha256':hashlib.sha256(scene_bytes).hexdigest(),
                    'triangles':len(triangles),'library':str(args.library.resolve()),
                    'library_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),
                    'omp_threads':os.getenv('OMP_NUM_THREADS'),'measured_passes':args.passes,'warmup_passes':1,
                    'method':'Standalone native render calls; same seeded paths per backend, one warm-up followed by measured passes. CUDA event compute time excludes readback; native call includes it. No Unity texture upload, display, or Editor FPS. Initialization and BVH build are separate.',
                    'results':results},indent=2)+'\n')
    finally:
        lib.RtDestroy(session)

if __name__ == '__main__': main()
