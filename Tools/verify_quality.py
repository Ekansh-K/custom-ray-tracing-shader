#!/usr/bin/env python3
"""Compare PCG, scrambled Sobol and OptiX against an independent high-sample reference.

Example: python3 Tools/verify_quality.py Benchmarks/embree/scene.json scratch/quality/suzanne \
  --before scratch/quality/before/renderer.so
Close Unity while measuring. Output includes raw/filtered PNGs and equal-sample,
end-to-end quality measurements. This does not modify Unity scenes.
"""
import argparse
import ctypes as c
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import statistics
import time
import numpy as np
from PIL import Image
from native_api import Triangle, Camera, Settings, LIBRARY, load_library, unpack

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('snapshot',type=Path)
parser.add_argument('output',type=Path)
parser.add_argument('--before',type=Path,required=True)
parser.add_argument('--size',default='320x180')
parser.add_argument('--reference-samples',type=int,default=2048)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
assert not (args.output/'results.json').exists(),'Use a fresh output directory'
snapshot=args.snapshot.read_bytes()
if args.snapshot.suffix=='.gz': snapshot=gzip.decompress(snapshot)
scene=json.loads(snapshot)
width,height=map(int,args.size.split('x'))
triangles=(Triangle*len(scene['triangles']))(*(unpack(Triangle,t) for t in scene['triangles']))
camera=unpack(Camera,scene['camera']);camera.aspect=width/height
settings=unpack(Settings,scene['settings'])
old,new=load_library(args.before),load_library()
contexts={};output=(c.c_ubyte*(width*height*4))();ms=c.c_float()
def check(lib,session,ok):
    assert ok,lib.RtLastError(session).decode()
def image(lib,session,samples,seed):
    lib.RtReset(session);settings.seed=seed;count=0;begin=time.perf_counter()
    while count<samples:
        settings.raysPerPixel=min(16,samples-count)
        check(lib,session,lib.RtRender(session,c.byref(camera),c.byref(settings),output,len(output),2,c.byref(ms)))
        count+=settings.raysPerPixel
    return bytes(output),(time.perf_counter()-begin)*1000

def save(name,data):
    Image.frombytes('RGBA',(width,height),data).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(args.output/(name+'.png'))

def rgb(data):return np.frombuffer(data,dtype=np.uint8).reshape(height,width,4)[:,:,:3].astype(np.float64)
def rmse(data,reference):return float(np.sqrt(np.mean((rgb(data)-rgb(reference))**2)))
try:
    for label,lib in [('PCG',old),('Sobol',new)]:
        session=lib.RtCreate();contexts[label]=(lib,session)
        check(lib,session,lib.RtInitialize(session,triangles,len(triangles),width,height))
    lib,session=contexts['Sobol']
    reference,reference_ms=image(lib,session,args.reference_samples,20261023)
    save('reference',reference)
    print('Independent reference finished',flush=True)
    rows=[]
    for samples in (16,64):
        for seed in (42,73,314,1234):
            for label in ('PCG','Sobol'):
                lib,session=contexts[label]
                raw,render_ms=image(lib,session,samples,seed)
                row=dict(method=label,samples=samples,seed=seed,render_ms=render_ms,rmse=rmse(raw,reference))
                rows.append(row)
                if seed==42:save(label.lower()+f'-{samples}',raw)
                if label=='Sobol':
                    begin=time.perf_counter()
                    check(lib,session,lib.RtDenoise(session,output,len(output),c.byref(ms)))
                    ai=bytes(output);wall=(time.perf_counter()-begin)*1000
                    assert lib.RtSampleCount(session)==samples,'Denoiser changed sample count'
                    rows.append(dict(method='Sobol + AI',samples=samples,seed=seed,render_ms=render_ms,
                                     denoise_ms=wall,total_ms=render_ms+wall,rmse=rmse(ai,reference)))
                    if seed==42:save('ai-'+str(samples),ai)
        print(f'{samples} samples, four seeds finished',flush=True)
    summaries=[]
    for samples in (16,64):
        for method in ('PCG','Sobol','Sobol + AI'):
            selected=[r for r in rows if r['method']==method and r['samples']==samples]
            summaries.append(dict(method=method,samples=samples,
                mean_rmse=statistics.mean(r['rmse'] for r in selected),
                median_render_ms=statistics.median(r['render_ms'] for r in selected),
                median_total_ms=statistics.median(r.get('total_ms',r['render_ms']) for r in selected)))
    # Filtering must not alter the next raw sample, and repeated inference must
    # not consume samples. Verify with two independently accumulated contexts.
    settings.seed=42;settings.raysPerPixel=1
    a=new.RtCreate();b=new.RtCreate()
    try:
        for ctx in (a,b):check(new,ctx,new.RtInitialize(ctx,triangles,len(triangles),width,height))
        for i in range(8):
            for ctx in (a,b):check(new,ctx,new.RtRender(ctx,c.byref(camera),c.byref(settings),output,len(output),2,c.byref(ms)))
        for _ in range(2):check(new,a,new.RtDenoise(a,output,len(output),c.byref(ms)))
        assert new.RtSampleCount(a)==8
        check(new,a,new.RtRender(a,c.byref(camera),c.byref(settings),output,len(output),2,c.byref(ms)));after=bytes(output)
        check(new,b,new.RtRender(b,c.byref(camera),c.byref(settings),output,len(output),2,c.byref(ms)))
        assert bytes(output)==after,'AI output fed back into raw accumulation'
        assert not new.RtDenoise(a,output,len(output)-1,c.byref(ms))
        new.RtShutdown(a)
        assert not new.RtDenoise(a,output,len(output),c.byref(ms))
    finally:
        new.RtDestroy(a);new.RtDestroy(b)
    result=dict(captured_at=datetime.now(timezone.utc).isoformat(),
        snapshot=str(args.snapshot),snapshot_sha256=hashlib.sha256(snapshot).hexdigest(),
        before_library_sha256=hashlib.sha256(args.before.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
        width=width,height=height,bounces=settings.maxBounces,
        reference_samples=args.reference_samples,reference_seed=20261023,reference_ms=reference_ms,
        notes='RGB display-space RMSE against independent Sobol reference; four seeds per low-sample method. CUDA render wall time includes raw readback. AI total includes inference, transfers and first-use model initialization when incurred. No fixed speedup assumed.',
        summaries=summaries,measurements=rows,raw_accumulation_unchanged=True)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)
    print('PASS AI preserves sample count, raw accumulation, independent sessions, and validates output sizes')
finally:
    for lib,session in contexts.values():lib.RtDestroy(session)
