"""Check progressive path tracing on the actual CPU, OpenMP and CUDA backends."""
import ctypes as c
import math
import os

from native_api import Vector, Triangle, Camera, Settings, load_library
lib = load_library()
lit = Settings(8, 1, 0, 3, 0, 42, 1)
session, preview = lib.RtCreate(), lib.RtCreate()

def check(ok, ctx=session):
    assert ok, lib.RtLastError(ctx).decode()

def initialize(triangles, w, h, ctx=session):
    packed = (Triangle * len(triangles))(*triangles)
    check(lib.RtInitialize(ctx, packed, len(packed), w, h), ctx)

def render(camera, w, h, backend=2, settings=lit, ctx=session):
    output = (c.c_ubyte * (w*h*4))()
    ms = c.c_float()
    check(lib.RtRender(ctx, c.byref(camera), c.byref(settings), output, len(output), backend, c.byref(ms)), ctx)
    assert ms.value >= 0 and math.isfinite(ms.value)
    return bytes(output)

def agree(camera, w, h, settings, samples=4):
    def batch(backend):
        lib.RtReset(session)
        for _ in range(samples): image = render(camera, w, h, backend, settings)
        assert lib.RtSampleCount(session) == samples * settings.raysPerPixel
        return image
    cpu = batch(0)
    assert lib.RtCpuThreadCount(session) == 1
    multi = batch(1)
    threads = lib.RtCpuThreadCount(session)
    requested = os.environ.get('OMP_NUM_THREADS')
    if requested and os.environ.get('OMP_DYNAMIC', 'FALSE').upper() == 'FALSE':
        assert threads == int(requested.split(',')[0])
    elif len(os.sched_getaffinity(0)) > 1:
        assert threads > 1
    gpu = batch(2)
    assert lib.RtCpuThreadCount(session) == 0
    assert max(abs(a-b) for a, b in zip(cpu, multi)) <= 1
    assert max(abs(a-b) for a, b in zip(cpu, gpu)) <= 1
    embree = batch(3)
    assert lib.RtCpuThreadCount(session) == threads
    assert max(abs(a-b) for a, b in zip(cpu, embree)) <= 1
    return gpu

def triangle(a, b, d, color=(1,1,1), emission=0):
    e1 = tuple(y-x for x, y in zip(a, b))
    e2 = tuple(y-x for x, y in zip(a, d))
    n = (e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0])
    length = math.sqrt(sum(x*x for x in n))
    normal = Vector(*(x/length for x in n))
    return Triangle(Vector(*a), Vector(*e1), Vector(*e2), Vector(*color), emission, 0, normal, normal, normal, 0, Vector(1,1,1), .7, 0)

def quad(x0, x1, y0, y1, z, **kwargs):
    return [triangle((x0,y0,z), (x1,y0,z), (x1,y1,z), **kwargs),
            triangle((x0,y0,z), (x1,y1,z), (x0,y1,z), **kwargs)]

def centre(image, w, h):
    offset = ((h//2)*w+w//2)*4
    return tuple(image[offset:offset+4])

def accumulate(camera, w, h, samples, settings=lit):
    lib.RtReset(session)
    for _ in range(samples): image = render(camera,w,h,settings=settings)
    return image

def rmse(a,b):
    values=[(x-y)**2 for i,(x,y) in enumerate(zip(a,b)) if i%4!=3]
    return math.sqrt(sum(values)/len(values))

try:
    red = quad(-1,1,-1,1,0,color=(1,0,0),emission=1)
    for w,h in [(37,25),(17,19)]:
        camera=Camera(Vector(0,0,-2),Vector(1,0,0),Vector(0,1,0),Vector(0,0,1),.5,w/h,1,0)
        for sided,inside,expected in [(0,False,(208,0,0,255)),(1,False,(0,0,0,255)),(1,True,(208,0,0,255))]:
            for t in red: t.oneSided=sided
            camera.origin.z=2 if inside else -2
            camera.forward.z=-1 if inside else 1
            initialize(red,w,h)
            image=agree(camera,w,h,lit)
            assert centre(image,w,h)==expected,(centre(image,w,h),expected)
            assert all(alpha==255 for alpha in image[3::4])
        print(f'PASS known emissive colour, culling, all backends, non-block-aligned size {w}x{h}')
    initialize(red,17,19)
    reference=render(camera,17,19)
    green=quad(-1,1,-1,1,0,color=(0,1,0),emission=1)
    initialize(green,7,9,preview)
    other=Camera(Vector(0,0,-2),Vector(1,0,0),Vector(0,1,0),Vector(0,0,1),.5,7/9,1,0)
    assert centre(render(other,7,9,ctx=preview),7,9)==(0,208,0,255)
    lib.RtShutdown(preview)
    assert render(camera,17,19)==reference
    print('PASS independent geometry, sizes, accumulation buffers and session shutdown')

    w,h=33,25
    camera=Camera(Vector(0,0,-3),Vector(1,0,0),Vector(0,1,0),Vector(0,0,1),.3,w/h,1,0)
    floor=quad(-2,2,-2,2,0)
    light=quad(.8,1.2,-.2,.2,-2,emission=30)
    blocker=quad(.25,.75,-.5,.5,-1)
    initialize(floor+light,w,h)
    unblocked=agree(camera,w,h,lit,16)
    initialize(floor+light+blocker,w,h)
    blocked=agree(camera,w,h,lit,16)
    assert centre(unblocked,w,h)[0]>centre(blocked,w,h)[0]+20
    one=accumulate(camera,w,h,1)
    sixteen=accumulate(camera,w,h,16)
    reference=accumulate(camera,w,h,256)
    assert rmse(sixteen,reference)<rmse(one,reference)*.6
    print(f'PASS area light, occlusion and progressive convergence: RMSE {rmse(one,reference):.2f} -> {rmse(sixteen,reference):.2f}')
    mirror=quad(-2,2,-2,2,0)
    for t in mirror: t.specularProbability=1; t.smoothness=1
    behind=quad(-1,1,-1,1,-4,color=(0,1,0),emission=2)
    initialize(mirror+behind,w,h)
    reflected=agree(camera,w,h,lit)
    assert centre(reflected,w,h)[1]>230 and centre(reflected,w,h)[0]==0
    print('PASS perfect reflection reaches a green emitter behind the camera')
    initialize(floor,w,h)
    assert centre(agree(camera,w,h,lit),w,h)==(0,0,0,255)
    lit.skyStrength=1
    assert centre(agree(camera,w,h,lit,16),w,h)[0]>100
    print('PASS diffuse bounce receives sky illumination without an artificial ambient term')
    camera.origin.x=20
    empty=agree(camera,w,h,lit)
    assert all(empty[i:i+4]==bytes((0,0,0,255)) for i in range(0,len(empty),4))
    camera.origin.x=0
    print('PASS empty background is pure black with sky lighting enabled on all backends')
    separate=accumulate(camera,w,h,12)
    for rays in (2,4,6):
        lit.raysPerPixel=rays
        batched=agree(camera,w,h,lit,12//rays)
        assert max(abs(a-b) for a,b in zip(separate,batched))<=1
        assert lib.RtSampleCount(session)==12
    lit.raysPerPixel=1
    print('PASS batched 2/4/6 rays reproduce twelve separate samples on CPU, OpenMP and CUDA')
    camera.orthographic=1
    agree(camera,w,h,lit)
    camera.orthographic=0
    lit.aperture=.2
    agree(camera,w,h,lit)
    print('PASS perspective, orthographic and depth-of-field rays agree across backends')
    lib.RtReset(session)
    a=render(camera,w,h)
    render(camera,w,h)
    assert lib.RtSampleCount(session)==2
    lib.RtReset(session)
    assert render(camera,w,h)==a and lib.RtSampleCount(session)==1
    camera.origin.x+=.1
    render(camera,w,h)
    assert lib.RtSampleCount(session)==1
    lit.exposure=2
    render(camera,w,h)
    assert lib.RtSampleCount(session)==1
    render(camera,w,h,1)
    assert lib.RtSampleCount(session)==1
    print('PASS deterministic reset and automatic reset on camera, settings and backend changes')
    output=(c.c_ubyte*(w*h*4))()
    for backend in (-1,4):
        assert not lib.RtRender(session,c.byref(camera),c.byref(lit),output,len(output),backend,c.byref(c.c_float()))
    assert not lib.RtRender(session,c.byref(camera),c.byref(lit),output,4,2,c.byref(c.c_float()))
    render(camera,w,h)
    lib.RtShutdown(session)
    assert not lib.RtRender(session,c.byref(camera),c.byref(lit),output,len(output),2,c.byref(c.c_float()))
    print('PASS invalid backends, undersized buffers and rendering after shutdown rejected')
finally:
    lib.RtDestroy(session)
    lib.RtDestroy(preview)
