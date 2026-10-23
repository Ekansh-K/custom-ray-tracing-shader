#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include "sampling.cuh"
#include <omp.h>

// Match these plain structs to RayTraceSession.cs and Tools/verify_native.py.
struct Triangle {
    float3 origin, edge1, edge2, color;
    float emission; int oneSided;
    float3 normal0, normal1, normal2;
    int mesh;
    float3 specularColor;
    float smoothness, specularProbability;
};
#include "bvh.cuh"

struct CameraData {
    float3 origin, right, up, forward;
    float tanHalfFov, aspect, orthographicSize; int orthographic;
};
struct RenderSettings {
    int maxBounces;
    float exposure, aperture, focusDistance, skyStrength;
    unsigned seed;
    int raysPerPixel;
};
static_assert(sizeof(Triangle)==116 && sizeof(CameraData)==64 && sizeof(RenderSettings)==28);
struct Emitter { int triangle; float cumulativeArea; };
struct Scene {
    const Triangle* triangles;
    const BvhNode* nodes;
    const int* order;
    const Emitter* lights;
    int triangleCount, lightCount;
    float lightArea;
    const unsigned* directions=nullptr;
    const void* embree=nullptr; // CPU only; null for our CPU/CUDA BVH.
};
struct Hit { int index=-1; float distance=1e30f, u=0, v=0; };
#include "embree.cuh"
#include "denoiser.cuh"
struct PixelSum { float3 radiance,albedo,normal; unsigned covered; };
struct Session {
    EmbreeScene embree;
    AIDenoiser denoiser;
    std::string hostError;
    std::vector<Triangle> triangles;
    std::vector<BvhNode> nodes;
    std::vector<int> order;
    std::vector<Emitter> lights;
    std::vector<PixelSum> sum;
    unsigned* deviceDirections=nullptr;
    Triangle* deviceTriangles=nullptr;
    BvhNode* deviceNodes=nullptr;
    int* deviceOrder=nullptr;
    Emitter* deviceLights=nullptr;
    PixelSum* deviceSum=nullptr;
    uchar4* devicePixels=nullptr;
    float lightArea=0, buildMilliseconds=0;
    int bvhDepth=0;
    int width=0, height=0, cpuThreads=0, lastBackend=-1;
    unsigned samples=0, accumulationVersion=0;
    CameraData lastCamera{};
    RenderSettings lastSettings{};
    cudaEvent_t startEvent, stopEvent;
    bool ready=false, eventsReady=false;
    cudaError_t error=cudaSuccess;
};

#define HD __host__ __device__ static inline
constexpr float Pi=3.14159265358979323846f;
HD float3 add(float3 a,float3 b) { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
HD float3 sub(float3 a,float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
HD float3 mul(float3 a,float b) { return make_float3(a.x*b,a.y*b,a.z*b); }
HD float3 product(float3 a,float3 b) { return make_float3(a.x*b.x,a.y*b.y,a.z*b.z); }
HD float dot(float3 a,float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
HD float3 cross(float3 a,float3 b) { return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
HD float3 unit(float3 a) { return mul(a,1/sqrtf(fmaxf(dot(a,a),1e-20f))); }
HD float3 reflect(float3 d,float3 n) { return sub(d,mul(n,2*dot(d,n))); }
HD float linear(float c) { return c<=.04045f ? c/12.92f : powf((c+.055f)/1.055f,2.4f); }
HD float3 toLinear(float3 c) { return make_float3(linear(c.x),linear(c.y),linear(c.z)); }
HD unsigned char display(float c,float exposure) {
    c=1-expf(-fmaxf(c,0)*exposure);
    c=c<=.0031308f ? 12.92f*c : 1.055f*powf(c,1/2.4f)-.055f;
    return (unsigned char)(255*fminf(c,1)+.5f);
}
HD float3 around(float3 axis,float cosine,float angle) {
    float3 tangent=unit(cross(fabsf(axis.y)<.999f ? make_float3(0,1,0) : make_float3(1,0,0),axis));
    float sine=sqrtf(fmaxf(0,1-cosine*cosine));
    return unit(add(mul(axis,cosine),add(mul(tangent,sine*cosf(angle)),mul(cross(axis,tangent),sine*sinf(angle)))));
}
HD float powerWeight(float a,float b) { return a*a/fmaxf(a*a+b*b,1e-30f); }

// All imported shapes use this same Moller-Trumbore triangle intersection.
HD float intersect(float3 origin,float3 ray,const Triangle& t,float& u,float& v) {
    float3 p=cross(ray,t.edge2);
    float determinant=dot(t.edge1,p);
    if (fabsf(determinant)<1e-7f || (t.oneSided && determinant<0)) return -1;
    float inverse=1/determinant;
    float3 offset=sub(origin,t.origin);
    u=dot(offset,p)*inverse;
    if (u<0 || u>1) return -1;
    float3 q=cross(offset,t.edge1);
    v=dot(ray,q)*inverse;
    if (v<0 || u+v>1) return -1;
    float distance=dot(t.edge2,q)*inverse;
    return distance>1e-4f ? distance : -1;
}
// Return entry distance for near-first traversal. Parallel rays on a box face
// remain valid; avoid the NaN produced by zero times an infinite reciprocal.
HD float boxDistance(float3 origin,float3 inverse,const BvhNode& box,float limit) {
    float near=0,far=limit;
    for (int axis=0;axis<3;axis++) {
        float o=(&origin.x)[axis],inv=(&inverse.x)[axis];
        float low=(&box.low.x)[axis],high=(&box.high.x)[axis];
        if (inv==0) { if (o<low || o>high) return FLT_MAX; }
        else {
            float a=(low-o)*inv,b=(high-o)*inv;
#ifdef __CUDA_ARCH__
            near=fmaxf(near,fminf(a,b)); far=fminf(far,fmaxf(a,b));
#else
            // Inline comparisons avoid CPU libm calls. CUDA has hardware min/max.
            if (a>b) { float swap=a; a=b; b=swap; }
            if (a>near) near=a;
            if (b<far) far=b;
#endif
            if (near>far) return FLT_MAX;
        }
    }
    return near;
}
HD Hit collision(float3 origin,float3 ray,Scene scene,float limit=1e30f,bool any=false) {
#ifndef __CUDA_ARCH__
    if (scene.embree) return embreeCollision(origin,ray,static_cast<const EmbreeScene*>(scene.embree),limit,any);
#endif
    Hit hit; hit.distance=limit;
    // Subnormal components cannot cross the padded bounds within the ray limit.
    float3 inverse=make_float3(fabsf(ray.x)<FLT_MIN ? 0 : 1/ray.x,
        fabsf(ray.y)<FLT_MIN ? 0 : 1/ray.y,fabsf(ray.z)<FLT_MIN ? 0 : 1/ray.z);
    if (boxDistance(origin,inverse,scene.nodes[0],limit)==FLT_MAX) return hit;
    int stack[BvhMaxDepth],top=0,nodeIndex=0;
    while (true) {
        const BvhNode& node=scene.nodes[nodeIndex];
        if (node.count>0) {
            for (int j=node.first;j<node.first+node.count;j++) {
                int i=scene.order[j];
                float u,v,distance=intersect(origin,ray,scene.triangles[i],u,v);
                // Original triangle order breaks exact ties, regardless of traversal order.
                if (distance>0 && (distance<hit.distance || (distance==hit.distance && i<hit.index))) {
                    hit.index=i; hit.distance=distance; hit.u=u; hit.v=v;
                    if (any) return hit;
                }
            }
        } else {
            int near=node.first,far=near+1;
            float a=boxDistance(origin,inverse,scene.nodes[near],hit.distance);
            float b=boxDistance(origin,inverse,scene.nodes[far],hit.distance);
            if (b<a) { int swap=near; near=far; far=swap; float d=a; a=b; b=d; }
            if (a!=FLT_MAX) {
                if (b!=FLT_MAX) stack[top++]=far;
                nodeIndex=near;
                continue;
            }
        }
        // A nearer triangle may have made deferred branches irrelevant.
        bool pending=false;
        while (top>0) {
            nodeIndex=stack[--top];
            if (boxDistance(origin,inverse,scene.nodes[nodeIndex],hit.distance)!=FLT_MAX) { pending=true; break; }
        }
        if (!pending) break;
    }
    return hit;
}

// A diffuse + normalized Phong mixture gives matte, glossy and mirror materials.
HD float exponent(const Triangle& t) { float rough=fmaxf(1-t.smoothness,.015f); return fmaxf(0,2/(rough*rough)-2); }
HD float3 evaluate(const Triangle& t,float3 n,float3 reflection,float3 direction,float& pdf) {
    float cosine=fmaxf(0,dot(n,direction)), p=t.specularProbability;
    pdf=(1-p)*cosine/Pi;
    float3 value=mul(t.color,(1-p)/Pi);
    if (p>0 && t.smoothness<.999f) {
        float e=exponent(t), lobe=powf(fmaxf(0,dot(reflection,direction)),e);
        pdf+=p*(e+1)*lobe/(2*Pi);
        value=add(value,mul(t.specularColor,p*(e+2)*lobe/(2*Pi)));
    }
    return value;
}
HD float3 environment(float3 ray,float strength) {
    float t=fminf(1,fmaxf(0,ray.y));
    return mul(add(mul(make_float3(.65f,.72f,.82f),1-t),mul(make_float3(.15f,.32f,.65f),t)),strength);
}
HD PixelSum trace(int x,int y,CameraData camera,Scene scene,RenderSettings settings,int w,int h,unsigned sample) {
    PathSampler rng{scene.directions,(unsigned)(y*w+x),sample,settings.seed};
    float px=2*(x+rng.get(0))/w-1,py=2*(y+rng.get(1))/h-1;
    float3 origin=camera.origin;
    float3 direction=unit(add(camera.forward,add(mul(camera.right,px*camera.aspect*camera.tanHalfFov),mul(camera.up,py*camera.tanHalfFov))));
    if (camera.orthographic) {
        origin=add(origin,add(mul(camera.right,px*camera.aspect*camera.orthographicSize),mul(camera.up,py*camera.orthographicSize)));
        direction=camera.forward;
    } else if (settings.aperture>0) {
        float angle=2*Pi*rng.get(2), radius=.5f*settings.aperture*sqrtf(rng.get(3));
        float3 offset=add(mul(camera.right,radius*cosf(angle)),mul(camera.up,radius*sinf(angle)));
        float3 focus=add(origin,mul(direction,settings.focusDistance/dot(direction,camera.forward)));
        origin=add(origin,offset); direction=unit(sub(focus,origin));
    }
    PixelSum result{};
    float3& radiance=result.radiance;
    float3 throughput=make_float3(1,1,1);
    float previousPdf=0;
    for (int bounce=0;bounce<settings.maxBounces;bounce++) {
        rng.base=4+8*bounce;
        Hit hit=collision(origin,direction,scene);
        if (hit.index<0) {
            // Show empty space as black; retain the sky for lighting and reflections.
            if (bounce>0) radiance=add(radiance,product(throughput,environment(direction,settings.skyStrength)));
            break;
        }
        const Triangle& t=scene.triangles[hit.index];
        float3 geometric=unit(cross(t.edge1,t.edge2));
        if (dot(geometric,direction)>0) geometric=mul(geometric,-1);
        if (bounce==0) {
            result.covered=1;
            result.albedo=t.emission>0 ? make_float3(1,1,1) :
                add(mul(t.color,1-t.specularProbability),mul(t.specularColor,t.specularProbability));
            result.normal=geometric;
        }
        if (t.emission>0) {
            float lightPdf=scene.lightArea>0 ? hit.distance*hit.distance/(scene.lightArea*fmaxf(1e-6f,-dot(geometric,direction))) : 0;
            float weight=previousPdf>0 ? powerWeight(previousPdf,lightPdf) : 1;
            radiance=add(radiance,mul(product(throughput,t.color),t.emission*weight));
            break;
        }
        float3 n=unit(add(mul(t.normal0,1-hit.u-hit.v),add(mul(t.normal1,hit.u),mul(t.normal2,hit.v))));
        if (dot(n,geometric)<0) n=mul(n,-1);
        if (dot(n,direction)>=0) n=geometric;
        if (bounce==0) result.normal=n;
        float3 point=add(origin,mul(direction,hit.distance));
        float3 reflection=reflect(direction,n);
        origin=add(point,mul(geometric,.0005f));
        // Sample a new point on an emissive mesh every time. MIS prevents counting
        // the same light twice when a later bounce also reaches the emitter.
        if (scene.lightCount>0) {
            float selected=rng.get(0)*scene.lightArea;
            int l=0;
            while (l<scene.lightCount-1 && selected>=scene.lights[l].cumulativeArea) l++;
            const Triangle& light=scene.triangles[scene.lights[l].triangle];
            float root=sqrtf(rng.get(1)),v=rng.get(2);
            float3 target=add(light.origin,add(mul(light.edge1,root*(1-v)),mul(light.edge2,root*v)));
            float3 offset=sub(target,origin);
            float distance2=dot(offset,offset),distance=sqrtf(distance2);
            float3 incoming=mul(offset,1/distance),ln=unit(cross(light.edge1,light.edge2));
            float lightCos=-dot(ln,incoming);
            if (!light.oneSided) lightCos=fabsf(lightCos);
            float surfaceCos=fmaxf(0,dot(n,incoming));
            if (lightCos>0 && surfaceCos>0 && dot(geometric,incoming)>0 &&
                collision(origin,incoming,scene,distance-.001f,true).index<0) {
                float pdf,lightPdf=distance2/(scene.lightArea*lightCos);
                float3 brdf=evaluate(t,n,reflection,incoming,pdf);
                float weight=bounce+1==settings.maxBounces ? 1 : powerWeight(lightPdf,pdf);
                radiance=add(radiance,mul(product(product(throughput,brdf),light.color),light.emission*surfaceCos*weight/lightPdf));
            }
        }
        if (bounce+1==settings.maxBounces) break;
        bool specular=rng.get(3)<t.specularProbability;
        if (specular && t.smoothness>=.999f) {
            direction=reflection; throughput=product(throughput,t.specularColor); previousPdf=0;
        } else {
            float cosine=specular ? powf(rng.get(4),1/(exponent(t)+1)) : sqrtf(rng.get(4));
            direction=around(specular ? reflection : n,cosine,2*Pi*rng.get(5));
            if (dot(geometric,direction)<=0 || dot(n,direction)<=0) break;
            float pdf;
            float3 brdf=evaluate(t,n,reflection,direction,pdf);
            if (pdf<1e-12f) break;
            throughput=product(throughput,mul(brdf,dot(n,direction)/pdf)); previousPdf=pdf;
        }
        if (bounce>=3) {
            float survival=fminf(.95f,fmaxf(throughput.x,fmaxf(throughput.y,throughput.z)));
            if (rng.get(6)>=survival) break;
            throughput=mul(throughput,1/survival);
        }
    }
    return result;
}
HD void pixel(int x,int y,CameraData camera,Scene scene,RenderSettings settings,
        PixelSum* sum,uchar4* pixels,int w,int h,unsigned sample) {
    int i=y*w+x;
    PixelSum total=sample ? sum[i] : PixelSum{};
    for (int r=0;r<settings.raysPerPixel;r++) {
        PixelSum path=trace(x,y,camera,scene,settings,w,h,sample+r);
        total.radiance=add(total.radiance,path.radiance);
        total.albedo=add(total.albedo,path.albedo); total.normal=add(total.normal,path.normal);
        total.covered+=path.covered;
    }
    sum[i]=total;
    float3 average=mul(total.radiance,1.0f/(sample+settings.raysPerPixel));
    pixels[i]=make_uchar4(display(average.x,settings.exposure),display(average.y,settings.exposure),display(average.z,settings.exposure),255);
}
__global__ void renderKernel(CameraData camera,Scene scene,RenderSettings settings,
        PixelSum* sum,uchar4* pixels,int w,int h,unsigned sample) {
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x<w && y<h) pixel(x,y,camera,scene,settings,sum,pixels,w,h,sample);
}

extern "C" void* RtCreate() { return new Session(); }
extern "C" void RtReset(Session* s) { if (s) { s->samples=0; s->accumulationVersion++; } }
extern "C" unsigned RtAccumulationVersion(Session* s) { return s ? s->accumulationVersion : 0; }
extern "C" unsigned RtSampleCount(Session* s) { return s ? s->samples : 0; }
extern "C" float RtEmbreeBuildMilliseconds(Session* s) { return s ? s->embree.buildMilliseconds : 0; }
extern "C" int RtCpuThreadCount(Session* s) { return s ? s->cpuThreads : 0; }
extern "C" void RtBvhInfo(Session* s,int* nodes,int* depth,float* buildMs) {
    *nodes=s ? (int)s->nodes.size() : 0; *depth=s ? s->bvhDepth : 0; *buildMs=s ? s->buildMilliseconds : 0;
}
extern "C" const char* RtLastError(Session* s) { return s && !s->hostError.empty() ? s->hostError.c_str() : cudaGetErrorString(s ? s->error : cudaErrorInvalidValue); }
extern "C" void RtShutdown(Session* s) {
    if (!s) return;
    s->denoiser.clear();
    s->embree.clear(); s->hostError.clear();
    cudaFree(s->deviceDirections); s->deviceDirections=nullptr;
    cudaFree(s->deviceTriangles); cudaFree(s->deviceNodes); cudaFree(s->deviceOrder); cudaFree(s->deviceLights);
    cudaFree(s->deviceSum); cudaFree(s->devicePixels);
    s->deviceTriangles=nullptr; s->deviceNodes=nullptr; s->deviceOrder=nullptr; s->deviceLights=nullptr;
    s->deviceSum=nullptr; s->devicePixels=nullptr;
    if (s->eventsReady) { cudaEventDestroy(s->startEvent); cudaEventDestroy(s->stopEvent); }
    s->ready=s->eventsReady=false; s->triangles.clear(); s->nodes.clear(); s->order.clear(); s->lights.clear(); s->sum.clear();
    s->width=s->height=s->samples=0; s->lightArea=s->buildMilliseconds=0; s->bvhDepth=0; s->lastBackend=-1;
}
extern "C" void RtDestroy(Session* s) { if (s) { RtShutdown(s); delete s; } }

template<class T> static bool upload(Session* s,T*& device,const std::vector<T>& host) {
    if (host.empty()) return true;
    s->error=cudaMalloc(&device,host.size()*sizeof(T)); if (s->error!=cudaSuccess) return false;
    s->error=cudaMemcpy(device,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice);
    return s->error==cudaSuccess;
}
extern "C" int RtInitialize(Session* s,const Triangle* triangles,int count,int w,int h) {
    if (!s) return 0;
    RtShutdown(s);
    if (!triangles || count<=0 || w<=0 || h<=0 || w>4096 || h>4096) { s->error=cudaErrorInvalidValue; return 0; }
    s->triangles.assign(triangles,triangles+count); s->width=w; s->height=h; s->sum.resize((size_t)w*h);
    for (int i=0;i<count;i++) {
        Triangle& t=s->triangles[i];
        t.color=toLinear(t.color); t.specularColor=toLinear(t.specularColor);
        if (t.emission>0) {
            float3 areaNormal=cross(t.edge1,t.edge2);
            s->lightArea+=.5f*sqrtf(dot(areaNormal,areaNormal));
            s->lights.push_back({i,s->lightArea});
        }
    }
    auto buildStart=std::chrono::steady_clock::now();
    BvhBuilder bvh(s->triangles);
    s->nodes=std::move(bvh.nodes); s->order=std::move(bvh.order); s->bvhDepth=bvh.maxDepth;
    s->buildMilliseconds=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-buildStart).count();
    if (!upload(s,s->deviceTriangles,s->triangles) || !upload(s,s->deviceNodes,s->nodes) ||
        !upload(s,s->deviceOrder,s->order) || !upload(s,s->deviceLights,s->lights)) return 0;
    try {
        const unsigned* directions=sobolDirections();
        s->error=cudaMalloc(&s->deviceDirections,SobolDimensions*32*sizeof(unsigned));
        if (s->error!=cudaSuccess) return 0;
        s->error=cudaMemcpy(s->deviceDirections,directions,SobolDimensions*32*sizeof(unsigned),cudaMemcpyHostToDevice);
        if (s->error!=cudaSuccess) return 0;
    } catch (const std::exception& e) { s->hostError=e.what(); return 0; }
    s->error=cudaMalloc(&s->deviceSum,(size_t)w*h*sizeof(PixelSum)); if (s->error!=cudaSuccess) return 0;
    s->error=cudaMalloc(&s->devicePixels,(size_t)w*h*sizeof(uchar4)); if (s->error!=cudaSuccess) return 0;
    s->error=cudaEventCreate(&s->startEvent); if (s->error!=cudaSuccess) return 0;
    s->error=cudaEventCreate(&s->stopEvent);
    if (s->error!=cudaSuccess) { cudaEventDestroy(s->startEvent); return 0; }
    s->ready=s->eventsReady=true; return 1;
}
extern "C" int RtRender(Session* s,const CameraData* camera,const RenderSettings* settings,
        unsigned char* output,int outputBytes,int backend,float* computeMs) {
    if (!s) return 0;
    if (!camera || !settings || !output || !computeMs || !s->ready || backend<0 || backend>3 ||
        outputBytes<0 || (size_t)outputBytes<(size_t)s->width*s->height*4 ||
        settings->raysPerPixel<1 || settings->raysPerPixel>16 || settings->maxBounces<1 || settings->maxBounces>32 || !(settings->exposure>0) ||
        !(settings->aperture>=0) || !(settings->focusDistance>0) || !(settings->skyStrength>=0)) {
        s->error=cudaErrorInvalidValue; return 0;
    }
    s->hostError.clear();
    // Build only when this backend is selected; exclude construction from compute.
    if (backend==3 && !s->embree.scene) {
        if (!s->embree.build(s->triangles,s->hostError)) { s->embree.clear(); return 0; }
    }
    if (s->lastBackend!=backend || std::memcmp(camera,&s->lastCamera,sizeof(CameraData)) ||
        std::memcmp(settings,&s->lastSettings,sizeof(RenderSettings))) RtReset(s);
    s->lastBackend=backend; s->lastCamera=*camera; s->lastSettings=*settings; s->cpuThreads=0;
    Scene scene{ s->triangles.data(),s->nodes.data(),s->order.data(),s->lights.data(),(int)s->triangles.size(),(int)s->lights.size(),s->lightArea,sobolDirections() };
    if (backend==3) scene.embree=&s->embree;
    if (backend==2) {
        scene.directions=s->deviceDirections;
        scene.triangles=s->deviceTriangles; scene.nodes=s->deviceNodes; scene.order=s->deviceOrder; scene.lights=s->deviceLights;
        s->error=cudaEventRecord(s->startEvent); if (s->error!=cudaSuccess) return 0;
        dim3 block(16,16),grid((s->width+15)/16,(s->height+15)/16);
        renderKernel<<<grid,block>>>(*camera,scene,*settings,s->deviceSum,s->devicePixels,s->width,s->height,s->samples);
        s->error=cudaGetLastError(); if (s->error!=cudaSuccess) return 0;
        s->error=cudaEventRecord(s->stopEvent); if (s->error!=cudaSuccess) return 0;
        s->error=cudaEventSynchronize(s->stopEvent); if (s->error!=cudaSuccess) return 0;
        s->error=cudaEventElapsedTime(computeMs,s->startEvent,s->stopEvent); if (s->error!=cudaSuccess) return 0;
        s->error=cudaMemcpy(output,s->devicePixels,(size_t)s->width*s->height*4,cudaMemcpyDeviceToHost);
        if (s->error!=cudaSuccess) return 0;
    } else {
        auto start=std::chrono::steady_clock::now();
        #pragma omp parallel if(backend==1 || backend==3)
        {
            #pragma omp single
            s->cpuThreads=omp_get_num_threads();
            #pragma omp for schedule(static)
            for (int y=0;y<s->height;y++) for (int x=0;x<s->width;x++)
                pixel(x,y,*camera,scene,*settings,s->sum.data(),reinterpret_cast<uchar4*>(output),s->width,s->height,s->samples);
        }
        *computeMs=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
    s->samples+=settings->raysPerPixel; s->error=cudaSuccess; return 1;
}

// Convert the accumulated HDR lighting and matching first-hit guides for inference.
// This is a separate stage: filtered values never enter the progressive sum.
__global__ void prepareDenoising(const PixelSum* sum,float3* color,float3* albedo,float3* normal,int count,unsigned samples) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    color[i]=mul(sum[i].radiance,1.0f/samples);
    albedo[i]=mul(sum[i].albedo,1.0f/samples);
    normal[i]=unit(sum[i].normal);
}
__global__ void presentDenoising(const PixelSum* sum,const float3* color,uchar4* pixels,int count,float exposure) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    // Do not let the image filter introduce light into the empty black background.
    float3 c=sum[i].covered ? color[i] : make_float3(0,0,0);
    pixels[i]=make_uchar4(display(c.x,exposure),display(c.y,exposure),display(c.z,exposure),255);
}
extern "C" int RtDenoise(Session* s,unsigned char* output,int outputBytes,float* milliseconds) {
    if (!s) return 0;
    if (!s->ready || !s->samples || !output || !milliseconds || outputBytes<0 ||
        (size_t)outputBytes<(size_t)s->width*s->height*4) { s->error=cudaErrorInvalidValue; return 0; }
    auto begin=std::chrono::steady_clock::now();
    s->hostError.clear();
    if (!s->denoiser.initialize(s->width,s->height,s->hostError)) { s->denoiser.clear(); return 0; }
    size_t count=(size_t)s->width*s->height;
    if (s->lastBackend!=2) {
        s->error=cudaMemcpy(s->deviceSum,s->sum.data(),count*sizeof(PixelSum),cudaMemcpyHostToDevice);
        if (s->error!=cudaSuccess) return 0;
    }
    auto& d=s->denoiser;
    prepareDenoising<<<(count+255)/256,256>>>(s->deviceSum,d.color,d.albedo,d.normal,count,s->samples);
    s->error=cudaGetLastError(); if (s->error!=cudaSuccess) return 0;
    if (!d.run(s->width,s->height,s->hostError)) { d.clear(); return 0; }
    presentDenoising<<<(count+255)/256,256>>>(s->deviceSum,d.output,s->devicePixels,count,s->lastSettings.exposure);
    s->error=cudaGetLastError(); if (s->error!=cudaSuccess) return 0;
    s->error=cudaMemcpy(output,s->devicePixels,count*sizeof(uchar4),cudaMemcpyDeviceToHost);
    if (s->error!=cudaSuccess) return 0;
    *milliseconds=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-begin).count();
    s->error=cudaSuccess; return 1;
}
