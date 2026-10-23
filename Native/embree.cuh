#pragma once
#include <embree3/rtcore.h>
#include <string>
#include <dlfcn.h>

// Unity exports an older Embree ABI too. Resolve our official release locally
// with deep binding, so neither its C nor C++ symbols bind to Unity's library.
struct EmbreeApi {
#define EMBREE_FUNCTIONS(F) \
    F(rtcNewDevice) F(rtcNewScene) F(rtcSetSceneFlags) F(rtcSetSceneBuildQuality) \
    F(rtcNewGeometry) F(rtcSetNewGeometryBuffer) F(rtcSetGeometryUserData) \
    F(rtcSetGeometryIntersectFilterFunction) F(rtcSetGeometryOccludedFilterFunction) \
    F(rtcCommitGeometry) F(rtcAttachGeometry) F(rtcReleaseGeometry) F(rtcCommitScene) \
    F(rtcGetDeviceError) F(rtcReleaseScene) F(rtcReleaseDevice) F(rtcIntersect1) F(rtcOccluded1)
#define DECLARE(name) decltype(&::name) name=nullptr;
    EMBREE_FUNCTIONS(DECLARE)
#undef DECLARE
    std::string error;
    EmbreeApi() {
        // Native libraries remain resident for the process lifetime. Each session
        // still releases its own scene and device when the room is unloaded.
        void* library=dlopen("libembree3.so.3",RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);
        if (!library) { error=dlerror(); return; }
#define BIND(name) name=reinterpret_cast<decltype(name)>(dlsym(library,#name)); if (!name) { error="Missing Embree symbol: " #name; return; }
        EMBREE_FUNCTIONS(BIND)
#undef BIND
#undef EMBREE_FUNCTIONS
    }
    static EmbreeApi& get() { static EmbreeApi api; return api; }
};

// Embree owns the CPU BVH and its built-in triangle intersections. Our path
// integrator remains shared, so lights, materials and sample counts stay equal.
struct EmbreeScene {
    EmbreeApi* api=nullptr;
    RTCDevice device=nullptr;
    RTCScene scene=nullptr;
    float buildMilliseconds=0;

    void clear() {
        if (scene) api->rtcReleaseScene(scene);
        if (device) api->rtcReleaseDevice(device);
        scene=nullptr; device=nullptr; buildMilliseconds=0;
    }
    static void filter(const RTCFilterFunctionNArguments* args) {
        const Triangle* triangles=static_cast<const Triangle*>(args->geometryUserPtr);
        for (unsigned i=0;i<args->N;i++) {
            if (!args->valid[i]) continue;
            const Triangle& t=triangles[RTCHitN_primID(args->hit,args->N,i)];
            float determinant=-(RTCRayN_dir_x(args->ray,args->N,i)*RTCHitN_Ng_x(args->hit,args->N,i)
                +RTCRayN_dir_y(args->ray,args->N,i)*RTCHitN_Ng_y(args->hit,args->N,i)
                +RTCRayN_dir_z(args->ray,args->N,i)*RTCHitN_Ng_z(args->hit,args->N,i));
            if (fabsf(determinant)<1e-7f || (t.oneSided && determinant<0)) args->valid[i]=0;
        }
    }
    bool build(const std::vector<Triangle>& triangles,std::string& error) {
        api=&EmbreeApi::get();
        if (!api->error.empty()) { error=api->error; return false; }
        auto begin=std::chrono::steady_clock::now();
        device=api->rtcNewDevice(nullptr);
        if (!device) { error="Could not create Embree CPU device"; return false; }
        scene=api->rtcNewScene(device);
        api->rtcSetSceneFlags(scene,RTC_SCENE_FLAG_ROBUST);
        api->rtcSetSceneBuildQuality(scene,RTC_BUILD_QUALITY_HIGH);
        RTCGeometry geometry=api->rtcNewGeometry(device,RTC_GEOMETRY_TYPE_TRIANGLE);
        auto vertices=static_cast<float3*>(api->rtcSetNewGeometryBuffer(geometry,RTC_BUFFER_TYPE_VERTEX,0,
            RTC_FORMAT_FLOAT3,sizeof(float3),triangles.size()*3));
        auto indices=static_cast<unsigned*>(api->rtcSetNewGeometryBuffer(geometry,RTC_BUFFER_TYPE_INDEX,0,
            RTC_FORMAT_UINT3,sizeof(unsigned)*3,triangles.size()));
        if (!vertices || !indices) {
            api->rtcReleaseGeometry(geometry); error="Could not allocate Embree geometry"; return false;
        }
        for (size_t i=0;i<triangles.size();i++) {
            const Triangle& t=triangles[i];
            vertices[i*3]=t.origin;
            vertices[i*3+1]=make_float3(t.origin.x+t.edge1.x,t.origin.y+t.edge1.y,t.origin.z+t.edge1.z);
            vertices[i*3+2]=make_float3(t.origin.x+t.edge2.x,t.origin.y+t.edge2.y,t.origin.z+t.edge2.z);
            for (int j=0;j<3;j++) indices[i*3+j]=(unsigned)(i*3+j);
        }
        api->rtcSetGeometryUserData(geometry,const_cast<Triangle*>(triangles.data()));
        api->rtcSetGeometryIntersectFilterFunction(geometry,filter);
        api->rtcSetGeometryOccludedFilterFunction(geometry,filter);
        api->rtcCommitGeometry(geometry);
        api->rtcAttachGeometry(scene,geometry); api->rtcReleaseGeometry(geometry);
        api->rtcCommitScene(scene);
        if (api->rtcGetDeviceError(device)!=RTC_ERROR_NONE) { error="Embree scene build failed"; return false; }
        buildMilliseconds=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-begin).count();
        return true;
    }
};

static inline Hit embreeCollision(float3 origin,float3 direction,const EmbreeScene* acceleration,float limit,bool any) {
    RTCRayHit query{};
    query.ray.org_x=origin.x; query.ray.org_y=origin.y; query.ray.org_z=origin.z;
    query.ray.dir_x=direction.x; query.ray.dir_y=direction.y; query.ray.dir_z=direction.z;
    query.ray.tnear=nextafterf(1e-4f,INFINITY); query.ray.tfar=nextafterf(limit,0);
    query.ray.mask=~0u;
    query.hit.geomID=RTC_INVALID_GEOMETRY_ID;
    RTCIntersectContext context; rtcInitIntersectContext(&context);
    Hit hit; hit.distance=limit;
    if (any) {
        acceleration->api->rtcOccluded1(acceleration->scene,&context,&query.ray);
        if (query.ray.tfar<0) hit.index=0;
    } else {
        acceleration->api->rtcIntersect1(acceleration->scene,&context,&query);
        if (query.hit.geomID!=RTC_INVALID_GEOMETRY_ID) {
            hit.index=(int)query.hit.primID; hit.distance=query.ray.tfar;
            hit.u=query.hit.u; hit.v=query.hit.v;
        }
    }
    return hit;
}
