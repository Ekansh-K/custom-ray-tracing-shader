#pragma once
#define OPTIX_ENABLE_SDK_MIXING
#include <optix_function_table.h>
// Unity can use another OptiX SDK. Keep our ABI table and stubs private.
#undef OPTIX_FUNCTION_TABLE_SYMBOL
#define OPTIX_FUNCTION_TABLE_SYMBOL cudaRoomOptixFunctions
extern "C" { __attribute__((visibility("hidden"))) OptixFunctionTable cudaRoomOptixFunctions{}; }
#include <optix_stubs.h>

struct AIDenoiser {
    OptixDeviceContext context=nullptr;
    OptixDenoiser handle=nullptr;
    OptixDenoiserSizes sizes{};
    void* state=nullptr;
    void* scratch=nullptr;
    float3 *color=nullptr,*albedo=nullptr,*normal=nullptr,*output=nullptr;

    static bool check(OptixResult result,std::string& error) {
        if (result==OPTIX_SUCCESS) return true;
        error=std::string("OptiX AI denoiser: ")+optixGetErrorName(result); return false;
    }
    static bool allocate(void** pointer,size_t bytes,std::string& error) {
        auto result=cudaMalloc(pointer,bytes);
        if (result==cudaSuccess) return true;
        error=std::string("AI denoiser memory: ")+cudaGetErrorString(result); return false;
    }
    bool initialize(int width,int height,std::string& error) {
        if (handle && output) return true;
        static const OptixResult initialized=optixInit();
        if (!check(initialized,error)) return false;
        OptixDeviceContextOptions contextOptions{};
        if (!check(optixDeviceContextCreate(nullptr,&contextOptions,&context),error)) return false;
        OptixDenoiserOptions options{}; options.guideAlbedo=options.guideNormal=1;
        if (!check(optixDenoiserCreate(context,OPTIX_DENOISER_MODEL_KIND_AOV,&options,&handle),error) ||
            !check(optixDenoiserComputeMemoryResources(handle,width,height,&sizes),error)) return false;
        size_t bytes=(size_t)width*height*sizeof(float3);
        if (!allocate(&state,sizes.stateSizeInBytes,error) || !allocate(&scratch,sizes.withoutOverlapScratchSizeInBytes,error) ||
            !allocate((void**)&color,bytes,error) || !allocate((void**)&albedo,bytes,error) ||
            !allocate((void**)&normal,bytes,error) || !allocate((void**)&output,bytes,error)) return false;
        return check(optixDenoiserSetup(handle,nullptr,width,height,(CUdeviceptr)state,sizes.stateSizeInBytes,
            (CUdeviceptr)scratch,sizes.withoutOverlapScratchSizeInBytes),error);
    }
    static OptixImage2D image(float3* data,int width,int height) {
        OptixImage2D result{};
        result.data=(CUdeviceptr)data; result.width=width; result.height=height;
        result.rowStrideInBytes=width*sizeof(float3); result.pixelStrideInBytes=sizeof(float3);
        result.format=OPTIX_PIXEL_FORMAT_FLOAT3; return result;
    }
    bool run(int width,int height,std::string& error) {
        OptixDenoiserGuideLayer guides{}; guides.albedo=image(albedo,width,height); guides.normal=image(normal,width,height);
        OptixDenoiserLayer layer{}; layer.input=image(color,width,height); layer.output=image(output,width,height);
        layer.type=OPTIX_DENOISER_AOV_TYPE_BEAUTY;
        OptixDenoiserParams params{}; // Automatic HDR exposure, full denoised output.
        return check(optixDenoiserInvoke(handle,nullptr,&params,(CUdeviceptr)state,sizes.stateSizeInBytes,
            &guides,&layer,1,0,0,(CUdeviceptr)scratch,sizes.withoutOverlapScratchSizeInBytes),error);
    }
    void clear() {
        if (handle) optixDenoiserDestroy(handle);
        if (context) optixDeviceContextDestroy(context);
        cudaFree(state); cudaFree(scratch); cudaFree(color); cudaFree(albedo); cudaFree(normal); cudaFree(output);
        handle=nullptr; context=nullptr; state=scratch=nullptr; color=albedo=normal=output=nullptr;
    }
};
