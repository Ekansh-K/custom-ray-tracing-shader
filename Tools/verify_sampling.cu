#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <set>
#include <cstdio>
#include <cstring>
#include "../Native/sampling.cuh"
__global__ void samples(float* output,const unsigned* directions) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=4096) return;
    PathSampler sampler{directions,(unsigned)(i/1024), (unsigned)(i%1024),42};
    for(unsigned d=0;d<SobolDimensions;d++) output[i*SobolDimensions+d]=sampler.get(d);
}
int main() {
    const unsigned* directions=sobolDirections();
    for (unsigned pixel: {0u,177u}) for(unsigned seed: {42u,1234u})
        for(unsigned d=0;d<SobolDimensions;d++) for(unsigned count: {16u,64u,256u}) {
            std::set<unsigned> strata;
            for(unsigned sample=0;sample<count;sample++) {
                PathSampler s{directions,pixel,sample,seed}; float value=s.get(d);
                if (!(value>=0 && value<1)) throw std::runtime_error("Sample outside [0,1)");
                strata.insert((unsigned)(value*count));
            }
            if (strata.size()!=count) throw std::runtime_error("Sobol projection lost stratification");
        }
    unsigned* deviceDirections; float* output;
    if(cudaMalloc(&deviceDirections,SobolDimensions*32*sizeof(unsigned)) ||
       cudaMalloc(&output,4096*SobolDimensions*sizeof(float))) return 1;
    cudaMemcpy(deviceDirections,directions,SobolDimensions*32*sizeof(unsigned),cudaMemcpyHostToDevice);
    samples<<<16,256>>>(output,deviceDirections);
    std::vector<float> data(4096*SobolDimensions);
    if(cudaMemcpy(data.data(),output,data.size()*sizeof(float),cudaMemcpyDeviceToHost)) return 1;
    for(unsigned i=0;i<4096;i++) {
        PathSampler s{directions,i/1024,i%1024,42};
        for(unsigned d=0;d<SobolDimensions;d++) {
            float value=s.get(d);
            if(std::memcmp(&value,&data[i*SobolDimensions+d],sizeof(float))) throw std::runtime_error("CPU/CUDA sampling mismatch");
        }
    }
    cudaFree(output);cudaFree(deviceDirections);
    puts("PASS Sobol projection coverage for 260 dimensions, bounds, seeds and 1,064,960 identical CPU/CUDA samples");
}
