#pragma once
#include <curand.h>

// Four camera dimensions, then eight fixed dimensions per surface bounce.
// Fixed dimensions keep lens/light/BRDF choices independent of skipped branches.
constexpr int SobolDimensions=4+8*32;
static const unsigned* sobolDirections() {
    static const unsigned* directions=[] {
        curandDirectionVectors32_t* table=nullptr;
        if (curandGetDirectionVectors32(&table,CURAND_DIRECTION_VECTORS_32_JOEKUO6)!=CURAND_STATUS_SUCCESS)
            throw std::runtime_error("Could not load cuRAND Sobol direction vectors");
        return &table[0][0];
    }();
    return directions;
}

struct PathSampler {
    const unsigned* directions;
    unsigned pixel, sample, seed, base=0;

    __host__ __device__ static unsigned hash(unsigned value) {
        value^=value>>16; value*=0x7feb352du;
        value^=value>>15; value*=0x846ca68bu;
        return value^(value>>16);
    }
    __host__ __device__ float get(unsigned dimension) const {
        dimension+=base;
        // A digital shift decorrelates pixels and dimensions without destroying
        // the Sobol net. The Gray-code index supports indefinite progression.
        unsigned value=hash(pixel^hash(seed+dimension*0x9e3779b9u));
        unsigned index=sample^(sample>>1);
        for (unsigned bit=0;index;index>>=1,bit++)
            if (index&1) value^=directions[dimension*32+bit];
        return (value>>8)*(1.0f/16777216.0f);
    }
};
