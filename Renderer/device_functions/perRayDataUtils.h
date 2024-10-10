#include <cuda_runtime.h>
#include <optix.h>

#include "perRayData.h"
namespace nert_renderer {
    //------------------------------------------------------------------------------
    // Load Store PRD Data
    //------------------------------------------------------------------------------
    static __forceinline__ __device__ PRD loadClosesthitRadiancePRD()
    {
        unsigned int recursive_depth = optixGetPayload_9();
        float3 light_contribution;
        float3 pixel_color;

        pixel_color.x = __uint_as_float(optixGetPayload_6());
        pixel_color.y = __uint_as_float(optixGetPayload_7());
        pixel_color.z = __uint_as_float(optixGetPayload_8());

        light_contribution.x = __uint_as_float(optixGetPayload_10());
        light_contribution.y = __uint_as_float(optixGetPayload_11());
        light_contribution.z = __uint_as_float(optixGetPayload_12());

        unsigned int randomState = optixGetPayload_13();
        unsigned int u15 = optixGetPayload_15();

        unsigned int i0 = optixGetPayload_16();;
        unsigned int i1 = optixGetPayload_17();;
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        PRDExtra* ptr = reinterpret_cast<PRDExtra*>(uptr);

        PRD prd(recursive_depth, pixel_color, light_contribution, randomState, ptr);
        prd.firstHitUpdated = u15 & 1u;
        prd.pixelDirectSampleHitEnvMapFlag = (u15 & (1u << 1)) >> 1;
        prd.pixelPerfectReflectionFlag = (u15 & (1u << 2)) >> 2;
        prd.done = (u15 & (1u << 3)) >> 3;
        return prd;
    }

    static __forceinline__ __device__ PRD loadMissRadiancePRD()
    {
        unsigned int recursive_depth = optixGetPayload_9();
        float3 light_contribution;
        float3 pixel_color;

        pixel_color.x = __uint_as_float(optixGetPayload_6());
        pixel_color.y = __uint_as_float(optixGetPayload_7());
        pixel_color.z = __uint_as_float(optixGetPayload_8());

        light_contribution.x = __uint_as_float(optixGetPayload_10());
        light_contribution.y = __uint_as_float(optixGetPayload_11());
        light_contribution.z = __uint_as_float(optixGetPayload_12());

        float pdf = __uint_as_float(optixGetPayload_14());
        unsigned int u15 = optixGetPayload_15();

        unsigned int i0 = optixGetPayload_16();;
        unsigned int i1 = optixGetPayload_17();;
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        PRDExtra* ptr = reinterpret_cast<PRDExtra*>(uptr);

        PRD prd(recursive_depth, pixel_color, light_contribution, 0, ptr);
        prd.pdf = pdf;
        prd.firstHitUpdated = u15 & 1u;
        prd.pixelDirectSampleHitEnvMapFlag = (u15 & (1u << 1)) >> 1;
        prd.pixelPerfectReflectionFlag = (u15 & (1u << 2)) >> 2;
        prd.done = (u15 & (1u << 3)) >> 3;
        return prd;
    }

    static __forceinline__ __device__ void storeClosesthitRadiancePRD(PRD& prd)
    {
        optixSetPayload_0(__float_as_uint(prd.ray_origin.x));
        optixSetPayload_1(__float_as_uint(prd.ray_origin.y));
        optixSetPayload_2(__float_as_uint(prd.ray_origin.z));

        optixSetPayload_3(__float_as_uint(prd.ray_dir.x));
        optixSetPayload_4(__float_as_uint(prd.ray_dir.y));
        optixSetPayload_5(__float_as_uint(prd.ray_dir.z));

        optixSetPayload_6(__float_as_uint(prd.pixelColor.x));
        optixSetPayload_7(__float_as_uint(prd.pixelColor.y));
        optixSetPayload_8(__float_as_uint(prd.pixelColor.z));

        optixSetPayload_10(__float_as_uint(prd.light_contribution.x));
        optixSetPayload_11(__float_as_uint(prd.light_contribution.y));
        optixSetPayload_12(__float_as_uint(prd.light_contribution.z));

        optixSetPayload_13(prd.random.state);

        optixSetPayload_14(__float_as_uint(prd.pdf));

        unsigned int u15 = prd.firstHitUpdated | prd.pixelDirectSampleHitEnvMapFlag << 1 |
            prd.pixelPerfectReflectionFlag << 2 | prd.done << 3;

        optixSetPayload_15(u15);
    }

    static __forceinline__ __device__ void storeMissRadiancePRD(PRD& prd)
    {
        optixSetPayload_6(__float_as_uint(prd.pixelColor.x));
        optixSetPayload_7(__float_as_uint(prd.pixelColor.y));
        optixSetPayload_8(__float_as_uint(prd.pixelColor.z));


        unsigned int u15 = prd.firstHitUpdated | prd.pixelDirectSampleHitEnvMapFlag << 1 |
            prd.pixelPerfectReflectionFlag << 2 | prd.done << 3;

        optixSetPayload_15(u15);
    }
}