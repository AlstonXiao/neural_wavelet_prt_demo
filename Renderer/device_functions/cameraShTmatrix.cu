#include <cuda_runtime.h>
#include <optix.h>

#include "LaunchParams.h"
#include "perRayData.h"
#include "shaderUtils.h"

#define RR_DEPTH 3

namespace nert_renderer
{
    extern "C" __constant__ LaunchParams optixLaunchParams;

    static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3 ray_origin,
        float3 ray_direction,
        PRD &prd)
    {
        unsigned int
            u0,
            u1, u2,        // Origin (R)
            u3, u4, u5,    // Direction (R)
            u6, u7, u8,    // pixelcolor (R)
            u9,            // Ray Recursive depth (W)
            u10, u11, u12, // Light contribution (RW)
            u13,           // Random (RW)
            u14,           // PDF (RW)
            u15,           // Flags (R)
            u16, u17;      // Pointer to extra (W)

        u6 = __float_as_uint(prd.pixelColor.x);
        u7 = __float_as_uint(prd.pixelColor.y);
        u8 = __float_as_uint(prd.pixelColor.z);
        u9 = prd.ray_recursie_depth;
        u10 = __float_as_uint(prd.light_contribution.x);
        u11 = __float_as_uint(prd.light_contribution.y);
        u12 = __float_as_uint(prd.light_contribution.z);
        u13 = prd.random.state;
        u14 = __float_as_uint(prd.pdf);

        u15 = prd.firstHitUpdated | prd.pixelDirectSampleHitEnvMapFlag << 1 |
              prd.pixelPerfectReflectionFlag << 2 | prd.done << 3;

        const uint64_t uptr = reinterpret_cast<uint64_t>(prd.prdExtraInfo);
        u16 = uptr >> 32;
        u17 = uptr & 0x00000000ffffffff;

        optixTrace(
            PAYLOAD_TYPE_RADIANCE,
            handle,
            ray_origin,
            ray_direction,
            1e-2f, // tmin
            1e20f, // tmax
            0.0f,  // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
            RADIANCE_RAY_TYPE,             // SBT offset
            RAY_TYPE_COUNT,                // SBT stride
            RADIANCE_RAY_TYPE,             // missSBTIndex
            u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17);

        prd.ray_origin = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
        prd.ray_dir = make_float3(__uint_as_float(u3), __uint_as_float(u4), __uint_as_float(u5));
        prd.pixelColor = make_float3(__uint_as_float(u6), __uint_as_float(u7), __uint_as_float(u8));

        prd.light_contribution = make_float3(__uint_as_float(u10), __uint_as_float(u11), __uint_as_float(u12));
        prd.random.state = u13;
        prd.pdf = __uint_as_float(u14);

        prd.firstHitUpdated = u15 & 1u;
        prd.pixelDirectSampleHitEnvMapFlag = (u15 & (1u << 1)) >> 1;
        prd.pixelPerfectReflectionFlag = (u15 & (1u << 2)) >> 2;
        prd.done = (u15 & (1u << 3)) >> 3;
    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__ComputeTMatrix()
    {
        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        int vertexID = ix + optixLaunchParams.frame.size.x * iy;

        PRDExtra prdE;
        prdE.cleanForNextLaunch();
        prdE.ix = ix;
        prdE.iy = iy;

        // frame ID will make the realtime visualization constant
        PRD prd(ix + optixLaunchParams.frame.size.x * iy, optixLaunchParams.frame.frameID, &prdE);

        float3 initialRayOrigin = optixLaunchParams.sh.vertexLocations[vertexID];
        int numPixelSamples = optixLaunchParams.samplingParameter.samplesPerPixel;
        int sh_degree = optixLaunchParams.sh.shTerms;
        int sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
        // accumuate these values, others will only depend on the final hit pos
        float3 accumulatedOverSampleColor = make_float3(0.f);
        float3 accumulatedOverSampleDirectColor = make_float3(0.f);
        float3 accumulatedOverSampleIndirectColor = make_float3(0.f);
        for (int i = 0; i < sh_coeffs; i++)
        {
            optixLaunchParams.sh.T_matrix[vertexID * sh_coeffs + i] = make_float3(0.f);
        }
        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++)
        {
            float3 initialRayDir;
            sample_sphere(prd.random(), prd.random(), initialRayDir);

            prdE.cleanForNextLaunch();
            prd.cleanForNextLaunch();

            float3 currentRayOrigin = initialRayOrigin;
            float3 currentRayDir = initialRayDir;
            float3 tentativeFirstHitColor = make_float3(0.f, 0.f, 0.f);
            for (;;)
            {
                traceRadiance(optixLaunchParams.traversable, currentRayOrigin, currentRayDir, prd);
                currentRayOrigin = prd.ray_origin;
                currentRayDir = prd.ray_dir;
                prd.ray_recursie_depth++;

                // First case, directly hit envmap
                if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y) && !isnan(prd.pixelColor.z) && prd.ray_recursie_depth == 1 && prd.pixelDirectSampleHitEnvMapFlag)
                {
                    tentativeFirstHitColor = prd.pixelColor;
                }

                // Second case, hit mirror and hit envmap
                if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y) && !isnan(prd.pixelColor.z) && prd.ray_recursie_depth == 2 && prd.pixelPerfectReflectionFlag && prd.pixelDirectSampleHitEnvMapFlag)
                {
                    tentativeFirstHitColor = prd.pixelColor;
                }

                if (prd.ray_recursie_depth > RR_DEPTH)
                {
                    float z = prd.random(); // halton().y;
                    float rr_prob = fmin(fmaxf(prd.light_contribution), 0.95f);

                    if (z < rr_prob)
                    {
                        prd.light_contribution = prd.light_contribution / fmax(rr_prob, 1e-10f);
                    }
                    else
                    {
                        prd.done = true;
                    }
                }

                // Hit the light source or exceed the max depth
                if (prd.done || prd.ray_recursie_depth >= optixLaunchParams.samplingParameter.maximumBounce)
                    break;

                // Ray dir and Ray origin will be updated in shader
            }
            float3 indirect_color = prd.pixelColor - tentativeFirstHitColor;

            // This result eval will always be in full degree mode
            float result[MAX_SH_COEFFICIENTS];
            SHEvaluate(initialRayDir, result);
            
            for (int l = 0; l <= MAX_SH_DEGREE; ++l)
                for (int m = -l; m <= l; ++m)
                    result[SHIndex(l, m)] = 1.f / (1.f + 0.005f*l*l*(l+1.f)*(l+1.f)) *result[SHIndex(l, m)];

            if (!isnan(indirect_color.x) && !isnan(indirect_color.y) && !isnan(indirect_color.z)) {
                for (int i = 0; i < sh_coeffs; i++) {
                    optixLaunchParams.sh.T_matrix[vertexID * sh_coeffs + i] += result[i] * indirect_color * 4 * M_PIf / numPixelSamples;
                }
            }
            else {
                for (int i = 0; i < sh_coeffs; i++) {
                    optixLaunchParams.sh.T_matrix[vertexID * sh_coeffs + i] += optixLaunchParams.sh.T_matrix[vertexID * sh_coeffs + i] / numPixelSamples;
                }
            }
        }
    }
}