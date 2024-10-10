#include <cuda_runtime.h>
#include <optix.h>

#include "LaunchParams.h"
#include "perRayData.h"
#include "shaderUtils.h"

#define NUM_LIGHT_SAMPLES 16
#define MAX_TRACE_DEPTH 12
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

    extern "C" __global__ void __raygen__SHRenderCamera()
    {

        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        // if (ix == 0 && iy == 0)
        // {
        //     // printf("%f %f %f\n", optixLaunchParams.camera.position.x, optixLaunchParams.camera.position.y, optixLaunchParams.camera.position.z);
        //     // printf("%f %f %f\n", optixLaunchParams.camera.direction.x, optixLaunchParams.camera.direction.y, optixLaunchParams.camera.direction.z);
        //     // printf("%d %d\n", optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);
        //     float3 test = optixLaunchParams.sh.T_matrix[73742 * 81 * 81 + 80 + 80 * 81];
        //     printf("%f %f %f\n", test.x, test.y, test.z);
        // }
        const auto &camera = optixLaunchParams.camera;

        PRDExtra prdE;
        prdE.cleanForNextLaunch();
        prdE.ix = ix;
        prdE.iy = iy;

        // frame ID will make the realtime visualization constant
        PRD prd(ix + optixLaunchParams.frame.size.x * iy, optixLaunchParams.frame.frameID, &prdE);

        // normalized screen plane position, in [0,1]^2

        // iw: note for denoising that's not actually correct - if we
        // assume that the camera should only(!) cover the denoised
        // screen then the actual screen plane we shuld be using during
        // rendreing is slightly larger than [0,1]^2
        float2 screen = make_float2(ix + prd.random(), iy + prd.random()) / make_float2(optixLaunchParams.frame.size);

        // generate ray direction
        float3 initialRayOrigin = (float3)camera.position;
        float3 initialRayDir = (float3)normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);

        prdE.cleanForNextLaunch();
        prd.cleanForNextLaunch();

        float3 currentRayOrigin = initialRayOrigin;
        float3 currentRayDir = initialRayDir;
        traceRadiance(optixLaunchParams.traversable, currentRayOrigin, currentRayDir, prd);

        uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

        float3 pixelNormal = normalize(prdE.pixelNormalFirstHit);
        float3 worldNormal(pixelNormal / 2 + make_float3(0.5));
        // move to camera space
        float3 cameraNormal = normalize(to_local(normalize(pixelNormal), normalize(camera.direction), normalize(camera.horizontal)));
        cameraNormal = cameraNormal / 2 + make_float3(0.5);

        optixLaunchParams.frame.cameraNormalBuffer[fbIndex] = make_float4(cameraNormal, 1.f);
        optixLaunchParams.frame.worldNormalBuffer[fbIndex] = make_float4(worldNormal, 1.f);
        optixLaunchParams.frame.colorBuffer[fbIndex] = make_float4(prd.pixelColor, 1.f);

        optixLaunchParams.frame.FirstHitKdBuffer[fbIndex] = make_float4(prdE.pixelFirstHitKd, 1.f);
        optixLaunchParams.frame.FirstHitKsBuffer[fbIndex] = make_float4(prdE.pixelFirstHitKs, 1.f);
    }
}