#include <cuda_runtime.h>
#include <optix.h>

#include "LaunchParams.h"
#include "perRayData.h"
#include "shaderUtils.h"

#define RR_DEPTH 3

namespace nert_renderer {
    extern "C" __constant__ LaunchParams optixLaunchParams;

    static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        PRD&                   prd
    )
    {
        unsigned int 
            u0, u1, u2,     // Origin (R)
            u3, u4, u5,     // Direction (R)
            u6, u7, u8,     // pixelcolor (R)
            u9,             // Ray Recursive depth (W)
            u10, u11, u12,  // Light contribution (RW)
            u13,            // Random (RW)
            u14,            // PDF (RW)
            u15,            // Flags (R)
            u16, u17;       // Pointer to extra (W)

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
            1e-2f,                         // tmin
            1e20f,                         // tmax
            0.0f,                          // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            RADIANCE_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            RADIANCE_RAY_TYPE,            // missSBTIndex 
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
    extern "C" __global__ void __raygen__fovCamera()
    {
        
        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        const auto& camera = optixLaunchParams.camera;

        PRDExtra prdE;
        prdE.cleanForNextLaunch();
        prdE.ix = ix;
        prdE.iy = iy;

        // frame ID will make the realtime visualization constant
        PRD prd(ix + optixLaunchParams.frame.size.x * iy, optixLaunchParams.frame.frameID, &prdE);
        
        int numPixelSamples = optixLaunchParams.samplingParameter.samplesPerPixel;
        int numIndirectPerSample = 1;// optixLaunchParams.samplingParameter.indirectSamplesPerDirect;

        // accumuate these values, others will only depend on the final hit pos
        float3 accumulatedOverSampleColor = make_float3(0.f);
        float3 accumulatedOverSampleDirectColor = make_float3(0.f);
        float3 accumulatedOverSampleIndirectColor = make_float3(0.f);
        float3 accumulatedOverSampleFirstHitKd = make_float3(0.f);
        float3 accumulatedOverSampleFirstHitKs = make_float3(0.f);

        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
            // normalized screen plane position, in [0,1]^2

            // iw: note for denoising that's not actually correct - if we
            // assume that the camera should only(!) cover the denoised
            // screen then the actual screen plane we shuld be using during
            // rendreing is slightly larger than [0,1]^2
            float2 screen;
            screen = make_float2(ix + prd.random() - 0.5 - optixLaunchParams.camera.cx, iy + prd.random() - 0.5 - optixLaunchParams.camera.cy);
            screen = screen / make_float2(optixLaunchParams.camera.fx, optixLaunchParams.camera.fy);
            // generate ray direction
            float3 initialRayOrigin = (float3)camera.position;
            float3 initialRayDir = (float3)normalize(camera.direction
                + screen.x * camera.horizontal
                + screen.y * camera.vertical);
            
            // bool sampleInvariantUpdated = false;
            float3 accumulatedColorPerSample = make_float3(0.);
            float3 accumulatedDirectColorPerSample = make_float3(0.);
            float3 accumulatedIndirectColorPerSample = make_float3(0.);

            for (int indirectSampleID = 0; indirectSampleID < numIndirectPerSample; indirectSampleID++) {
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

                    // First case, EnvMap importance sampling
                    if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y)&& !isnan(prd.pixelColor.z) 
                        && prd.ray_recursie_depth == 1) {
                        tentativeFirstHitColor = prd.pixelColor;
                    }

                    // second case, BRDF sampling that hits the envmap
                    if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y) && !isnan(prd.pixelColor.z) 
                        && prd.ray_recursie_depth == 2 && prd.pixelDirectSampleHitEnvMapFlag) {
                        tentativeFirstHitColor = prd.pixelColor;
                    }
                    
                    // Mirrors
                    if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y) && !isnan(prd.pixelColor.z)
                        && prd.ray_recursie_depth == 2 && prd.pixelPerfectReflectionFlag) {
                        tentativeFirstHitColor = prd.pixelColor;
                    }

                    if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y) && !isnan(prd.pixelColor.z)
                        && prd.ray_recursie_depth == 3 && prd.pixelPerfectReflectionFlag && prd.pixelDirectSampleHitEnvMapFlag) {
                        tentativeFirstHitColor = prd.pixelColor;
                    }

                    if (prd.ray_recursie_depth > RR_DEPTH) {
                        float z = prd.random(); // halton().y;
                        float rr_prob = fmin(fmaxf(prd.light_contribution), 0.95f);

                        if (z < rr_prob) {
                            prd.light_contribution = prd.light_contribution / fmax(rr_prob, 1e-10f);
                        }
                        else {
                            prd.done = true;
                        }
                    }

                    // Hit the light source or exceed the max depth
                    if (prd.done || prd.ray_recursie_depth >= optixLaunchParams.samplingParameter.maximumBounce)
                        break;

                    // Ray dir and Ray origin will be updated in shader
                }
                accumulatedDirectColorPerSample += tentativeFirstHitColor;
                // finished tracing, updating the values
                if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y)&& !isnan(prd.pixelColor.z)) {
                    accumulatedColorPerSample += prd.pixelColor;
                    accumulatedIndirectColorPerSample += prd.pixelColor - tentativeFirstHitColor;
                }

            }

            accumulatedOverSampleColor += accumulatedColorPerSample / (float)numIndirectPerSample;
            accumulatedOverSampleDirectColor += accumulatedDirectColorPerSample / (float)numIndirectPerSample;
            accumulatedOverSampleIndirectColor += accumulatedIndirectColorPerSample / (float)numIndirectPerSample;            
            accumulatedOverSampleFirstHitKd += prdE.pixelFirstHitKd;
            accumulatedOverSampleFirstHitKs += prdE.pixelFirstHitKs;
        }

        // and write/accumulate to frame buffer ...
        // bconst uint32_t 
        uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

        //////////////
        /// Colors
        //////////////
        
        accumulatedOverSampleColor = accumulatedOverSampleColor / (float)numPixelSamples;
        accumulatedOverSampleDirectColor = accumulatedOverSampleDirectColor / (float)numPixelSamples;
        accumulatedOverSampleIndirectColor = accumulatedOverSampleIndirectColor / (float)numPixelSamples;
        accumulatedOverSampleFirstHitKd = accumulatedOverSampleFirstHitKd / (float)numPixelSamples;
        accumulatedOverSampleFirstHitKs = accumulatedOverSampleFirstHitKs / (float)numPixelSamples;
      
        // Do the accumulation
        if (optixLaunchParams.frame.frameID > 0) {
            accumulatedOverSampleColor += float(optixLaunchParams.frame.frameID)
                * make_float3(optixLaunchParams.frame.colorBuffer[fbIndex]);
            accumulatedOverSampleColor /= (optixLaunchParams.frame.frameID + 1.f);
        }

        if (optixLaunchParams.frame.frameID > 0) {
            accumulatedOverSampleDirectColor += float(optixLaunchParams.frame.frameID)
                * make_float3(optixLaunchParams.frame.FirstHitDirectBuffer[fbIndex]);
            accumulatedOverSampleDirectColor /= (optixLaunchParams.frame.frameID + 1.f);
        }

        if (optixLaunchParams.frame.frameID > 0) {
            accumulatedOverSampleIndirectColor += float(optixLaunchParams.frame.frameID)
                * make_float3(optixLaunchParams.frame.FirstHitIndirectBuffer[fbIndex]);
            accumulatedOverSampleIndirectColor /= (optixLaunchParams.frame.frameID + 1.f);
        }

        // float4 vectexIDcolor = (float4)(vec4f((float)lastVertexID.x / 69150, (float)lastVertexID.y / 69150, (float)lastVertexID.z / 69150, 1.f));

        optixLaunchParams.frame.colorBuffer[fbIndex] = make_float4(accumulatedOverSampleColor, 1.f);
        //optixLaunchParams.frame.colorBuffer[fbIndex] = make_float4_vec3f(accumulatedOverSampleIndirectColor / numPixelSamples);
        optixLaunchParams.frame.FirstHitDirectBuffer[fbIndex] = make_float4(accumulatedOverSampleDirectColor, 1.f);
        optixLaunchParams.frame.FirstHitIndirectBuffer[fbIndex] = make_float4(accumulatedOverSampleIndirectColor, 1.f);

        optixLaunchParams.frame.albedoBuffer[fbIndex] = make_float4(prdE.pixelAlbedoFirstHit, 1.f);

        //////////////
        /// Normals
        //////////////
        
        // This is not correct...
        float3 pixelNormal = normalize(prdE.pixelNormalFirstHit);
        float3 worldNormal(pixelNormal / 2 + make_float3(0.5));
        // move to camera space
        float3 cameraNormal = normalize(to_local(normalize(pixelNormal), normalize(camera.direction), normalize(camera.horizontal)));
        cameraNormal = cameraNormal / 2 + make_float3(0.5);

        optixLaunchParams.frame.cameraNormalBuffer[fbIndex] = make_float4(cameraNormal, 1.f);
        optixLaunchParams.frame.worldNormalBuffer[fbIndex] = make_float4(worldNormal, 1.f);

        ////////////////
        /// Positions
        ////////////////
        float3 pixelFirstHitPos = (prdE.pixelFirstHitPos - optixLaunchParams.lowerBound) / optixLaunchParams.boxSize;
        // keep in world space?
        float3 pixelReflectDir = normalize(prdE.pixelFirstHitReflectDir) / 2 + make_float3(0.5);

        optixLaunchParams.frame.FirstHitPositionBuffer[fbIndex] = make_float4(pixelFirstHitPos, 1.f);
        optixLaunchParams.frame.FirstHitReflectDirBuffer[fbIndex] = make_float4(pixelReflectDir, 1.f);
        optixLaunchParams.frame.FirstHitKdBuffer[fbIndex] = make_float4(accumulatedOverSampleFirstHitKd, 1.f);
        optixLaunchParams.frame.FirstHitKsBuffer[fbIndex] = make_float4(accumulatedOverSampleFirstHitKs, 1.f);

        optixLaunchParams.frame.roughnessBuffer[fbIndex] = prdE.pixelFirstHitRoughness;
        optixLaunchParams.frame.envMapFlagBuffer[fbIndex] = prdE.pixelEnvMapFlag;

        ////////////////
        /// Extras
        ////////////////
        if (optixLaunchParams.frame.extraRender) {
            optixLaunchParams.frame.FirstHitBary[fbIndex] = make_float4(prdE.pixelFirstHitBary, 1.f);
            optixLaunchParams.frame.FirstHitVertexID[fbIndex] = make_int4(prdE.pixelFirstHitVertexID, 0);
        }

        ////////////////
        /// Not needed
        ////////////////
        optixLaunchParams.frame.depthBuffer[fbIndex] = prdE.pixeldepth;
        optixLaunchParams.frame.FirstHitUVCordBuffer[fbIndex] = make_float4(prdE.pixelUV_cord, 0.f, 1.f);

        optixLaunchParams.frame.perfectReflectionFlagBuffer[fbIndex] = prd.pixelPerfectReflectionFlag;
        optixLaunchParams.frame.emissiveFlagBuffer[fbIndex] = prdE.pixelEmissiveFlag;


    }
}