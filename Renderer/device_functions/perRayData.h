#pragma once
#include "mathUtils.h"

#define PAYLOAD_TYPE_RADIANCE  OPTIX_PAYLOAD_TYPE_ID_0
#define PAYLOAD_TYPE_OCCLUSION OPTIX_PAYLOAD_TYPE_ID_1

namespace nert_renderer {
    struct PRDExtra {

        // information for denoising
        float3  pixelNormalFirstHit;
        float3  pixelAlbedoFirstHit;

        // Advanced information - Geometry
       
        float3  pixelFirstHitBary;
        int3  pixelFirstHitVertexID;

        float2  pixelUV_cord;
        float3  pixelFirstHitPos;
        float3  pixelFirstHitReflectDir;
        float  pixeldepth;

        // Advanced information - Materials
        float3  pixelFirstHitKd;
        float3  pixelFirstHitKs;
        float  pixelFirstHitRoughness;

        bool   pixelEmissiveFlag;
        bool   pixelEnvMapFlag;
        

        // debug information
        int ix;
        int iy;

        __forceinline__ __device__ void cleanForNextLaunch() {
            pixelNormalFirstHit = make_float3(0.f);
            pixelAlbedoFirstHit = make_float3(0.f);

            pixelFirstHitBary = make_float3(0.f);
            pixelFirstHitVertexID = make_int3(0);

            pixelUV_cord = make_float2(0.f);
            pixelFirstHitPos = make_float3(0.f);
            pixelFirstHitReflectDir = make_float3(0.f);
            pixeldepth = 0.f;

            pixelFirstHitKd = make_float3(0.f);
            pixelFirstHitKs = make_float3(0.f);
            pixelFirstHitRoughness = 0.f;

            pixelEmissiveFlag = false;
            pixelEnvMapFlag = false;
        }
    };

    struct PRD {
        // Random number generator
        // ONLY one will be passed to the prd
        Random random;
        Halton halton;

        // Ray Info
        float3  ray_dir;
        float3  ray_origin;

        // basic information per pixel
        float3  pixelColor;

        // Recursion info
        unsigned int    ray_recursie_depth;
        float3  light_contribution;
        float  pdf;

        // More detail about the recursion
        bool   firstHitUpdated; // used for mirror
        bool   pixelDirectSampleHitEnvMapFlag;
        bool   pixelPerfectReflectionFlag;
        bool   done;
    
        /////
        // Extra information
        /////
        PRDExtra* prdExtraInfo;

        __forceinline__ __device__ PRD(unsigned int randomSeed1, unsigned int randomSeed2, PRDExtra* extraPointer)
        {
            random.init(randomSeed1, randomSeed2);
            unsigned int R11 = 2, R12 = 3, R13 = 5, R14 = 7;
            unsigned int R21 = 3, R22 = 5, R23 = 7, R24 = 2;
            unsigned int iterations = randomSeed1 % 31;
            if (randomSeed2 % 4 == 0) {
                halton.init(R11, R21, iterations);
            }
            else if (randomSeed2 % 4 == 1) {
                halton.init(R12, R22, iterations);
            }
            else if (randomSeed2 % 4 == 2) {
                halton.init(R13, R23, iterations);
            }
            else {
                halton.init(R14, R24, iterations);
            }
            prdExtraInfo = extraPointer;
            cleanForNextLaunch();
        }

        __forceinline__ __device__ PRD(unsigned int _recursive_depth, float3 _pixel_color, float3 _light_contribution, unsigned int randomState, PRDExtra* extraPointer)
        {
            random.state = randomState;
            pixelColor = _pixel_color;

            ray_recursie_depth = _recursive_depth;
            light_contribution = _light_contribution;
            pdf = 1.f;

            firstHitUpdated = false;
            pixelDirectSampleHitEnvMapFlag = false;
            pixelPerfectReflectionFlag = false;
            done = false;

            prdExtraInfo = extraPointer;
        }

        __forceinline__ __device__ void cleanForNextLaunch() {
            pixelColor = make_float3(0.f);
            
            ray_recursie_depth = 0.f;
            light_contribution = make_float3(1.f);
            pdf = 0.f;

            firstHitUpdated = false;
            pixelDirectSampleHitEnvMapFlag = false;
            pixelPerfectReflectionFlag = false;
            done = false;
        }
    };

    //------------------------------------------------------------------------------
    // PRD payload related 
    //------------------------------------------------------------------------------
    

    const unsigned int radiancePayloadSemantics[18] =
    {
        // RadiancePRD::origin, we only cares about where the next trace direction is
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
        // RadiancePRD::direction, we only cares about where the next trace direction is
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,

        // RadiancePRD::pixelColor CH, MS: only write to the current trace color 
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE,

        // RadiancePRD::ray_recursie_depth, caller: keep track of depth, CH: debug only, MS: for direct illumination 
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,

        // RadiancePRD::light_contribution caller: RR, CH: Update using BRDF, MS: Only use it to compute pixelcolor
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ,

        // RadiancePRD::random
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
        
        // RadiancePRD::pdf 
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ,

        // RadiancePRD::done & RadiancePRD::firstHitUpdated & RadiancePRD::pixelDirectSampleHitEnvMapFlag
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE,

        // RadiancePRD::ptr_to_extraInfo
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ | OPTIX_PAYLOAD_SEMANTICS_MS_READ
    };

    const unsigned int occlusionPayloadSemantics[3]
    {
        // occlduded rgb
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
        OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
    };

}