#pragma once
#include"gdt/math/AffineSpace.h"
#include"optix7.h"
#include "CUDABuffer.h"
#include "device_functions/LaunchParams.h"
#include "Scene.h"

namespace nert_renderer {
    using namespace gdt;
    struct fullframe {
        fullframe(vec2i size) {
            assignSize(size);
        }

        ~fullframe() {
            delete[]perfectReflectionFlag;
            delete[]environmentMapFlag;
            delete[]emissiveFlag;
        }
        std::vector<vec4f> FinalColor;
        std::vector<vec4f> DirectColor;
        std::vector<vec4f> IndirectColor;

        std::vector<vec4f> worldNormalBuffer;
        std::vector<vec4f> cameraNormalBuffer;
        std::vector<vec4f> albedoBuffer;

        std::vector<vec4f> FirstHitBary;
        std::vector<vec4i> FirstHitVertexID;

        std::vector<vec4f> FirstHitUVCoord;
        std::vector<vec4f> firstHitPos;
        std::vector<vec4f> firstHitReflecDir;

        std::vector<vec4f> firstHitKd;
        std::vector<vec4f> firstHitKs;

        std::vector<vec4f> Speccolor;
        std::vector<vec4f> Diffcolor;

        std::vector<float> depth;
        std::vector<float> roughness;

        std::vector<vec4f> predictedIndirect;
        std::vector<vec4f> predictedIndirectWithDirect;

        bool* perfectReflectionFlag;
        bool* emissiveFlag;
        bool* environmentMapFlag;

        vec2i currentSize = vec2i(0);

        void assignSize(vec2i newSize) {
            if (newSize.x * newSize.y == currentSize.x * currentSize.y)
                return;
            currentSize = newSize;
            size_t totalSize = newSize.x * newSize.y;

            FinalColor.resize(totalSize);
            DirectColor.resize(totalSize);
            IndirectColor.resize(totalSize);

            worldNormalBuffer.resize(totalSize);
            cameraNormalBuffer.resize(totalSize);
            albedoBuffer.resize(totalSize);

            FirstHitBary.resize(totalSize);
            FirstHitVertexID.resize(totalSize);

            FirstHitUVCoord.resize(totalSize);
            firstHitPos.resize(totalSize);
            firstHitReflecDir.resize(totalSize);

            firstHitKd.resize(totalSize);
            firstHitKs.resize(totalSize);

            Speccolor.resize(totalSize);
            Diffcolor.resize(totalSize);

            depth.resize(totalSize);
            roughness.resize(totalSize);

            predictedIndirect.resize(totalSize);
            predictedIndirectWithDirect.resize(totalSize);
            
            // if (perfectReflectionFlag) delete[] perfectReflectionFlag; 
            // if (emissiveFlag) delete[] emissiveFlag; 
            // if (environmentMapFlag) delete[] environmentMapFlag; 

            perfectReflectionFlag = new bool[totalSize];
            emissiveFlag = new bool[totalSize];
            environmentMapFlag = new bool[totalSize];
        }
    };

    struct fullFrameCUDA {
        /*! the color buffer we use during _rendering_, which is a bit
        larger than the actual displayed frame buffer (to account for
        the border), and in float4 format (the denoiser requires
        floats) */
        CUDABuffer fbColor;
        CUDABuffer fbFirstHitDirect;
        CUDABuffer fbFirstHitIndirect;

        CUDABuffer fbworldNormal;
        CUDABuffer fbCameraNormal;
        CUDABuffer fbAlbedo;

        CUDABuffer fbFirstHitBary;
        CUDABuffer fbFirstHitVertexID;

        CUDABuffer fbFirstHitUV;
        CUDABuffer fbFirstHitPos;
        CUDABuffer fbFirstHitReflecDir;

        CUDABuffer fbFirstHitKd;
        CUDABuffer fbFirstHitKs;

        CUDABuffer fbSpecColor;
        CUDABuffer fbDiffColor;

        CUDABuffer fbDepth;
        CUDABuffer fbRoughness;

        CUDABuffer fbPerfectReflectionFlag;
        CUDABuffer fbEmissiveFlag;
        CUDABuffer fbEnvironmentMapFlag;

        /*! output of the denoiser pass, in float4 */
        CUDABuffer fbDenoised;
        CUDABuffer fbDirectDenoised;
        CUDABuffer fbIndirectDenoised;

        /*! output of the TCNN result */
        CUDABuffer fbPredictedIndirect;
        CUDABuffer fbPredictedIndirectDenoised;
        CUDABuffer fbPredictedIndirectWithDirect;
        CUDABuffer fbPredictedIndirectWithDirectDenoised;

        /* the actual tonemapped final color buffer used for display, in rgba8 */
        CUDABuffer fbFinalColor;

        void resize(const vec2i& newSize) {
            size_t totalSize = static_cast<size_t>(newSize.x) *
                static_cast<size_t>(newSize.y);
            fbColor.resize(totalSize * sizeof(float4));
            fbFirstHitDirect.resize(totalSize * sizeof(float4));
            fbFirstHitIndirect.resize(totalSize * sizeof(float4));

            fbworldNormal.resize(totalSize * sizeof(float4));
            fbCameraNormal.resize(totalSize * sizeof(float4));
            fbAlbedo.resize(totalSize * sizeof(float4));

            fbFirstHitBary.resize(totalSize * sizeof(float4));
            fbFirstHitVertexID.resize(totalSize * sizeof(int4));

            fbFirstHitUV.resize(totalSize * sizeof(float4));
            fbFirstHitPos.resize(totalSize * sizeof(float4));
            fbFirstHitReflecDir.resize(totalSize * sizeof(float4));

            fbFirstHitKd.resize(totalSize * sizeof(float4));
            fbFirstHitKs.resize(totalSize * sizeof(float4));

            fbSpecColor.resize(totalSize * sizeof(float4));
            fbDiffColor.resize(totalSize * sizeof(float4));

            fbDepth.resize(totalSize * sizeof(float));
            fbRoughness.resize(totalSize * sizeof(float));
            fbPerfectReflectionFlag.resize(totalSize * sizeof(bool));
            fbEmissiveFlag.resize(totalSize * sizeof(bool));
            fbEnvironmentMapFlag.resize(totalSize * sizeof(bool));

            fbDenoised.resize(totalSize * sizeof(float4));
            fbDirectDenoised.resize(totalSize * sizeof(float4));
            fbIndirectDenoised.resize(totalSize * sizeof(float4));

            fbPredictedIndirect.resize(totalSize * sizeof(float4));
            fbPredictedIndirectDenoised.resize(totalSize * sizeof(float4));
            fbPredictedIndirectWithDirect.resize(totalSize * sizeof(float4));
            fbPredictedIndirectWithDirectDenoised.resize(totalSize * sizeof(float4));

            fbFinalColor.resize(totalSize * sizeof(uint32_t));
        }

        void assignBufferToLaunch(LaunchParams& Params) {
            Params.frame.colorBuffer = (float4*)fbColor.d_pointer();
            Params.frame.FirstHitDirectBuffer = (float4*)fbFirstHitDirect.d_pointer();
            Params.frame.FirstHitIndirectBuffer = (float4*)fbFirstHitIndirect.d_pointer();

            Params.frame.worldNormalBuffer = (float4*)fbworldNormal.d_pointer();
            Params.frame.cameraNormalBuffer = (float4*)fbCameraNormal.d_pointer();
            Params.frame.albedoBuffer = (float4*)fbAlbedo.d_pointer();

            Params.frame.FirstHitBary = (float4*)fbFirstHitBary.d_pointer();
            Params.frame.FirstHitVertexID = (int4*)fbFirstHitVertexID.d_pointer();

            Params.frame.FirstHitUVCordBuffer = (float4*)fbFirstHitUV.d_pointer();
            Params.frame.FirstHitPositionBuffer = (float4*)fbFirstHitPos.d_pointer();
            Params.frame.FirstHitReflectDirBuffer = (float4*)fbFirstHitReflecDir.d_pointer();

            Params.frame.FirstHitKdBuffer = (float4*)fbFirstHitKd.d_pointer();
            Params.frame.FirstHitKsBuffer = (float4*)fbFirstHitKs.d_pointer();

            Params.frame.SpecColorBuffer = (float4*)fbSpecColor.d_pointer();
            Params.frame.DiffColorBuffer = (float4*)fbDiffColor.d_pointer();

            Params.frame.depthBuffer = (float*)fbDepth.d_pointer();
            Params.frame.roughnessBuffer = (float*)fbRoughness.d_pointer();

            Params.frame.perfectReflectionFlagBuffer = (bool*)fbPerfectReflectionFlag.d_pointer();
            Params.frame.emissiveFlagBuffer = (bool*)fbEmissiveFlag.d_pointer();
            Params.frame.envMapFlagBuffer = (bool*)fbEnvironmentMapFlag.d_pointer();
        }
    };
}