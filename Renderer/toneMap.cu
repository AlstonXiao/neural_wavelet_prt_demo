// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "RenderEngine.h"

using namespace nert_renderer;

namespace nert_renderer {

    inline __device__ float4 sqrt(float4 f)
    {
    return make_float4(sqrtf(f.x),
                        sqrtf(f.y),
                        sqrtf(f.z),
                        sqrtf(f.w));
    }
    inline __device__ float  clampf(float f) { return min(1.f, max(0.f, f)); }
    inline __device__ float4 power(float4 val, float f) 
    {
        return make_float4(pow(val.x, f),
            pow(val.y, f),
            pow(val.z, f),
            pow(val.w, f));
    }
    inline __device__ float4 clamp(float4 f)
    {
    return make_float4(clampf(f.x),
                        clampf(f.y),
                        clampf(f.z),
                        clampf(f.w));
    }
  
    /*! runs a cuda kernel that performs gamma correction and float4-to-rgba conversion */
    __global__ void computeFinalPixelColorsKernel(uint32_t* __restrict__ finalColorBuffer,
                                                const float4* __restrict__ denoisedBuffer,
                                                vec2i     size)
    {
    int pixelX = threadIdx.x + blockIdx.x*blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y*blockDim.y;
    if (pixelX >= size.x) return;
    if (pixelY >= size.y) return;

    int pixelID = pixelX + size.x*pixelY;

    float4 f4 = denoisedBuffer[pixelID];
    float srgb_exponential_coeff = 1.055;
    float srgb_exponent = 2.4;

    float4 tensor_nonlinear = srgb_exponential_coeff * (power(f4 + 1e-6, 1 / srgb_exponent)) - (srgb_exponential_coeff - 1);
    tensor_nonlinear.w = 1.f;
    f4 = clamp(tensor_nonlinear);
    uint32_t rgba = 0;
    rgba |= (uint32_t)(f4.x * 255.9f) <<  0;
    rgba |= (uint32_t)(f4.y * 255.9f) <<  8;
    rgba |= (uint32_t)(f4.z * 255.9f) << 16;
    rgba |= (uint32_t)255             << 24;
    finalColorBuffer[pixelID] = rgba;
    }

    void RenderEngine::computeFinalPixelColors(int frameID, CUstream stream, renderConfig config)
    {
        vec2i fbSize = launchParams.frame.size;
        vec2i blockSize = 32;
        vec2i numBlocks = divRoundUp(fbSize,blockSize);
        CUdeviceptr source_ptr = NULL;

        if (config.waveletMode) {
            if (config.renderingMode == finalColorBufferContent::full)
                source_ptr = fullframeCUDABuffer[frameID].fbPredictedIndirectWithDirectDenoised.d_pointer();
            else if (config.renderingMode == finalColorBufferContent::indirect)
                source_ptr = fullframeCUDABuffer[frameID].fbPredictedIndirectDenoised.d_pointer();
        }
        else {
            if (config.renderingMode == finalColorBufferContent::full)
                source_ptr = fullframeCUDABuffer[frameID].fbDenoised.d_pointer();
            else if (config.renderingMode == finalColorBufferContent::indirect)
                source_ptr = fullframeCUDABuffer[frameID].fbIndirectDenoised.d_pointer();
            else if (config.renderingMode == finalColorBufferContent::direct)
                source_ptr = fullframeCUDABuffer[frameID].fbDirectDenoised.d_pointer();
        }
        computeFinalPixelColorsKernel
        <<< dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y), 0, stream >>>
        ((uint32_t*)fullframeCUDABuffer[frameID].fbFinalColor.d_pointer(),
            (float4*)source_ptr,fbSize);

    }  
} // ::nert_renderer
