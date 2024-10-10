#pragma once
#include "gdt/math/AffineSpace.h"
#include "prt_globals.h"

namespace prt {
    template <typename T>
    __global__ void waveletDataCopy(
        const int2 size,
        const float* __restrict__ locationBuffer,
        const float* __restrict__ normalBuffer,
        const float* __restrict__ reflectBuffer,
        const float* __restrict__ roughnessBuffer,
        const float* __restrict__ kdBuffer,
        const float* __restrict__ ksBuffer,
        T* __restrict__ out
    ) {
        const uint32_t pixelX = threadIdx.x + blockIdx.x * blockDim.x;
        const uint32_t pixelY = threadIdx.y + blockIdx.y * blockDim.y;
        if (pixelX >= size.x) return;
        if (pixelY >= size.y) return;
        const float rgh_den = 1.f / logf(25.f + 1.f);

        const uint32_t idx = pixelX + size.x * pixelY;
        #pragma unroll
        for (int i = 0; i < WAVELETNUM; i++) {
            const uint32_t start_idx = idx * INPUT_BUFFER_SIZE * WAVELETNUM + i * INPUT_BUFFER_SIZE;
            out[start_idx + INPUT_BUFFER_LOCATION_IDX + 0] = (T)locationBuffer[idx * 4 + 0];
            out[start_idx + INPUT_BUFFER_LOCATION_IDX + 1] = (T)locationBuffer[idx * 4 + 1];
            out[start_idx + INPUT_BUFFER_LOCATION_IDX + 2] = (T)locationBuffer[idx * 4 + 2];

            out[start_idx + INPUT_BUFFER_NORMAL_IDX + 0] = (T)normalBuffer[idx * 4 + 0];
            out[start_idx + INPUT_BUFFER_NORMAL_IDX + 1] = (T)normalBuffer[idx * 4 + 1];
            out[start_idx + INPUT_BUFFER_NORMAL_IDX + 2] = (T)normalBuffer[idx * 4 + 2];

            out[start_idx + INPUT_BUFFER_REFLECT_IDX + 0] = (T)reflectBuffer[idx * 4 + 0];
            out[start_idx + INPUT_BUFFER_REFLECT_IDX + 1] = (T)reflectBuffer[idx * 4 + 1];
            out[start_idx + INPUT_BUFFER_REFLECT_IDX + 2] = (T)reflectBuffer[idx * 4 + 2];

            out[start_idx + INPUT_BUFFER_ROUGHNESS_IDX] = (T)logf(25.f*roughnessBuffer[idx]+1.f) * rgh_den;

            out[start_idx + INPUT_BUFFER_KD_IDX + 0] = (T)(2*kdBuffer[idx * 4 + 0]-1);
            out[start_idx + INPUT_BUFFER_KD_IDX + 1] = (T)(2*kdBuffer[idx * 4 + 1]-1);
            out[start_idx + INPUT_BUFFER_KD_IDX + 2] = (T)(2*kdBuffer[idx * 4 + 2]-1);

            out[start_idx + INPUT_BUFFER_KS_IDX + 0] = (T)(2*ksBuffer[idx * 4 + 0]-1);
            out[start_idx + INPUT_BUFFER_KS_IDX + 1] = (T)(2*ksBuffer[idx * 4 + 1]-1);
            out[start_idx + INPUT_BUFFER_KS_IDX + 2] = (T)(2*ksBuffer[idx * 4 + 2]-1);
        }
    }

    // In the naive implementation, we add first hit with respone. 
    __global__ void computeResponse(
        const int2 size,
        const float* __restrict__ responseBuffer,
        const float* __restrict__ firstHitBuffer,
        const float* __restrict__ waveletLightBuffer,
        const bool* __restrict__ envMapFlag,
        float* __restrict__ predictedIndirect,
        float* __restrict__ predictedIndirectWithDirect
    ) {
        int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
        int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
        if (pixelX >= size.x) return;
        if (pixelY >= size.y) return;

        int idx = pixelX + size.x * pixelY;
        const uint32_t startIdx = idx * WAVELETNUM * 3;

        float waveletColorR = 0.f;
        float waveletColorG = 0.f;
        float waveletColorB = 0.f;
        if (envMapFlag[idx]) {
            #pragma unroll
            for (int i = 0; i < WAVELETNUM; i++) {
                waveletColorR += fmaxf(waveletLightBuffer[i * 3 + 0] * responseBuffer[startIdx + i * 3 + 0], 0.0f);
                waveletColorG += fmaxf(waveletLightBuffer[i * 3 + 1] * responseBuffer[startIdx + i * 3 + 1], 0.0f);
                waveletColorB += fmaxf(waveletLightBuffer[i * 3 + 2] * responseBuffer[startIdx + i * 3 + 2], 0.0f);
            }
        }
        predictedIndirect[idx * 4 + 0] = waveletColorR;
        predictedIndirect[idx * 4 + 1] = waveletColorG;
        predictedIndirect[idx * 4 + 2] = waveletColorB;
        predictedIndirect[idx * 4 + 3] = 1.0f;

        predictedIndirectWithDirect[idx * 4 + 0] = waveletColorR + firstHitBuffer[idx * 4 + 0];
        predictedIndirectWithDirect[idx * 4 + 1] = waveletColorG + firstHitBuffer[idx * 4 + 1];
        predictedIndirectWithDirect[idx * 4 + 2] = waveletColorB + firstHitBuffer[idx * 4 + 2];
        predictedIndirectWithDirect[idx * 4 + 3] = 1.0f;

    }

    /// <summary>
    /// Obtain the wavelet feature vector, and multiply with the 
    /// vertex feature vector
    /// only god knows how it is done now...
    /// </summary>
    template <typename T>
    __global__ void ptwise_mlt(
        const uint32_t n_elements,
        const uint32_t batchSize,
        const T* __restrict__ inp_a,
        const T* __restrict__ waveletWeigthVector,
        const int* __restrict__ waveletCoefs,
        T* __restrict__ out
    ) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        int hiddenVectorLocation = i / (batchSize); // [0, WAVELETNUM]
        int whichInput = i % (batchSize); // location in the input
        int whichWavelet = whichInput % WAVELETNUM; // which wavelet 0 - numwavelet

        // 24576 = 6 * 64 * 64, note this is in dimension major order, not feature major order
        out[i] = inp_a[i] * waveletWeigthVector[waveletCoefs[whichWavelet] + hiddenVectorLocation * 24576];
    }

    template <typename T>
    __global__ void scale_by_arr(
        const uint32_t n_elements,
        const T* __restrict__ inp,
        T* __restrict__ out
    ) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        out[i] *= inp[i];
    }

    template <typename T>
    __global__ void cpy(
        const uint32_t n_elements,
        const T* __restrict__ inp,
        T* __restrict__ out
    ) {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        out[i] = inp[i];
    }
}