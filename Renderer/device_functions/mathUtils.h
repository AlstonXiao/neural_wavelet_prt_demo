#pragma once
#include <OptiXToolkit/ShaderUtil/color.h>
#include <cstdint>

__device__ __inline__ float3 sqrt(const float3& v) {
    return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}

// This is the random number generator specifically for 
// nert renderer
namespace nert_renderer {
    /*! simple 24-bit linear congruence generator */
    template<unsigned int N = 16>
    struct LCG {

        inline __host__ __device__ LCG()
        { /* intentionally empty so we can use it in device vars that
             don't allow dynamic initialization (ie, PRD) */
        }
        inline __host__ __device__ LCG(unsigned int val0, unsigned int val1)
        {
            init(val0, val1);
        }

        inline __host__ __device__ void init(unsigned int val0, unsigned int val1)
        {
            unsigned int v0 = val0;
            unsigned int v1 = val1;
            unsigned int s0 = 0;

            for (unsigned int n = 0; n < N; n++) {
                s0 += 0x9e3779b9;
                v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
                v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
            }
            state = v0;
        }

        // Generate random unsigned int in [0, 2^24)
        inline __host__ __device__ float operator() ()
        {
            const uint32_t LCG_A = 1664525u;
            const uint32_t LCG_C = 1013904223u;
            state = (LCG_A * state + LCG_C);
            return (state & 0x00FFFFFF) / (float)0x01000000;
        }

        uint32_t state;
    };

    /*! simple 24-bit linear congruence generator */
    template<unsigned int N = 16>
    struct halton {

        inline __host__ __device__ halton()
        { /* intentionally empty so we can use it in device vars that
             don't allow dynamic initialization (ie, PRD) */
            init(2, 3, N);
        }
        inline __host__ __device__ halton(unsigned int val0, unsigned int val1)
        {
            init(val0, val1, N);
        }

        inline __host__ __device__ void init(unsigned int base0, unsigned int base1, unsigned int iterations)
        {
            n0 = n1 = 0.0f;
            d0 = d1 = 1.0f;

            b0 = base0; // should be base 2
            b1 = base1; // should be base 3

            for (unsigned int n = 0; n < iterations; n++) {
                auto ret = operator()();
            }
        }

        // Generate random unsigned int in [0, 2^24)
        inline __host__ __device__ float2 operator() ()
        {
            int x0 = d0 - n0;
            if (x0 == 1) {
                n0 = 1;
                d0 *= b0;
            }
            else {
                int y0 = d0 / b0;
                while (x0 <= y0) {
                    y0 /= b0;
                }
                n0 = (b0 + 1) * y0 - x0;
            }
            state_x = (double)n0 / d0;

            int x1 = d1 - n1;
            if (x1 == 1) {
                n1 = 1;
                d1 *= b1;
            }
            else {
                int y1 = d1 / b1;
                while (x1 <= y1) {
                    y1 /= b1;
                }
                n1 = (b1 + 1) * y1 - x1;
            }
            state_y = (double)n1 / d1;

            return make_float2(state_x, state_y);
        }

        double state_x, state_y;
        uint32_t n0, n1, d0, d1, b0, b1;
    };

    typedef LCG<16> Random;
    typedef halton<16> Halton;
}
