
/*
    pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


// core/sh.cpp*
#include "sh.h"
#include <float.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "LaunchParams.h"
#include "util.h"
#include <assert.h>

// Spherical Harmonics Local Definitions
static void legendrep(float x, int lmax, float *out) {
    #define P(l,m) out[SHIndex(l,m)]
    // Compute $m=0$ Legendre values using recurrence
    P(0,0) = 1.f;
    P(1,0) = x;
    for (int l = 2; l <= lmax; ++l)
    {
        P(l, 0) = ((2*l-1)*x*P(l-1,0) - (l-1)*P(l-2,0)) / l;
        assert(!isnan(P(l, 0)));
        assert(!isinf(P(l, 0)));
    }

    // Compute $m=l$ edge using Legendre recurrence
    float neg = -1.f;
    float dfact = 1.f;
    float xroot = sqrtf(fmaxf(0.f, 1.f - x*x));
    float xpow = xroot;
    for (int l = 1; l <= lmax; ++l) {
        P(l, l) = neg * dfact * xpow;
        assert(!isnan(P(l, l)));
        assert(!isinf(P(l, l)));
        neg *= -1.f;      // neg = (-1)^l
        dfact *= 2*l + 1; // dfact = (2*l-1)!!
        xpow *= xroot;    // xpow = powf(1.f - x*x, float(l) * 0.5f);
    }

    // Compute $m=l-1$ edge using Legendre recurrence
    for (int l = 2; l <= lmax; ++l)
    {
        P(l, l-1) = x * (2*l-1) * P(l-1, l-1);
        assert(!isnan(P(l, l-1)));
        assert(!isinf(P(l, l-1)));
    }

    // Compute $m=1, \ldots, l-2$ values using Legendre recurrence
    for (int l = 3; l <= lmax; ++l)
        for (int m = 1; m <= l-2; ++m)
        {
            P(l, m) = ((2 * (l-1) + 1) * x * P(l-1,m) -
                       (l-1+m) * P(l-2,m)) / (l - m);
            assert(!isnan(P(l, m)));
            assert(!isinf(P(l, m)));
        }
    #if 0
        // wrap up with the negative m ones now
        // P(l,-m)(x) = -1^m (l-m)!/(l+m)! P(l,m)(x)
        for (int l = 1; l <= lmax; ++l) {
            float fa = 1.f, fb = fact(2*l);
            // fa = fact(l+m), fb = fact(l-m)
            for (int m = -l; m < 0; ++m) {
                float neg = ((-m) & 0x1) ? -1.f : 1.f;
                P(l,m) = neg * fa/fb * P(l,-m);
                fb /= l-m;
                fa *= (l+m+1) > 1 ? (l+m+1) : 1.;
            }
        }
    #endif
    #undef P
}


static inline float fact(float v);
static inline float divfact(int a, int b);
static inline float K(int l, int m) {
    return sqrtf((2.f * l + 1.f) * divfact(l, m) / (4 * M_PI));
}


static inline float divfact(int a, int b) {
    if (b == 0) return 1.f;
    float fa = a, fb = fabsf(b);
    float v = 1.f;
    for (float x = fa-fb+1.f; x <= fa+fb; x += 1.f)
        v *= x;
    return 1.f / v;
}


// n!! = 1 if n==0 or 1, otherwise n * (n-2)!!
static float dfact(float v) {
    if (v <= 1.f) return 1.f;
    return v * dfact(v - 2.f);
}


static inline float fact(float v) {
    if (v <= 1.f) return 1.f;
    return v * fact(v - 1.f);
}


static void sinCosIndexed(float s, float c, int n,
                          float *sout, float *cout) {
    float si = 0, ci = 1;
    for (int i = 0; i < n; ++i) {
        // Compute $\sin{}i\phi$ and $\cos{}i\phi$ using recurrence
        *sout++ = si;
        *cout++ = ci;
        float oldsi = si;
        si = si * c + ci * s;
        ci = ci * c - oldsi * s;
    }
}


static void toZYZ(const float* &m, float *alpha, float *beta, float *gamma) {
    #define M(a, b) m[4*a + b]

    float sy = sqrtf(M(2,1)*M(2,1) + M(2,0)*M(2,0));
    if (sy > 16*FLT_EPSILON) {
        *gamma = -atan2f(M(1,2), -M(0,2));
        *beta  = -atan2f(sy, M(2,2));
        *alpha = -atan2f(M(2,1), M(2,0));
    } else {
        *gamma =  0;
        *beta  = -atan2f(sy, M(2,2));
        *alpha = -atan2f(-M(1,0), M(1,1));
    }
#undef M
}


static inline float lambda(float l) {
    return sqrtf((4.f * M_PI) / (2.f * l + 1.));
}



// Spherical Harmonics Definitions
void SHEvaluate(const vec3f &w, int lmax, float *out) {
//    if (lmax > 28) {
//        Error("SHEvaluate() runs out of numerical precision for lmax > 28. "
//               "If you need more bands, try recompiling using doubles.");
//    }
    assert(lmax <= 28);

    // Compute Legendre polynomial values for $\cos\theta$
    assert(w.Length() > .995f && w.Length() < 1.005f);
    legendrep(w.z, lmax, out);

    // Compute $K_l^m$ coefficients
    float Klm[SHTerms(lmax)];
    for (int l = 0; l <= lmax; ++l)
        for (int m = -l; m <= l; ++m)
            Klm[SHIndex(l, m)] = K(l, m);

    // Compute $\sin\phi$ and $\cos\phi$ values
    float sins[lmax + 1], coss[lmax + 1]; 
    // float *sins = ALLOCA(float, lmax+1), *coss = ALLOCA(float, lmax+1);
    float xyLen = sqrtf(fmaxf(0.f, 1.f - w.z*w.z));
    if (xyLen == 0.f) {
        for (int i = 0; i <= lmax; ++i) sins[i] = 0.f;
        for (int i = 0; i <= lmax; ++i) coss[i] = 1.f;
    }
    else
        sinCosIndexed(w.y / xyLen, w.x / xyLen, lmax+1, sins, coss);

    // Apply SH definitions to compute final $(l,m)$ values
    static const float sqrt2 = sqrtf(2.f);
    for (int l = 0; l <= lmax; ++l) {
        for (int m = -l; m < 0; ++m)
        {
            out[SHIndex(l, m)] = sqrt2 * Klm[SHIndex(l, m)] *
                out[SHIndex(l, -m)] * sins[-m];
            assert(!isnan(out[SHIndex(l,m)]));
            assert(!isinf(out[SHIndex(l,m)]));
        }
        out[SHIndex(l, 0)] *= Klm[SHIndex(l, 0)];
        for (int m = 1; m <= l; ++m)
        {
            out[SHIndex(l, m)] *= sqrt2 * Klm[SHIndex(l, m)] * coss[m];
            assert(!isnan(out[SHIndex(l,m)]));
            assert(!isinf(out[SHIndex(l,m)]));
        }
    }
}

float* SHWriteImage(const char *filename, const vec3f *c, int lmax, int yres) {
    int xres = 2 * yres;
    float *rgb = new float[xres * yres * 3];
    float *Ylm = new float[SHTerms(lmax)]; 
    for (int y = 0; y < yres; ++y) {
        float theta = (float(y) + 0.5f) / float(yres) * M_PI;
        for (int x = 0; x < xres; ++x) {
            float phi = (float(x) + 0.5f) / float(xres) * 2.f * M_PI;
            // Compute RGB color for direction $(\theta,\phi)$ from SH coefficients
            vec3f w = SphericalDirection(sinf(theta), cosf(theta), phi);
            SHEvaluate(w, lmax, Ylm);
            vec3f val = vec3f(0.0f);
            for (int i = 0; i < SHTerms(lmax); ++i) {
                val += Ylm[i] * c[i];
            }
            rgb[3*(y * yres + xres) + 0] = val[0];
            rgb[3*(y * yres + xres) + 1] = val[1];
            rgb[3*(y * yres + xres) + 2] = val[2];
        }
    }

    delete Ylm;
    return rgb;
}

void SHProjectIncidentIndirectRadiance(vec3f p, int lmax, int ns, vec3f *c_i,
        unsigned int random_seed_1, unsigned int random_seed_2, int nSamples) {
    /* SET UP PER-RAY DATA */
    PRD prd;
    initPRD(prd, random_seed_1, random_seed_2);
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);
    vec3f p = prd.ray_origin;

    /* SET UP MONTE CARLO VARIABLES */
    float *Ylm = new float[SHTerms(lmax)]; 
    vec3f accumulatedIndirectColorPerSample = vec3f(0.f);

    /* COMPUTE THE LIGHT CONTRIBUTION AT p */
    for (int i = 0; i < nSamples; ++i) {
        // Sample incident direction for radiance probe
        float u0 = prd.random(); 
        float u1 = prd.random(); 
        vec3f wi;
        sample_cos_hemisphere_dir(u0, u1, wi);

        // Compute incident radiance along direction for probe
        vec3f tentativeFirstHitColor(0.f);
        for (;;) {
            resetPRD(prd, p, wi);
            optixTrace(optixLaunchParams.traversable,
                prd.ray_origin,
                prd.ray_dir,
                1e-2f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                RADIANCE_RAY_TYPE,            // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                RADIANCE_RAY_TYPE,            // missSBTIndex 
                u0, u1);
            prd.ray_recursie_depth++;
            // Assuming here that optixTrace will call the kernels
            // to update ray_dir, ray_origin and shadow rays
            // as well as compositing the light accumulation

            // First case, EnvMap importance sampling
            if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y)&& !isnan(prd.pixelColor.z) 
                && prd.ray_recursie_depth == 1 && prd.pixelDirectSampleHitEnvMapFlag) {
                // Hit the environment map on the first try.
                // This direct illumination should not contribute.
                break;
            }

            // second case, BRDF sampling that hits the envmap
            // since this is an indirect ray, we can calculate the contribution
            if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y) && !isnan(prd.pixelColor.z) 
                && prd.ray_recursie_depth == 2 && prd.pixelDirectSampleHitEnvMapFlag) {
                tentativeFirstHitColor = prd.pixelColor;
            }

            if (prd.ray_recursie_depth > RR_DEPTH) {
                float z = prd.random();
                float rr_prob = fmin(fmaxf(prd.light_contribution), 0.95f);

                if (z < rr_prob) {
                    prd.light_contribution = prd.light_contribution / fmax(rr_prob, 1e-10);
                }
                else {
                    prd.done = true;
                }
            }

            // Hit the light source or exceed the max depth
            if (prd.done || prd.ray_recursie_depth >= MAX_TRACE_DEPTH || prd.ray_recursie_depth >= optixLaunchParams.samplingParameter.maximumBounce)
                break;

        }
        // finished tracing, updating the values
        if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y)&& !isnan(prd.pixelColor.z)) {
            accumulatedIndirectColorPerSample += prd.pixelColor;
        }

        /* PROJECT LIGHT CONTRIBUTION TO SH */
        // Correct to do division by nSamples after?
        SHEvaluate(wi, lmax, Ylm);
        for (int j = 0; j < SHTerms(lmax); ++j)
            c_i[j] += Ylm[j] * accumulatedIndirectColorPerSample / (prd.pdf * nSamples);

    }
    delete Ylm;
}

void SHReduceRinging(vec3f *c, int lmax, float lambda) {
    for (int l = 0; l <= lmax; ++l) {
        float scale = 1.f / (1.f + lambda * l * l * (l + 1) * (l + 1));
        for (int m = -l; m <= l; ++m)
            c[SHIndex(l, m)] *= scale;
    }
}

void SHRotate(const vec3f *c_in, vec3f *c_out, const float* &m,
              int lmax) {
    float alpha, beta, gamma;
    toZYZ(m, &alpha, &beta, &gamma);
    vec3f *work = new vec3f[SHTerms(max)];
    SHRotateZ(c_in, c_out, gamma, lmax);
    SHRotateXPlus(c_out, work, lmax);
    SHRotateZ(work, c_out, beta, lmax);
    SHRotateXMinus(c_out, work, lmax);
    SHRotateZ(work, c_out, alpha, lmax);
    delete work;
}


void SHRotateZ(const vec3f *c_in, vec3f *c_out, float alpha,
               int lmax) {
    assert(c_in != c_out);
    c_out[0] = c_in[0];
    if (lmax == 0) return;
    // Precompute sine and cosine terms for $z$-axis SH rotation
    float *ct = new float[lmax + 1];
    float *st = new float[lmax + 1];
    sinCosIndexed(sinf(alpha), cosf(alpha), lmax+1, st, ct);
    for (int l = 1; l <= lmax; ++l) {
        // Rotate coefficients for band _l_ about $z$
        for (int m = -l; m < 0; ++m)
            c_out[SHIndex(l, m)] =
                ( ct[-m] * c_in[SHIndex(l,  m)] +
                 -st[-m] * c_in[SHIndex(l, -m)]);
        c_out[SHIndex(l, 0)] = c_in[SHIndex(l, 0)];
        for (int m = 1; m <= l; ++m)
            c_out[SHIndex(l, m)] =
                (ct[m] * c_in[SHIndex(l,  m)] +
                 st[m] * c_in[SHIndex(l, -m)]);
    }
}


void SHConvolveCosTheta(int lmax, const vec3f *c_in,
                        vec3f *c_out) {
    static const float c_costheta[18] = { 0.8862268925, 1.0233267546,
        0.4954159260, 0.0000000000, -0.1107783690, 0.0000000000,
        0.0499271341, 0.0000000000, -0.0285469331, 0.0000000000,
        0.0185080823, 0.0000000000, -0.0129818395, 0.0000000000,
        0.0096125342, 0.0000000000, -0.0074057109, 0.0000000000 };
    for (int l = 0; l <= lmax; ++l)
        for (int m = -l; m <= l; ++m) {
            int o = SHIndex(l, m);
            if (l < 18) c_out[o] = lambda(l) * c_in[o] * c_costheta[l];
            else        c_out[o] = 0.f;
        }
}


void SHConvolvePhong(int lmax, float n, const vec3f *c_in,
        vec3f *c_out) {
    for (int l = 0; l <= lmax; ++l) {
        float c_phong = expf(-(l*l) / (2.f * n));
        for (int m = -l; m <= l; ++m) {
            int o = SHIndex(l, m);
            c_out[o] = lambda(l) * c_in[o] * c_phong;
        }
    }
}


// If glossy doesn't work, then maybe we try with just diffuse
// as a sanity check

//void SHComputeDiffuseTransfer(const Point &p, const Normal &n,
//        float rayEpsilon, const Scene *scene, RNG &rng, int nSamples,
//        int lmax, vec3f *c_transfer) {
//    for (int i = 0; i < SHTerms(lmax); ++i)
//        c_transfer[i] = 0.f;
//    uint32_t scramble[2] = { rng.RandomUInt(), rng.RandomUInt() };
//    float *Ylm = ALLOCA(float, SHTerms(lmax));
//    for (int i = 0; i < nSamples; ++i) {
//        // Sample _i_th direction and compute estimate for transfer coefficients
//        float u[2];
//        Sample02(i, scramble, u);
//        Vector w = UniformSampleSphere(u[0], u[1]);
//        float pdf = UniformSpherePdf();
//        if (Dot(w, n) > 0.f && !scene->IntersectP(Ray(p, w, rayEpsilon))) {
//            // Accumulate contribution of direction $\w{}$ to transfer coefficients
//            SHEvaluate(w, lmax, Ylm);
//            for (int j = 0; j < SHTerms(lmax); ++j)
//                c_transfer[j] += (Ylm[j] * AbsDot(w, n)) / (pdf * nSamples);
//        }
//    }
//}
//

void SHComputeTransferMatrix(const vec3f &p, int nSamples, int lmax, float *T) {
    /* SET UP PER-RAY DATA */
    PRD prd;
    initPRD(prd, random_seed_1, random_seed_2);
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);
    vec3f p = prd.ray_origin;

    /* SET UP MONTE CARLO VARIABLES */
    float *Ylm = new float[SHTerms(lmax)]; 
    int nSamples = 1024;
    vec3f accumulatedIndirectColorPerSample = vec3f(0.f);
    for (int i = 0; i < SHTerms(lmax)*SHTerms(lmax); ++i)
        T[i] = 0.f;

    /* Compute Monte Carlo estimate of $i$th sample for transfer matrix */
    for (int i = 0; i < nSamples; ++i) {
        u0 = prd.random();
        u1 = prd.random();
        vec3f wi;
        sample_cos_hemisphere_dir(u0, u1, wi);
        vec3f tentativeFirstHitColor(0.f);
        for (;;) {
            resetPRD(prd, p, wi);
            optixTrace(optixLaunchParams.traversable,
                prd.ray_origin,
                prd.ray_dir,
                1e-2f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                RADIANCE_RAY_TYPE,            // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                RADIANCE_RAY_TYPE,            // missSBTIndex 
                u0, u1);
            prd.ray_recursie_depth++;
            // Assuming here that optixTrace will call the kernels
            // to update ray_dir, ray_origin and shadow rays
            // as well as compositing the light accumulation

            // First case, EnvMap importance sampling
            if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y)&& !isnan(prd.pixelColor.z) 
                && prd.ray_recursie_depth == 1 && prd.pixelDirectSampleHitEnvMapFlag) {
                // Hit the environment map on the first try.
                // This direct illumination should not contribute.
                break;
            }

            // second case, BRDF sampling that hits the envmap
            // since this is an indirect ray, we can calculate the contribution
            if (!isnan(prd.pixelColor.x) && !isnan(prd.pixelColor.y) && !isnan(prd.pixelColor.z) 
                && prd.ray_recursie_depth == 2 && prd.pixelDirectSampleHitEnvMapFlag) {
                tentativeFirstHitColor = prd.pixelColor;
            }

            if (prd.ray_recursie_depth > RR_DEPTH) {
                float z = prd.random();
                float rr_prob = fmin(fmaxf(prd.light_contribution), 0.95f);

                if (z < rr_prob) {
                    prd.light_contribution = prd.light_contribution / fmax(rr_prob, 1e-10);
                }
                else {
                    prd.done = true;
                }
            }

            // Hit the light source or exceed the max depth
            if (prd.done || prd.ray_recursie_depth >= MAX_TRACE_DEPTH || prd.ray_recursie_depth >= optixLaunchParams.samplingParameter.maximumBounce)
                break;

        }
        // Done tracing
        SHEvaluate(wi, lmax, Ylm);
        for (int j = 0; j < SHTerms(lmax); ++j)
            for (int k = 0; k < SHTerms(lmax); ++k)
                T[j*SHTerms(lmax)+k] += (Ylm[j] * Ylm[k]) / (pdf * nSamples);
    }
    delete Ylm;
}


void SHComputeBSDFMatrix(const vec3f &Kd, const vec3f &Ks,
        float roughness, RNG &rng, int nSamples, int lmax, float *B,
        Random random) {
    // Create _BSDF_ for computing BSDF transfer matrix
    for (int i = 0; i < SHTerms(lmax)*SHTerms(lmax); ++i)
        B[i] = 0.f;

    // Precompute directions $\w{}$ and SH values for directions
    float *Ylm = new float[SHTerms(lmax) * nSamples];
    vec3f *w = new vec3f[nSamples];
    for (int i = 0; i < nSamples; ++i) {
        w[i] = cos_sample_hemisphere(random(), random());
        SHEvaluate(w[i], lmax, &Ylm[SHTerms(lmax)*i]);
    }

    // Compute double spherical integral for BSDF matrix
    for (int osamp = 0; osamp < nSamples; ++osamp) {
        const vec3f &wo = w[osamp];
        for (int isamp = 0; isamp < nSamples; ++isamp) {
            const vec3f &wi = w[isamp];
            // Update BSDF matrix elements for sampled directions
            vec3f f_val = bsdf(wo, wi); // Sample BSDF value

            // If the BRDF value is not black in this direction
            float pdf = 1 / (4 * M_PI);
            pdf = pdf * pdf;

            // Set this term to zero if it's below, else cosine term
            f_val *= fmaxf(cosf(wi.z), 0.f);

            f_val /= pdf * nSamples * nSamples;
            for (int i = 0; i < SHTerms(lmax); ++i)
                for (int j = 0; j < SHTerms(lmax); ++j)
                    B[i*SHTerms(lmax)+j] += f_val * Ylm[isamp*SHTerms(lmax)+j] *
                                            Ylm[osamp*SHTerms(lmax)+i];
        }
    }
    // Free memory allocated for SH matrix computation
    delete[] w;
    delete[] Ylm;
}

void SHMatrixVectorMultiply(const vec3f *M, const vec3f *v,
        vec3f *vout, int lmax) {
    for (int i = 0; i < SHTerms(lmax); ++i) {
        vout[i] = 0.f;
        for (int j = 0; j < SHTerms(lmax); ++j)
            vout[i] += M[SHTerms(lmax) * i + j] * v[j];
    }
}


