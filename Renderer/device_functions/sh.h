
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

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "util.h"

// core/sh.h*

// Spherical Harmonics Declarations
inline int SHTerms(int lmax) {
    return (lmax + 1) * (lmax + 1);
}


inline int SHIndex(int l, int m) {
    return l*l + l + m;
}

inline vec3f SphericalDirection(float sintheta, float costheta, float phi) {
    return vec3f(sintheta * cosf(phi),
                 sintheta * sinf(phi),
                 costheta);
}

void SHEvaluate(const vec3f &v, int lmax, float *out);
void SHWriteImage(const char *filename, const vec3f *c, int lmax, int yres);
template <typename Func>
void SHProjectCube(Func func, const vec3f &p, int res, int lmax,
                   vec3f *coeffs) {
    float* Ylm = new float[SHTerms(max)];
    for (int u = 0; u < res; ++u) {
        float fu = -1.f + 2.f * (float(u) + 0.5f) / float(res);
        for (int v = 0; v < res; ++v) {
            float fv = -1.f + 2.f * (float(v) + 0.5f) / float(res);
            // Incorporate results from $+z$ face to coefficients
            vec3f w(fu, fv, 1);
            SHEvaluate(normalize(w), lmax, Ylm);
            Spectrum f = func(u, v, p, w);
            float dA = 1.f / powf(Dot(w, w), 3.f/2.f);
            for (int k = 0; k < SHTerms(lmax); ++k)
                coeffs[k] += f * Ylm[k] * dA * (4.f / (res * res));

            // Incorporate results from other faces to coefficients
            w = Vector(fu, fv, -1);
            SHEvaluate(Normalize(w), lmax, Ylm);
            f = func(u, v, p, w);
            for (int k = 0; k < SHTerms(lmax); ++k)
                coeffs[k] += f * Ylm[k] * dA * (4.f / (res * res));
            w = Vector(fu, 1, fv);
            SHEvaluate(Normalize(w), lmax, Ylm);
            f = func(u, v, p, w);
            for (int k = 0; k < SHTerms(lmax); ++k)
                coeffs[k] += f * Ylm[k] * dA * (4.f / (res * res));
            w = Vector(fu, -1, fv);
            SHEvaluate(Normalize(w), lmax, Ylm);
            f = func(u, v, p, w);
            for (int k = 0; k < SHTerms(lmax); ++k)
                coeffs[k] += f * Ylm[k] * dA * (4.f / (res * res));
            w = Vector(1, fu, fv);
            SHEvaluate(Normalize(w), lmax, Ylm);
            f = func(u, v, p, w);
            for (int k = 0; k < SHTerms(lmax); ++k)
                coeffs[k] += f * Ylm[k] * dA * (4.f / (res * res));
            w = Vector(-1, fu, fv);
            SHEvaluate(Normalize(w), lmax, Ylm);
            f = func(u, v, p, w);
            for (int k = 0; k < SHTerms(lmax); ++k)
                coeffs[k] += f * Ylm[k] * dA * (4.f / (res * res));
        }
    }
    delete Ylm;
}


void SHProjectIncidentDirectRadiance(const Point &p, float pEpsilon, float time,
    MemoryArena &arena, const Scene *scene, bool computeLightVisibility,
    int lmax, RNG &rng, Spectrum *c_d);
void SHProjectIncidentIndirectRadiance(const Point &p, float pEpsilon,
    float time, const Renderer *renderer, Sample *origSample,
    const Scene *scene, int lmax, RNG &rng, int nSamples, Spectrum *c_i);
void SHReduceRinging(Spectrum *c, int lmax, float lambda = .005f);
void SHRotate(const Spectrum *c_in, Spectrum *c_out, const Matrix4x4 &m,
              int lmax, MemoryArena &arena);
void SHRotateZ(const Spectrum *c_in, Spectrum *c_out, float alpha, int lmax);
void SHRotateXMinus(const Spectrum *c_in, Spectrum *c_out, int lmax);
void SHRotateXPlus(const Spectrum *c_in, Spectrum *c_out, int lmax);
//void SHSwapYZ(const Spectrum *c_in, Spectrum *c_out, int lmax);
void SHConvolveCosTheta(int lmax, const Spectrum *c_in, Spectrum *c_out);
void SHConvolvePhong(int lmax, float n, const Spectrum *c_in, Spectrum *c_out);
void SHComputeDiffuseTransfer(const Point &p, const Normal &n, float rayEpsilon,
    const Scene *scene, RNG &rng, int nSamples, int lmax, Spectrum *c_transfer);
void SHComputeTransferMatrix(const Point &p, float rayEpsilon,
    const Scene *scene, RNG &rng, int nSamples, int lmax, Spectrum *T);
void SHComputeBSDFMatrix(const Spectrum &Kd, const Spectrum &Ks,
    float roughness, RNG &rng, int nSamples, int lmax, Spectrum *B);
void SHMatrixVectorMultiply(const Spectrum *M, const Spectrum *v,
                            Spectrum *vout, int lmax);
