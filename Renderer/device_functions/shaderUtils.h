#pragma once
#include "mathUtils.h"

namespace nert_renderer {

    /////////////////////
    /// Transformations
    /////////////////////
    __device__ __inline__ float3 to_local(const float3& dir, const float3& z, const float3& x) {
        if (dot(z, x) > 1e-3) optixThrowException(0);
        float3 y = cross(x, z);
        y = normalize(y);
        return normalize(make_float3(dot(dir, x), dot(dir, y), dot(dir, z)));
    }

    __device__ __inline__ float3 to_local(const float3& dir, const float3& normal) {
        float3 norm = normalize(normal);
        float3 tangent, binormal;
        if (norm.z < -1.f + 1e-6f) {
            tangent = make_float3(0.f, -1.f, 0.f);
            binormal = make_float3(-1.f, 0.f, 0.f);
        }
        else {
            float a = 1 / (1 + norm.z);
            float b = -norm.x * norm.y * a;
            tangent = make_float3(1 - norm.x * norm.x * a, b, -norm.x);
            binormal = make_float3(b, 1 - norm.y * norm.y * a, -norm.y);
        }
        return make_float3(dot(dir, tangent), dot(dir, binormal), dot(dir, norm));
    }

    __device__ __inline__ float3 to_world(const float3& dir, const float3& normal) {
        float3 norm = normalize(normal);
        float3 tangent, binormal;
        if (norm.z < -1.f + 1e-6f) {
            tangent = make_float3(0.f, -1.f, 0.f);
            binormal = make_float3(-1.f, 0.f, 0.f);
        }
        else {
            float a = 1 / (1 + norm.z);
            float b = -norm.x * norm.y * a;
            tangent = normalize(make_float3(1 - norm.x * norm.x * a, b, -norm.x));
            binormal = normalize(make_float3(b, 1 - norm.y * norm.y * a, -norm.y));
        }
        return float3(tangent * dir.x + binormal * dir.y + norm * dir.z);
    }

    //////////////////
    /// Shader Related 
    //////////////////
    __device__ __inline__ float luminance(const float3& s) {
        return s.x * float(0.212671) + s.y * float(0.715160) + s.z * float(0.072169);
    }

    template <typename T>
    __device__ __inline__ T schlick_fresnel(const T& F0, float cos_theta) {
        return F0 + (1.f - F0) *
            pow(max(1.f - cos_theta, 0.f), 5.f);
    }

    __device__ __inline__ float3 schlick_fresnel(const float3& F0, float cos_theta, float eta) {
        float h_dot_out_sq = 1 - (1 / (eta * eta)) * (1 - cos_theta * cos_theta);
        float3 F = make_float3(1, 1, 1);
        if (h_dot_out_sq > 0) {
            F = schlick_fresnel(F0, eta > 1 ? cos_theta : sqrt(h_dot_out_sq));
        }
        return F;
    }

    __device__ __inline__ float fresnel_dielectric(float n_dot_i, float n_dot_t, float eta) {
        // assert(n_dot_i >= 0 && n_dot_t >= 0 && eta > 0);
        float rs = (n_dot_i - eta * n_dot_t) / (n_dot_i + eta * n_dot_t);
        float rp = (eta * n_dot_i - n_dot_t) / (eta * n_dot_i + n_dot_t);
        float F = (rs * rs + rp * rp) / 2;
        return F;
    }

    __device__ __inline__ float fresnel_dielectric(float n_dot_i, float eta) {
        // assert(eta > 0);
        float n_dot_t_sq = 1 - (1 - n_dot_i * n_dot_i) / (eta * eta);
        if (n_dot_t_sq < 0) {
            // total internal reflection
            return 1;
        }
        float n_dot_t = sqrt(n_dot_t_sq);
        return fresnel_dielectric(fabs(n_dot_i), n_dot_t, eta);
    }

    __device__ __inline__ float smith_masking_gtr2(const float3& v_local, float roughness) {
        float alpha = roughness * roughness;
        float a2 = alpha * alpha;
        float3 v2 = v_local * v_local;
        float Lambda = (-1 + sqrt(1 + (v2.x * a2 + v2.y * a2) / v2.z)) / 2;
        return 1 / (1 + Lambda);
    }

    __device__ __inline__ float smith_masking_gtr2(const float3& v_local, float alpha_x, float alpha_y) {
        float ax2 = alpha_x * alpha_x;
        float ay2 = alpha_y * alpha_y;
        float3 v2 = v_local * v_local;
        float Lambda = (-1 + sqrt(1 + (v2.x * ax2 + v2.y * ay2) / v2.z)) / 2;
        return 1 / (1 + Lambda);
    }

    __device__ __inline__ float GTR2(float n_dot_h, float roughness) {
        float alpha = roughness * roughness;
        float a2 = alpha * alpha;
        float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
        return a2 / (M_PIf * t * t);
    }

    __device__ __inline__ float GTR2(const float3& h_local, float alpha_x, float alpha_y) {
        float3 h_local_scaled{ h_local.x / alpha_x, h_local.y / alpha_y, h_local.z };
        float h_local_scaled_len_sq = dot(h_local_scaled, h_local_scaled);
        return 1 / (M_PIf * alpha_x * alpha_y * h_local_scaled_len_sq * h_local_scaled_len_sq);
    }


    __device__ __inline__ float smith_masking_gtr1(const float3& v_local) {
        return smith_masking_gtr2(v_local, 0.25f, 0.25f);
    }

    __device__ __inline__ float GTR1(float n_dot_h, float alpha) {
        float a2 = alpha * alpha;
        float t = 1 + (a2 - 1) * n_dot_h * n_dot_h;
        return (a2 - 1) / (M_PIf * log(a2) * t);
    }

    ///////////////
    /// Sampling
    ///////////////
    __device__ __inline__ void sample_hemisphere_dir(const float& z1, const float& z2, float3& normal) {
        const float radius = sqrtf(z1);
        const float theta = 2.f * M_PIf * z2;
        float x = radius * cosf(theta);
        float y = radius * sinf(theta);
        float z = sqrtf(fmax(0.f, 1.f - x * x - y * y));
        normal.x = x;
        normal.y = y;
        normal.z = z;
    }

    __device__ __inline__ void sample_cos_hemisphere_dir(const float& z1, const float& z2, float3& normal) {
        float phi = 2 * M_PIf * z1;
        float tmp = sqrtf(clamp(1 - z2, float(0), float(1)));
        normal.x = cos(phi) * tmp;
        normal.y = sin(phi) * tmp;
        normal.z = sqrtf(clamp(z2, float(0), float(1)));
    }

    __device__ __inline__ void sample_sphere(const float& z1, const float& z2, float3& normal) {

        float theta = 2 * M_PIf * z1;
        float phi = acos(1 - 2 * z2);
        normal.x = sin(phi) * cos(theta);
        normal.y = sin(phi) * sin(theta);
        normal.z = cos(phi);
    }

    __device__ __inline__ float3 sample_visible_normals2(const float3& local_dir_in,
        float alpha_x, float alpha_y, const float2& rnd_param) {
        // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
        if (local_dir_in.z < 0) {
            // Ensure the input is on top of the surface.
            return make_float3(0.f, 1.f, 0.f);
        }

        // Transform the incoming direction to the "hemisphere configuration".
        float3 hemi_dir_in = normalize(
            make_float3(alpha_x * local_dir_in.x, alpha_y * local_dir_in.y, local_dir_in.z));

        // Parameterization of the projected area of a hemisphere.
        // First, sample a disk.
        float r = sqrt(rnd_param.x);
        float phi = 2 * M_PIf * rnd_param.y;
        float t1 = r * cos(phi);
        float t2 = r * sin(phi);
        // Vertically scale the position of a sample to account for the projection.
        float s = (1 + hemi_dir_in.z) / 2;
        t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
        // Point in the disk space
        float3 disk_N = make_float3(t1, t2, sqrt(fmax(float(0), 1 - t1 * t1 - t2 * t2)));

        // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
        float3 hemi_N = to_world(disk_N, hemi_dir_in);

        // Transforming the normal back to the ellipsoid configuration
        return normalize(make_float3(alpha_x * hemi_N.x, alpha_y * hemi_N.y, fmax(float(0), hemi_N.z)));
    }

    __device__ __inline__ float3 sample_visible_normals(const float3& local_dir_in,
        float alpha_x, float alpha_y, const float2& rnd_param) {
        // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
        if (local_dir_in.z < 0) {
            // Ensure the input is on top of the surface.
            return -sample_visible_normals2(-local_dir_in, alpha_x, alpha_y, rnd_param);
        }

        // Transform the incoming direction to the "hemisphere configuration".
        float3 hemi_dir_in = normalize(
            float3{ alpha_x * local_dir_in.x, alpha_y * local_dir_in.y, local_dir_in.z });

        // Parameterization of the projected area of a hemisphere.
        // First, sample a disk.
        float r = sqrt(rnd_param.x);
        float phi = 2 * M_PIf * rnd_param.y;
        float t1 = r * cos(phi);
        float t2 = r * sin(phi);
        // Vertically scale the position of a sample to account for the projection.
        float s = (1 + hemi_dir_in.z) / 2;
        t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
        // Point in the disk space
        float3 disk_N = make_float3(t1, t2, sqrt(fmax(float(0), 1 - t1 * t1 - t2 * t2)));

        // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
        float3 hemi_N = to_world(disk_N, hemi_dir_in);

        // Transforming the normal back to the ellipsoid configuration
        return normalize(make_float3(alpha_x * hemi_N.x, alpha_y * hemi_N.y, fmax(float(0), hemi_N.z)));
    }

    #define INV_PI     0.31830988618379067154f
    #define INV_TWOPI  0.15915494309189533577f
    #define INV_FOURPI 0.07957747154594766788f
    
    __device__ __inline__ int SHTerms(int lmax) {
        return (lmax + 1) * (lmax + 1);
    }


    __device__ __inline__ int SHIndex(int l, int m) {
        return l*l + l + m;
    }


    // Spherical Harmonics Local Definitions
    __device__ __inline__ void legendrep(float x, int lmax, float *out) {
        #define P(l,m) out[SHIndex(l,m)]
        // Compute $m=0$ Legendre values using recurrence
        P(0,0) = 1.f;
        P(1,0) = x;
        for (int l = 2; l <= lmax; ++l)
        {
            P(l, 0) = ((2*l-1)*x*P(l-1,0) - (l-1)*P(l-2,0)) / l;
        }

        // Compute $m=l$ edge using Legendre recurrence
        float neg = -1.f;
        float dfact = 1.f;
        float xroot = sqrtf(fmaxf(0.f, 1.f - x*x));
        float xpow = xroot;
        for (int l = 1; l <= lmax; ++l) {
            P(l, l) = neg * dfact * xpow;
            neg *= -1.f;      // neg = (-1)^l
            dfact *= 2*l + 1; // dfact = (2*l-1)!!
            xpow *= xroot;    // xpow = powf(1.f - x*x, float(l) * 0.5f);
        }

        // Compute $m=l-1$ edge using Legendre recurrence
        for (int l = 2; l <= lmax; ++l)
        {
            P(l, l-1) = x * (2*l-1) * P(l-1, l-1);
        }

        // Compute $m=1, \ldots, l-2$ values using Legendre recurrence
        for (int l = 3; l <= lmax; ++l)
            for (int m = 1; m <= l-2; ++m)
            {
                P(l, m) = ((2 * (l-1) + 1) * x * P(l-1,m) -
                        (l-1+m) * P(l-2,m)) / (l - m);
            }
        #undef P
    }
    __device__ __inline__ float divfact(int a, int b) {
        if (b == 0) return 1.f;
        float fa = a, fb = fabsf(b);
        float v = 1.f;
        for (float x = fa-fb+1.f; x <= fa+fb; x += 1.f)
            v *= x;
        return 1.f / v;
    }

    __device__ __inline__ float K(int l, int m) {
        return sqrtf((2.f * l + 1.f) * INV_FOURPI * divfact(l, m));
    }

     __device__ __inline__ void sinCosIndexed(float s, float c, int n,
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
    // Spherical Harmonics Definitions
    __device__ __inline__ void SHEvaluate(const float3 &w, float *out) {
        int lmax = 8;
        legendrep(w.z, lmax, out);

        // Compute $K_l^m$ coefficients
        float Klm[81];
        for (int l = 0; l <= lmax; ++l)
            for (int m = -l; m <= l; ++m)
                Klm[SHIndex(l, m)] = K(l, m);

        // Compute $\sin\phi$ and $\cos\phi$ values
        float sins[9], coss[9]; 
        // float *sins = ALLOCA(float, lmax+1), *coss = ALLOCA(float, lmax+1);
        float xyLen = sqrtf(fmaxf(0.f, 1.f - w.z*w.z));
        if (xyLen == 0.f) {
            for (int i = 0; i <= lmax; ++i) sins[i] = 0.f;
            for (int i = 0; i <= lmax; ++i) coss[i] = 1.f;
        }
        else
            sinCosIndexed(w.y / xyLen, w.x / xyLen, lmax+1, sins, coss);

        // Apply SH definitions to compute final $(l,m)$ values
        float sqrt2 = sqrtf(2.f);
        for (int l = 0; l <= lmax; ++l) {
            for (int m = -l; m < 0; ++m)
            {
                out[SHIndex(l, m)] = sqrt2 * Klm[SHIndex(l, m)] *
                    out[SHIndex(l, -m)] * sins[-m];
                // assert(!isnan(out[SHIndex(l,m)]));
                // assert(!isinf(out[SHIndex(l,m)]));
            }
            out[SHIndex(l, 0)] *= Klm[SHIndex(l, 0)];
            for (int m = 1; m <= l; ++m)
            {
                out[SHIndex(l, m)] *= sqrt2 * Klm[SHIndex(l, m)] * coss[m];
                // assert(!isnan(out[SHIndex(l,m)]));
                // assert(!isinf(out[SHIndex(l,m)]));
            }
        }
    }
}