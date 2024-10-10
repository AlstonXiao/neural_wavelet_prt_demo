#include <cuda_runtime.h>
#include <optix.h>

#include "LaunchParams.h"
#include "perRayData.h"
#include "shaderUtils.h"

namespace nert_renderer {
    extern "C" __device__ void __direct_callable__packSBT(const TriangleMeshSBTData * sbt, PRD & prd)
    {        
        prd.prdExtraInfo->pixelAlbedoFirstHit = sbt->kd.val;
        prd.prdExtraInfo->pixelFirstHitKd = sbt->kd.val;
        prd.prdExtraInfo->pixelFirstHitKs = sbt->ks.val;
        prd.prdExtraInfo->pixelFirstHitRoughness = sbt->roughness.val;
        prd.prdExtraInfo->pixelEmissiveFlag = false;
        
    }

    static __forceinline__ __device__ void getSbt(roughPlasticShaderInfo& sbt, const TriangleMeshSBTData * sbtRaw ) {
        sbt.kd = sbtRaw->kd.val;
        sbt.ks = sbtRaw->ks.val;
        sbt.roughness = sbtRaw->roughness.val;
        sbt.eta = sbtRaw->eta.val;
    } 

    extern "C" __device__ float __direct_callable__pdf(const TriangleMeshSBTData * sbtRaw,
        const float3& L, const float3& V, const float3& Ns, const float3 & Ng)
    {
        roughPlasticShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        if (dot(Ng, V) < 0 || dot(Ng, L) < 0) {
            // No light below the surface
            return 0.f;
        }
        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) < 0) {
            normal = -normal;
        }

        float3 half_vector = normalize(V + L);
        float n_dot_h = dot(normal, half_vector);
        float n_dot_in = dot(normal, V);
        float n_dot_out = dot(normal, L);
        if (n_dot_out <= 0 || n_dot_h <= 0) {
            return 0;
        }
        float lS = luminance(sbt.ks), lR = luminance(sbt.kd);
        if (lS + lR <= 0) {
            return 0;
        }

        float roughness = sbt.roughness;
        // Clamp roughness to avoid numerical issues.
        roughness = clamp(roughness, (0.01f), (1.f));

        // We use the reflectance to determine whether to choose specular sampling lobe or diffuse.
        float spec_prob = lS / (lS + lR);
        float diff_prob = 1 - spec_prob;
        // For the specular lobe, we use the ellipsoidal sampling from Heitz 2018
        // "Sampling the GGX Distribution of Visible Normals"
        // https://jcgt.org/published/0007/04/01/
        // this importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, roughness) * cos_theta_out
        float G = smith_masking_gtr2(to_local(V, normal), roughness);
        float D = GTR2(n_dot_h, roughness);
        // (4 * cos_theta_v) is the Jacobian of the reflectiokn
        spec_prob *= (G * D) / (4 * n_dot_in);
        // For the diffuse lobe, we importance sample cos_theta_out
        diff_prob *= n_dot_out / M_PIf;
        return spec_prob + diff_prob;
    }

    extern "C" __device__ float3 __direct_callable__evaluate(const TriangleMeshSBTData * sbtRaw,
        const float3& L, const float3& V, const float3& Ns, const float3 & Ng)
    {
        roughPlasticShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        if (dot(Ng, V) < 0 || dot(Ng, L) < 0) {
            // No light below the surface
            // return float3(0.f);
        }
        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) < 0) {
            normal = -normal;
        }
        float3 half_vector = normalize(V + L);
        float n_dot_h = dot(normal, half_vector);
        float n_dot_in = dot(normal, V);
        float n_dot_out = dot(normal, L);
        if (n_dot_out <= 0 || n_dot_h <= 0) {
            return make_float3(0.f);
        }

        // Clamp roughness to avoid numerical issues.
        float roughness = clamp(sbt.roughness, (0.01f), (1.f));

        // We first account for the dielectric layer.

        // Fresnel equation determines how much light goes through, 
        // and how much light is reflected for each wavelength.
        // Fresnel equation is determined by the angle between the (micro) normal and 
        // both incoming and outgoing directions (dir_out & dir_in).
        // However, since they are related through the Snell-Descartes law,
        // we only need one of them.
        float F_o = fresnel_dielectric(dot(half_vector, L), sbt.eta); // F_o is the reflection percentage.
        float D = GTR2(n_dot_h, roughness); // "Generalized Trowbridge Reitz", GTR2 is equivalent to GGX.
        float G = smith_masking_gtr2(to_local(V, normal), roughness) *
            smith_masking_gtr2(to_local(L, normal), roughness);

        float3 spec_contrib = sbt.ks * (G * F_o * D) / (4 * n_dot_in * n_dot_out);

        // Next we account for the diffuse layer.
        // In order to reflect from the diffuse layer,
        // the photon needs to bounce through the dielectric layers twice.
        // The transmittance is computed by 1 - fresnel.
        float F_i = fresnel_dielectric(dot(half_vector, V), sbt.eta);
        // Multiplying with Fresnels leads to an overly dark appearance at the 
        // object boundaries. Disney BRDF proposes a fix to this -- we will implement this in problem set 1.
        float3 diffuse_contrib = sbt.kd * ((1.f) - F_o) * ((1.f) - F_i) / (float)M_PIf;
        // if (depth == 0)
        return (diffuse_contrib + spec_contrib) * n_dot_out;
        // else
        //    return (diffuse_contrib)*n_dot_out;
    }

    extern "C" __device__ float3 __direct_callable__sample(const TriangleMeshSBTData * sbtRaw,
         const float3& V, const float3& Ns, const float3 & Ng, Random& random)
    {
        roughPlasticShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        if (dot(Ng, V) < 0) {
            // No light below the surface
            return make_float3(-2);
        }
        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) < 0) {
            normal = -normal;
        }
        float3 L;
        float lS = luminance(sbt.ks), lR = luminance(sbt.kd);
        if (lS + lR <= 0) {
            // dont care
            return make_float3(-2);
        }
        float spec_prob = lS / (lS + lR);
        float z1 = random();
        float z2 = random();

        if (z1 > spec_prob) {
            sample_cos_hemisphere_dir(z1, z2, L);
            return to_world(L, normal);
        }
        else {
            float3 local_dir_in = to_local(V, normal);
            float roughness = clamp(sbt.roughness, (0.01f), (1.f));
            float alpha = roughness * roughness;
            float3 hemi_dir_in = normalize(make_float3(alpha * local_dir_in.x, alpha * local_dir_in.y, local_dir_in.z));
            float r = sqrt(z1);
            float phi = 2 * M_PIf * z2;
            float t1 = r * cos(phi);
            float t2 = r * sin(phi);
            float s = (1 + hemi_dir_in.z) / 2;
            t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
            float3 disk_N= make_float3(t1, t2, sqrt(fmax(0.0f, 1.f - t1 * t1 - t2 * t2)));
            float3 hemi_N = to_world(disk_N, hemi_dir_in);
            float3 local_micro_normal = normalize(make_float3(alpha * hemi_N.x, alpha * hemi_N.y, max(0.f, hemi_N.z)));

            float3 H = normalize(to_world(local_micro_normal, normal));
            return normalize(-V + 2 * dot(V, H) * H);
        }
    }
}