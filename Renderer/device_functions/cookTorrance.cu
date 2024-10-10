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

    static __forceinline__ __device__ void getSbt(cookTorranceShaderInfo& sbt, const TriangleMeshSBTData * sbtRaw ) {
        sbt.kd = sbtRaw->kd.val;
        sbt.ks = sbtRaw->ks.val;
        sbt.roughness = sbtRaw->roughness.val;
    } 


    extern "C" __device__ float __direct_callable__pdf(const TriangleMeshSBTData * sbtRaw,
        const float3 & L, const float3 & V, const float3 & Ns, const float3 & Ng)
    {
        cookTorranceShaderInfo sbt;
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
        roughness = clamp(roughness, (0.05f), (1.f));

        // We use the reflectance to determine whether to choose specular sampling lobe or diffuse.
        float spec_prob = lS / (lS + lR);
        float diff_prob = 1 - spec_prob;
        // For the specular lobe, we use the ellipsoidal sampling from Heitz 2018
        // "Sampling the GGX Distribution of Visible Normals"
        // https://jcgt.org/published/0007/04/01/
        // this importance samples smith_masking(cos_theta_in) * GTR2(cos_theta_h, roughness) * cos_theta_out
        float G = smith_masking_gtr2(to_local(V, normal), roughness*roughness, roughness*roughness);
        float D = GTR2(to_local(half_vector, normal), roughness*roughness, roughness*roughness);
        // float D = GTR2(n_dot_h, roughness);
        // (4 * cos_theta_v) is the Jacobian of the reflectiokn
        // printf("Spec prob before: %f\n", spec_prob);
        spec_prob *= (G * D) / (4 * n_dot_in);
        // For the diffuse lobe, we importance sample cos_theta_out
        diff_prob *= n_dot_out / M_PIf;
        // printf("Diff prob after: %f\n", diff_prob);

        //elete sbt;

        return spec_prob + diff_prob;

    }

    extern "C" __device__ float3 __direct_callable__evaluate(const TriangleMeshSBTData * sbtRaw,
        const float3 & L, const float3 & V, const float3 & Ns, const float3 & Ng)
    {
        cookTorranceShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        if (dot(Ng, V) < 0 || dot(Ng, L) < 0) {
            // No light below the surface
            return make_float3(0.f);
        }
        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) < 0) {
            normal = -normal;
        }

        float lS = luminance(sbt.ks), lR = luminance(sbt.kd);
        if (lS + lR <= 0) {
            return make_float3(0.f);
        }

        float3 half_vector = normalize(V + L);
        float n_dot_h = dot(normal, half_vector);
        float n_dot_in = dot(normal, V);
        float n_dot_out = dot(normal, L);

        constexpr float min_alpha = 0.0025f;
        float alpha_x = max(min_alpha, sbt.roughness * sbt.roughness);
        float alpha_y = max(min_alpha, sbt.roughness * sbt.roughness);

        // Flip half-vector if it's below surface
        if (dot(half_vector, normal) < 0) {
            half_vector = -half_vector;
        }
        n_dot_h = dot(normal, half_vector);
        float h_dot_in = dot(half_vector, V);
        float h_dot_out = dot(half_vector, L);

        float3 diffuse_contrib = make_float3(0.f);
        float3 metallic_contrib = make_float3(0.f);

        // Metallic component
        float3 F = schlick_fresnel(sbt.ks, h_dot_out);
        float D = GTR2(to_local(half_vector, normal), alpha_x, alpha_y);
        float G_in = smith_masking_gtr2(to_local(V, normal), alpha_x, alpha_y);
        float G_out = smith_masking_gtr2(to_local(L, normal), alpha_x, alpha_y);
        float G = G_in * G_out;
        metallic_contrib = F * D * G / (4 * n_dot_in * n_dot_out);


        // The base diffuse model, computing view independent second diffuse layer
        float Fd90 = float(0.5) + 2 * sbt.roughness * h_dot_out * h_dot_out;

        float schlick_n_dot_out = pow(1 - n_dot_out, float(5));
        float schlick_n_dot_in = pow(1 - n_dot_in, float(5));

        float3 base_diffuse = make_float3(21.f / 20.f) * (-sbt.ks + 1.f) * make_float3(1.f - schlick_n_dot_out) * make_float3(1.f - schlick_n_dot_in);
        diffuse_contrib = base_diffuse * sbt.kd / ((float)M_PIf) ;

        return (diffuse_contrib + metallic_contrib) * n_dot_out;
    }

    extern "C" __device__ float3 __direct_callable__sample(const TriangleMeshSBTData * sbtRaw,
        const float3 & V, const float3 & Ns, const float3 & Ng, Random & random)
    {
        cookTorranceShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) < 0) {
            normal = normal * -1.f;
        }

        float rnd_param_w = random();
        float rnd_param_u = random();
        float rnd_param_v = random();

        float lS = luminance(sbt.ks), lR = luminance(sbt.kd);
        if (lS + lR <= 0) {
            // dont care
            return make_float3(-2, -2, -2);
        }


        float spec_prob = lS / (lS + lR);
        if (rnd_param_w > spec_prob) {
            float3 L;
            sample_cos_hemisphere_dir(rnd_param_u, rnd_param_v, L);
            return to_world(L, normal);
        }
        else { 
            float3 local_dir_in = to_local(V, normal);
            auto org_local_dir_in = local_dir_in;
            if (org_local_dir_in.z < 0) local_dir_in = -local_dir_in;
            float roughness = clamp(sbt.roughness, (0.05f), (1.f));
            float alpha = roughness * roughness;
            float3 hemi_dir_in = normalize(make_float3(alpha * local_dir_in.x, alpha * local_dir_in.y, local_dir_in.z));
            float r = sqrt(rnd_param_u);
            float phi = 2 * M_PIf * rnd_param_v;
            float t1 = r * cos(phi);
            float t2 = r * sin(phi);
            float s = (1 + hemi_dir_in.z) / 2;
            t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
            float3 disk_N = make_float3(t1, t2, sqrt(fmax(0.0f, 1.f - t1 * t1 - t2 * t2)));
            float3 hemi_N = to_world(disk_N, hemi_dir_in);
            float3 local_micro_normal = normalize(make_float3(alpha * hemi_N.x, alpha * hemi_N.y, max(0.f, hemi_N.z)));

            float3 H = normalize(to_world(local_micro_normal, normal));
            float3 ret = normalize(-V + 2 * dot(V, H) * H);

            if (org_local_dir_in.z < 0) return -ret;
            else return ret;
        }
    }
}