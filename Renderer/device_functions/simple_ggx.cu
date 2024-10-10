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
        prd.prdExtraInfo->pixelFirstHitKs = make_float3(1.0f);
        prd.prdExtraInfo->pixelFirstHitRoughness = sbt->roughness.val;
        prd.prdExtraInfo->pixelEmissiveFlag = false;
        
    }

    static __forceinline__ __device__ void getSbt(simpleGGXShaderInfo& sbt, const TriangleMeshSBTData * sbtRaw ) {
        sbt.kd = sbtRaw->kd.val;
        sbt.roughness = sbtRaw->roughness.val;
    } 


    extern "C" __device__ float __direct_callable__pdf(const TriangleMeshSBTData * sbtRaw,
        const float3 & L, const float3 & V, const float3 & Ns, const float3 & Ng)
    {
        simpleGGXShaderInfo sbt;
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

        float roughness = sbt.roughness;
        // Clamp roughness to avoid numerical issues.
        roughness = clamp(roughness, (0.05f), (1.f));

        // We use the reflectance to determine whether to choose specular sampling lobe or diffuse.
        float spec_prob = clamp(1 - roughness, (0.05f), (0.75f)); 
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

        return spec_prob + diff_prob;

    }

    extern "C" __device__ float3 __direct_callable__evaluate(const TriangleMeshSBTData * sbtRaw,
        const float3 & L, const float3 & V, const float3 & Ns, const float3 & Ng)
    {
        simpleGGXShaderInfo sbt;
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

        //L = F.normalize(pts2l, dim = -1)  # [nrays, nlights, 3]
        //V = F.normalize(pts2c, dim = -1)  # [nrays, 3]
        //H = F.normalize((L + V[:, None, : ]) / 2.0, dim = -1)  # [nrays, nlights, 3]
        //N = F.normalize(normal, dim = -1)  # [nrays, 3]

        float3 half_vector = normalize(V + L);

        ////NoL = torch.sum(N[:, None, : ] * L, dim = -1, keepdim = True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
        ////NoV = torch.sum(N * V, dim = -1, keepdim = True).clamp_(1e-6, 1)  # [nrays, 1]
        ////NoH = torch.sum(N[:, None, : ] * H, dim = -1, keepdim = True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
        ////VoH = torch.sum(V[:, None, : ] * H, dim = -1, keepdim = True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
        float n_dot_l = clamp(dot(normal, L), 1e-6f, 1.f);
        float n_dot_v = clamp(dot(normal, V), 1e-6f, 1.f);
        float n_dot_h = clamp(dot(normal, half_vector), 1e-6f, 1.f);
        float v_dot_h = clamp(dot(half_vector, V), 1e-6f, 1.f);

        ////alpha = roughness * roughness  # [nrays, 3]
        ////alpha2 = alpha * alpha  # [nrays, 3]
        ////k = (alpha + 2 * roughness + 1.0) / 8.0
        ////FMi = ((-5.55473) * VoH - 6.98316) * VoH
        ////frac0 = fresnel[:, None, : ] + (1 - fresnel[:, None, : ]) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
        float alpha = sbt.roughness * sbt.roughness;
        float alpha2 = alpha * alpha;
        float k = (alpha + 2. * sbt.roughness + 1.) / 8.;
        float FMi = ((-5.55473) * v_dot_h - 6.98316) * v_dot_h;
        float3 fresnel = make_float3(0.04);
        float3 frac0 = fresnel + (1. - fresnel) * pow(2.f, FMi);

        ////frac = frac0 * alpha2[:, None, : ]  # [nrays, 1]
        ////nom0 = NoH * NoH * (alpha2[:, None, : ] - 1) + 1
        float3 frac = frac0 * alpha2;
        float nom0 = n_dot_h * n_dot_h * (alpha2 - 1) + 1;

        ////nom1 = NoV * (1 - k) + k
        ////nom2 = NoL * (1 - k[:, None, : ]) + k[:, None, : ]
        ////nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, : ] * nom2).clamp_(1e-6, 4 * np.pi)
        ////spec = frac / nom
        ////return spec

        float nom1 = n_dot_v * (1 - k) + k;
        float nom2 = n_dot_l * (1 - k) + k;
        float nom = clamp(4. * M_PIf * nom0 * nom0 * nom1 * nom2, 1e-6f, 4.f * M_PIf);
        float3 spec = frac / nom;
        // std::cout << spec << std::endl;
        return (spec * n_dot_l) + sbt.kd / M_PIf * n_dot_l;
        // return half_vector; // make_float3(nom0, nom, spec.x);// (spec * n_dot_l) + sbt.kd / M_PIf * n_dot_l;
    }

    extern "C" __device__ float3 __direct_callable__sample(const TriangleMeshSBTData * sbtRaw,
        const float3 & V, const float3 & Ns, const float3 & Ng, Random & random)
    {
        simpleGGXShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) < 0) {
            normal = normal * -1.f;
        }

        float rnd_param_w = random();
        float rnd_param_u = random();
        float rnd_param_v = random();

        float roughness = clamp(sbt.roughness, (0.05f), (1.f));
        float spec_prob = clamp(1 - roughness, (0.05f), (0.75f));  
        if (rnd_param_w > spec_prob) {
            float3 L;
            sample_cos_hemisphere_dir(rnd_param_u, rnd_param_v, L);
            return to_world(L, normal);
        }
        else { 
            float3 local_dir_in = to_local(V, normal);
            auto org_local_dir_in = local_dir_in;
            if (org_local_dir_in.z < 0) local_dir_in = -local_dir_in;
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