#include <cuda_runtime.h>
#include <optix.h>

#include "LaunchParams.h"
#include "perRayData.h"
#include "shaderUtils.h"

namespace nert_renderer {
    // TODO
    extern "C" __device__ void __direct_callable__packSBT(const TriangleMeshSBTData * sbt, PRD & prd)
    {        
        prd.prdExtraInfo->pixelAlbedoFirstHit = sbt->kd.val;
        prd.prdExtraInfo->pixelFirstHitKd = sbt->kd.val;

        float3 Ctint = luminance(sbt->kd.val) > 0 ? sbt->kd.val / luminance(sbt->kd.val) : make_float3(1.);
        float3 spec_color =
            ((1 - sbt->specular_tint.val) * make_float3(1.) + sbt->specular_tint.val * Ctint);
        float spec_f0 = (sbt->eta.val - 1) * (sbt->eta.val - 1) / ((sbt->eta.val + 1) * (sbt->eta.val + 1));
        prd.prdExtraInfo->pixelFirstHitKs =
            sbt->specular.val * spec_f0 * (1 - sbt->metallic.val) * spec_color + 
            sbt->metallic.val * sbt->kd.val;

        prd.prdExtraInfo->pixelFirstHitRoughness = sbt->roughness.val;
        prd.prdExtraInfo->pixelEmissiveFlag = false;
        
    }

    static __forceinline__ __device__ void getSbt(DisneyBSDFShaderInfo& sbt, const TriangleMeshSBTData * sbtRaw ) {
        sbt.baseColor = sbtRaw->kd.val;
        sbt.roughness = sbtRaw->roughness.val;
        sbt.specular_transmission = sbtRaw->specular_transmission.val;
        sbt.metallic = sbtRaw->metallic.val;
        sbt.subsurface = sbtRaw->subsurface.val;
        sbt.specular = sbtRaw->specular.val;
        sbt.specular_tint = sbtRaw->specular_tint.val;
        sbt.anisotropic = sbtRaw->anisotropic.val;
        sbt.sheen = sbtRaw->sheen.val;
        sbt.sheen_tint = sbtRaw->sheen_tint.val;
        sbt.clearcoat = sbtRaw->clearcoat.val;
        sbt.clearcoat_gloss = sbtRaw->clearcoat_gloss.val;
        sbt.eta = sbtRaw->eta.val;

    } 

    extern "C" __device__ float __direct_callable__pdf(const TriangleMeshSBTData * sbtRaw,
        const float3 & L, const float3 & V, const float3 & Ns, const float3 & Ng)
    {
        DisneyBSDFShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        bool reflect = dot(Ng, L) * dot(Ng, V) > 0;
        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) * dot(Ng, V) < 0) {
            normal = normal * -1.f;
        }

        // Fetch the texture values for later use
        float specular_transmission = sbt.specular_transmission;
        float metallic = sbt.metallic;
        float roughness = sbt.roughness;
        float anisotropic = sbt.anisotropic;
        float clearcoat = float(0.25) * sbt.clearcoat;
        float clearcoat_gloss = sbt.clearcoat_gloss;

        float aspect = sqrt(1 - anisotropic * float(0.9));
        constexpr float min_alpha = float(0.0001);
        float alpha_x = max(min_alpha, roughness * roughness / aspect);
        float alpha_y = max(min_alpha, roughness * roughness * aspect);
        float alpha_c = (1 - clearcoat_gloss) * float(0.1) + clearcoat_gloss * float(0.001);

        float diffuse_weight = (1 - metallic) * (1 - specular_transmission);
        float metallic_weight = (1 - specular_transmission * (1 - metallic));
        float glass_weight = (1 - metallic) * specular_transmission;
        float clearcoat_weight = clearcoat;
        float total_weight = diffuse_weight + metallic_weight + glass_weight + clearcoat_weight;
        float diffuse_prob = diffuse_weight / total_weight;
        float metallic_prob = metallic_weight / total_weight;
        float glass_prob = glass_weight / total_weight;
        float clearcoat_prob = clearcoat_weight / total_weight;

        if (dot(Ng, V) < 0) {
            // Our incoming ray is coming from inside,
            // so the probability of sampling the glass lobe is 1 if glass_prob is not 0.
            diffuse_prob = 0;
            metallic_prob = 0;
            clearcoat_prob = 0;
            if (glass_prob > 0) {
                glass_prob = 1;
            }
        }

        if (reflect) {
            // For metallic: visible normal sampling -> D * G_in
            float3 half_vector = normalize(V + L);
            // Flip half-vector if it's below surface
            if (dot(half_vector, normal) < 0) {
                half_vector = -half_vector;
            }
            float n_dot_in = dot(normal, V);
            float n_dot_h = dot(half_vector, Ns);
            float h_dot_in = dot(half_vector, V);
            float h_dot_out = dot(half_vector, L);

            // For diffuse, metallic, and clearcoat, the light bounces
            // only at the top of the surface.
            if (dot(Ng, L) >= 0 && dot(Ng, V) >= 0) {
                diffuse_prob *= fmax(dot(normal, V), float(0)) / M_PIf;

                if (n_dot_in > 0) {
                    float D = GTR2(to_local(half_vector, normal), alpha_x, alpha_y);
                    float G_in = smith_masking_gtr2(to_local(V, normal), alpha_x, alpha_y);
                    metallic_prob *= (D * G_in / (4 * n_dot_in));
                }
                else {
                    metallic_prob = 0;
                }

                // For clearcoat: D importance sampling
                if (n_dot_h > 0 && h_dot_out > 0) {
                    float Dc = GTR1(n_dot_h, alpha_c);
                    clearcoat_prob *= (Dc * n_dot_h / (4 * h_dot_out));
                }
                else {
                    clearcoat_prob = 0;
                }
            }

            // For glass: F * visible normal
            float eta = dot(Ng, V) > 0 ? sbt.eta : 1 / sbt.eta;
            float Fg = fresnel_dielectric(h_dot_in, eta);
            float D = GTR2(to_local(half_vector, normal), alpha_x, alpha_y);
            float G_in = smith_masking_gtr2(to_local(V, normal), alpha_x, alpha_y);
            glass_prob *= (Fg * D * G_in / (4 * fabs(n_dot_in)));
        }
        else {
            // Only glass component for refraction
            float eta = dot(Ng, V) > 0 ? sbt.eta : 1 / sbt.eta;
            float3 half_vector = normalize(V + L * eta);
            // Flip half-vector if it's below surface
            if (dot(half_vector, normal) < 0) {
                half_vector = -half_vector;
            }
            float h_dot_in = dot(half_vector, V);
            float h_dot_out = dot(half_vector, L);
            float D = GTR2(to_local(half_vector, normal), alpha_x, alpha_y);
            float G_in = smith_masking_gtr2(to_local(V, normal), alpha_x, alpha_y);
            float Fg = fresnel_dielectric(h_dot_in, eta);
            float sqrt_denom = h_dot_in + eta * h_dot_out;
            float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
            glass_prob *= (1 - Fg) * D * G_in * fabs(dh_dout * h_dot_in / dot(normal, V));
        }

        return diffuse_prob + metallic_prob + glass_prob + clearcoat_prob;
    }

    extern "C" __device__ float3 __direct_callable__evaluate(const TriangleMeshSBTData * sbtRaw,
        const float3 & L, const float3 & V, const float3 & Ns, const float3 & Ng)
    {
        DisneyBSDFShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        bool reflect = dot(Ng, L) * dot(Ng, V) > 0;
        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) * dot(Ng, V) < 0) {
            normal = normal * make_float3(- 1.f);
        }
                
        float3 base_color = sbt.baseColor;
        //float specular_transmission =
        //    eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
        //float metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
        //float subsurface = eval(bsdf.subsurface, vertex.uv, vertex.uv_screen_size, texture_pool);
        //float specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
        //float roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
        //float specular_tint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
        //float anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
        //float sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);
        //float sheen_tint = eval(bsdf.sheen_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
        float clearcoat = (0.25f) * sbt.clearcoat;
        //float clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);

        float aspect = sqrt(1 - sbt.anisotropic * 0.9f);
        constexpr float min_alpha = 0.0001f;
        float alpha_x = max(min_alpha, sbt.roughness * sbt.roughness / aspect);
        float alpha_y = max(min_alpha, sbt.roughness * sbt.roughness * aspect);
        float alpha_c = (1.f - sbt.clearcoat_gloss) * 0.1f + sbt.clearcoat_gloss * 0.001f;
        // return normal / 2.f + 0.5f;
        if (reflect) {
            float n_dot_in = dot(normal, V);
            float n_dot_out = dot(normal, L);
            float3 half_vector = normalize(V + L);
            // Flip half-vector if it's below surface
            if (dot(half_vector, normal) < 0) {
                half_vector = -half_vector;
            }
            float n_dot_h = dot(Ns, half_vector);
            float h_dot_in = dot(half_vector, V);
            float h_dot_out = dot(half_vector, L);

            float3 diffuse_contrib = make_float3(0.f);
            float3 metallic_contrib = make_float3(0.f);
            float clearcoat_contrib = float(0);
            float3 sheen_contrib = make_float3(0.f);
            // return make_float3(h_dot_out, h_dot_out, h_dot_out);
            // For diffuse, metallic, sheen, and clearcoat, the light bounces
            // only at the top of the surface.
            if (dot(Ng, V) >= 0 && dot(Ng, L) >= 0 && n_dot_out > 0) {
                // Diffuse component
                
                // The base diffuse model
                float Fd90 = float(0.5) + 2 * sbt.roughness * h_dot_out * h_dot_out;
                
                float schlick_n_dot_out = pow(1 - n_dot_out, float(5));
                float schlick_n_dot_in = pow(1 - n_dot_in, float(5));
                float schlick_h_dot_out = pow(1 - h_dot_out, float(5));
                float base_diffuse = (1 + (Fd90 - 1) * schlick_n_dot_out) *
                    (1 + (Fd90 - 1) * schlick_n_dot_in);
                
                // The subsurface model
                // Disney's hack to increase the response at grazing angle
                float Fss90 = h_dot_out * h_dot_out * sbt.roughness;
                float Fss = (1 + (Fss90 - 1) * schlick_n_dot_out) *
                    (1 + (Fss90 - 1) * schlick_n_dot_in);
                // Lommel-Seeliger law (modified/rescaled)
                float ss = float(1.25) * (Fss * (1 / (n_dot_out + n_dot_in) - float(0.5)) + float(0.5));

                diffuse_contrib =
                    (1 - sbt.specular_transmission) * (1 - sbt.metallic) * base_color *
                    ((base_diffuse * (1 - sbt.subsurface) + ss * sbt.subsurface) / (float)M_PIf) * n_dot_out;
                
                // Sheen component
                float3 Ctint =
                    luminance(base_color) > 0 ? base_color / luminance(base_color) : make_float3(1.);
                float3 Csheen = (1 - sbt.sheen_tint) * make_float3(1.) + sbt.sheen_tint * Ctint;
                sheen_contrib =
                    (1 - sbt.metallic) * sbt.sheen * Csheen * schlick_h_dot_out * n_dot_out;

                // Metallic component
                if (n_dot_in > 0 && h_dot_out > 0 && n_dot_h > 0) {
                    float eta = sbt.eta; // we're always going inside
                    float spec_f0 = (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
                    float3 spec_color =
                        ((1 - sbt.specular_tint) * make_float3(1.) + sbt.specular_tint * Ctint);
                    float3 Cspec0 =
                        sbt.specular * spec_f0 * (1 - sbt.metallic) * spec_color + sbt.metallic * base_color;
                    float spec_weight = (1 - sbt.specular_transmission * (1 - sbt.metallic));

                    float3 F = schlick_fresnel(Cspec0, h_dot_out);
                    float D = GTR2(to_local(half_vector, normal), alpha_x, alpha_y);
                    float G_in = smith_masking_gtr2(to_local(V, normal), alpha_x, alpha_y);
                    float G_out = smith_masking_gtr2(to_local(L, normal), alpha_x, alpha_y);
                    float G = G_in * G_out;
                    metallic_contrib = spec_weight * F * D * G / (4 * n_dot_in);
                }

                // Clearcoat component
                if (n_dot_in > 0 && n_dot_h > 0) {
                    float Fc = schlick_fresnel(float(0.04), h_dot_out); 
                    // Generalized Trowbridge-Reitz distribution
                    float Dc = GTR1(n_dot_h, alpha_c);
                    // SmithG with fixed alpha
                    float Gc_in = smith_masking_gtr1(to_local(V, normal));
                    float Gc_out = smith_masking_gtr1(to_local(L, normal));
                    float Gc = Gc_in * Gc_out;

                    clearcoat_contrib = clearcoat * Fc * Dc * Gc / (4 * n_dot_in);
                }
            }
            
            // For glass, lights bounce at both sides of the surface.
            // return diffuse_contrib;
            // Glass component
            float eta = dot(Ng, V) > 0 ? sbt.eta : 1 / sbt.eta;
            
            float Fg = fresnel_dielectric(h_dot_in, eta);
            float D = GTR2(to_local(half_vector, normal), alpha_x, alpha_y);
            float G_in = smith_masking_gtr2(to_local(V, normal), alpha_x, alpha_y);
            float G_out = smith_masking_gtr2(to_local(L, normal), alpha_x, alpha_y);
            
            float G = G_in * G_out;
            float3 glass_contrib =
                base_color * ((1 - sbt.metallic) * sbt.specular_transmission * (Fg * D * G) / (4 * fabs(n_dot_in)));
            // printf("diffuse %f, %f, %f\n", diffuse_contrib.x, diffuse_contrib.y, diffuse_contrib.z);
            // return float3(1.f, 1.f, 1.f);
            return diffuse_contrib +sheen_contrib + metallic_contrib + glass_contrib + make_float3(clearcoat_contrib);
        }
        else {
            // Only the glass component for refraction
            float eta = dot(Ng, V) > 0 ? sbt.eta : 1 / sbt.eta;
            float3 half_vector = normalize(V + L * eta);
            // Flip half-vector if it's below surface
            if (dot(half_vector, normal) < 0) {
                half_vector = -half_vector;
            }

            float eta_factor = (1 / (eta * eta));
            float h_dot_in = dot(half_vector, V);
            float h_dot_out = dot(half_vector, L);
            float sqrt_denom = h_dot_in + eta * h_dot_out;

            float Fg = fresnel_dielectric(h_dot_in, eta);
            float D = GTR2(to_local(half_vector, normal), alpha_x, alpha_y);
            float G_in = smith_masking_gtr2(to_local(V, normal), alpha_x, alpha_y);
            float G_out = smith_masking_gtr2(to_local(L, normal), alpha_x, alpha_y);
            float G = G_in * G_out;

            // Burley propose to take the square root of the base color to preserve albedoF
            return sqrt(base_color) * ((1 - sbt.metallic) * sbt.specular_transmission *
                (eta_factor * (1 - Fg) * D * G * eta * eta * fabs(h_dot_out * h_dot_in)) /
                (fabs(dot(normal, V)) * sqrt_denom * sqrt_denom));
        }
    }

    extern "C" __device__ float3 __direct_callable__sample(const TriangleMeshSBTData * sbtRaw,
        const float3 & V, const float3 & Ns, const float3 & Ng, Random & random)
    {
        DisneyBSDFShaderInfo sbt;
        getSbt(sbt, sbtRaw);

        // Flip the shading frame if it is inconsistent with the geometry normal
        float3 normal = Ns;
        if (dot(Ns, V) * dot(Ng, V) < 0) {
            normal = normal * -1.f;
        }

        // Fetch the texture values for later use
        float specular_transmission = sbt.specular_transmission;
        float metallic = sbt.metallic;
        float roughness = sbt.roughness;
        float anisotropic = sbt.anisotropic;
        float clearcoat = float(0.25) * sbt.clearcoat;
        float clearcoat_gloss = sbt.clearcoat_gloss;

        float aspect = sqrt(1 - anisotropic * float(0.9));
        constexpr float min_alpha = float(0.0001);
        float alpha_x = max(min_alpha, roughness * roughness / aspect);
        float alpha_y = max(min_alpha, roughness * roughness * aspect);
        float alpha_c = (1 - clearcoat_gloss) * float(0.1) + clearcoat_gloss * float(0.001);

        float diffuse_weight = (1 - metallic) * (1 - specular_transmission);
        float metallic_weight = (1 - specular_transmission * (1 - metallic));
        float glass_weight = (1 - metallic) * specular_transmission;
        float clearcoat_weight = clearcoat;

        // Two cases: 1) if we are coming from "outside" the surface, 
        // sample all lobes
        float rnd_param_w = random();
        float rnd_param_u = random();
        float rnd_param_v = random();

        if (dot(Ng, V) >= 0) {
            float total_weight = diffuse_weight + metallic_weight + glass_weight + clearcoat_weight;
            float diffuse_prob = diffuse_weight / total_weight;
            float metallic_prob = metallic_weight / total_weight;
            float glass_prob = glass_weight / total_weight;
            // float clearcoat_prob = clearcoat_weight / total_weight;
            if (rnd_param_w <= diffuse_prob) {
                float3 L;
                sample_cos_hemisphere_dir(rnd_param_u, rnd_param_v, L);
                return to_world(L, normal);
            }
            else if (rnd_param_w <= (diffuse_prob + metallic_prob)) { // metallic
             // Visible normal sampling

             // Convert the incoming direction to local coordinates
                float3 local_dir_in = to_local(V, Ns);
                float3 local_micro_normal =
                    sample_visible_normals(local_dir_in, alpha_x, alpha_y, make_float2(rnd_param_u, rnd_param_v));

                // Transform the micro normal to world space
                float3 half_vector = to_world(local_micro_normal, Ns);
                // Reflect over the world space normal
                float3 reflected = normalize(-V + 2 * dot(V, half_vector) * half_vector);
                return reflected;
            }
            else if (rnd_param_w <= (diffuse_prob + metallic_prob + glass_prob)) { // glass
                if (glass_prob <= 0) {
                    // Just to be safe numerically.
                    return make_float3(-2);
                }
                // Visible normal sampling

                // Convert the incoming direction to local coordinates
                float3 local_dir_in = to_local(V, Ns);
                float3 local_micro_normal =
                    sample_visible_normals(local_dir_in, alpha_x, alpha_y, make_float2(rnd_param_u, rnd_param_v));

                // Transform the micro normal to world space
                float3 half_vector = to_world(local_micro_normal, Ns);
                // Flip half-vector if it's below surface
                if (dot(half_vector, normal) < 0) {
                    half_vector = -half_vector;
                }

                // Now we need to decide whether to reflect or refract.
                // We do this using the Fresnel term.
                float h_dot_in = dot(half_vector, V);
                float eta = dot(Ng, V) > 0 ? sbt.eta : 1 / sbt.eta;
                float F = fresnel_dielectric(h_dot_in, eta);
                // rescale rnd_param_w from
                // (diffuse_prob + metallic_prob, diffuse_prob + metallic_prob + glass_prob]
                // to
                // (0, 1]
                float u = (rnd_param_w - (diffuse_prob + metallic_prob)) / glass_prob;
                if (u <= F) {
                    // Reflect over the world space normal
                    float3 reflected = normalize(-V + 2 * dot(V, half_vector) * half_vector);
                    return reflected;
                }
                else {
                    // Refraction
                    float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
                    if (h_dot_out_sq <= 0) {
                        return make_float3(-2, -2, -2);
                    }
                    // flip half_vector if needed
                    if (h_dot_in < 0) {
                        half_vector = -half_vector;
                    }
                    float h_dot_out = sqrt(h_dot_out_sq);
                    float3 refracted = -V / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
                    return refracted;
                }
            }
            else { // clearcoat
             // Only importance sampling D

             // Appendix B.2 Burley's note
                float alpha2 = alpha_c * alpha_c;
                // Equation 5
                float cos_h_elevation =
                    sqrt(max(float(0), (1 - pow(alpha2, 1 - rnd_param_u)) / (1 - alpha2)));
                float sin_h_elevation = sqrt(max(1 - cos_h_elevation * cos_h_elevation, float(0)));
                float h_azimuth = 2 * M_PIf * rnd_param_v;
                float3 local_micro_normal{
                    sin_h_elevation * cos(h_azimuth),
                    sin_h_elevation * sin(h_azimuth),
                    cos_h_elevation
                };
                // Transform the micro normal to world space
                float3 half_vector = to_world(local_micro_normal, normal);

                // Reflect over the world space normal
                float3 reflected = normalize(-V + 2 * dot(V, half_vector) * half_vector);
                return reflected;
            }
        }
        else {
            // 2) otherwise, only consider the glass lobes.

            // Convert the incoming direction to local coordinates
            float3 local_dir_in = to_local(V, Ns);
            float3 local_micro_normal =
                sample_visible_normals(local_dir_in, alpha_x, alpha_y, make_float2(rnd_param_u, rnd_param_v));

            // Transform the micro normal to world space
            float3 half_vector = to_world(local_micro_normal, Ns);
            // Flip half-vector if it's below surface
            if (dot(half_vector, normal) < 0) {
                half_vector = -half_vector;
            }

            // Now we need to decide whether to reflect or refract.
            // We do this using the Fresnel term.
            float h_dot_in = dot(half_vector, V);
            float eta = dot(Ng, V) > 0 ? sbt.eta : 1 / sbt.eta;
            float F = fresnel_dielectric(h_dot_in, eta);
            float u = rnd_param_w;
            if (u <= F) {
                // Reflect over the world space normal
                float3 reflected = normalize(-V + 2 * dot(V, half_vector) * half_vector);
                return  reflected;
            }
            else {
                // Refraction
                float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
                if (h_dot_out_sq <= 0) {
                    return {};
                }
                // flip half_vector if needed
                if (h_dot_in < 0) {
                    half_vector = -half_vector;
                }
                float h_dot_out = sqrt(h_dot_out_sq);
                float3 refracted = -V / eta + (fabs(h_dot_in) / eta - h_dot_out) * half_vector;
                return refracted;
            }
        }
    }
}
