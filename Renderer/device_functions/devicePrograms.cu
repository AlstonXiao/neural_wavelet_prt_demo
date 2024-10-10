#include <cuda_runtime.h>
#include <optix.h>

#include "LaunchParams.h"
#include "perRayDataUtils.h"
#include "shaderUtils.h"

#define NUM_LIGHT_SAMPLES 16
#define MAX_TRACE_DEPTH 12
#define RR_DEPTH 3

#define EVAL 0
#define PDF 1
#define SAMPLE 2
#define PACKINFO 3

#define ENVMAPBASE 0
#define NUMENVFUNC 3

#define MATBASE 12
#define NUMMATFUNC 4

namespace nert_renderer {
  
    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;
  
    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------
    extern "C" __global__ void __closesthit__shadow()
    {
        optixSetPayloadTypes(PAYLOAD_TYPE_OCCLUSION);
        optixSetPayload_0(__float_as_uint(0.f));
        optixSetPayload_1(__float_as_uint(0.f));
        optixSetPayload_2(__float_as_uint(0.f));
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
     
        const TriangleMeshSBTData &sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        PRD prd = loadClosesthitRadiancePRD();

        TriangleMeshSBTData sbtDataCopy = sbtData;
    
        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int   primID = optixGetPrimitiveIndex();
        const int3 index  = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        float2 tc = make_float2(0,0);
        bool hasTexcoord = sbtData.texcoord;

        if (hasTexcoord)
            tc = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        // ------------------------------------------------------------------
        // compute normal, using either shading normal (if avail), or
        // geometry normal (fallback)
        // ------------------------------------------------------------------
        const float3 &A     = sbtData.vertex[index.x];
        const float3 &B     = sbtData.vertex[index.y];
        const float3 &C     = sbtData.vertex[index.z];
        float3 Ng = cross(B-A,C-A);

        // Texture Normal > Vertex Normal > Geometry Normal
        float3 Ns = (sbtData.normal)
            ? ((1.f-u-v) * sbtData.normal[index.x]
                +       u * sbtData.normal[index.y]
                +       v * sbtData.normal[index.z])
            : Ng;
    
        /*if (sbtData.normalMap.hasTexture && hasTexcoord) {
            float4 fromTexture = tex2D<float4>(sbtData.normalMap.texture, tc.x, tc.y);
            float3 tangentNormal = make_float3(fromTexture) * 2 - 1;
            Ns = make_float3(fromTexture);
        }*/
    
        Ng = normalize(Ng);
        Ns = normalize(Ns);

        // TODO?
        float4 worldToObject[3];
        optix_impl::optixGetWorldToObjectTransformMatrix(worldToObject[0], worldToObject[1], worldToObject[2]);
        float3 transfromedNg = optix_impl::optixTransformNormal(worldToObject[0], worldToObject[1], worldToObject[2], (float3)Ng);
        float3 transfromedNs = optix_impl::optixTransformNormal(worldToObject[0], worldToObject[1], worldToObject[2], (float3)Ns);
        
        Ng = normalize(transfromedNg);
        Ns = normalize(transfromedNs);


        // ------------------------------------------------------------------
        // check the hitpoint
        // ------------------------------------------------------------------
        const float3 rayDir = normalize(optixGetWorldRayDirection());
        float depth = optixGetRayTmax();

        const float3 surfPos = (float3)optixGetWorldRayOrigin() + rayDir * depth + make_float3(0.001);
        prd.ray_origin = surfPos; 

        // ------------------------------------------------------------------
        // compute shader specific parameters
        // ------------------------------------------------------------------
   
        if (hasTexcoord) {
            #define readTextureField(field) field.val = field.hasTexture? \
                    tex2D<float4>(field.texture, tc.x, tc.y).x : field.val;

            sbtDataCopy.kd.val = sbtDataCopy.kd.val;
            sbtDataCopy.ks.val = sbtDataCopy.ks.val;

            if (sbtDataCopy.kd.hasTexture) {
                float4 tex = tex2D<float4>(sbtDataCopy.kd.texture, tc.x, tc.y);
                sbtDataCopy.kd.val = make_float3(tex.x, tex.y, tex.z);
            }
            if (sbtDataCopy.ks.hasTexture) {
                float4 tex = tex2D<float4>(sbtDataCopy.ks.texture, tc.x, tc.y);
                sbtDataCopy.ks.val = make_float3(tex.x, tex.y, tex.z);
            }

            readTextureField(sbtDataCopy.roughness)
            readTextureField(sbtDataCopy.specular_transmission)
            readTextureField(sbtDataCopy.metallic)
            readTextureField(sbtDataCopy.subsurface)
            readTextureField(sbtDataCopy.specular)
            readTextureField(sbtDataCopy.specular_tint)
            readTextureField(sbtDataCopy.anisotropic)
            readTextureField(sbtDataCopy.sheen)
            readTextureField(sbtDataCopy.sheen_tint)
            readTextureField(sbtDataCopy.clearcoat)
            readTextureField(sbtDataCopy.clearcoat_gloss)
            readTextureField(sbtDataCopy.eta)
            #undef readTextureField
        }

        int mattype = sbtData.type;

        // mirror is a special case here
        if (mattype == 3) {
            prd.ray_dir = reflect(rayDir, Ns);
            if (!prd.firstHitUpdated) {
                // add the depth from camera to the mirror
                prd.prdExtraInfo->pixeldepth += depth;
                prd.pixelPerfectReflectionFlag = true;
            }
            return;
        }
    
        float3 radianceInc = make_float3(0.f);
        // Avoid quaduple increase in sampling in later stage
        int samplesPerLight = prd.firstHitUpdated ? 1 : optixLaunchParams.samplingParameter.samplesPerLight;

        for (int i = 0; i < samplesPerLight; i++) {
            float3 L = optixDirectCall<float3, const light_data&, Random&>(ENVMAPBASE + SAMPLE + optixLaunchParams.env.sampling_type * NUMENVFUNC, optixLaunchParams.env.sampling_data, prd.random);
            float pdfSolidEnv = optixDirectCall<float, const light_data&, const float3&>(ENVMAPBASE + PDF + optixLaunchParams.env.sampling_type * NUMENVFUNC, optixLaunchParams.env.sampling_data, L);

            // the values we store the PRD pointer in:
            uint32_t u0, u1, u2;
            u0 = __float_as_uint(0.f);
            u1 = __float_as_uint(0.f);
            u2 = __float_as_uint(0.f);
            optixTrace(
                PAYLOAD_TYPE_OCCLUSION,
                optixLaunchParams.traversable,
                surfPos,
                L,
                1e-2f,      // tmin
                1e20f,      // tmax
                0.0f,       // rayTime
                OptixVisibilityMask(255),
                // For shadow rays: skip any/closest hit shaders and terminate on first
                // intersection with anything. The miss shader is used to mark if the
                // light was visible.
                OPTIX_RAY_FLAG_DISABLE_ANYHIT
                | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                // | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                SHADOW_RAY_TYPE,            // SBT offset
                RAY_TYPE_COUNT,               // SBT stride
                SHADOW_RAY_TYPE,            // missSBTIndex 
                u0, u1, u2);
            float3 lightVisibility = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));
            if (fmaxf(lightVisibility) > 0.1 && pdfSolidEnv > 1e-4)
            {
                float pdfSolidBRDF = optixDirectCall <float, const TriangleMeshSBTData*, const float3&, const float3&, const float3&, const float3&>
                    (MATBASE + sbtData.type * NUMMATFUNC + PDF, &sbtDataCopy, L, -rayDir, Ns, Ng);
                float3 intensity = optixDirectCall <float3, const TriangleMeshSBTData*, const float3&, const float3&, const float3&, const float3&>
                    (MATBASE + sbtData.type * NUMMATFUNC + EVAL, &sbtDataCopy, L, -rayDir, Ns, Ng);
                float3 radiance = optixDirectCall<float3, const light_data&, const float3&>(ENVMAPBASE + EVAL + optixLaunchParams.env.lighting_type * NUMENVFUNC, optixLaunchParams.env.lighting_data, L);
                float pdfSolidBRDF2 = pdfSolidBRDF * pdfSolidBRDF;
                float pdfSolidEnv2 = pdfSolidEnv * pdfSolidEnv;
                float MIS_weight = pdfSolidEnv /
                    fmax(pdfSolidEnv2 + pdfSolidBRDF2, 1e-8f);
                float3 radianceIncPR = intensity * radiance * prd.light_contribution * MIS_weight;
                if (isnan(radianceIncPR.x) || isnan(radianceIncPR.y) || isnan(radianceIncPR.z)) {
                    radianceIncPR = make_float3(0.f);
                    // printf("gotu\n");
                } else {
                    radianceInc += radianceIncPR;
                }
            }
        }
        prd.pixelColor += radianceInc / (float)samplesPerLight;

        // If direct, importance sample the brdf
        if (optixLaunchParams.samplingParameter.maximumBounce == 1) {
            float3 radianceInc = make_float3(0.);
            for (int i = 0; i < samplesPerLight; i++) {
                float3 L = optixDirectCall<float3, const  TriangleMeshSBTData*, const float3&, const float3&, const float3&, Random&>
                    (MATBASE + sbtData.type * NUMMATFUNC + SAMPLE, &sbtDataCopy, -rayDir, Ns, Ng, prd.random);
                float pdfSolidBRDF = optixDirectCall <float, const  TriangleMeshSBTData*, const float3&, const float3&, const float3&, const float3&>
                    (MATBASE + sbtData.type * NUMMATFUNC + PDF, &sbtDataCopy, L, -rayDir, Ns, Ng);
                float pdfSolidEnv = optixDirectCall<float, const light_data&, const float3&>(ENVMAPBASE + PDF + optixLaunchParams.env.sampling_type * NUMENVFUNC, optixLaunchParams.env.sampling_data, L);

                uint32_t u0, u1, u2;
                u0 = __float_as_uint(0.f);
                u1 = __float_as_uint(0.f);
                u2 = __float_as_uint(0.f);
                optixTrace(
                    PAYLOAD_TYPE_OCCLUSION,
                    optixLaunchParams.traversable,
                    surfPos,
                    L,
                    1e-2f,      // tmin
                    1e20f,      // tmax
                    0.0f,       // rayTime
                    OptixVisibilityMask(255),
                    // For shadow rays: skip any/closest hit shaders and terminate on first
                    // intersection with anything. The miss shader is used to mark if the
                    // light was visible.
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                    // | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                    SHADOW_RAY_TYPE,            // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    SHADOW_RAY_TYPE,            // missSBTIndex 
                    u0, u1, u2);
                float3 lightVisibility = make_float3(__uint_as_float(u0), __uint_as_float(u1), __uint_as_float(u2));

                if (fmaxf(lightVisibility) > 0.1 && pdfSolidEnv > 1e-4)
                {
                    float3 intensity = optixDirectCall <float3, const TriangleMeshSBTData*, const float3&, const float3&, const float3&, const float3&>
                        (MATBASE + sbtData.type * NUMMATFUNC + EVAL, &sbtDataCopy, L, -rayDir, Ns, Ng);
                    float3 radiance = optixDirectCall<float3, const light_data&, const float3&>(ENVMAPBASE + EVAL + optixLaunchParams.env.lighting_type * NUMENVFUNC, optixLaunchParams.env.lighting_data, L);
                    float pdfSolidBRDF2 = pdfSolidBRDF * pdfSolidBRDF;
                    float pdfSolidEnv2 = pdfSolidEnv * pdfSolidEnv;
                    float MIS_weight = pdfSolidBRDF /
                        fmax(pdfSolidEnv2 + pdfSolidBRDF2, 1e-8f);
                    float3 radianceIncPR = intensity * radiance * prd.light_contribution * MIS_weight;
                    if (isnan(radianceIncPR.x) || isnan(radianceIncPR.y) || isnan(radianceIncPR.z)) {}
                    else {
                        radianceInc += radianceIncPR;
                    }
                }
            }
            prd.pixelColor += radianceInc / (float)samplesPerLight;
        } 

        float3 ray_dir;
        float pdf;
        ray_dir = optixDirectCall<float3, const  TriangleMeshSBTData*, const float3&, const float3&, const float3&, Random&>
            (MATBASE + sbtData.type * NUMMATFUNC + SAMPLE, &sbtDataCopy, -rayDir, Ns, Ng, prd.random);
    
        if (ray_dir.x == -2) prd.done = true;
        pdf = optixDirectCall <float, const  TriangleMeshSBTData*, const float3&, const float3&, const float3&, const float3&>
            (MATBASE + sbtData.type * NUMMATFUNC + PDF, &sbtDataCopy, ray_dir, -rayDir, Ns, Ng);
        float3 intensity = optixDirectCall <float3, const TriangleMeshSBTData*, const float3&, const float3&, const float3&, const float3&>
            (MATBASE + sbtData.type * NUMMATFUNC + EVAL, &sbtDataCopy, ray_dir, -rayDir, Ns, Ng);
     
        if (pdf <= 1e-4) {
            // Numerical issue -- we generated some invalid rays.
            prd.done = true;
        }

        prd.ray_dir = ray_dir;
        prd.pdf = pdf;

        prd.light_contribution = prd.light_contribution * intensity / prd.pdf; 
        if (!prd.firstHitUpdated) {
            prd.firstHitUpdated = true;
            prd.prdExtraInfo->pixelNormalFirstHit = Ns;

            prd.prdExtraInfo->pixelFirstHitBary = make_float3(1 - u - v, u, v);
            // prd.prdExtraInfo->pixelFirstHitVertexID = make_int3(sbtData.globalIndex[index.x], sbtData.globalIndex[index.y], sbtData.globalIndex[index.z]);

            prd.prdExtraInfo->pixelUV_cord = tc;
            prd.prdExtraInfo->pixelFirstHitPos = surfPos;
            prd.prdExtraInfo->pixelFirstHitReflectDir = reflect(rayDir, Ns);
            prd.prdExtraInfo->pixeldepth += depth;

            prd.prdExtraInfo->pixelEnvMapFlag = true;
            optixDirectCall<void, const TriangleMeshSBTData *, PRD &>(MATBASE + sbtData.type * NUMMATFUNC + PACKINFO, &sbtDataCopy, prd);
        }
 
        storeClosesthitRadiancePRD(prd);
    }
    
    extern "C" __global__ void __closesthit__SHradiance()
    {
        optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
        PRD prd = loadClosesthitRadiancePRD();

        TriangleMeshSBTData sbtDataCopy = sbtData;

        // ------------------------------------------------------------------
        // gather some basic hit information
        // ------------------------------------------------------------------
        const int   primID = optixGetPrimitiveIndex();
        const int3 index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        float2 tc = make_float2(0, 0);
        bool hasTexcoord = sbtData.texcoord;

        if (hasTexcoord)
            tc = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        // ------------------------------------------------------------------
        // compute normal, using either shading normal (if avail), or
        // geometry normal (fallback)
        // ------------------------------------------------------------------
        const float3& A = sbtData.vertex[index.x];
        const float3& B = sbtData.vertex[index.y];
        const float3& C = sbtData.vertex[index.z];
        float3 Ng = cross(B - A, C - A);

        // Texture Normal > Vertex Normal > Geometry Normal
        float3 Ns = (sbtData.normal)
            ? ((1.f - u - v) * sbtData.normal[index.x]
                + u * sbtData.normal[index.y]
                + v * sbtData.normal[index.z])
            : Ng;

        if (sbtData.normalMap.hasTexture && hasTexcoord) {
            float4 fromTexture = tex2D<float4>(sbtData.normalMap.texture, tc.x, tc.y);
            float3 tangentNormal = make_float3(fromTexture) * 2 - 1;
            Ns = make_float3(fromTexture);
        }

        Ng = normalize(Ng);
        Ns = normalize(Ns);

        // TODO?
        float4 worldToObject[3];
        optix_impl::optixGetWorldToObjectTransformMatrix(worldToObject[0], worldToObject[1], worldToObject[2]);
        float3 transfromedNg = optix_impl::optixTransformNormal(worldToObject[0], worldToObject[1], worldToObject[2], (float3)Ng);
        float3 transfromedNs = optix_impl::optixTransformNormal(worldToObject[0], worldToObject[1], worldToObject[2], (float3)Ns);

        Ng = normalize(transfromedNg);
        Ns = normalize(transfromedNg);


        // ------------------------------------------------------------------
        // check the hitpoint
        // ------------------------------------------------------------------
        const float3 rayDir = normalize(optixGetWorldRayDirection());
        float depth = optixGetRayTmax();

        const float3 surfPos = (float3)optixGetWorldRayOrigin() + rayDir * depth + make_float3(0.001);
        prd.ray_origin = surfPos;

        // ------------------------------------------------------------------
        // compute shader specific parameters
        // ------------------------------------------------------------------

        if (hasTexcoord) {
            #define readTextureField(field) field.val = field.hasTexture? \
                    tex2D<float4>(field.texture, tc.x, tc.y).x : field.val;

            sbtDataCopy.kd.val = sbtDataCopy.kd.val;
            sbtDataCopy.ks.val = sbtDataCopy.ks.val;

            if (sbtDataCopy.kd.hasTexture) {
                float4 tex = tex2D<float4>(sbtDataCopy.kd.texture, tc.x, tc.y);
                sbtDataCopy.kd.val = make_float3(tex.x, tex.y, tex.z);
            }
            if (sbtDataCopy.ks.hasTexture) {
                float4 tex = tex2D<float4>(sbtDataCopy.ks.texture, tc.x, tc.y);
                sbtDataCopy.ks.val = make_float3(tex.x, tex.y, tex.z);
            }

            readTextureField(sbtDataCopy.roughness)
            readTextureField(sbtDataCopy.specular_transmission)
            readTextureField(sbtDataCopy.metallic)
            readTextureField(sbtDataCopy.subsurface)
            readTextureField(sbtDataCopy.specular)
            readTextureField(sbtDataCopy.specular_tint)
            readTextureField(sbtDataCopy.anisotropic)
            readTextureField(sbtDataCopy.sheen)
            readTextureField(sbtDataCopy.sheen_tint)
            readTextureField(sbtDataCopy.clearcoat)
            readTextureField(sbtDataCopy.clearcoat_gloss)
            readTextureField(sbtDataCopy.eta)
            #undef readTextureField
        }

        int mattype = sbtData.type;
        int sh_max_degree = optixLaunchParams.sh.shTerms;
        int sh_max_coeffs = (sh_max_degree + 1) * (sh_max_degree + 1);

        // Compute the response of each point
        float3 sh_response_vertex_brdf[MAX_SH_COEFFICIENTS];
        float3 lightResponse[MAX_SH_COEFFICIENTS];

        float3 colorTriangleVertices[3];
        for (int i = 0; i < 3; i++) {
            colorTriangleVertices[i] = make_float3(0.f);
        }

        int spp = optixLaunchParams.samplingParameter.samplesPerPixel;

        for (int vertex_id = 0; vertex_id < 3; vertex_id++) {
            int current_index = index.x;
            if (vertex_id == 1)
                current_index = index.y;
            if (vertex_id == 2)
                current_index = index.z;

            for (int sh_coeff_id = 0; sh_coeff_id < MAX_SH_COEFFICIENTS; sh_coeff_id++) {
                sh_response_vertex_brdf[sh_coeff_id] = make_float3(0.f);
            }

            for (int sh_coeff_id = 0; sh_coeff_id < MAX_SH_COEFFICIENTS; sh_coeff_id++) {
                lightResponse[sh_coeff_id] = make_float3(0.f);
            }

            // Compute the BRDF in sh space
            float3 normal = normalize(sbtData.normal[current_index]);
            float3 viewDir = normalize(optixLaunchParams.camera.position - sbtData.vertex[current_index]);
            int valid_samples = 0;
            for (int j = 0; j < spp; j++) {
                float3 ray_dir = optixDirectCall<float3, const  TriangleMeshSBTData*, const float3&, const float3&, const float3&, Random&>
                    (MATBASE + sbtData.type * NUMMATFUNC + SAMPLE, &sbtDataCopy, viewDir, normal, normal, prd.random);
                float pdf = optixDirectCall <float, const  TriangleMeshSBTData*, const float3&, const float3&, const float3&, const float3&>
                    (MATBASE + sbtData.type * NUMMATFUNC + PDF, &sbtDataCopy, ray_dir, viewDir, normal, normal);

                if (ray_dir.x == -2 || pdf < 1e-6)
                    continue;

                float3 intensity = optixDirectCall <float3, const TriangleMeshSBTData*, const float3&, const float3&, const float3&, const float3&>
                    (MATBASE + sbtData.type * NUMMATFUNC + EVAL, &sbtDataCopy, ray_dir, viewDir, normal, normal);

                if (!isnan(intensity.x) && !isnan(intensity.y) && !isnan(intensity.z)){
                    float sh[MAX_SH_COEFFICIENTS];
                    SHEvaluate(ray_dir, sh);
                    for (int sh_coeff_id = 0; sh_coeff_id < MAX_SH_COEFFICIENTS; sh_coeff_id++) {
                        sh_response_vertex_brdf[sh_coeff_id] += sh[sh_coeff_id] * intensity / pdf;
                    }
                    valid_samples += 1;
                }
                
            }
            if (valid_samples > 0)
                for (int sh_coeff_id = 0; sh_coeff_id < MAX_SH_COEFFICIENTS; sh_coeff_id++) {
                    sh_response_vertex_brdf[sh_coeff_id] = sh_response_vertex_brdf[sh_coeff_id] / valid_samples;
                }

            // Compute the incoming light
            int globalIndex = sbtData.globalIndex[current_index];
            for (int k = 0; k < sh_max_coeffs; k++) {
                for (int j = 0; j < sh_max_coeffs; j++) {
                    lightResponse[k] += optixLaunchParams.sh.T_matrix[globalIndex * sh_max_coeffs * sh_max_coeffs + j + k * sh_max_coeffs] *
                        optixLaunchParams.sh.light_vector[j];
                }
            }
            for (int sh_coeff_id = 0; sh_coeff_id < sh_max_coeffs; sh_coeff_id++) {
                colorTriangleVertices[vertex_id] += lightResponse[sh_coeff_id] * sh_response_vertex_brdf[sh_coeff_id];
            }
            colorTriangleVertices[vertex_id] = make_float3(fabs(colorTriangleVertices[vertex_id].x), fabs(colorTriangleVertices[vertex_id].y), fabs(colorTriangleVertices[vertex_id].z));
        }

        prd.pixelColor =  (1 - u - v) * colorTriangleVertices[0]
            + u * colorTriangleVertices[1]
            + v * colorTriangleVertices[2];

        prd.firstHitUpdated = true;
        prd.prdExtraInfo->pixelNormalFirstHit = Ns;
        optixDirectCall<void, const TriangleMeshSBTData *, PRD &>(MATBASE + sbtData.type * NUMMATFUNC + PACKINFO, &sbtDataCopy, prd);
        storeClosesthitRadiancePRD(prd);
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */ }

    extern "C" __global__ void __anyhit__shadow()
    { /*! not going to be used */ }
  
    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------
  
    extern "C" __global__ void __miss__radiance()
    {
        optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
        PRD prd = loadMissRadiancePRD();

        if (prd.ray_recursie_depth == 0) {
            prd.pixelColor = optixDirectCall<float3, const light_data&, const float3& >(ENVMAPBASE + EVAL + optixLaunchParams.env.view_type * NUMENVFUNC, optixLaunchParams.env.view_data, optixGetWorldRayDirection());

        } else {
            float3 radiance = optixDirectCall<float3, const light_data&, const float3& >(ENVMAPBASE + EVAL + optixLaunchParams.env.lighting_type * NUMENVFUNC, optixLaunchParams.env.lighting_data, optixGetWorldRayDirection());
            // Multiple Importance Sampling 
            if (prd.pdf < 0) {
                prd.pixelColor = radiance * prd.light_contribution + prd.pixelColor;
            } else {
                float pdfSolidEnv = optixDirectCall<float, const light_data&, const float3& >(ENVMAPBASE + PDF + optixLaunchParams.env.sampling_type * NUMENVFUNC, optixLaunchParams.env.sampling_data, optixGetWorldRayDirection());
                float pdfSolidBRDF = prd.pdf;
                float pdfSolidEnv2 = pdfSolidEnv * pdfSolidEnv;
                float pdfSolidBRDF2 = pdfSolidBRDF * pdfSolidBRDF;
                float MIS_weight = pdfSolidBRDF2 / fmax(pdfSolidBRDF2 + pdfSolidEnv2, 1e-14f);
                float3 radianceInc = radiance * prd.light_contribution * MIS_weight;
                prd.pixelColor = radianceInc + prd.pixelColor;
            }
        }
        
        //if (prd.ray_recursie_depth == 2) {
        //    prd.pixelColor += vec3f(0.f, 1.f, 0.f);
        //}
        if (!prd.firstHitUpdated) {
            prd.firstHitUpdated = true;
            prd.prdExtraInfo->pixelNormalFirstHit = -optixGetWorldRayDirection();
            prd.prdExtraInfo->pixelAlbedoFirstHit = prd.pixelColor;

            prd.prdExtraInfo->pixelUV_cord = make_float2(0.f);
            prd.prdExtraInfo->pixelFirstHitPos = make_float3(-100000);
            prd.prdExtraInfo->pixelFirstHitReflectDir = -optixGetWorldRayDirection();
            prd.prdExtraInfo->pixeldepth = -1;

            // assuming environment mat is a diffuse surface
            prd.prdExtraInfo->pixelFirstHitKs = make_float3(0.f);
            prd.prdExtraInfo->pixelFirstHitKd = prd.pixelColor;
            prd.prdExtraInfo->pixelFirstHitRoughness= 1;

            prd.prdExtraInfo->pixelEmissiveFlag = true;
            prd.prdExtraInfo->pixelEnvMapFlag = false;
        }
        prd.done = true;
        prd.pixelDirectSampleHitEnvMapFlag = true;
        storeMissRadiancePRD(prd);
    }


    extern "C" __global__ void __miss__shadow()
    {
        optixSetPayloadTypes(PAYLOAD_TYPE_OCCLUSION);
        // we didn't hit anything, so the light is visible
        optixSetPayload_0(__float_as_uint(1.f));
        optixSetPayload_1(__float_as_uint(1.f));
        optixSetPayload_2(__float_as_uint(1.f));
    }

    extern "C" __global__ void __exception__exception()
    { 
        const int exceptionCode = optixGetExceptionCode();
        const uint3 index = optixGetLaunchIndex();
        
        if (exceptionCode == 0) {
            printf("(%4i,%4i,%4i) error: To_local z and x not orthogonal", index.x, index.y, index.z);
        }
        else {
            optix_impl::optixDumpExceptionDetails();
        }

    }
} // ::nert_renderer
