#pragma once
#include "mathUtils.h"
#ifdef __linux__ 
#include <bits/stdc++.h>
#endif
#define MAX_SH_DEGREE 8
#define MAX_SH_COEFFICIENTS 81

namespace nert_renderer {

    // for this simple example, we have a single ray type
    enum { RADIANCE_RAY_TYPE=0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

    struct floatSBT {
        float val;
        bool hasTexture;
        cudaTextureObject_t texture;
    };

    struct vec3SBT {
        float3 val;
        bool hasTexture;
        cudaTextureObject_t texture;
    };

    // This is the motherSBT for all materials, each material will have its own interpretation
    struct TriangleMeshSBTData {
        float3 *vertex;
        float3 *normal;
        float2 *texcoord;
        int3 *index;
        int* globalIndex;

        vec3SBT kd;
        vec3SBT ks;
        vec3SBT normalMap;
        
        floatSBT roughness;
        floatSBT specular_transmission;
        floatSBT metallic;
        floatSBT subsurface;
        floatSBT specular;
        floatSBT specular_tint;
        floatSBT anisotropic;
        floatSBT sheen;
        floatSBT sheen_tint;
        floatSBT clearcoat;
        floatSBT clearcoat_gloss;
        floatSBT eta;

        //{ roughPlastic, DisneyBSDF, cookTorrance, mirror };
        int type;
    };

    struct roughPlasticShaderInfo{
        float3 kd;
        float3 ks;
        float roughness;
        float eta;
    };

    struct DisneyBSDFShaderInfo{
        float3 baseColor;
        float roughness;
        float specular_transmission;
        float metallic;
        float subsurface;
        float specular;
        float specular_tint;
        float anisotropic;
        float sheen;
        float sheen_tint;
        float clearcoat;
        float clearcoat_gloss;
        float eta;
    };
    
    struct cookTorranceShaderInfo{
        float3 kd;
        float3 ks;
        float roughness;
    };
    
    struct simpleGGXShaderInfo{
        float3 kd;
        float roughness;
    };

    struct mirrorShaderInfo{

    };


    ////////////////// 
    /// Launch params
    //////////////////    

    struct TableDist1D_device {
        double* pmf;
        double* cdf;
        int size;
    };

    struct TableDist2D_device {
        double* cdf_rows;
        double* pdf_rows;

        double* cdf_marginals;
        double* pdf_marginals;

        double total_values;
        int width, height;
    };

    struct cubeEnvMap {
        cudaTextureObject_t faces[6];

        TableDist1D_device face_dist;
        TableDist2D_device sampling_dist[6];
    };

    struct sgEnvMap {
        float3* sg_location = nullptr;
        float3* sg_color= nullptr;
        float* sg_roughness= nullptr;
        int sg_count = 128;
    };

    union light_data {
        #ifdef __linux__ 
        light_data() {
            memset( this, 0, sizeof( light_data ) ); 
        };
        #endif
        float3 color = { 0.0f, 0.0f, 0.0f };
        cudaTextureObject_t latlong;
        struct cubeEnvMap cubeMap;
        struct sgEnvMap sgMap;
    };

    struct envLight {
        // Only used for sampling
        // Color, latlong, cubemap, sgmap
        int sampling_type = 0;
        light_data sampling_data;

        // Only used for direct interaction between ray and bg
        int view_type = 0;
        light_data view_data;

        // Used to light up the scene
        int lighting_type = 0;
        light_data lighting_data;
    };

    struct LaunchParams
    {


        struct {
            int samplesPerPixel = 1;
            int samplesPerLight = 1;
            int indirectSamplesPerDirect = 1;
            int maximumBounce = 12;
        } samplingParameter;

        struct {
            int      frameID = 0;
            bool     extraRender = true;

            float4*  colorBuffer;
            float4*  FirstHitDirectBuffer;
            float4*  FirstHitIndirectBuffer;

            float4*  cameraNormalBuffer;
            float4*  worldNormalBuffer;
            float4*  albedoBuffer;
            
            float4*  FirstHitBary;
            int4*    FirstHitVertexID;

            float4*  FirstHitUVCordBuffer;
            float4*  FirstHitPositionBuffer;
            float4*  FirstHitReflectDirBuffer;
            
            /*! Not used */
            float4*  FirstHitKdBuffer;
            float4*  FirstHitKsBuffer;

            float4*  SpecColorBuffer;
            float4*  DiffColorBuffer;
            
            float*   depthBuffer;
            float*   roughnessBuffer;
            
            bool*    perfectReflectionFlagBuffer;
            bool*    emissiveFlagBuffer;
            bool*    envMapFlagBuffer;

            /*! the size of the frame buffer to render */
            int2     size;
        } frame;
    
        struct {
            float3 position;
            float3 direction;
            float3 horizontal;
            float3 vertical;
            float cx;
            float cy;
            float fx;
            float fy;
        } camera;

        envLight env;

        float3 lowerBound;
        float3 boxSize;

        struct {
            float3* vertexLocations;
            float3* vertexNormals;
            int shTerms;

            float3* T_matrix;
            float3* light_vector;
        } sh;
        

        OptixTraversableHandle traversable;
    };

} // ::nert_renderer
