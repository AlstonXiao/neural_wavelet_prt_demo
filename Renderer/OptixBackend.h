#pragma once

#include "optix7.h"
#include <cuda_runtime.h>

#include "CUDABuffer.h"
#include "device_functions/LaunchParams.h"
#include "Scene.h"
#include "frames.h"

namespace nert_renderer {
	struct optixState
	{
        /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
        CUcontext          cudaContext;
        int                deviceID;
        /*! @} */

        //! the optix context that our pipeline will run in.
        OptixDeviceContext optixContext;
        OptixDeviceContextOptions contextOption;
        /*! @{ the pipeline and module we're building */
        OptixPipeline               pipeline;
        std::map<std::string, OptixModule> moduels;

        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions    pipelineLinkOptions = {};        
        OptixModuleCompileOptions   moduleCompileOptions = {};
        /* @} */

        /*! vector of all our program(group)s, and the SBT built around
        them */
        std::vector<OptixProgramGroup> raygenPGs;
        std::vector<OptixProgramGroup> missPGs;
        std::vector<OptixProgramGroup> hitgroupPGs;
        std::vector<OptixProgramGroup> exceptionPGs;
        std::vector<OptixProgramGroup> directCallablePGs;

        OptixDenoiser denoiser = nullptr;
        vec2i denoiserSize = {0, 0};
        bool aovMode = false;
        OptixDenoiserParams denoiserParams = {};
        CUDABuffer    denoiserScratch;
        CUDABuffer    denoiserState;
        CUDABuffer    denoiserHdrIntensity;
        CUDABuffer    denoiserHdrAverageColor;
	};

    struct sceneStateStruct
    {
        CUDABuffer raygenRecordsBuffer;
        CUDABuffer exceptionRecordsBuffer;
        CUDABuffer missRecordsBuffer;
        CUDABuffer hitgroupRecordsBuffer;
        CUDABuffer directCallabeRecordsBuffer;

        OptixShaderBindingTable sbt = {};

        /*! @{ one buffer per input mesh */
        std::vector<std::vector<CUDABuffer>> vertexBuffer;
        std::vector<std::vector<CUDABuffer>> normalBuffer;
        std::vector<std::vector<CUDABuffer>> texcoordBuffer;
        std::vector<std::vector<CUDABuffer>> indexBuffer;
        std::vector<CUDABuffer> globalIndexBuffer;
        /*! @} */
        
        
        //! buffer that keeps the (final, compacted) accel structure
        CUDABuffer asBuffer;

        std::vector<CUDABuffer> GasBuffer;
        std::vector<OptixTraversableHandle> GasHandle;

        std::vector<OptixInstance> instanceList;

        /*! @{ one texture object and pixel array per used texture */
        std::vector<cudaTextureObject_t> textureObjects;
        /*! @} */
        
        std::vector<float3> vertexBufferHost;
        CUDABuffer shVertexBuffer;
        CUDABuffer launchParamsBuffer;
    };

    enum class renderEngineMode { normal, shTmatrix, shRender, };

    class OptixBackend {
    public:
        OptixBackend(bool debugFlag = false, renderEngineMode mode = renderEngineMode::normal, int deviceID = 0);
    
        /*! creates and configures a optix device context (in this simple
        example, only for the primary GPU device) */
        void createContext();

        /*! creates the module that contains all the programs we are going
          to use. in this simple example, we use a single module from a
          single .cu file, using a single embedded ptx string */
        void createModule(std::map<const std::string, std::string> programs);

        /* does all setup for the callable program(s) we are going to use */
        void createDirectCallablePrograms(const std::string moduleName, std::vector<std::string>functionNames);

        /* does all setup for the raygen program(s) we are going to use */
        void createRaygenMissHitPrograms(const std::string RayModuleName, const std::string MissModuleName, const std::string HitModuleName, const std::string expModuleName);
 
        /*! assembles the full pipeline of all programs */
        void createPipeline();

        void buildGAS(const std::shared_ptr<Scene> scene);

        OptixTraversableHandle buildIAS(const std::shared_ptr<Scene> scene);
        
        // TODO update IAS

        /*! constructs the shader binding table */
        void buildSBT(const std::shared_ptr<Scene> scene);

        void createTexturesForSceneObjects(const std::shared_ptr<Scene> scene);

        /* Setup denoiser according to the new size. This function will destory previous denoiser*/
        void setupDenoiser(vec2i newsize, bool aov = false);

        /*! render one frame */
        void render(LaunchParams& launchParams, CUstream stream = nullptr);
        
        void shComputeTMatrix(LaunchParams& launchParams);
        int getNumVertex();

        void denoise(CUdeviceptr color, CUdeviceptr normal, CUdeviceptr albedo, CUdeviceptr output, float blendFactor, vec2i size, CUstream stream = nullptr);
        void AOVDenoise(std::vector<std::pair<CUdeviceptr, CUdeviceptr>>& inputOuputPair, CUdeviceptr normal, CUdeviceptr albedo, float blendFactor, vec2i size, CUstream stream);

    private:
        bool debug;
        renderEngineMode m_mode;
        optixState state;
        sceneStateStruct sceneState;
        const uint32_t tileSize = 256;
    };
}