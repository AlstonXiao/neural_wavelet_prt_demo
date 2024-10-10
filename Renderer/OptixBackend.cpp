#include "OptixBackend.h"

#include <optix_stack_size.h>
#include "device_functions/perRayData.h"

inline float3 make_float3(const gdt::vec3f in) { return make_float3(in.x, in.y, in.z); }
#define putSBTFloat3(mat, sbt) sbt.val = make_float3(mat.val); if (mat.map_id != -1) {sbt.hasTexture = true; sbt.texture = sceneState.textureObjects[mat.map_id];} else {sbt.hasTexture = false;}
#define putSBT(mat, sbt) sbt.val = mat.val; if (mat.map_id != -1) {sbt.hasTexture = true; sbt.texture = sceneState.textureObjects[mat.map_id];} else {sbt.hasTexture = false;}

namespace nert_renderer {
    using namespace gdt;
    static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
        fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
    }

    /*! SBT record for a raygen program. We use launchParams not sbt to pass info */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void* data;
    };

    /*! SBT record for a miss program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        void* data;
    };

    /*! SBT record for a hitgroup program */
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        TriangleMeshSBTData data;
    };

    /*! SBT record for envmap program, it is a direct callable so no record*/
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) directCallableRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    /*! SBT record for envmap program, it is a direct callable so no record*/
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) exceptionRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };


	OptixBackend::OptixBackend(bool debugFlag, renderEngineMode mode, int deviceID)
        : debug(debugFlag), m_mode(mode) {
        if (debug)
            std::cout << "# Optix Renderer: Initializing optix backend..." << std::endl;

        cudaFree(0);
        
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0)
            throw std::runtime_error("no CUDA capable devices found!");
        if (debug)
            std::cout << "# Optix Renderer: Found " << numDevices << " CUDA devices" << std::endl;
        OPTIX_CHECK(optixInit());
        state.deviceID = deviceID;
        createContext();
	}

    void OptixBackend::createContext() {
        CUDA_CHECK(SetDevice(state.deviceID));

        if (debug) {
            cudaDeviceProp deviceProps;
            cudaGetDeviceProperties(&deviceProps, state.deviceID);
            std::cout << "#Optix Renderer: running on device: " << deviceProps.name << std::endl;
        }

        CUresult  cuRes = cuCtxGetCurrent(&state.cudaContext);
        if (cuRes != CUDA_SUCCESS)
            fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

        
        if (debug) {
            state.contextOption.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
            state.contextOption.logCallbackFunction = context_log_cb;
            state.contextOption.logCallbackLevel = 4;
            state.contextOption.logCallbackData = nullptr;
            OPTIX_CHECK(optixDeviceContextCreate(state.cudaContext, &state.contextOption, &state.optixContext));
        }
        else {
            state.contextOption.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
            state.contextOption.logCallbackFunction = context_log_cb;
            state.contextOption.logCallbackLevel = 1;
            state.contextOption.logCallbackData = nullptr;
            OPTIX_CHECK(optixDeviceContextCreate(state.cudaContext, &state.contextOption, &state.optixContext));
        }

    }

    void OptixBackend::createModule(std::map<const std::string, std::string> programs) {
        state.moduleCompileOptions = {};
        state.moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        if (debug) {
            state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            state.moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

        } else{
            state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            state.moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        }

        OptixPayloadType payloadTypes[2] = {};
        // radiance prd
        payloadTypes[RADIANCE_RAY_TYPE].numPayloadValues = sizeof(radiancePayloadSemantics) / sizeof(radiancePayloadSemantics[0]);
        payloadTypes[RADIANCE_RAY_TYPE].payloadSemantics = radiancePayloadSemantics;
        // occlusion prd
        payloadTypes[SHADOW_RAY_TYPE].numPayloadValues = sizeof(occlusionPayloadSemantics) / sizeof(occlusionPayloadSemantics[0]);
        payloadTypes[SHADOW_RAY_TYPE].payloadSemantics = occlusionPayloadSemantics;

        state.moduleCompileOptions.numPayloadTypes = 2;
        state.moduleCompileOptions.payloadTypes = payloadTypes;

        state.pipelineCompileOptions = {};
        state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        state.pipelineCompileOptions.usesMotionBlur = false;
        // state.pipelineCompileOptions.allowOpacityMicromaps = false;
        state.pipelineCompileOptions.numPayloadValues = 0;
        state.pipelineCompileOptions.numAttributeValues = 2;
        state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
        state.pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        if (debug)
            state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH 
            | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_USER;
        else
            state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

        char log[8192];
        size_t sizeof_log = sizeof(log);
       
        for (auto const& [programName, programContent] : programs) {
            state.moduels[programName] = {};
            OPTIX_CHECK(optixModuleCreateFromPTX(state.optixContext,
                &state.moduleCompileOptions,
                &state.pipelineCompileOptions,
                programContent.c_str(),
                programContent.size(),
                log, &sizeof_log,
                &state.moduels[programName]
            ));
            if (debug) if (sizeof_log > 1) PRINT(log);
        }
    }

    void OptixBackend::createDirectCallablePrograms(const std::string moduleName, std::vector<std::string>functionNames)
    {
        int currentSize = state.directCallablePGs.size();
        state.directCallablePGs.resize(state.directCallablePGs.size() + functionNames.size());

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc* pgDesc = new OptixProgramGroupDesc[functionNames.size()];
        for (int i = 0; i < functionNames.size(); i++) {
            pgDesc[i] = {};
            pgDesc[i].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            pgDesc[i].flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
            pgDesc[i].callables.moduleDC = state.moduels[moduleName];
            pgDesc[i].raygen.entryFunctionName = functionNames[i].c_str();
        }

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            pgDesc,
            functionNames.size(),
            &pgOptions,
            log, &sizeof_log,
            &state.directCallablePGs[currentSize]
        ));
        if (debug) if (sizeof_log > 1) PRINT(log);
    }

    void OptixBackend::createRaygenMissHitPrograms(const std::string RayModuleName, const std::string MissModuleName, const std::string HitModuleName, const std::string expModuleName)
    {
        // we do a single ray gen program in this example:
        state.raygenPGs.resize(1);
        state.missPGs.resize(RAY_TYPE_COUNT);
        state.hitgroupPGs.resize(RAY_TYPE_COUNT);
        state.exceptionPGs.resize(1);

        OptixProgramGroupOptions RpgOptions = {};
        OptixProgramGroupDesc RpgDesc = {};
        RpgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        RpgDesc.raygen.module = state.moduels[RayModuleName];

        if (m_mode == renderEngineMode::normal)
            RpgDesc.raygen.entryFunctionName = "__raygen__fovCamera";
        else if (m_mode == renderEngineMode::shTmatrix)
            RpgDesc.raygen.entryFunctionName = "__raygen__ComputeTMatrix";
        else
            RpgDesc.raygen.entryFunctionName = "__raygen__SHRenderCamera";

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &RpgDesc,
            1,
            &RpgOptions,
            log, &sizeof_log,
            &state.raygenPGs[0]
        ));
        if (debug) if (sizeof_log > 1) PRINT(log);

        OptixPayloadType payloadTypes[2] = {};
        // radiance prd
        payloadTypes[RADIANCE_RAY_TYPE].numPayloadValues = sizeof(radiancePayloadSemantics) / sizeof(radiancePayloadSemantics[0]);
        payloadTypes[RADIANCE_RAY_TYPE].payloadSemantics = radiancePayloadSemantics;
        // occlusion prd
        payloadTypes[SHADOW_RAY_TYPE].numPayloadValues = sizeof(occlusionPayloadSemantics) / sizeof(occlusionPayloadSemantics[0]);
        payloadTypes[SHADOW_RAY_TYPE].payloadSemantics = occlusionPayloadSemantics;

        OptixProgramGroupOptions MpgOptions = {};
        // MpgOptions.payloadType = &payloadTypes[SHADOW_RAY_TYPE];

        OptixProgramGroupDesc MpgDesc[2] = {};
        MpgDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        MpgDesc[0].miss.module = state.moduels[MissModuleName];
        MpgDesc[0].miss.entryFunctionName = "__miss__radiance";

        MpgDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        MpgDesc[1].miss.module = state.moduels[MissModuleName];
        MpgDesc[1].miss.entryFunctionName = "__miss__shadow";

        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &MpgDesc[0],
            1,
            &MpgOptions,
            log, &sizeof_log,
            &state.missPGs[0]
        ));

        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &MpgDesc[1],
            1,
            &MpgOptions,
            log, &sizeof_log,
            &state.missPGs[1]
        ));

        if (debug) if (sizeof_log > 1) PRINT(log);

        OptixProgramGroupOptions HpgOptions = {};
        // HpgOptions.payloadType = &payloadTypes[RADIANCE_RAY_TYPE];

        OptixProgramGroupDesc HpgDesc[2] = {};
        HpgDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        HpgDesc[0].hitgroup.moduleCH = state.moduels[HitModuleName];
        HpgDesc[0].hitgroup.moduleAH = state.moduels[HitModuleName];
        if (m_mode == renderEngineMode::shRender)
            HpgDesc[0].hitgroup.entryFunctionNameCH = "__closesthit__SHradiance";
        else
            HpgDesc[0].hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        HpgDesc[0].hitgroup.entryFunctionNameAH = "__anyhit__radiance";
        
        HpgDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        HpgDesc[1].hitgroup.moduleCH = state.moduels[HitModuleName];
        HpgDesc[1].hitgroup.moduleAH = state.moduels[HitModuleName];
        HpgDesc[1].hitgroup.entryFunctionNameCH = "__closesthit__shadow";
        HpgDesc[1].hitgroup.entryFunctionNameAH = "__anyhit__shadow";
        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &HpgDesc[0],
            1,
            &HpgOptions,
            log, &sizeof_log,
            &state.hitgroupPGs[0]
        ));

        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &HpgDesc[1],
            1,
            &HpgOptions,
            log, &sizeof_log,
            &state.hitgroupPGs[1]
        ));
        if (debug) if (sizeof_log > 1) PRINT(log);

        OptixProgramGroupDesc exceptionDesc = {};
        OptixProgramGroupOptions exceptionOptions = {};
        exceptionDesc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        exceptionDesc.exception.entryFunctionName = "__exception__exception";
        exceptionDesc.exception.module = state.moduels[expModuleName];

        OPTIX_CHECK(optixProgramGroupCreate(state.optixContext,
            &exceptionDesc,
            1,
            &exceptionOptions,
            log, &sizeof_log,
            &state.exceptionPGs[0]
        ));

        if (debug) if (sizeof_log > 1) PRINT(log);
    }

    void OptixBackend::createPipeline()
    {
        std::vector<OptixProgramGroup> programGroups;
        for (auto pg : state.raygenPGs)
            programGroups.push_back(pg);
        for (auto pg : state.hitgroupPGs)
            programGroups.push_back(pg);
        for (auto pg : state.missPGs)
            programGroups.push_back(pg);
        for (auto pg : state.exceptionPGs)
            programGroups.push_back(pg);
        for (auto pg : state.directCallablePGs)
            programGroups.push_back(pg);

        state.pipelineLinkOptions.maxTraceDepth = 15;
        if (debug)
            state.pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        else
            state.pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        char log[2048];
        size_t sizeof_log = sizeof(log);
        if (debug) {
            PING;
            PRINT(programGroups.size());
        }
        OPTIX_CHECK(optixPipelineCreate(state.optixContext,
            &state.pipelineCompileOptions,
            &state.pipelineLinkOptions,
            programGroups.data(),
            (int)programGroups.size(),
            log, &sizeof_log,
            &state.pipeline
        ));
        if (debug) if (sizeof_log > 1) PRINT(log);

        OptixStackSizes stack_sizes = {};
        for (auto pg : programGroups)
            OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes));

        uint32_t max_trace_depth = 2;
        uint32_t max_cc_depth = 0;
        uint32_t max_dc_depth = 1;
        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(
            &stack_sizes,
            max_trace_depth,
            max_cc_depth,
            max_dc_depth,
            &direct_callable_stack_size_from_traversal,
            &direct_callable_stack_size_from_state,
            &continuation_stack_size
        ));

        if (debug)
        std::cout << "DC " << direct_callable_stack_size_from_traversal << " DC2 " <<
            direct_callable_stack_size_from_state << " CC " << continuation_stack_size << std::endl;

        const uint32_t max_traversal_depth = 2;
        OPTIX_CHECK(optixPipelineSetStackSize(
            state.pipeline,
            direct_callable_stack_size_from_traversal,
            direct_callable_stack_size_from_state,
            continuation_stack_size,
            max_traversal_depth
        ));
       
    }

    void OptixBackend::buildGAS(const std::shared_ptr<Scene> scene){
        // collecting number of meshes
        int totalNumberOfTriMesh = 0;
        for (auto instance: scene->instance_list){
            totalNumberOfTriMesh += instance->meshes.size();
        }
        if (debug)
            std::cout<<"Building GAS: " << scene->instance_list.size() << " instance, " << totalNumberOfTriMesh << " meshes in total." << std::endl;
        sceneState.vertexBuffer.resize(scene->instance_list.size());
        sceneState.normalBuffer.resize(scene->instance_list.size());
        sceneState.texcoordBuffer.resize(scene->instance_list.size());
        sceneState.indexBuffer.resize(scene->instance_list.size());
        sceneState.GasBuffer.resize(scene->instance_list.size());
        sceneState.GasHandle.resize(scene->instance_list.size());

        std::vector<std::vector<OptixBuildInput>> triangleInput(scene->instance_list.size());
        std::vector<std::vector<CUdeviceptr>> d_vertices(scene->instance_list.size());
        std::vector<std::vector<CUdeviceptr>> d_indices(scene->instance_list.size());
        // Flags shared by everyone
        unsigned int triangleFlag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
            | OPTIX_BUILD_FLAG_ALLOW_COMPACTION 
            | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

        // TODO: primitiveIndexOffset and  OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS Might be useful

        accelOptions.motionOptions.numKeys = 0;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Unsync update
        std::vector<CUDABuffer> compactedSizeBuffer(scene->instance_list.size());
        std::vector<CUDABuffer> tempBuffer(scene->instance_list.size());
        std::vector<CUDABuffer> outputBuffer(scene->instance_list.size());
        std::vector<OptixAccelEmitDesc> emitDesc(scene->instance_list.size());
        
        for (int instanceID = 0; instanceID < scene->instance_list.size(); instanceID++) {
            compactedSizeBuffer[instanceID].alloc(sizeof(uint64_t));
            emitDesc[instanceID].type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitDesc[instanceID].result = compactedSizeBuffer[instanceID].d_pointer();
        }
        int maxStream = 2;
        std::vector<CUstream> unSyncStreams(maxStream);
        for (int i = 0; i < maxStream; i++) {
            cudaStreamCreate(&unSyncStreams[i]);
        }

        for (int instanceID = 0; instanceID < scene->instance_list.size(); instanceID++) {
            auto instance =  scene->instance_list[instanceID];
            int numMeshes = (int)instance->meshes.size();
            triangleInput[instanceID].resize(numMeshes);
            d_vertices[instanceID].resize(numMeshes);
            d_indices[instanceID].resize(numMeshes);

            sceneState.vertexBuffer[instanceID].resize(numMeshes);
            sceneState.normalBuffer[instanceID].resize(numMeshes);
            sceneState.texcoordBuffer[instanceID].resize(numMeshes);
            sceneState.indexBuffer[instanceID].resize(numMeshes);

            for (int meshID = 0; meshID < numMeshes; meshID++) {
                // upload the model to the device: the builder
                TriangleMesh& mesh = *instance->meshes[meshID];
                sceneState.vertexBuffer[instanceID][meshID].alloc_and_upload(mesh.vertex);
                sceneState.indexBuffer[instanceID][meshID].alloc_and_upload(mesh.index);
                if (!mesh.normal.empty())
                    sceneState.normalBuffer[instanceID][meshID].alloc_and_upload(mesh.normal);
                if (!mesh.texcoord.empty())
                    sceneState.texcoordBuffer[instanceID][meshID].alloc_and_upload(mesh.texcoord);

                triangleInput[instanceID][meshID] = {};
                triangleInput[instanceID][meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

                // create local variables, because we need a *pointer* to the
                // device pointers
                d_vertices[instanceID][meshID] = sceneState.vertexBuffer[instanceID][meshID].d_pointer();
                d_indices[instanceID][meshID] = sceneState.indexBuffer[instanceID][meshID].d_pointer();

                triangleInput[instanceID][meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                triangleInput[instanceID][meshID].triangleArray.vertexStrideInBytes = sizeof(vec3f);
                triangleInput[instanceID][meshID].triangleArray.numVertices = (int)mesh.vertex.size();
                triangleInput[instanceID][meshID].triangleArray.vertexBuffers = &d_vertices[instanceID][meshID];

                triangleInput[instanceID][meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                triangleInput[instanceID][meshID].triangleArray.indexStrideInBytes = sizeof(vec3i);
                triangleInput[instanceID][meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
                triangleInput[instanceID][meshID].triangleArray.indexBuffer = d_indices[instanceID][meshID];

                // in this example we have one SBT entry, and no per-primitive
                // materials:
                triangleInput[instanceID][meshID].triangleArray.flags = &triangleFlag;
                triangleInput[instanceID][meshID].triangleArray.numSbtRecords = 1;
                triangleInput[instanceID][meshID].triangleArray.sbtIndexOffsetBuffer = 0;
                triangleInput[instanceID][meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
                triangleInput[instanceID][meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;

                
            }
        
            OptixAccelBufferSizes blasBufferSizes = {};
            OPTIX_CHECK(optixAccelComputeMemoryUsage
            (state.optixContext,
                &accelOptions,
                triangleInput[instanceID].data(),
                (int)numMeshes,  // num_build_inputs
                &blasBufferSizes
            ));

            // ==================================================================
            // execute build (main stage)
            // ==================================================================

            tempBuffer[instanceID].alloc(blasBufferSizes.tempSizeInBytes);
            outputBuffer[instanceID].alloc(blasBufferSizes.outputSizeInBytes);

            OPTIX_CHECK(optixAccelBuild(state.optixContext,
                unSyncStreams[instanceID % maxStream],
                &accelOptions,
                triangleInput[instanceID].data(),
                (int)numMeshes,
                tempBuffer[instanceID].d_pointer(),
                tempBuffer[instanceID].sizeInBytes,

                outputBuffer[instanceID].d_pointer(),
                outputBuffer[instanceID].sizeInBytes,

                &sceneState.GasHandle[instanceID],
                &emitDesc[instanceID], 1
            ));
        }
        
        CUDA_SYNC_CHECK();
        // ==================================================================
        // perform compaction
        // ==================================================================
        for (int instanceID = 0; instanceID < scene->instance_list.size(); instanceID++) {
            uint64_t compactedSize;
            compactedSizeBuffer[instanceID].download(&compactedSize, 1);

            sceneState.GasBuffer[instanceID].alloc(compactedSize);
            OPTIX_CHECK(optixAccelCompact(state.optixContext,
                /*stream:*/unSyncStreams[instanceID % maxStream],
                sceneState.GasHandle[instanceID],
                sceneState.GasBuffer[instanceID].d_pointer(),
                sceneState.GasBuffer[instanceID].sizeInBytes,
                &sceneState.GasHandle[instanceID]));
            CUDA_SYNC_CHECK();
        }

        CUDA_SYNC_CHECK();
        // ==================================================================
        // Clean up
        // ==================================================================
        for (int i = 0; i < maxStream; i++) {
            cudaStreamDestroy(unSyncStreams[i]);
        }
        for (int instanceID = 0; instanceID < scene->instance_list.size(); instanceID++) {
            compactedSizeBuffer[instanceID].free();
            tempBuffer[instanceID].free();
            outputBuffer[instanceID].free();
        }
    }

    OptixTraversableHandle OptixBackend::buildIAS(const std::shared_ptr<Scene> scene) {
        std::vector<std::shared_ptr<instanceNode>> todoList;
        std::vector<AffineSpace3f> parentTransformation;

        todoList.push_back(scene->root);
        AffineSpace3f baseSpace(one);
        parentTransformation.push_back(baseSpace);
        int currentSBTOffset = 0;
        while (!todoList.empty()) {
            std::shared_ptr<instanceNode> currentInstance = todoList.back();
            todoList.pop_back();
            AffineSpace3f currentTransform = parentTransformation.back() * currentInstance->transformation;
            parentTransformation.pop_back();

            for (int childrenInstance = 0; childrenInstance < currentInstance->childrenInstanceNode.size(); childrenInstance++) {
                todoList.push_back(currentInstance->childrenInstanceNode[childrenInstance]);
                parentTransformation.push_back(currentTransform);
            }

            for (int instanceID : currentInstance->instanceIDs) {
                OptixInstance newInstance = {};
                newInstance.traversableHandle = sceneState.GasHandle[instanceID];
                newInstance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
                newInstance.instanceId = 0;
                newInstance.sbtOffset = currentSBTOffset;
                newInstance.visibilityMask = 1;

                newInstance.transform[0] = currentTransform.l.vx.x;
                newInstance.transform[4] = currentTransform.l.vx.y;
                newInstance.transform[8] = currentTransform.l.vx.z; 
                
                newInstance.transform[1] = currentTransform.l.vy.x;
                newInstance.transform[5] = currentTransform.l.vy.y;
                newInstance.transform[9] = currentTransform.l.vy.z;

                newInstance.transform[2] = currentTransform.l.vz.x;
                newInstance.transform[6] = currentTransform.l.vz.y;
                newInstance.transform[10] = currentTransform.l.vz.z;

                newInstance.transform[3] = currentTransform.p.x;
                newInstance.transform[7] = currentTransform.p.y;
                newInstance.transform[11] = currentTransform.p.z;
                sceneState.instanceList.push_back(newInstance);
                
                // Compute SBT offset for next instance, *2 for two different rays
                currentSBTOffset += scene->instance_list[instanceID]->meshes.size() * 2;
            }
        }

        CUDABuffer instanceBuffer;
        instanceBuffer.alloc_and_upload(sceneState.instanceList);

        OptixBuildInput instance_input = {};

        instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances    = instanceBuffer.d_pointer();
        instance_input.instanceArray.numInstances = sceneState.instanceList.size();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( state.optixContext, &accel_options, &instance_input,
                                                1,  // num build inputs
                                                &ias_buffer_sizes ) );

        CUDABuffer tempBuffer;
        tempBuffer.alloc(ias_buffer_sizes.tempSizeInBytes);
        sceneState.asBuffer.alloc(ias_buffer_sizes.outputSizeInBytes);

        OptixTraversableHandle asHandle{ 0 };
        OPTIX_CHECK( optixAccelBuild( state.optixContext,
                                  0,  // CUDA stream
                                  &accel_options,
                                  &instance_input,
                                  1,  // num build inputs
                                  tempBuffer.d_pointer(),
                                  ias_buffer_sizes.tempSizeInBytes,
                                  sceneState.asBuffer.d_pointer(),
                                  ias_buffer_sizes.outputSizeInBytes,
                                  &asHandle,
                                  nullptr,  // emitted property list
                                  0         // num emitted properties
                                  ) );

        CUDA_SYNC_CHECK();
        instanceBuffer.free();
        tempBuffer.free();

        return asHandle;
    }

    void OptixBackend::buildSBT(const std::shared_ptr<Scene> scene) {
        // ------------------------------------------------------------------
        // build callables records
        // ------------------------------------------------------------------
        std::vector<directCallableRecord> directCallableRecords;
        for (int i = 0; i < state.directCallablePGs.size(); i++) {
            directCallableRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.directCallablePGs[i], &rec));
            directCallableRecords.push_back(rec);
        }
        sceneState.directCallabeRecordsBuffer.alloc_and_upload(directCallableRecords);
        sceneState.sbt.callablesRecordBase = sceneState.directCallabeRecordsBuffer.d_pointer();
        sceneState.sbt.callablesRecordStrideInBytes = (unsigned int)sizeof(directCallableRecord);
        sceneState.sbt.callablesRecordCount = state.directCallablePGs.size();
        // ------------------------------------------------------------------
        // build raygen records
        // ------------------------------------------------------------------
        std::vector<RaygenRecord> raygenRecords;
        for (int i = 0; i < state.raygenPGs.size(); i++) {
            RaygenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        sceneState.raygenRecordsBuffer.alloc_and_upload(raygenRecords);
        sceneState.sbt.raygenRecord = sceneState.raygenRecordsBuffer.d_pointer();

        // ------------------------------------------------------------------
        // build except records
        // ------------------------------------------------------------------
        std::vector<exceptionRecord> exceptionRecords;
        for (int i = 0; i < state.exceptionPGs.size(); i++) {
            exceptionRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.exceptionPGs[i], &rec));
            exceptionRecords.push_back(rec);
        }
        sceneState.exceptionRecordsBuffer.alloc_and_upload(exceptionRecords);
        sceneState.sbt.exceptionRecord = sceneState.exceptionRecordsBuffer.d_pointer();
        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<MissRecord> missRecords;
        for (int i = 0; i < state.missPGs.size(); i++) {
            MissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.missPGs[i], &rec));
            rec.data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        sceneState.missRecordsBuffer.alloc_and_upload(missRecords);
        sceneState.sbt.missRecordBase = sceneState.missRecordsBuffer.d_pointer();
        sceneState.sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sceneState.sbt.missRecordCount = (int)missRecords.size();

        // ------------------------------------------------------------------
        // build hitgroup records
        // ------------------------------------------------------------------
        std::vector<HitgroupRecord> hitgroupRecords;
        std::vector<std::shared_ptr<instanceNode>> todoList;
        todoList.push_back(scene->root);
        int currentIndexOffset = 0;
        sceneState.globalIndexBuffer.resize(0);

        while(!todoList.empty()) {
            std::shared_ptr<instanceNode> currentInstance = todoList.back();
            todoList.pop_back();

            for (int childrenInstance = 0; childrenInstance < currentInstance->childrenInstanceNode.size(); childrenInstance++) {
                todoList.push_back(currentInstance->childrenInstanceNode[childrenInstance]);
            }

            for (int instanceID : currentInstance->instanceIDs) {

                for (int meshID = 0; meshID < scene->instance_list[instanceID]->meshes.size(); meshID++) {
                    std::vector<int> globalID;
                    // Make sure global ID is align with the vertex buffer location. 
                    for (int i = currentIndexOffset; i < currentIndexOffset + scene->instance_list[instanceID]->meshes[meshID]->vertex.size(); i++) {
                        globalID.push_back(i);
                    }
                    for (int vexID = 0; vexID < scene->instance_list[instanceID]->meshes[meshID]->vertex.size(); vexID++){
                        sceneState.vertexBufferHost.push_back(make_float3(scene->instance_list[instanceID]->meshes[meshID]->vertex[vexID]));
                        // sceneState.normalBufferHost.push_back(make_float3(scene->instance_list[instanceID]->meshes[meshID]->normal[vexID]));
                    }
                    sceneState.globalIndexBuffer.push_back(CUDABuffer());
                    sceneState.globalIndexBuffer.back().alloc_and_upload(globalID);
                    for (int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++) {
                        auto mesh = scene->instance_list[instanceID]->meshes[meshID];

                        HitgroupRecord rec;
                        OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupPGs[rayID], &rec));
                        putSBTFloat3(mesh->mat->kd, rec.data.kd);
                        putSBTFloat3(mesh->mat->ks, rec.data.ks);

                        putSBT(mesh->mat->roughness, rec.data.roughness);
                        putSBT(mesh->mat->specular_transmission, rec.data.specular_transmission);
                        putSBT(mesh->mat->metallic, rec.data.metallic);
                        putSBT(mesh->mat->subsurface, rec.data.subsurface);
                        putSBT(mesh->mat->specular, rec.data.specular);
                        putSBT(mesh->mat->specular_tint, rec.data.specular_tint);
                        putSBT(mesh->mat->anisotropic, rec.data.anisotropic);
                        putSBT(mesh->mat->sheen, rec.data.sheen);
                        putSBT(mesh->mat->sheen_tint, rec.data.sheen_tint);
                        putSBT(mesh->mat->clearcoat, rec.data.clearcoat);
                        putSBT(mesh->mat->clearcoat_gloss, rec.data.clearcoat_gloss);
                        putSBT(mesh->mat->eta, rec.data.eta);
                        rec.data.type = (int)mesh->mat->type;

                        if (mesh->mat->normalMap.map_id != -1) {
                            rec.data.normalMap.hasTexture = true;
                            rec.data.normalMap.texture = sceneState.textureObjects[mesh->mat->normalMap.map_id];
                        }
                        else {
                            rec.data.normalMap.hasTexture = false;
                        }

                        rec.data.index = reinterpret_cast<int3*>(sceneState.indexBuffer[instanceID][meshID].d_pointer());
                        rec.data.vertex = reinterpret_cast<float3*>(sceneState.vertexBuffer[instanceID][meshID].d_pointer());
                        rec.data.normal = reinterpret_cast<float3*>(sceneState.normalBuffer[instanceID][meshID].d_pointer());

                        rec.data.texcoord = reinterpret_cast<float2*>(sceneState.texcoordBuffer[instanceID][meshID].d_pointer());
                        rec.data.globalIndex = reinterpret_cast<int*>(sceneState.globalIndexBuffer.back().d_pointer());

                        hitgroupRecords.push_back(rec);
                    }
                    currentIndexOffset += scene->instance_list[instanceID]->meshes[meshID]->vertex.size();
                }
            }
        }
        std::cout << "# Optix Renderer: total number of vertices " << currentIndexOffset << std::endl;
        if (debug) std::cout << "# Optix Renderer: total number of hit records = " << hitgroupRecords.size() << std::endl;

        sceneState.hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        sceneState.sbt.hitgroupRecordBase = sceneState.hitgroupRecordsBuffer.d_pointer();
        sceneState.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sceneState.sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
        sceneState.shVertexBuffer.alloc_and_upload(sceneState.vertexBufferHost);
    }

    void OptixBackend::createTexturesForSceneObjects(const std::shared_ptr<Scene> scene) {
        sceneState.textureObjects.clear();
        for (auto i : scene->texture_manager.texture_list)
            sceneState.textureObjects.push_back(i->toCUDA());
    }

    void OptixBackend::setupDenoiser(vec2i newSize, bool aov) {
        if (newSize == state.denoiserSize) {
            return;
        }
        state.denoiserSize = newSize;

        if (state.denoiser) {
            OPTIX_CHECK(optixDenoiserDestroy(state.denoiser));
        };
        // ------------------------------------------------------------------
        // create the denoiser:
        OptixDenoiserOptions denoiserOptions = {};
        denoiserOptions.guideAlbedo = 1;
        denoiserOptions.guideNormal = 1;
        if (aov) {
            OPTIX_CHECK(optixDenoiserCreate(state.optixContext, OPTIX_DENOISER_MODEL_KIND_AOV, &denoiserOptions, &state.denoiser));
            state.aovMode = true;
        } else {
            state.aovMode = false;
            OPTIX_CHECK(optixDenoiserCreate(state.optixContext, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &state.denoiser));
        }

        // .. then compute and allocate memory resources for the denoiser
        OptixDenoiserSizes denoiserReturnSizes;
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(state.denoiser,newSize.x,newSize.y,
                                                        &denoiserReturnSizes));

        state.denoiserScratch.resize(denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
        state.denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

        OPTIX_CHECK(optixDenoiserSetup(state.denoiser, 0,
            newSize.x, newSize.y,
            state.denoiserState.d_pointer(),
            state.denoiserState.size(),
            state.denoiserScratch.d_pointer(),
            state.denoiserScratch.size()));

        if (state.denoiserHdrIntensity.sizeInBytes != sizeof(float))
            state.denoiserHdrIntensity.alloc(sizeof(float));

        if (state.denoiserHdrAverageColor.sizeInBytes != sizeof(float) * 3)
            state.denoiserHdrAverageColor.alloc(sizeof(float) * 3);

        state.denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
        state.denoiserParams.hdrIntensity = state.denoiserHdrIntensity.d_pointer();
        state.denoiserParams.hdrAverageColor = state.denoiserHdrAverageColor.d_pointer();
        state.denoiserParams.temporalModeUsePreviousLayers = 0;
    }

    void OptixBackend::render(LaunchParams& launchParams, CUstream stream) {
        if (sceneState.launchParamsBuffer.size() == 0){
            sceneState.launchParamsBuffer.alloc(sizeof(launchParams));
        }
        sceneState.launchParamsBuffer.upload(&launchParams, 1, stream);
   
        // std::cout << shFlag << " " << shRenderFlag << std::endl;
        OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            state.pipeline, stream,
            /*! parameters and SBT */
            sceneState.launchParamsBuffer.d_pointer(),
            sceneState.launchParamsBuffer.sizeInBytes,
            &sceneState.sbt,
            /*! dimensions of the launch: */
            launchParams.frame.size.x,
            launchParams.frame.size.y,
            1
        ));
    }

    void OptixBackend::denoise(CUdeviceptr color, CUdeviceptr normal, CUdeviceptr albedo, CUdeviceptr output, float blendFactor, vec2i size, CUstream stream) {
        state.denoiserParams.blendFactor = blendFactor;
        if (state.aovMode) {
            std::cout << "denoising using the wrong AOV mode" << std::endl;
            throw std::runtime_error("normal denoising with AOV mode");
        }
        // In this mode, we disable AOV features
        state.denoiserParams.hdrAverageColor = reinterpret_cast<CUdeviceptr>(nullptr);

        std::vector< OptixDenoiserLayer > layers;
        auto createOptixImage = [size](CUdeviceptr pointer, OptixImage2D& layer) -> void {
            layer.data = pointer;
            /// Width of the image (in pixels)
            layer.width = size.x;
            /// Height of the image (in pixels)
            layer.height = size.y;
            /// Stride between subsequent rows of the image (in bytes).
            layer.rowStrideInBytes = size.x * sizeof(float4);
            /// Stride between subsequent pixels of the image (in bytes).
            /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
            layer.pixelStrideInBytes = sizeof(float4);
            /// Pixel format.
            layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        };

        // -------------------------------------------------------
        OptixImage2D colorImage;
        OptixImage2D albedoImage;
        OptixImage2D normalImage;
        createOptixImage(color, colorImage);
        createOptixImage(albedo, albedoImage);
        createOptixImage(normal, normalImage);

        OptixImage2D outputImage;
        createOptixImage(output, outputImage);

        OPTIX_CHECK(optixDenoiserComputeIntensity
        (state.denoiser,
            /*stream*/stream,
            &colorImage,
            state.denoiserHdrIntensity.d_pointer(),
            state.denoiserScratch.d_pointer(),
            state.denoiserScratch.size()));


        OptixDenoiserGuideLayer denoiserGuideLayer = {};
        denoiserGuideLayer.albedo = albedoImage;
        denoiserGuideLayer.normal = normalImage;

        OptixDenoiserLayer denoiserLayer = {};
        denoiserLayer.input = colorImage;
        denoiserLayer.output = outputImage;
        layers.push_back(denoiserLayer);

        OPTIX_CHECK(optixDenoiserInvoke(state.denoiser,
            /*stream*/stream,
            &state.denoiserParams,
            state.denoiserState.d_pointer(),
            state.denoiserState.size(),
            &denoiserGuideLayer,
            layers.data(), static_cast<unsigned int>(layers.size()),
            /*inputOffsetX*/0,
            /*inputOffsetY*/0,
            state.denoiserScratch.d_pointer(),
            state.denoiserScratch.size()));
    }

    void OptixBackend::AOVDenoise(std::vector<std::pair<CUdeviceptr, CUdeviceptr>>& inputOuputPair, CUdeviceptr normal, CUdeviceptr albedo, 
        float blendFactor, vec2i size, CUstream stream) {
        std::cout << "AOV mode should be disabled" << std::endl;
        throw std::runtime_error("AOV mode should be disabled");
        return;
        state.denoiserParams.blendFactor = blendFactor;
        if (!state.aovMode) {
            std::cout << "denoising using the wrong normal mode" << std::endl;
            throw std::runtime_error("AOV denoising with normal mode");
        }

        auto createOptixImage = [size](CUdeviceptr pointer, OptixImage2D& layer) -> void {
            layer.data = pointer;
            /// Width of the image (in pixels)
            layer.width = size.x;
            /// Height of the image (in pixels)
            layer.height = size.y;
            /// Stride between subsequent rows of the image (in bytes).
            layer.rowStrideInBytes = size.x * sizeof(float4);
            /// Stride between subsequent pixels of the image (in bytes).
            /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
            layer.pixelStrideInBytes = sizeof(float4);
            /// Pixel format.
            layer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        };

        std::vector< OptixDenoiserLayer > layers;
        // -------------------------------------------------------
        OptixImage2D albedoImage;
        OptixImage2D normalImage;
        createOptixImage(albedo, albedoImage);
        createOptixImage(normal, normalImage);

        for (const auto& [inputPointer, outputPointer] : inputOuputPair) {
            OptixImage2D inputImage;
            OptixImage2D outputImage;
            createOptixImage(inputPointer, inputImage);
            createOptixImage(outputPointer, outputImage);

            OptixDenoiserLayer denoiserLayer = {};
            denoiserLayer.input = inputImage;
            denoiserLayer.output = outputImage;
            layers.push_back(denoiserLayer);
        }

        OPTIX_CHECK(optixDenoiserComputeIntensity
        (state.denoiser,
            /*stream*/stream,
            &layers[0].input,
            state.denoiserHdrIntensity.d_pointer(),
            state.denoiserScratch.d_pointer(),
            state.denoiserScratch.size()));

        OPTIX_CHECK(optixDenoiserComputeAverageColor(
            state.denoiser,
            stream, // CUDA stream
            &layers[0].input,
            state.denoiserHdrAverageColor.d_pointer(),
            state.denoiserScratch.d_pointer(),
            state.denoiserScratch.size()));

        OptixDenoiserGuideLayer denoiserGuideLayer = {};
        denoiserGuideLayer.albedo = albedoImage;
        denoiserGuideLayer.normal = normalImage;


        OPTIX_CHECK(optixDenoiserInvoke(state.denoiser,
            /*stream*/stream,
            &state.denoiserParams,
            state.denoiserState.d_pointer(),
            state.denoiserState.size(),
            &denoiserGuideLayer,
            layers.data(), static_cast<unsigned int>(layers.size()),
            /*inputOffsetX*/0,
            /*inputOffsetY*/0,
            state.denoiserScratch.d_pointer(),
            state.denoiserScratch.size()));
    }

    int OptixBackend::getNumVertex(){
        return sceneState.vertexBufferHost.size();
    }

    void OptixBackend::shComputeTMatrix(LaunchParams& launchParams){
        launchParams.sh.vertexLocations = reinterpret_cast<float3*>(sceneState.shVertexBuffer.d_pointer());

        if (sceneState.launchParamsBuffer.size() == 0){
            sceneState.launchParamsBuffer.alloc(sizeof(launchParams));
        }

        sceneState.launchParamsBuffer.upload(&launchParams, 1);
        OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            state.pipeline, 0,
            /*! parameters and SBT */
            sceneState.launchParamsBuffer.d_pointer(),
            sceneState.launchParamsBuffer.sizeInBytes,
            &sceneState.sbt,
            /*! dimensions of the launch: */
            sceneState.vertexBufferHost.size(),
            1,
            1
        ));
    }    
}