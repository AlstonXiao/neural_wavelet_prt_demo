#include "RenderEngine.h"
#include <optix_function_table_definition.h>
#include <DeviceKernelPTX.h>
#include "npy.h"

namespace nert_renderer {
    inline vec2i makeVec2i(int2 in) {
        return vec2i(in.x, in.y);
    }
    inline float3 make_float3(const vec3f& in) {
        return ::make_float3(in.x, in.y, in.z);
    }

    RenderEngine::RenderEngine(const std::shared_ptr<Scene> scene, renderEngineMode mode, bool debug_i)
        : scene(scene), engine_mode(mode)
    {
        debug = debug_i;
        BackendRender = new OptixBackend(debug, engine_mode);
        auto embedded_ptx_code = devicePrograms_ptx_text();
        auto roughPlastic = roughPlastic_ptx_text();
        auto disneyBSDF = disneyBSDF_ptx_text();
        auto fovCamera = camera_ptx_text();
        auto cookTorrance = cookTorrance_ptx_text();
        auto simple_ggx = simple_ggx_ptx_text();

        auto constant_color = constantColorMap_ptx_text();
        auto latlong_map = latlongMap_ptx_text();
        auto cubemap = cubeMap_ptx_text();
        auto sgmap = sgMap_ptx_text();

        auto camera_tmatrix = cameraShTmatrix_ptx_text();
        auto camera_shrender = cameraShRender_ptx_text();

        std::map<const std::string, std::string> moduleNameToCodes;
        moduleNameToCodes["camera"] = fovCamera;
        moduleNameToCodes["camera_tmatrix"] = camera_tmatrix;
        moduleNameToCodes["camera_shrender"] = camera_shrender;

        moduleNameToCodes["shader"] = embedded_ptx_code;
        moduleNameToCodes["roughp"] = roughPlastic;
        moduleNameToCodes["disney"] = disneyBSDF;
        moduleNameToCodes["cookTorrance"] = cookTorrance;
        moduleNameToCodes["simple_ggx"] = simple_ggx;

        moduleNameToCodes["constant_color"] = constant_color;
        moduleNameToCodes["latlong_map"] = latlong_map;
        moduleNameToCodes["cubemap"] = cubemap;
        moduleNameToCodes["sgmap"] = sgmap;

        BackendRender->createModule(moduleNameToCodes);
        std::vector<std::string> envmapFunctions;
        envmapFunctions.push_back("__direct_callable__evaluate");
        envmapFunctions.push_back("__direct_callable__pdf");
        envmapFunctions.push_back("__direct_callable__sampleEnvironmapLight");
        BackendRender->createDirectCallablePrograms("constant_color", envmapFunctions);
        BackendRender->createDirectCallablePrograms("latlong_map", envmapFunctions);
        BackendRender->createDirectCallablePrograms("cubemap", envmapFunctions);
        BackendRender->createDirectCallablePrograms("sgmap", envmapFunctions);

        std::vector<std::string> roughFunctions;
        roughFunctions.push_back("__direct_callable__evaluate");
        roughFunctions.push_back("__direct_callable__pdf");
        roughFunctions.push_back("__direct_callable__sample");
        roughFunctions.push_back("__direct_callable__packSBT");
        BackendRender->createDirectCallablePrograms("roughp", roughFunctions);

        std::vector<std::string> disneyFunctions;
        disneyFunctions.push_back("__direct_callable__evaluate");
        disneyFunctions.push_back("__direct_callable__pdf");
        disneyFunctions.push_back("__direct_callable__sample");
        disneyFunctions.push_back("__direct_callable__packSBT");
        BackendRender->createDirectCallablePrograms("disney", disneyFunctions);

        std::vector<std::string> cookFunctions;
        cookFunctions.push_back("__direct_callable__evaluate");
        cookFunctions.push_back("__direct_callable__pdf");
        cookFunctions.push_back("__direct_callable__sample");
        cookFunctions.push_back("__direct_callable__packSBT");
        BackendRender->createDirectCallablePrograms("cookTorrance", cookFunctions);

        std::vector<std::string> simple_ggxFunctions;
        simple_ggxFunctions.push_back("__direct_callable__evaluate");
        simple_ggxFunctions.push_back("__direct_callable__pdf");
        simple_ggxFunctions.push_back("__direct_callable__sample");
        simple_ggxFunctions.push_back("__direct_callable__packSBT");

        BackendRender->createDirectCallablePrograms("simple_ggx", simple_ggxFunctions);

        if (engine_mode == renderEngineMode::shRender)
            BackendRender->createRaygenMissHitPrograms("camera_shrender", "shader", "shader", "shader");
        else if (engine_mode == renderEngineMode::shTmatrix)
            BackendRender->createRaygenMissHitPrograms("camera_tmatrix", "shader", "shader", "shader");
        else if (engine_mode == renderEngineMode::normal)
            BackendRender->createRaygenMissHitPrograms("camera", "shader", "shader", "shader");
        BackendRender->createPipeline();
    }

    void RenderEngine::buildScene() {
        BackendRender->buildGAS(scene);
        launchParams.traversable = BackendRender -> buildIAS(scene);
        BackendRender->createTexturesForSceneObjects(scene);
        BackendRender->buildSBT(scene);
        setCamera(scene->initialCamera);
        launchParams.lowerBound = make_float3(scene->bounds.lower);
        launchParams.boxSize = make_float3(scene->bounds.upper - scene->bounds.lower);

        std::cout << GDT_TERMINAL_GREEN;
        std::cout << "# Optix Renderer: Optix 7 Scene fully set up" << std::endl;
        std::cout << GDT_TERMINAL_DEFAULT;

    }

    void RenderEngine::buildLights() {
        if (scene->EnvMap.size() == 0) return;

        envMap_host_Buffer.resize(scene->EnvMap.size());
        envMap_device_Buffer.resize(scene->EnvMap.size());
        
        EnvmapSetup = true;

        for (int i = 0; i < scene->EnvMap.size(); i++)
        {
            envMap_device_Buffer[i].sampling_type = 2;
            envMap_device_Buffer[i].sampling_data.cubeMap.face_dist.size = 6;

            envMap_host_Buffer[i].face_dist.cdf.alloc_and_upload(scene->EnvMap[i]->face_dist.cdf);
            envMap_host_Buffer[i].face_dist.pmf.alloc_and_upload(scene->EnvMap[i]->face_dist.pmf);

            envMap_device_Buffer[i].sampling_data.cubeMap.face_dist.cdf = (double*)envMap_host_Buffer[i].face_dist.cdf.d_pointer();
            envMap_device_Buffer[i].sampling_data.cubeMap.face_dist.pmf = (double*)envMap_host_Buffer[i].face_dist.pmf.d_pointer();

            for (int j = 0; j < 6; j++) {
                envMap_device_Buffer[i].sampling_data.cubeMap.faces[j] = scene->EnvMap[i]->faces[j]->toCUDA_Point();

                envMap_host_Buffer[i].sampling_dist[j].cdf_marginals.alloc_and_upload(scene->EnvMap[i]->sampling_dist[j].cdf_marginals);
                envMap_host_Buffer[i].sampling_dist[j].cdf_rows.alloc_and_upload(scene->EnvMap[i]->sampling_dist[j].cdf_rows);
                envMap_host_Buffer[i].sampling_dist[j].pdf_marginals.alloc_and_upload(scene->EnvMap[i]->sampling_dist[j].pdf_marginals);
                envMap_host_Buffer[i].sampling_dist[j].pdf_rows.alloc_and_upload(scene->EnvMap[i]->sampling_dist[j].pdf_rows);

                envMap_device_Buffer[i].sampling_data.cubeMap.sampling_dist[j].cdf_marginals = (double*)envMap_host_Buffer[i].sampling_dist[j].cdf_marginals.d_pointer();
                envMap_device_Buffer[i].sampling_data.cubeMap.sampling_dist[j].cdf_rows = (double*)envMap_host_Buffer[i].sampling_dist[j].cdf_rows.d_pointer();
                envMap_device_Buffer[i].sampling_data.cubeMap.sampling_dist[j].pdf_marginals = (double*)envMap_host_Buffer[i].sampling_dist[j].pdf_marginals.d_pointer();
                envMap_device_Buffer[i].sampling_data.cubeMap.sampling_dist[j].pdf_rows = (double*)envMap_host_Buffer[i].sampling_dist[j].pdf_rows.d_pointer();

                envMap_device_Buffer[i].sampling_data.cubeMap.sampling_dist[j].total_values = scene->EnvMap[i]->sampling_dist[j].total_values;
                envMap_device_Buffer[i].sampling_data.cubeMap.sampling_dist[j].width = scene->EnvMap[i]->sampling_dist[j].width;
                envMap_device_Buffer[i].sampling_data.cubeMap.sampling_dist[j].height = scene->EnvMap[i]->sampling_dist[j].height;
            }
            envMap_device_Buffer[i].lighting_type = 2;
            envMap_device_Buffer[i].lighting_data = envMap_device_Buffer[i].sampling_data;

            if (scene->EnvMap[i]->latlongMap != NULL) {
                envMap_device_Buffer[i].view_type = 1;

                envMap_device_Buffer[i].view_data.latlong = scene->EnvMap[i]->latlongMap->toCUDA();
            }
            else {
                envMap_device_Buffer[i].view_type = 2;
                envMap_device_Buffer[i].view_data = envMap_device_Buffer[i].sampling_data;
            }

            if (scene->EnvMap[i]->topWaveletCoefficients.size() == 0 || scene->EnvMap[i]->topWaveletStrength.size() == 0) 
                EnvmapSetup = false;
        }

        // We only allows wavelet rendering if all envmaps in a batch have the wavelet coeff. 
        if (EnvmapSetup) {
            waveletCoefBuffer.resize(scene->EnvMap.size());
            waveletStrengthBuffer.resize(scene->EnvMap.size());

            for (int i = 0; i < scene->EnvMap.size(); i++) {
                waveletCoefBuffer[i].alloc_and_upload(scene->EnvMap[i]->topWaveletCoefficients);
                waveletStrengthBuffer[i].alloc_and_upload(scene->EnvMap[i]->topWaveletStrength);
            }
        }

        if (debug) {
            std::cout << GDT_TERMINAL_GREEN;
            std::cout << "# Optix Renderer: " << envMap_host_Buffer.size() << " Environment map " << (waveletCoefBuffer.size() > 0 ? "with wavelets" : "") << " fully set up" << std::endl;

            std::cout << GDT_TERMINAL_DEFAULT;
        }

        /*
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<float> data;

        npy::LoadArrayFromNumpy("C:/Users/Alsto/OneDrive - UC San Diego/Sony/Testing Python Script/location.npy", shape, fortran_order, data);
        SG_position.alloc_and_upload(data);

        std::vector<float> data2;

        npy::LoadArrayFromNumpy("C:/Users/Alsto/OneDrive - UC San Diego/Sony/Testing Python Script/roughness.npy", shape, fortran_order, data2);
        SG_roughness.alloc_and_upload(data2);

        std::vector<float> data3;
        npy::LoadArrayFromNumpy("C:/Users/Alsto/OneDrive - UC San Diego/Sony/Testing Python Script/color.npy", shape, fortran_order, data3);

        std::cout << data[0] << " " << data[1] << " " << data[2] << std::endl;
        std::cout << data2[0] << std::endl;
        std::cout << data3[0] << " " << data3[1] << " " << data3[2] << std::endl;
        std::cout << shape[0] << " " << shape[1] << std::endl;
        SG_color.alloc_and_upload(data3);
        

        launchParams.sg_color = (float3*)SG_color.d_pointer();
        launchParams.sg_location = (float3*)SG_position.d_pointer();
        launchParams.sg_roughness = (float*)SG_roughness.d_pointer();
        launchParams.sg_count = 128;

        */
    }

    void RenderEngine::destroyLights() {
        for (int i = 0; i < waveletStrengthBuffer.size(); i++) {
            waveletCoefBuffer[i].free();
            waveletStrengthBuffer[i].free();
        }
        waveletCoefBuffer.resize(0);
        waveletStrengthBuffer.resize(0);

        for (int i = 0; i < envMap_device_Buffer.size(); i++) {
            if (envMap_device_Buffer[i].view_type == 1) {
                cudaDestroyTextureObject(envMap_device_Buffer[i].view_data.latlong);
            }
            for (int j = 0; j < 6; j++) {
                cudaDestroyTextureObject(envMap_device_Buffer[i].sampling_data.cubeMap.faces[j]);
            }
        }
        for (int i = 0; i < envMap_host_Buffer.size(); i++) {
            envMap_host_Buffer[i].face_dist.cdf.free();
            envMap_host_Buffer[i].face_dist.pmf.free();
            for (int j = 0; j < 6; j++) {
                envMap_host_Buffer[i].sampling_dist[j].cdf_marginals.free();
                envMap_host_Buffer[i].sampling_dist[j].cdf_rows.free();
                envMap_host_Buffer[i].sampling_dist[j].pdf_marginals.free();
                envMap_host_Buffer[i].sampling_dist[j].pdf_rows.free();
            }
        }
        envMap_device_Buffer.resize(0);
        envMap_host_Buffer.resize(0);
        EnvmapSetup = false;
    }

    void RenderEngine::setCamera(const Camera &camera)
    {
        lastSetCamera = camera;
        vec3f position = camera.from;
        vec3f direction = normalize(camera.at - camera.from);
        const float cosFovy = 0.9094f;
        const float aspect
            = float(launchParams.frame.size.x)
            / float(launchParams.frame.size.y);
        vec3f horizontal = aspect * normalize(cross(direction, camera.up));
        vec3f vertical = normalize(cross(horizontal,direction));

        launchParams.camera.cx = float(launchParams.frame.size.x) / 2.f;
        launchParams.camera.cy = float(launchParams.frame.size.y) / 2.f;
        launchParams.camera.fx = 0.5 * launchParams.frame.size.x / tan(0.5 * 0.8575560548920328f);
        launchParams.camera.fy = 0.5 * launchParams.frame.size.y / tan(0.5 * 0.8575560548920328f);

        launchParams.camera.position = make_float3(position);
        launchParams.camera.direction = make_float3(direction);
        launchParams.camera.horizontal = make_float3(horizontal);
        launchParams.camera.vertical = make_float3(vertical);
    }
  
    /*! resize frame buffer to given resolution */
    void RenderEngine::resize(const vec2i &newSize)
    {
        BackendRender->setupDenoiser(newSize, aovMode);
        // ------------------------------------------------------------------
        
        // resize our cuda frame buffer
        fullframeCUDABuffer[0].resize(newSize);
        fullframeCUDABuffer[1].resize(newSize);

        launchParams.frame.size = make_int2(newSize.x, newSize.y);

        // and re-set the camera, since aspect may have changed
        setCamera(lastSetCamera);
    }
  
    /*! download the rendered color buffer */
    void RenderEngine::downloadPixels(uint32_t h_pixels[], int frameID, CUstream stream)
    {
        size_t totalSize = static_cast<size_t>(launchParams.frame.size.x) *
            static_cast<size_t>(launchParams.frame.size.y);
        cudaMemcpyAsync(h_pixels, (const void*)fullframeCUDABuffer[frameID].fbFinalColor.d_pointer(), totalSize * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    }

    void RenderEngine::downloadframe(std::shared_ptr<fullframe> frame, int frameID) {
        size_t totalSize = static_cast<size_t>(launchParams.frame.size.x) *
            static_cast<size_t>(launchParams.frame.size.y);
        fullframeCUDABuffer[frameID].fbDenoised.download(frame->FinalColor.data(), totalSize);
        fullframeCUDABuffer[frameID].fbDirectDenoised.download(frame->DirectColor.data(), totalSize);
        fullframeCUDABuffer[frameID].fbIndirectDenoised.download(frame->IndirectColor.data(), totalSize);

        fullframeCUDABuffer[frameID].fbworldNormal.download(frame->worldNormalBuffer.data(), totalSize);
        fullframeCUDABuffer[frameID].fbCameraNormal.download(frame->cameraNormalBuffer.data(), totalSize);
        fullframeCUDABuffer[frameID].fbAlbedo.download(frame->albedoBuffer.data(), totalSize);

        fullframeCUDABuffer[frameID].fbFirstHitBary.download(frame->FirstHitBary.data(), totalSize);
        fullframeCUDABuffer[frameID].fbFirstHitVertexID.download(frame->FirstHitVertexID.data(), totalSize);

        fullframeCUDABuffer[frameID].fbFirstHitUV.download(frame->FirstHitUVCoord.data(), totalSize);
        fullframeCUDABuffer[frameID].fbFirstHitPos.download(frame->firstHitPos.data(), totalSize);
        fullframeCUDABuffer[frameID].fbFirstHitReflecDir.download(frame->firstHitReflecDir.data(), totalSize);

        fullframeCUDABuffer[frameID].fbFirstHitKd.download(frame->firstHitKd.data(), totalSize);
        fullframeCUDABuffer[frameID].fbFirstHitKs.download(frame->firstHitKs.data(), totalSize);

        fullframeCUDABuffer[frameID].fbSpecColor.download(frame->Speccolor.data(), totalSize);
        fullframeCUDABuffer[frameID].fbDiffColor.download(frame->Diffcolor.data(), totalSize);

        fullframeCUDABuffer[frameID].fbDepth.download(frame->depth.data(), totalSize);
        fullframeCUDABuffer[frameID].fbRoughness.download(frame->roughness.data(), totalSize);

        fullframeCUDABuffer[frameID].fbPredictedIndirectDenoised.download(frame->predictedIndirect.data(), totalSize);
        fullframeCUDABuffer[frameID].fbPredictedIndirectWithDirectDenoised.download(frame->predictedIndirectWithDirect.data(), totalSize);

        fullframeCUDABuffer[frameID].fbPerfectReflectionFlag.download(frame->perfectReflectionFlag, totalSize);
        fullframeCUDABuffer[frameID].fbEmissiveFlag.download(frame->emissiveFlag, totalSize);
        fullframeCUDABuffer[frameID].fbEnvironmentMapFlag.download(frame->environmentMapFlag, totalSize);
    }
    
    void RenderEngine::render(int frameID, int accumulateFrameID, uint32_t envMapID, renderConfig config, CUstream stream)
    {
        // sanity check: make sure we launch only after first resize is
        // already done:
        if (launchParams.frame.size.x == 0) return;

        fullframeCUDABuffer[frameID].assignBufferToLaunch(launchParams);
        
        // We never do direct accumulation without the wavelet mode.
        launchParams.frame.frameID = accumulateFrameID;

        launchParams.samplingParameter.samplesPerPixel = config.samplesPerPixel;
        launchParams.samplingParameter.indirectSamplesPerDirect = config.indirectSamplesPerDirect;
        launchParams.samplingParameter.samplesPerLight = config.directSamplesPerLight;

        if (config.renderingMode == finalColorBufferContent::direct || config.waveletMode)
            launchParams.samplingParameter.maximumBounce = 1;
        else
            launchParams.samplingParameter.maximumBounce = 12;

        if (envMapID > envMap_device_Buffer.size())
            throw std::overflow_error("Envmap ID for rendering is larger than cached envmap");
        launchParams.env = envMap_device_Buffer[envMapID];

        float blendingFactor = accumulateFrameID > 0 ? 1.f / (accumulateFrameID + 1) : 0;
        BackendRender->render(launchParams, stream);

        // Never used
        if (aovMode) {
            std::vector<std::pair<CUdeviceptr, CUdeviceptr>> inputOuputPair;
            inputOuputPair.push_back({ fullframeCUDABuffer[frameID].fbFirstHitDirect.d_pointer(), fullframeCUDABuffer[frameID].fbDirectDenoised.d_pointer() });
            inputOuputPair.push_back({ fullframeCUDABuffer[frameID].fbFirstHitIndirect.d_pointer(), fullframeCUDABuffer[frameID].fbIndirectDenoised.d_pointer() });
            inputOuputPair.push_back({ fullframeCUDABuffer[frameID].fbColor.d_pointer(), fullframeCUDABuffer[frameID].fbDenoised.d_pointer() });
            BackendRender->AOVDenoise(inputOuputPair, fullframeCUDABuffer[frameID].fbCameraNormal.d_pointer(), fullframeCUDABuffer[frameID].fbAlbedo.d_pointer(), blendingFactor, makeVec2i(launchParams.frame.size), stream);
        }

        // Only denoise what we need
        if (config.realtimeMode) {
            if (config.waveletMode) 
                denoiseOrMoveData(frameID, config.directDenoiser, fullframeCUDABuffer[frameID].fbFirstHitDirect.d_pointer(),
                    fullframeCUDABuffer[frameID].fbDirectDenoised.d_pointer(), 0);
            else if (config.renderingMode == finalColorBufferContent::full) 
                denoiseOrMoveData(frameID, config.finalDenoiser, fullframeCUDABuffer[frameID].fbColor.d_pointer(),
                    fullframeCUDABuffer[frameID].fbDenoised.d_pointer(), blendingFactor, stream);
            else if (config.renderingMode == finalColorBufferContent::direct)
                denoiseOrMoveData(frameID, config.directDenoiser, fullframeCUDABuffer[frameID].fbFirstHitDirect.d_pointer(),
                    fullframeCUDABuffer[frameID].fbDirectDenoised.d_pointer(), 0);
            else if (config.renderingMode == finalColorBufferContent::indirect)
                denoiseOrMoveData(frameID, config.indirectDenoiser, fullframeCUDABuffer[frameID].fbFirstHitIndirect.d_pointer(),
                    fullframeCUDABuffer[frameID].fbIndirectDenoised.d_pointer(), 0, stream);
        }
        // In offline mode, we denoise everything
        else {
            denoiseOrMoveData(frameID, config.directDenoiser, fullframeCUDABuffer[frameID].fbFirstHitDirect.d_pointer(),
                fullframeCUDABuffer[frameID].fbDirectDenoised.d_pointer(), 0);
            denoiseOrMoveData(frameID, config.finalDenoiser, fullframeCUDABuffer[frameID].fbColor.d_pointer(),
                fullframeCUDABuffer[frameID].fbDenoised.d_pointer(), blendingFactor, stream);
            denoiseOrMoveData(frameID, config.indirectDenoiser, fullframeCUDABuffer[frameID].fbFirstHitIndirect.d_pointer(),
                fullframeCUDABuffer[frameID].fbIndirectDenoised.d_pointer(), 0, stream);
        }
    }

    void RenderEngine::denoiseOrMoveData(int frameID, bool denoiserOn, CUdeviceptr inputBuffer, CUdeviceptr outputBuffer, int blendFactor, CUstream stream) {
        if (denoiserOn) {
            BackendRender->denoise(inputBuffer,
                fullframeCUDABuffer[frameID].fbCameraNormal.d_pointer(),
                fullframeCUDABuffer[frameID].fbAlbedo.d_pointer(),
                outputBuffer,
                blendFactor, makeVec2i(launchParams.frame.size), stream);
        }
        else {
            cudaMemcpyAsync((void*)outputBuffer, (void*)inputBuffer,
                launchParams.frame.size.x * launchParams.frame.size.y * sizeof(float4),
                cudaMemcpyDeviceToDevice, stream);
        }
    }

    void RenderEngine::denoise(CUdeviceptr color, CUdeviceptr normal, CUdeviceptr albedo, CUdeviceptr output, vec2i size) {
        BackendRender->setupDenoiser(size);
        BackendRender->denoise(color, normal, albedo, output, 0, size);
    }

    void RenderEngine::shComputeTMatrix(int shTerm, std::filesystem::path exportFileName) {
        launchParams.samplingParameter.samplesPerPixel = 4096;
        launchParams.sh.shTerms = shTerm;

        if (shTerm != 8) {
            std::cout << GDT_TERMINAL_RED;
            std::cout << "# For T matrix computation, SH other than 8 has not been fully tested! #" << std::endl;
            std::cout << GDT_TERMINAL_DEFAULT;
        }
        int total_sh_coeffs = (shTerm + 1) * (shTerm + 1);
        int num_vertices = BackendRender->getNumVertex();

        std::vector<float> T_matrix_host(num_vertices * total_sh_coeffs * total_sh_coeffs * 3, 0.f);
        std::vector<float> T_matrix_host_partial(num_vertices * total_sh_coeffs * 3, 0.f);
        T_matrix.alloc(num_vertices * total_sh_coeffs * 3  * sizeof(float));
        launchParams.sh.T_matrix = reinterpret_cast<float3*>(T_matrix.d_pointer());

        for (int i = 0; i < total_sh_coeffs; i++) {
            launchParams.env = envMap_device_Buffer[i];
            BackendRender->shComputeTMatrix(launchParams);
            T_matrix.download(T_matrix_host_partial.data(), num_vertices * total_sh_coeffs * 3);
            std::cout << "Rendering coefficients " << i << " done!" << std::endl;

            for (int j = 0; j < num_vertices; j++) { 
                for (int k = 0; k < total_sh_coeffs; k++) {
                    T_matrix_host[(j * total_sh_coeffs * total_sh_coeffs + i + k * total_sh_coeffs) * 3] = T_matrix_host_partial[(j * total_sh_coeffs + k) * 3];
                    T_matrix_host[(j * total_sh_coeffs * total_sh_coeffs + i + k * total_sh_coeffs) * 3 + 1] = T_matrix_host_partial[(j * total_sh_coeffs + k) * 3 + 1];
                    T_matrix_host[(j * total_sh_coeffs * total_sh_coeffs + i + k * total_sh_coeffs) * 3 + 2] = T_matrix_host_partial[(j * total_sh_coeffs + k) * 3 + 2];
                }
            }
        }

        std::cout <<"Rendering done, writing to file: " << exportFileName << std::endl;

        std::array<long unsigned, 3> leshape11{ {num_vertices, total_sh_coeffs * total_sh_coeffs, 3} };
        npy::SaveArrayAsNumpy(exportFileName.string(), false, leshape11.size(), leshape11.data(), T_matrix_host.data());
    }

    void RenderEngine::shLoadTMatrix(std::string exportFileName) {
        std::vector<unsigned long> shape;
        bool fortran_order;
        std::vector<float> data;

        npy::LoadArrayFromNumpy(exportFileName, shape, fortran_order, data);
        T_matrix.alloc_and_upload(data);
        launchParams.sh.T_matrix = reinterpret_cast<float3*>(T_matrix.d_pointer());
        if (debug) std::cout << "T matrix Loaded, with size [" <<shape[0] << ", " <<shape[1]<<", " <<shape[2] <<"]!" << std::endl;
        loaded_sh_degree = sqrt(sqrt(shape[1])) - 1;
    }

    void RenderEngine::shRender(int frameID, int accumulateFrameID, uint32_t envMapID, renderConfig config, CUstream stream){

        if (launchParams.frame.size.x == 0) return;
        if (debug) std::cout << "Using SH of degree" << loaded_sh_degree << std::endl;

        fullframeCUDABuffer[frameID].assignBufferToLaunch(launchParams);

        launchParams.frame.frameID = accumulateFrameID;
        launchParams.sh.shTerms = loaded_sh_degree;
        
        // std::cout << lightData.size() << std::endl;
        launchParams.samplingParameter.samplesPerPixel = config.samplesPerPixel;
        
        if (!envMap_vector.d_pointer())
            envMap_vector.resize(sizeof(float) * scene->EnvMap[envMapID]->shCoefficients.size());
        envMap_vector.upload(scene->EnvMap[envMapID]->shCoefficients.data(), scene->EnvMap[envMapID]->shCoefficients.size());
        launchParams.sh.light_vector = reinterpret_cast<float3*>(envMap_vector.d_pointer());

        cudaDeviceSynchronize();
        // std::cout << launchParams.has_env << std::endl;

        BackendRender->render(launchParams, stream);
        cudaDeviceSynchronize();

        denoiseOrMoveData(0, config.indirectDenoiser, fullframeCUDABuffer[frameID].fbColor.d_pointer(),
            fullframeCUDABuffer[frameID].fbDenoised.d_pointer(), 0, stream);
    }


} // ::nert_renderer
