#pragma once

#include "OptixBackend.h"
#include <filesystem>
/*! \namespace nert_renderer */
namespace nert_renderer {

    struct TableDist1D_host {
        CUDABuffer pmf;
        CUDABuffer cdf;
    };

    struct TableDist2D_host {
        CUDABuffer cdf_rows;
        CUDABuffer pdf_rows;

        CUDABuffer cdf_marginals;
        CUDABuffer pdf_marginals;

    };

    struct envLight_host {
        TableDist1D_host face_dist;
        TableDist2D_host sampling_dist[6];
    };

    enum class finalColorBufferContent { full, direct, indirect, };

    struct renderConfig {
        int samplesPerPixel = 1024;
        int directSamplesPerLight = 1;
        int indirectSamplesPerDirect = 1;
        
        finalColorBufferContent renderingMode = finalColorBufferContent::full;
        bool waveletMode = false;

        bool finalDenoiser = true;
        bool directDenoiser = true;
        bool indirectDenoiser = true;

        bool realtimeMode = false; // This will minimize all useless memcopy and denoiser call
    };

    class RenderEngine
    {
    // ------------------------------------------------------------------
    // publicly accessible interface
    // ------------------------------------------------------------------
    public:
    /*! constructor - performs all setup, including initializing
        optix, creates module, pipeline, programs, SBT, etc. */
        RenderEngine(const std::shared_ptr<Scene> scene, renderEngineMode mode, bool debug_i = false);
        ~RenderEngine() {
            destroyLights();
        }

        void buildScene();
        void buildLights();
        void destroyLights();

        /*! render one frame */
        void render(int frameID, int accumulateFrameID, uint32_t envMapID, renderConfig config, CUstream stream = nullptr);

        void denoiseOrMoveData(int frameID, bool denoiserOn, CUdeviceptr inputBuffer, CUdeviceptr outputBuffer, int blendFactor = 0, CUstream stream = nullptr);

        /*! resize frame buffer to given resolution */
        void resize(const vec2i &newSize);

        /*! download the rendered color buffer */
        void downloadPixels(uint32_t h_pixels[], int frameID, CUstream stream = nullptr);

        /*! download the rendered full frame buffer */
        void downloadframe(std::shared_ptr<fullframe> frame, int frameID);

        /*! set camera to render with */
        void setCamera(const Camera &camera);
        
        void initTCNN(const std::string modelPath);
        bool TCNNAvaliable();

        void TCNNPass_p1(int frameID, CUstream stream = nullptr);
        void TCNNPass_p2(int envMapID, int frameID, renderConfig config);
        void TCNNSync();

        void computeFinalPixelColors(int frameID = 1, CUstream stream = nullptr, renderConfig config = {});

        void denoise(CUdeviceptr color, CUdeviceptr normal, CUdeviceptr albedo, CUdeviceptr output, vec2i size);
        
        void shComputeTMatrix(int shTerm, std::filesystem::path exportFileName);
        void shLoadTMatrix(std::string exportFileName);
        void shRender(int frameID, int accumulateFrameID, uint32_t envMapID, renderConfig config, CUstream stream = nullptr);

        renderEngineMode engine_mode = renderEngineMode::normal;
        bool EnvmapSetup = false;
        bool TCNNSetup = false;
        // AOV mode should be disabled due to worse performance
        bool aovMode = false;
        bool debug;
        LaunchParams launchParams;

    protected:
        /* @{ complex light buffer*/
        std::vector<envLight_host> envMap_host_Buffer;
        std::vector<envLight> envMap_device_Buffer;

        std::vector<CUDABuffer> waveletCoefBuffer;
        std::vector<CUDABuffer> waveletStrengthBuffer;
        /*@}*/

        /* T Matrix, not used*/
        CUDABuffer T_matrix;
        unsigned int loaded_sh_degree;
        CUDABuffer envMap_vector;

        /* SG Env map, not used*/
        CUDABuffer SG_position;
        CUDABuffer SG_color;
        CUDABuffer SG_roughness;

        /* Full frame in cuda*/
        fullFrameCUDA fullframeCUDABuffer[2];
    
        /*! the camera we are to render with. */
        Camera lastSetCamera;
    
        /*! the scene we are going to trace rays against */
        const std::shared_ptr<Scene> scene;
        
        /* Backend */
        OptixBackend* BackendRender;
        
    };

} // ::nert_renderer
