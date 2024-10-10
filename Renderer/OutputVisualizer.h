// our helper library for window handling
#include "glfWindow/GLFWindow.h"
#include <GL/gl.h>

// Ours
#include "Scene.h"
#include "RenderEngine.h"

namespace nert_renderer {
    // Frame 1 is the default frame! Not 0
    enum class pipeStatus { idle, frameID0, frameID1 };

    struct SampleWindow : public GLFCameraWindow
    {
        SampleWindow(const std::string& title,
            std::shared_ptr<RenderEngine> engine,
            const std::shared_ptr<Scene> scene,
            const gdt::vec2i size,
            const float worldScale,
            renderEngineMode mode
        );

        virtual void render(float deltaTime) override;

        virtual void draw() override;

        virtual void resize(const gdt::vec2i& newSize);

        virtual void key(int key, int mods, int action) override;

        void print_mode();

        std::shared_ptr<RenderEngine> m_engine;
        renderEngineMode      m_mode;
        gdt::vec2i            fbSize;
        GLuint                fbTexture{ 0 };
        std::vector<uint32_t> pixels;

        CUstream              renderStream;
        CUstream              finalStream;
        pipeStatus            render_status = pipeStatus::idle;
        pipeStatus            tcnn_status = pipeStatus::idle;

        // Rendering Details
        bool                  accumulate = true;
        int                   accumulateFrameID_1 = 0;
        int                   accumulateFrameID_2 = 0;

        int                   lightID = 0;
        int                   numEnvLights;

        renderConfig launchConfig {
             1,     // samplesPerPixel
             1,     // directSamplesPerLight
             1,     // indirectSamplesPerDirect

             finalColorBufferContent::full,  // renderingMode
             false, // waveletMode

             true,  // finalDenoiser
             true,  // directDenoiser
             true,  // indirectDenoiser

             true, // Realtime mode
        };
    };
}