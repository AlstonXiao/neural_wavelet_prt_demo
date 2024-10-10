#include "OutputVisualizer.h"

namespace nert_renderer {

        SampleWindow::SampleWindow(const std::string& title,
            std::shared_ptr<RenderEngine> engine,
            const std::shared_ptr<Scene> scene,
            const vec2i size,
            const float worldScale,
            renderEngineMode mode
        )
            : GLFCameraWindow(title, scene->initialCamera.from, scene->initialCamera.at, scene->initialCamera.up, worldScale, size),
            m_mode(mode)
        {
            m_engine = engine;
            numEnvLights = scene->EnvMap.size();
            cudaStreamCreate(&renderStream);
            cudaStreamCreate(&finalStream);

            if (m_mode == renderEngineMode::normal) {
                std::cout << "Press 'k' to switch between modes." << std::endl;
                std::cout << "Press 'l' to toggle wavelet mode" << std::endl << std::endl;
                std::cout << "Press 'b' to toggle direct color denoiser. Note this feature is only useful in full wavelet rendering mode" << std::endl;
            }

            std::cout << "Press 'n' to toggle final color denoiser" << std::endl;
            std::cout << "Press 'm' to toggle accumulation" << std::endl << std::endl;
            std::cout << "Press 'g' to select the previous envmap" << std::endl;
            std::cout << "Press 'h' to select the next envmap" << std::endl << std::endl;
            std::cout << "Press ',' to reduce the number of paths/pixel" << std::endl;
            std::cout << "Press '.' to increase the number of paths/pixel" << std::endl;
            if (m_mode == renderEngineMode::normal) {
                std::cout << "Press 'q' to reduce the number of light samples at first hit" << std::endl;
                std::cout << "Press 'e' to increase the number of light samples at first hit" << std::endl;
                std::cout << "Press 'o' to reduce the number of indirect samples at first hit" << std::endl;
                std::cout << "Press 'p' to increase the number of indirect samples at first hit" << std::endl << std::endl;
            }
            std::cout << "Press 'f' to switch to Flying mode" << std::endl;
            std::cout << "Press 'i' to switch to inspect mode" << std::endl;
            std::cout << "Press 'xyz' to move camera in absolute frame" << std::endl;
            std::cout << "Press 'wsad' to move camera in local frame" << std::endl;
            std::cout << "Press 'c' to inspect camera stat" << std::endl;
        }

        void SampleWindow::render(float deltaTime)
        {
            // Persistent movements are handled here
            cameraFrameManip->keyUpdate(deltaTime);

            if (cameraFrame.modified) {
                m_engine->setCamera(Camera{ cameraFrame.get_from(),
                                         cameraFrame.get_at(),
                                         cameraFrame.get_up() });
                cameraFrame.modified = false;
                accumulateFrameID_1 = 0;
                accumulateFrameID_2 = 0;
            }

            cudaStreamSynchronize(renderStream);

            if (launchConfig.waveletMode) {
                if (!m_engine->EnvmapSetup || !m_engine->TCNNSetup)
                    throw std::runtime_error("Enviornment or TCNN not ready for rendering in wavelet mode");
                
                // sync the TCNN
                m_engine->TCNNSync();

                // Prepare to launch

                // First check the TCNN engine 
                if (tcnn_status == pipeStatus::frameID1) {
                    m_engine->computeFinalPixelColors(1, finalStream, launchConfig);
                    m_engine->downloadPixels(pixels.data(), 1, finalStream);
                }
                else if (tcnn_status == pipeStatus::frameID0) {
                    m_engine->computeFinalPixelColors(0, finalStream, launchConfig);
                    m_engine->downloadPixels(pixels.data(), 0, finalStream);
                }
                else {
                    // Do nothing. This should only be happen at second render engine call
                }

                // Switching from wavelet mode to frame mode
                if (tcnn_status == pipeStatus::idle && render_status == pipeStatus::frameID1) {
                    m_engine->TCNNPass_p1(1, renderStream);
                }

                // Second check the render+engine status

                if (render_status == pipeStatus::frameID1) {
                    m_engine->TCNNPass_p2(lightID, 1, launchConfig);
                    tcnn_status = pipeStatus::frameID1;
                }
                else if (render_status == pipeStatus::frameID0) {
                    m_engine->TCNNPass_p2(lightID, 0, launchConfig);
                    tcnn_status = pipeStatus::frameID0;
                }
                else {
                    // Do nothing. This should only be happen at first render engine call
                }

                // Finally Render the next frame
                if (!accumulate) {
                    accumulateFrameID_1 = 0;
                    accumulateFrameID_2 = 0;
                }
                if (render_status == pipeStatus::frameID0) {
                    m_engine->render(1, accumulateFrameID_1, lightID, launchConfig, renderStream);
                    m_engine->TCNNPass_p1(1, renderStream);
                    render_status = pipeStatus::frameID1;
                    accumulateFrameID_1++;
                }
                else if (render_status == pipeStatus::frameID1) {
                    m_engine->render(0, accumulateFrameID_2, lightID, launchConfig, renderStream);
                    m_engine->TCNNPass_p1(0, renderStream);
                    render_status = pipeStatus::frameID0;
                    accumulateFrameID_2++;
                }
                // initial call, This will only happen when the program is loaded with wavelet mode on
                else {
                    m_engine->render(1, accumulateFrameID_1, lightID, launchConfig, renderStream);
                    m_engine->TCNNPass_p1(1, renderStream);
                    render_status = pipeStatus::frameID1;
                    accumulateFrameID_1++;
                }
            }
            // Two different rendering mode
            else {
                // Normal rendering will happen in FB1
                tcnn_status = pipeStatus::idle;

                // Though we render in frame 1, we might need to use frame 2 at first if switching from TCNN
                if (render_status == pipeStatus::frameID1) {
                    m_engine->computeFinalPixelColors(1, finalStream, launchConfig);
                    m_engine->downloadPixels(pixels.data(), 1, finalStream);
                }
                else if (render_status == pipeStatus::frameID0) {
                    m_engine->computeFinalPixelColors(0, finalStream, launchConfig);
                    m_engine->downloadPixels(pixels.data(), 0, finalStream);
                }

                // It is always frame 1
                render_status = pipeStatus::frameID1;

                // launch
                if (!accumulate)
                    accumulateFrameID_1 = 0;

                if (m_mode == renderEngineMode::normal) 
                    m_engine->render(1, accumulateFrameID_1, lightID, launchConfig, renderStream);
                else
                    m_engine->shRender(1, accumulateFrameID_1, lightID, launchConfig, renderStream);

                accumulateFrameID_1++;
            }
            return;
        }

        void SampleWindow::draw()
        {
            cudaStreamSynchronize(finalStream);
            if (fbTexture == 0)
                glGenTextures(1, &fbTexture);

            glBindTexture(GL_TEXTURE_2D, fbTexture);
            GLenum texFormat = GL_RGBA;
            GLenum texelType = GL_UNSIGNED_BYTE;
            glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbSize.x, fbSize.y, 0, GL_RGBA,
                texelType, pixels.data());

            glDisable(GL_LIGHTING);
            glColor3f(1, 1, 1);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, fbTexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            glDisable(GL_DEPTH_TEST);

            glViewport(0, 0, fbSize.x, fbSize.y);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.f, 0.f);
                glVertex3f(0.f, 0.f, 0.f);

                glTexCoord2f(0.f, 1.f);
                glVertex3f(0.f, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 1.f);
                glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

                glTexCoord2f(1.f, 0.f);
                glVertex3f((float)fbSize.x, 0.f, 0.f);
            }
            glEnd();
        }

        void SampleWindow::resize(const vec2i& newSize)
        {
            cudaStreamSynchronize(renderStream);
            cudaStreamSynchronize(finalStream);
            m_engine->TCNNSync();

            fbSize = newSize;
            m_engine->resize(newSize);

            render_status = pipeStatus::idle;
            tcnn_status = pipeStatus::idle;
            
            pixels.resize(newSize.x * newSize.y);
            
            render(0);
            render(0);
            accumulateFrameID_1 = 0;
            accumulateFrameID_2 = 0;
        }

        void SampleWindow::print_mode() {
            if (launchConfig.renderingMode == finalColorBufferContent::full) {
                if (launchConfig.waveletMode)
                    std::cout << "Current Mode is: Ground Truth Direct + Predicted Indirect " << std::endl;
                else
                    std::cout << "Current Mode is: Ground Truth Direct + Ground Truth Indirect " << std::endl;
            }
            else if (launchConfig.renderingMode == finalColorBufferContent::direct) {
                if (launchConfig.waveletMode) {
                    launchConfig.renderingMode = finalColorBufferContent::indirect;
                    std::cout << "Current Mode is: Predicted Indirect " << std::endl;
                }
                else
                    std::cout << "Current Mode is: Ground Truth Direct " << std::endl;
            }
            else if (launchConfig.renderingMode == finalColorBufferContent::indirect) {
                if (launchConfig.waveletMode)
                    std::cout << "Current Mode is: Predicted Indirect " << std::endl;
                else
                    std::cout << "Current Mode is: Ground Truth Indirect  " << std::endl;
            }

        }

        void SampleWindow::key(int key, int mods, int action)
        {
            if (action != GLFW_PRESS) {
                cameraFrameManip->key(key, mods, action);
                return;
            }

            if (m_mode == renderEngineMode::normal) {
                if (key == 'L' || key == 'l') {
                    launchConfig.waveletMode = !launchConfig.waveletMode;
                    if (!m_engine->TCNNAvaliable())
                        launchConfig.waveletMode = false;
                    // direct denoiser will always be on
                    if (launchConfig.waveletMode) {
                        launchConfig.directDenoiser = true;
                    }
                    print_mode();
                    accumulateFrameID_1 = 0;
                    accumulateFrameID_2 = 0;
                }
                if (key == 'K' || key == 'k') {
                    if (launchConfig.renderingMode == finalColorBufferContent::full) {
                        if (launchConfig.waveletMode)
                            launchConfig.renderingMode = finalColorBufferContent::indirect;
                        else
                            launchConfig.renderingMode = finalColorBufferContent::direct;
                    }
                    else if (launchConfig.renderingMode == finalColorBufferContent::direct) {
                        launchConfig.renderingMode = finalColorBufferContent::indirect;
                    }
                    else if (launchConfig.renderingMode == finalColorBufferContent::indirect) {
                        launchConfig.renderingMode = finalColorBufferContent::full;
                    }

                    print_mode();
                    accumulateFrameID_1 = 0;
                    accumulateFrameID_2 = 0;
                }
            }

            if (key == 'N' || key == ' ' || key == 'n') {
                launchConfig.finalDenoiser = !launchConfig.finalDenoiser;
                launchConfig.indirectDenoiser = launchConfig.finalDenoiser;
                launchConfig.directDenoiser = launchConfig.finalDenoiser;
                // direct denoiser will always be on
                if (launchConfig.waveletMode) {
                    launchConfig.directDenoiser = true;
                }
                std::cout << "Final Denoising now " << (launchConfig.finalDenoiser ? "ON" : "OFF") << std::endl;
            }

            if (key == 'M' || key == 'm') {
                accumulate = !accumulate;
                std::cout << "accumulation/progressive refinement now " << (accumulate? "ON" : "OFF") << std::endl;
                accumulateFrameID_1 = 0;
                accumulateFrameID_2 = 0;
            }

            if (key == 'G' || key == 'g') {
                lightID++;
                if (lightID >= numEnvLights)
                    lightID = 0;
                std::cout << "light ID is " << lightID << std::endl;
            }
            if (key == 'H' || key == 'h') {
                lightID--;
                if (lightID < 0)
                    lightID = numEnvLights - 1;
                std::cout << "light ID is " << lightID << std::endl;
            }

            if (key == 'C' || key == 'c') {
                std::cout << GDT_TERMINAL_YELLOW;
                std::cout << "RENDERING STATS DUMP" << std::endl;

                if (m_mode == renderEngineMode::normal) {
                    print_mode();
                    std::cout << std::endl;
                    std::cout << "Wavelet rendering is " << (launchConfig.waveletMode ? "ON" : "OFF") << std::endl;
                }
                std::cout << "Final Denoising now " << (launchConfig.finalDenoiser ? "ON" : "OFF") << std::endl;
                std::cout << "accumulation/progressive refinement now " << (accumulate ? "ON" : "OFF") << std::endl << std::endl;

                std::cout << "Envmap ID " << lightID + 1 << " / " << numEnvLights << std::endl << std::endl;

                std::cout << "num samples/pixel now "
                    << launchConfig.samplesPerPixel << std::endl;
                std::cout << "num indirectSamplesPerDirect now "
                    << launchConfig.indirectSamplesPerDirect << std::endl;
                std::cout << "num samplesPerLight now "
                    << launchConfig.directSamplesPerLight << std::endl << std::endl;
                std::cout << "frame size at " << fbSize.x << "x" << fbSize.y << std::endl;
                std::cout << GDT_TERMINAL_DEFAULT;
            }
            if (key == ',') {
                launchConfig.samplesPerPixel = std::max(1, launchConfig.samplesPerPixel - 1);
                std::cout << "num samples/pixel now " << launchConfig.samplesPerPixel << std::endl;
            }
            if (key == '.') {
                launchConfig.samplesPerPixel = std::max(1, launchConfig.samplesPerPixel + 1);
                std::cout << "num samples/pixel now " << launchConfig.samplesPerPixel << std::endl;
            }

            if (key == 'O' || key == 'o') {
                launchConfig.indirectSamplesPerDirect = std::max(1, launchConfig.indirectSamplesPerDirect - 1);
                std::cout << "num indirectSamplesPerDirect now " << launchConfig.indirectSamplesPerDirect << std::endl;
            }
            if (key == 'P' || key == 'p') {
                launchConfig.indirectSamplesPerDirect = std::max(1, launchConfig.indirectSamplesPerDirect + 1);
                std::cout << "num indirectSamplesPerDirect now "<< launchConfig.indirectSamplesPerDirect << std::endl;
            }

            if (key == 'Q' || key == 'q') {
                launchConfig.directSamplesPerLight = std::max(1, launchConfig.directSamplesPerLight - 1);
                std::cout << "num samplesPerLight now " << launchConfig.directSamplesPerLight << std::endl;
            }
            if (key == 'E' || key == 'e') {
                launchConfig.directSamplesPerLight = std::max(1, launchConfig.directSamplesPerLight + 1);
                std::cout << "num samplesPerLight now " << launchConfig.directSamplesPerLight << std::endl;
            }
            switch (key) {
            case 'f':
            case 'F':
                std::cout << "Entering 'fly' mode" << std::endl;
                if (flyModeManip) cameraFrameManip = flyModeManip;
                break;
            case 'i':
            case 'I':
                std::cout << "Entering 'inspect' mode" << std::endl;
                if (inspectModeManip) cameraFrameManip = inspectModeManip;
                break;
            default:
                if (cameraFrameManip)
                    cameraFrameManip->key(key, mods, action);
            }
        }

}