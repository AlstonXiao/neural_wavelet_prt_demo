#include "offlineRenderer.h"
#include "files.h"
#include "npy.h"

namespace nert_renderer {
    OfflineRenderer::OfflineRenderer(std::shared_ptr<Scene> scene, std::shared_ptr<RenderEngine> engine, std::shared_ptr<OutputWriter> writer, bool debug) :
        m_scene(scene), m_engine(engine), m_writer(writer), debug(debug)
    { 

    }

    void OfflineRenderer::render_trajectory(
        fs::path trajectory_file,
        fs::path env_path,
        int start,
        int end
    ) {
        auto envFileName = readTrajectoryFile(trajectory_file);
        if (end > cameraList.size()) end = cameraList.size();
        for (int i = start; i < end; i++)
        {
            m_engine->setCamera(cameraList[i]);
            m_scene->EnvMap.push_back(std::make_shared<Environment_Map>(env_path.string(), envmapList[i], debug));
            m_engine->buildLights();

            renderOneFrame(cameraList[i], std::to_string(i) + "_" + envmapList[i]);
            m_scene->EnvMap.pop_back();
            m_engine->destroyLights();
            
            std::cout << "Trajectory: " << i << " Done. Total is: " << end << std::endl;
        }
    }

    void OfflineRenderer::render_trajectory_sh(
        fs::path trajectory_file,
        fs::path env_path,
        int start,
        int end
    ) {
        auto envFileName = readTrajectoryFile(trajectory_file);
        if (end > cameraList.size()) end = cameraList.size();
        for (int i = start; i < end; i++)
        {
            m_engine->setCamera(cameraList[i]);
            m_scene->EnvMap.push_back(std::make_shared<Environment_Map>(env_path.string(), envmapList[i], debug));
            m_engine->shRender(0, 0, 0, config);
            m_scene->EnvMap.pop_back();

            cudaDeviceSynchronize();
            
            m_engine->downloadframe(m_writer->frame, 0);
            m_writer->write_SH(std::to_string(i) + "_" + envmapList[i]);
            std::cout << "Trajectory: " << i << " Done. Total is: " << end << std::endl;
        }
    }
    void OfflineRenderer::render() {
        m_engine->buildLights();
        renderOneFrame(m_scene->initialCamera, "0");
    }

    void OfflineRenderer::render_sh() {
        if (debug) std::cout << "Start rendering" << std::endl;

        m_engine->shRender(0, 0, 0, config);
        cudaDeviceSynchronize();
        m_engine->downloadframe(m_writer->frame, 0);
        m_writer->write_SH("0");
    }

    std::string OfflineRenderer::readTrajectoryFile(fs::path trajectory_file_name) {
        std::ifstream infile(trajectory_file_name.c_str());
        
        bool trajectoryStart = false;
        std::string envmapFolder;
        std::string line;
        config.renderingMode = finalColorBufferContent::full;
        config.waveletMode = false; // TODO: not supporting wavelet mode for now 
        while (std::getline(infile, line)) {
            std::vector<std::string> commands;
            line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            if (line == "")
                continue;
            tokenize(line, ' ', commands);
            
            if (commands[0] == "trajectoryBegin")
                trajectoryStart = true;
            else if (commands[0] == "samplesPerPixel")
                config.samplesPerPixel = stoi(commands[1]);
            else if (commands[0] == "directSamplesPerLight")
                config.directSamplesPerLight = stoi(commands[1]);
            else if (commands[0] == "IndirectSamplesPerDirect")
                config.indirectSamplesPerDirect = stoi(commands[1]);
            else if (commands[0] == "finalDenoiser")
                config.finalDenoiser = commands[1] == "true";
            else if (commands[0] == "directIndirectDenoiser") {
                config.directDenoiser = commands[1] == "true";
                config.indirectDenoiser = commands[1] == "true";
            }
            else if (commands[0] == "trainingMode")
                trainingMode = commands[1] == "true";
            else if (commands[0] == "waveletSamplesPerPixel")
                waveletConfig.samplesPerPixel = stoi(commands[1]);
            else if (commands[0] == "waveletSamplesPerLight")
                waveletConfig.directSamplesPerLight = stoi(commands[1]);
            else if (commands[0] == "waveletDirectDenoiser")
                continue;
            else if (commands[0] == "waveletFinalDenoiser") {
                waveletConfig.finalDenoiser = commands[1] == "true";
                waveletConfig.indirectDenoiser = commands[1] == "true";
            }
            else if (commands[0] == "envmapFolder")
                envmapFolder = commands[1];
            else {
                if (!trajectoryStart)
                    throw std::runtime_error("Unknown things before trajectory start " + commands[0]);
                cameraList.push_back({ {stof(commands[0]), stof(commands[1]), stof(commands[2])}, {stof(commands[3]), stof(commands[4]),stof(commands[5])}, {0,1,0} });
                envmapList.push_back(commands[6]);
            }
        }
        return envmapFolder;
    }

    inline void OfflineRenderer::renderOneFrame(const Camera& camera, const std::string& outputName) {
        m_engine->render(1, 0, 0, config);

        if (m_engine->TCNNAvaliable()) {
            m_engine->TCNNPass_p1(1);
            m_engine->TCNNPass_p2(0, 1, waveletConfig);
         }
        cudaDeviceSynchronize();

        m_engine->downloadframe(m_writer->frame, 1);
        m_writer->write_frame(outputName); // TODO!! 
        if (trainingMode)
            m_writer->write_aux(outputName, camera);

        if (m_engine->TCNNAvaliable()) {
            m_writer->write_wavelet(outputName);
        }
    }
}