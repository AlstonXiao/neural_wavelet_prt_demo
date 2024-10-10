#pragma once
#include <vector>
#include <map>
#pragma once
#include <random>
#include <filesystem>

#include "RenderEngine.h"
#include "Scene.h"
#include "OutputWriter.h"

namespace nert_renderer {
	namespace fs = std::filesystem;

	class OfflineRenderer {

	public:
		OfflineRenderer(std::shared_ptr<Scene> scene, std::shared_ptr<RenderEngine> engine, std::shared_ptr<OutputWriter> writer, bool debug = false);
		void render_trajectory(fs::path trajectory_file, fs::path env_path, int start, int end);
		void render();

		void render_trajectory_sh(fs::path trajectory_file, fs::path env_path, int start, int end);
		void render_sh();
	private:
		std::shared_ptr<Scene> m_scene;
		std::shared_ptr<RenderEngine> m_engine;
		std::shared_ptr<OutputWriter> m_writer;

		renderConfig config;
		renderConfig waveletConfig;
		bool trainingMode = false;
		bool debug;
		std::vector<Camera> cameraList;
		std::vector<std::string> envmapList;

		std::string readTrajectoryFile(fs::path trajectory_file_name);
		void renderOneFrame(const Camera& camera, const std::string& outputName);
	};

}