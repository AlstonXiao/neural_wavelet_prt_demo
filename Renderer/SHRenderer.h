
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

	class SHRenderer {
	public:
		SHRenderer(std::shared_ptr<Scene> scene, fs::path envmap_folder, fs::path export_folder, bool debug = false, int sh_degree = 8);
	};
}