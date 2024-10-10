#include "SHRenderer.h"
#include "files.h"

namespace nert_renderer {
	namespace fs = std::filesystem;

	SHRenderer::SHRenderer(std::shared_ptr<Scene> scene, fs::path env_path, fs::path export_folder, bool debug, int sh_degree) {
        std::shared_ptr<RenderEngine> engine = std::make_shared<RenderEngine>(scene, renderEngineMode::shTmatrix, debug);
        assert(fs::is_directory(env_path));
        assert(export_folder.extension() == ".npy");

        std::vector<std::string> directory;
        for (auto& p : std::filesystem::directory_iterator(env_path)) {
            if (p.path().extension() == ".npy")
                directory.push_back(p.path().string());
        }
        std::sort(directory.begin(), directory.end(), compareNat);
        for (auto fileName : directory) {
            fs::path filses(fileName);
            if (debug) std::cout << "Imported filename: " << filses.string() << std::endl;
            scene->EnvMap.push_back(std::make_shared<Environment_Map>(filses.string(), debug));
        }
        engine->buildScene();
        engine->buildLights();

        std::cout << GDT_TERMINAL_GREEN;
        std::cout << "# Start Rendering! " << std::endl;
        std::cout << GDT_TERMINAL_DEFAULT;

        assert((sh_degree + 1) * (sh_degree + 1) <= scene->EnvMap.size());
        engine->shComputeTMatrix(sh_degree, export_folder);
	}
}