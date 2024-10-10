#include <filesystem>
#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h> 
#include "3rdParty/cxxopts.hpp"

#include "RenderEngine.h"
#include "Scene.h"
#include "OutputWriter.h"
#include "OutputVisualizer.h"
#include "npy.h"
#include "denoiser.h"
#include "offlineRenderer.h"
#include "files.h"
#include "SHRenderer.h"

inline float random(std::mt19937& rnd) {
    std::uniform_real_distribution<> dist(0, 1);
    return dist(rnd);
}

void affineSpaceExperiment() {
    using namespace gdt;
    AffineSpace3f a(one);
    a = AffineSpace3f::translate(vec3f(1,0,0)) * a;
    a = AffineSpace3f::rotate(vec3f(0,1,0), M_PI ) * a;
    a = AffineSpace3f::translate(vec3f(1,0,0)) * a;
    
    AffineSpace3f b(one);
    b = AffineSpace3f::rotate(vec3f(0,1,0), M_PI ) * b;
    b = AffineSpace3f::translate(vec3f(10,0,0)) * b;
    
    vec3f testingVector(1,0,1);
    std::cout << xfmPoint(b * a, testingVector)<< std::endl;
    return;
}

void NPYLoaderExperiment() {
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<double> data;

    const std::string path{ "0.npy" };
    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    std::cout << "The shape is: {";
    unsigned long total = 1;
    for (unsigned long s : shape) {
        std::cout << s << ", ";
        total *= s;
    }
    std::cout << "}" << std::endl;
    std::cout <<"The data is:" << std::endl;
    for (float s : data) {
        std::cout << s << ", ";
    }
    std::cout << "}" << std::endl;

}

namespace nert_renderer {

    extern "C" int main(int argc, char **argv)
    {
        //NPYLoaderExperiment();
        #pragma region Args Parser
        cxxopts::Options options("PathTracer", "A render enigne created for neural prt");
        options
            .positional_help("scene.pbrt results_dir")
            .show_positional_help();

        options.add_options()
            ("scene_file", "scene.pbrt file", cxxopts::value<std::string>())
            ("results", "directory to put all results", cxxopts::value<std::string>())
            ("m,envmap_file_or_folder", "path to a *.hdr or a folder containing the four additional folders, cube, latlong, waveletcoeffs, and waveletstrength", cxxopts::value<std::string>())
            ("n,envmap_fileName", "name of the envmap should be use to render, for example \"0\" means 0.hdr", cxxopts::value<std::string>())
            ("v,visualize", "create an iteractive OpenGL window", cxxopts::value<bool>()->default_value("false"))
            ("w,wavelet_model_folder", "path to the folder containing the TCNN model", cxxopts::value<std::string>())
            ("t,trajectory_fileName", "path to the trajectory file", cxxopts::value<std::string>())
            ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
            ("l,large", "rendering at 800x600", cxxopts::value<bool>()->default_value("false"))
            ("s,start", "Starting position of the trajectory", cxxopts::value<int>()->default_value("0"))
            ("e,end", "Ending position of the trajectory", cxxopts::value<int>()->default_value("2000"))
            ("p,sh_tmatrix", "Special function: generate the tmatrix for SH PRT for the scene with degree specified", cxxopts::value<int>())
            ("k,sh_render", "Provide the path to a T matrix and the engine will enter SH rendering mode", cxxopts::value<std::string>())
            ("o,denoise", "Special function: will generate the denoised image for all images in the \"render\" sub folder,"
                          "sub folder should have cameraNormals / firstKds. Image name in the folders must match", cxxopts::value<std::string>())
            ("h,help", "Print usage");
        options.parse_positional({ "scene_file", "results"});
        cxxopts::ParseResult result;
        try {
            result = options.parse(argc, argv);
        }
        catch (const cxxopts::OptionParseException& x) {
            std::cerr << "PathTracer: " << x.what() << '\n';
            std::cout << options.help() << std::endl;
            return EXIT_FAILURE;
        }
        #pragma endregion 
        
        bool debug = result["debug"].as<bool>();
        if (debug) std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

        #pragma region Special Modes
        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (result.count("denoise") > 0) {
            std::cout << GDT_TERMINAL_GREEN;
            std::cout << "# Entering denoise mode." << std::endl;
            std::cout << GDT_TERMINAL_DEFAULT;
            denoiseMode(result["denoise"].as<std::string>());
            exit(0);
        }

        if (result.count("scene_file") == 0)
        {
            std::cerr << "PathTracer: scene.pbrt missing" << '\n';
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (result.count("sh_tmatrix") > 0) {
            std::shared_ptr<Scene> testScene(new Scene(debug));
            testScene->loadPBRT(result["scene_file"].as<std::string>());

            std::cout << GDT_TERMINAL_GREEN;
            std::cout << "# Entering T Matrix Compute mode." << std::endl;
            std::cout << GDT_TERMINAL_DEFAULT;

            if (result.count("results") == 0) {
                std::cerr << "PathTracer: result directory needed for non-visualization task" << '\n';
                std::cout << options.help() << std::endl;
                exit(0);
            }

            fs::path env_path(result["envmap_file_or_folder"].as<std::string>());
            fs::path export_path(result["results"].as<std::string>());
            SHRenderer shrender(testScene, env_path, export_path, debug, result["sh_tmatrix"].as<int>());
            exit(0);
        }
        #pragma endregion 

        #pragma region loading information
        // Load Scene
        std::shared_ptr<Scene> testScene (new Scene(debug));
        testScene->loadPBRT(result["scene_file"].as<std::string>());
        if (debug) std::cout << "Base Scene Loaded" << std::endl;

        // Setup Engine
        renderEngineMode mode = result.count("sh_render") > 0 ? renderEngineMode::shRender : renderEngineMode::normal;
        vec2i dim = result["large"].as<bool>() ? vec2i(800, 600) : vec2i(512, 512);

        std::shared_ptr<RenderEngine> engine (new RenderEngine(testScene, mode, debug));
        engine->setCamera(testScene->initialCamera);
        engine->buildScene();
        engine->resize(dim);
        if (debug) std::cout << "Engine Built" << std::endl;

        // Setup TCNN Here
        if (result.count("wavelet_model_folder") > 0) {
            engine->initTCNN(result["wavelet_model_folder"].as<std::string>());
        }

        // Setup sh_render Here
        if (mode == renderEngineMode::shRender) {
            engine->shLoadTMatrix(result["sh_render"].as<std::string>());
        }
        #pragma endregion

        // Check contradictory
        if (result.count("trajectory_fileName") > 0 && result["visualize"].as<bool>()) {
            std::cerr << "PathTracer: visualization cannot be turn on with batch mode" << '\n';
            std::cout << options.help() << std::endl;
            exit(0);
        }

        // If visualize
        if (result["visualize"].as<bool>()) {
            const float worldScale = length(testScene->bounds.span());
            fs::path env_path(result["envmap_file_or_folder"].as<std::string>());
            if (fs::is_directory(env_path)) {
                // Option A: Just visualize one envmap with wavelet coeffs 
                if (result.count("envmap_fileName") > 0) {
                    testScene->EnvMap.push_back(std::make_shared<Environment_Map>(env_path.string(), result["envmap_fileName"].as<std::string>(), debug));
                }
                // Option B: visualize a group of envmap
                else {
                    fs::path cube_path = env_path / "cube";
                    std::vector<std::string> directory;
                    for (auto& p : fs::directory_iterator(cube_path))
                        directory.push_back(p.path().stem().string());

                    std::sort(directory.begin(), directory.end(), compareNat);
                    for (auto fileName : directory)
                        testScene->EnvMap.push_back(std::make_shared<Environment_Map>(env_path.string(), fileName, debug));
                }
            }

            else {
                // Option C: Just visualize one envmap without cube or wavelets 
                testScene->EnvMap.push_back(std::make_shared<Environment_Map>(env_path.string(), debug));
            }

            engine->buildLights();
            std::cout << GDT_TERMINAL_GREEN;
            std::cout << "# Optix Renderer: Start visualizing" << std::endl;
            std::cout << GDT_TERMINAL_DEFAULT;

            std::shared_ptr<SampleWindow> window = std::make_shared<SampleWindow>("NERT ", engine, testScene, dim, worldScale, mode);
            window->enableFlyMode();
            window->run();
            return 0;
        }

        if (result.count("results") == 0) {
            std::cerr << "PathTracer: result directory needed for non-visualization task" << '\n';
            std::cout << options.help() << std::endl;
            exit(0);
        }

        // If offline rendering
        std::shared_ptr<OutputWriter> writer = std::make_shared<OutputWriter>(dim, result["results"].as<std::string>());
        OfflineRenderer renderer(testScene, engine, writer, debug);

        if (result.count("trajectory_fileName") > 0) {
            fs::path env_path(result["envmap_file_or_folder"].as<std::string>());
            if (!fs::is_directory(env_path)) {
                std::cerr << "PathTracer: In trajectory rendering mode, envmap_file_or_folder should be a folder" << '\n';
                std::cout << options.help() << std::endl;
                exit(0);
            }

            if (mode == renderEngineMode::normal)
                renderer.render_trajectory(
                    fs::path(result["trajectory_fileName"].as<std::string>()),
                    env_path,
                    result["start"].as<int>(),
                    result["end"].as<int>()
                );
            else
                renderer.render_trajectory_sh(
                    fs::path(result["trajectory_fileName"].as<std::string>()),
                    env_path,
                    result["start"].as<int>(),
                    result["end"].as<int>()
                );
        }
        else {
            fs::path env_path(result["envmap_file_or_folder"].as<std::string>());
            if (fs::is_directory(env_path)) {
                if(result.count("envmap_fileName") > 0) {
                    testScene->EnvMap.push_back(std::make_shared<Environment_Map>(env_path.string(), result["envmap_fileName"].as<std::string>(), debug));
                }
                else {
                    std::cerr << "PathTracer: In normal rendering mode, if envmap_file_or_folder is folder, it needs to come with a envmap_name" << '\n';
                    std::cout << options.help() << std::endl;
                    exit(0);
                }
            } 
            else {
                testScene->EnvMap.push_back(std::make_shared<Environment_Map>(env_path.string(), result.count("sh_render") != 0, debug));
            }

            if (mode == renderEngineMode::normal)
                renderer.render();
            else 
                renderer.render_sh();
            
        }
        return 0;
    }
} // ::nert_renderer
