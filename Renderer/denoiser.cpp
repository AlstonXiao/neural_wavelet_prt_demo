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
#include "files.h"

namespace nert_renderer {
    inline void loadToCUDABuffer(HDRTexture& texture, CUDABuffer& cudaBuffer) {
        vec2i resolution = texture.resolution;
        int totalSize = resolution.x * resolution.y;

        std::vector<vec4f> textureVec4f(totalSize);

        for (int u = 0; u < resolution.y; ++u) {
            for (int v = 0; v < resolution.x; ++v) {
                textureVec4f[u * resolution.x + v] = vec4f(texture.pixel[u * resolution.x + v], 1.f);
            }
        }
        cudaBuffer.alloc_and_upload(textureVec4f);
    }

    void denoiseMode(std::string folderName) {
        RenderEngine* engine = new RenderEngine(NULL, renderEngineMode::normal, false);
        fs::path rootFolder(folderName);
        fs::path cameraNormal = rootFolder / "cameraNormals";
        fs::path cameraAlbedo = rootFolder / "firstKds";
        fs::path renders = rootFolder / "render";

        std::vector<std::string> normalDirectory;
        getFiles(normalDirectory, cameraNormal);
        std::vector<std::string> albedoDirectory;
        getFiles(albedoDirectory, cameraAlbedo);
        std::vector<std::string> renderDirectory;
        getFiles(renderDirectory, renders);

        int size = normalDirectory.size();
        if (normalDirectory.size() != albedoDirectory.size() || normalDirectory.size() != renderDirectory.size()) {
            std::cout << "normal and albdeo should have the same size" << std::endl;
            return;
        }

        for (int i = 0; i < size; i++) {
            std::cout << "Currently denoising: " << i << " Total is: " << size << std::endl;

            HDRTexture normalTexture;
            HDRTexture albedoTexure;

            normalTexture.loadTexture(normalDirectory[i]);
            albedoTexure.loadTexture(albedoDirectory[i]);

            if (normalTexture.resolution.x != albedoTexure.resolution.x) {
                std::cout << "resolution mismtach" << std::endl;
                return;
            }

            if (normalTexture.resolution.y != albedoTexure.resolution.y) {
                std::cout << "resolution mismtach" << std::endl;
                return;
            }

            CUDABuffer normal_buffer;
            CUDABuffer albedo_buffer;

            loadToCUDABuffer(normalTexture, normal_buffer);
            loadToCUDABuffer(albedoTexure, albedo_buffer);

            if (renderDirectory.size() > 0) {
                OutputWriter writer(albedoTexure.resolution, rootFolder.string());

                CUDABuffer color_buffer;
                HDRTexture colorTexure;
                colorTexure.loadTexture(renderDirectory[i]);

                if (colorTexure.resolution.y != colorTexure.resolution.y || colorTexure.resolution.x != colorTexure.resolution.x) {
                    std::cout << "resolution mismtach" << std::endl;
                    return;
                }
                loadToCUDABuffer(colorTexure, color_buffer);

                CUDABuffer final_buffer;
                final_buffer.alloc(sizeof(vec4f) * colorTexure.resolution.x * colorTexure.resolution.y);
                engine->denoise(color_buffer.d_pointer(), normal_buffer.d_pointer(), albedo_buffer.d_pointer(), final_buffer.d_pointer(), colorTexure.resolution);
                std::vector<vec4f> resultHost(colorTexure.resolution.x * colorTexure.resolution.y);
                final_buffer.download(resultHost.data(), resultHost.size());
                writer.write_buffer(resultHost, "denoised", renderDirectory[i]);

                color_buffer.free();
                final_buffer.free();
            }
            normal_buffer.free();
            albedo_buffer.free();
        }
    }
}