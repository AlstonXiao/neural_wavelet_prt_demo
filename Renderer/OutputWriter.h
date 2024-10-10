#pragma once

#include <filesystem>
#include <random>
#include <fstream>
#include <string>

namespace nert_renderer {
    namespace fs = std::filesystem;
    class OutputWriter
    {
    public:
        OutputWriter(vec2i ImageSize, const std::string& outputPath);
        ~OutputWriter() {}

        void resize(vec2i newSize);

        void write_buffer(std::vector<vec4f> color, const std::string& fileType, const std::string& frameName);
        void write_aux(const std::string& frameName, Camera camera);
        void write_frame(const std::string& frameName);
        void write_SH(const std::string& frameName);

        void write_wavelet(const std::string& frameName);
        void write_final_color(const std::string& frameName);
        void write_indirect_color(const std::string& frameName);
        
        void writeRGBA(float* data, std::string fileType, const std::string& frameName);
        void writeAccurateRGBA(float* data, std::string fileType, const std::string& frameName);
        void writeAccurateRGBA(int* data, std::string fileType, const std::string& frameName);
        void writeMask(bool* data, std::string fileType, const std::string& frameName);
        void writeOneChannel(float* data, std::string fileType, const std::string& frameName);
        void writePose(Camera camera, const std::string& fileType, const std::string& frameName);

        std::shared_ptr<fullframe> frame;
        int row, column;
        const fs::path outputDir;
    };
}