#include <filesystem>
#include <random>
#include <fstream>
#include <string>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"
#include "RenderEngine.h"
#include "OutputWriter.h"
#include "npy.h"

namespace nert_renderer {
    OutputWriter::OutputWriter(vec2i ImageSize, const std::string& outputPath) : outputDir(outputPath) {
        row = ImageSize.x;
        column = ImageSize.y;
        frame = std::make_shared<fullframe>(ImageSize);
    }

    void OutputWriter::resize(vec2i newSize) {
        frame->assignSize(newSize);
        row = newSize.x;
        column = newSize.y;
    }

    void OutputWriter::writeRGBA(float* data, std::string fileType, const std::string& frameName) {
        for (int y = 0; y < column / 2; y++) {
            float* line_y = data + y * row * 4;
            float* mirrored_y = data + ( column - 1 - y) * row * 4;
            for (int x = 0; x < row * 4; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }
        fs::path folder = outputDir / fileType;
        if (!fs::exists(folder)) { fs::create_directories(folder); }
        fs::path fileName = folder / (frameName + ".hdr");
#ifdef _WINDOWS
        stbi_write_hdr(fileName.string().c_str(), row, column, 4, data);
#else
        stbi_write_hdr(fileName.c_str(), row, column, 4, data);
#endif
    }

    void OutputWriter::writeAccurateRGBA(float* data , std::string fileType, const std::string& frameName) {
        for (int y = 0; y < column / 2; y++) {
            float* line_y = data + y * row * 4;
            float* mirrored_y = data + (column - 1 - y) * row * 4;
            for (int x = 0; x < row * 4; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }

        fs::path folder = outputDir / fileType;
        if (!fs::exists(folder)) { fs::create_directories(folder); }
        fs::path fileName = folder / (frameName + ".npy");

        std::ofstream ofs; // output file stream

        std::array<long unsigned, 3> leshape11{ {column, row,4} };
        npy::SaveArrayAsNumpy(fileName.string(), false, leshape11.size(), leshape11.data(), data);
    }

    void OutputWriter::writeAccurateRGBA(int* data, std::string fileType, const std::string& frameName) {
        for (int y = 0; y < column / 2; y++) {
            int* line_y = data + y * row * 4;
            int* mirrored_y = data + (column - 1 - y) * row * 4;
            for (int x = 0; x < row * 4; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }

        fs::path folder = outputDir / fileType;
        if (!fs::exists(folder)) { fs::create_directories(folder); }
        fs::path fileName = folder / (frameName + ".npy");

        std::ofstream ofs; // output file stream

        std::array<long unsigned, 3> leshape11{ {column, row, 4} };
        npy::SaveArrayAsNumpy(fileName.string(), false, leshape11.size(), leshape11.data(), data);
    }

    void OutputWriter::writeOneChannel(float* data, std::string fileType, const std::string& frameName) {
        for (int y = 0; y < column / 2; y++) {
            float* line_y = data + y * row;
            float* mirrored_y = data + (column - 1 - y) * row;
            for (int x = 0; x < row; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }
        fs::path folder = outputDir / fileType;
        if (!fs::exists(folder)) { fs::create_directories(folder); }
        fs::path fileName = folder / (frameName + ".hdr");
#ifdef _WINDOWS
        stbi_write_hdr(fileName.string().c_str(), row, column, 1, data);
#else
        stbi_write_hdr(fileName.c_str(), row, column, 1, data);
#endif
    }

    void OutputWriter::writeMask(bool* data, std::string fileType, const std::string& frameName) {
        std::vector<float> mask_in_float;
        for (int rowID = 0; rowID < row * column; rowID++) {
            mask_in_float.push_back(data[rowID] ? 1.f : 0.f);
        }
        float* dataF = mask_in_float.data();
        for (int y = 0; y < column / 2; y++) {
            float* line_y = dataF + y * row;
            float* mirrored_y = dataF + (column - 1 - y) * row;
            for (int x = 0; x < row; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }

        fs::path folder = outputDir / fileType;
        if (!fs::exists(folder)) { fs::create_directories(folder); }
        fs::path fileName = folder / (frameName + ".hdr");

#ifdef _WINDOWS
        stbi_write_hdr(fileName.string().c_str(), row, column, 1, dataF);
#else
        stbi_write_hdr(fileName.c_str(), row, column, 1, dataF);
#endif
    }
    void OutputWriter::writePose(Camera camera, const std::string& fileType, const std::string& frameName) {
        fs::path folder = outputDir / fileType;
        if (!fs::exists(folder)) {
            fs::create_directories(folder);
        }
        fs::path fileName = folder / (frameName + ".txt");
        std::ofstream poseFile;
#ifdef _WINDOWS
        poseFile.open(fileName.string().c_str());
#else
        poseFile.open(fileName.c_str());
#endif
        poseFile << camera.at.x << " " << camera.at.y << " " << camera.at.z << " " << std::endl;
        poseFile << camera.from.x << " " << camera.from.y << " " << camera.from.z << " " << std::endl;
        poseFile.close();
    }

    void OutputWriter::write_buffer(std::vector<vec4f> color, const std::string& fileType, const std::string& frameName) {
        if (color.size() != row * column) {
            throw std::runtime_error("Output Writter: Buffer size and row/column mismtach");
        }
        writeRGBA((float*)color.data(), fileType, frameName);
    }

    void OutputWriter::write_wavelet(const std::string& frameName) {
        writeRGBA((float*)frame->predictedIndirect.data(), "predictedIndirect", frameName);
        writeRGBA((float*)frame->predictedIndirectWithDirect.data(), "predictedWithDirect", frameName);
    }

    void OutputWriter::write_final_color(const std::string& frameName) {
        writeRGBA((float*)frame->FinalColor.data(), "render", frameName);
    }
    void OutputWriter::write_indirect_color(const std::string& frameName) {
        writeRGBA((float*)frame->IndirectColor.data(), "renderIndirect", frameName);
    }
    
    void OutputWriter::write_aux(const std::string& frameName, Camera camera) {
        writeAccurateRGBA((int*)frame->FirstHitVertexID.data(), "VertexID", frameName);
        writeAccurateRGBA((float*)frame->FirstHitBary.data(), "Bary", frameName);

        writeAccurateRGBA((float*)frame->FirstHitUVCoord.data(), "UV", frameName);
        writeRGBA((float*)frame->worldNormalBuffer.data(), "normal", frameName);
        writeRGBA((float*)frame->cameraNormalBuffer.data(), "cameraNormal", frameName);
        writeRGBA((float*)frame->firstHitReflecDir.data(), "reflection_dir", frameName);

        writeRGBA((float*)frame->firstHitKd.data(), "firstKd", frameName);
        writeRGBA((float*)frame->firstHitKs.data(), "firstKs", frameName);
        writeOneChannel((float*)frame->roughness.data(), "roughness", frameName);

        writeAccurateRGBA((float*)frame->firstHitPos.data(), "firstHitPos", frameName);
        writeMask(frame->environmentMapFlag, "mask", frameName);
        writePose(camera, "pose", frameName);
    }

    void OutputWriter::write_frame(const std::string& frameName) {
        writeRGBA((float*)frame->FinalColor.data(), "render", frameName);
        writeRGBA((float*)frame->DirectColor.data(), "renderDirect", frameName);
        writeRGBA((float*)frame->IndirectColor.data(), "renderIndirect", frameName);
    }

    void OutputWriter::write_SH(const std::string& frameName) {
        writeRGBA((float*)frame->FinalColor.data(), "SHIndirect", frameName);
    }
}