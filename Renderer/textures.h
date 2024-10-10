#pragma once
#include <map>
#include <vector>
#include <filesystem>
#include <optix7.h>

#include"gdt/math/AffineSpace.h"

namespace nert_renderer {
	struct Texture {
		Texture() = default;
		Texture(gdt::vec2i size) {
			resolution = size;
		}
		~Texture() {
			/*if (pixelArray != nullptr)
				CUDA_CHECK(Free(pixelArray));*/
		}
		// resolution is width x height. row major order first
		gdt::vec2i     resolution{ -1 };

		// We own the CUDA version of this texture as well! 
		// Backend renderer does not
		cudaArray_t pixelArray = nullptr;
		cudaTextureObject_t cuda_tex = 0;

		virtual cudaTextureObject_t toCUDA() {
			cudaTextureObject_t cuda_tex = 0;
			return cuda_tex;
		}
		
		virtual int loadTexture(const std::filesystem::path fileName) {
			return -1;
		}

		protected:
		cudaTextureObject_t toCUDA_Helper(int formatSize, cudaChannelFormatDesc channel_desc, cudaTextureFilterMode filterMode,
			cudaTextureReadMode readMode, void* pixels);
	};

	struct HDRTexture : Texture {
		HDRTexture() = default;
		// We do deep copy
		HDRTexture(std::vector<gdt::vec3f>& pixels, gdt::vec2i size) : Texture(size) {
			pixel.clear();
			for (auto& x : pixels) {
				pixel.emplace_back(x);
			}
		}

		std::vector<gdt::vec3f> pixel;

		int loadTexture(const std::filesystem::path fileName);
		std::shared_ptr<HDRTexture> get_submatrix(int from_c, int from_r, int size_c, int size_r);

		// Will linearly interpolate
		cudaTextureObject_t toCUDA() {
			std::vector<float> pixelArrayPadded = toCUDA_padding();
			return toCUDA_Helper(sizeof(float), cudaCreateChannelDesc<float4>(), cudaFilterModeLinear,
				cudaReadModeElementType, pixelArrayPadded.data());
		}

		// Will remain as it
		cudaTextureObject_t toCUDA_Point() {
			std::vector<float> pixelArrayPadded = toCUDA_padding();
			return toCUDA_Helper(sizeof(float), cudaCreateChannelDesc<float4>(), cudaFilterModePoint,
				cudaReadModeElementType, pixelArrayPadded.data());
		}

		private:
			std::vector<float> toCUDA_padding();
	};

	struct EightBitTexture : Texture {
		EightBitTexture() = default;
		EightBitTexture(std::vector<uint32_t>& pixels, gdt::vec2i size) : Texture(size) {
			pixel.clear();
			for (auto& x : pixels) {
				pixel.emplace_back(x);
			}
		}
		std::vector<uint32_t> pixel;

		int loadTexture(const std::filesystem::path fileName);

		cudaTextureObject_t toCUDA() {
			return toCUDA_Helper(sizeof(uint8_t), cudaCreateChannelDesc<uchar4>(), cudaFilterModeLinear, 
				cudaReadModeNormalizedFloat, pixel.data());
		}
	};

	class textureManager {
	public:
		textureManager() {}
		int addTexture(const std::filesystem::path texturePath);

		std::vector<std::shared_ptr<EightBitTexture>> texture_list;
		std::map<std::string, int> knownTextures;
	};
}