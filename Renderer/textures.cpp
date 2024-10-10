#include "textures.h"
#include "3rdParty/stb_image.h"
#include "HDRloader.h"

namespace nert_renderer {
	using namespace gdt;
	namespace fs = std::filesystem;
	cudaTextureObject_t Texture::toCUDA_Helper(int formatSize, cudaChannelFormatDesc channel_desc, cudaTextureFilterMode filterMode,
		cudaTextureReadMode readMode, void* pixels) {
		cudaResourceDesc res_desc = {};

		int32_t width = resolution.x;
		int32_t height = resolution.y;
		int32_t numComponents = 4;

		// Width is the column number
		int32_t pitch = width * numComponents * formatSize;
		CUDA_CHECK(MallocArray(&pixelArray,
			&channel_desc,
			width, height));

		CUDA_CHECK(Memcpy2DToArray(pixelArray,
			/* offset */0, 0,
			pixels,
			pitch, pitch, height,
			cudaMemcpyHostToDevice));

		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = pixelArray;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = filterMode;
		tex_desc.readMode = readMode;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 99;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.borderColor[0] = 1.0f;
		tex_desc.sRGB = 2.2;

		// Create texture object
		cuda_tex = 0;
		CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
		return cuda_tex;
	}

	int HDRTexture::loadTexture(const fs::path fileName) {
		HDRLoaderResult result;
		bool success = HDRLoader::load(fileName.string().c_str(), result);
		if (!success) {
			std::string result = success ? "success" : "failed";
			std::cout << "Loading Envmap texture: " << fileName.string() << " " << result << std::endl;
			return -1;
		}

		pixel.resize(result.width * result.height);
		for (int u = 0; u < result.height; ++u) {
			for (int v = 0; v < result.width; ++v) {
				pixel[u * result.width + v] = result.get(u, v);
				// pixel[u * result.width + v] = result.get(u, v) * 64*64*6;
			}
		}

		// NOTE: height is up&down, width is left&right
		resolution = vec2i(result.width, result.height);
		return 0;
	}

	std::shared_ptr<HDRTexture> HDRTexture::get_submatrix(int from_c, int from_r, int size_c, int size_r) {
		std::vector<vec3f> submatrix;
		for (int ii = from_r; ii < from_r + size_r; ++ii) {
			for (int jj = from_c; jj < from_c + size_c; ++jj) {
				submatrix.push_back(pixel[ii * resolution[0] + jj]);
			}
		}
		// width, height, which means column, row
		vec2i dims{ size_c, size_r };
		std::shared_ptr<HDRTexture> ret (new HDRTexture(submatrix, dims));
		return ret;
	}

	std::vector<float> HDRTexture::toCUDA_padding() {
		int32_t width = resolution.x;
		int32_t height = resolution.y;
		int32_t numComponents = 4;
		std::vector<float> pixelArrayPadded(width* numComponents* height);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				pixelArrayPadded[i * width * numComponents + j * numComponents + 0] = pixel[i * width + j].x;
				pixelArrayPadded[i * width * numComponents + j * numComponents + 1] = pixel[i * width + j].y;
				pixelArrayPadded[i * width * numComponents + j * numComponents + 2] = pixel[i * width + j].z;
				pixelArrayPadded[i * width * numComponents + j * numComponents + 3] = 1.f;
			}
		}
		return pixelArrayPadded;
	}

	int EightBitTexture::loadTexture(const fs::path fileName) {
		vec2i res;
		int   comp;
		unsigned char* image = stbi_load(fileName.string().c_str(),
			&res.x, &res.y, &comp, STBI_rgb_alpha);

		if (image) {
			resolution = res;
			pixel.resize(resolution.x * resolution.y);
			auto pixel_raw = reinterpret_cast<uint32_t*>(image);

			/* iw - actually, it seems that stbi loads the pictures
			mirrored along the y axis - mirror them here */
			// Top left most to lower left most
			for (int y = 0; y < res.y; y++) {
				uint32_t* line_y = pixel_raw + y * res.x;
				uint32_t* mirrored_y = pixel_raw + (res.y - 1 - y) * res.x;
				int mirror_y = res.y - 1 - y;
				for (int x = 0; x < res.x; x++) {
					pixel[y * res.x + x] = mirrored_y[x];
				}
			}
			delete[] image;
		}
		else {
			std::cout << GDT_TERMINAL_RED
				<< "Could not load texture from " << fileName << "!"
				<< GDT_TERMINAL_DEFAULT << std::endl;
			return -1;
		}
		return 0;
	}

	int textureManager::addTexture(const fs::path texturePath) {
		if (!fs::exists(texturePath))
			return -1;
		std::string absoluteTexturePath = fs::absolute(texturePath).string();
		if (knownTextures.find(absoluteTexturePath) != knownTextures.end())
			return knownTextures[absoluteTexturePath];

		std::shared_ptr<EightBitTexture> newTexture (new EightBitTexture);
		int result = newTexture->loadTexture(absoluteTexturePath);

		if (result == 0) {
			int textureID = (int)texture_list.size();
			texture_list.emplace_back(std::move(newTexture));
			knownTextures[absoluteTexturePath] = textureID;
			return textureID;
		}
		else {
			return -1;
		}
	}
}