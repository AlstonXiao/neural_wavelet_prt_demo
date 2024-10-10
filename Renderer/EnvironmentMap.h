#pragma once
#include <vector>
#include <map>
#include <random>

#include "table_dist.h"
#include "textures.h"

namespace nert_renderer {

	class Environment_Map {

	public:
		Environment_Map(bool debug = false) {
			m_debug = debug;
			latlongMap = NULL;
		}
		Environment_Map(std::string inFileName, bool sh_coeff = false, bool debug = false);
		Environment_Map(std::string envMapFolder, std::string envMapName, bool debug = false);
		~Environment_Map() {}

		std::vector<std::shared_ptr<HDRTexture>> faces;
		std::shared_ptr<HDRTexture> latlongMap; // only used in direct shader
		std::vector<int> topWaveletCoefficients;
		std::vector<float> topWaveletStrength;
		std::vector<float> shCoefficients;

		// PDF sampling
		std::vector<double> face_luminances;
		TableDist1D face_dist;
		std::vector<TableDist2D> sampling_dist;

	private:
		void loadCubeMap(std::filesystem::path envFileName);
		void initCubeFaces(std::shared_ptr<HDRTexture> allFaces);
		void init_sampling_dist(int w, int h);
		void load_sh(std::filesystem::path sh_path);
		bool m_debug;

	};

}