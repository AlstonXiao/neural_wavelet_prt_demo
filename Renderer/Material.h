#pragma once
#include "gdt/math/AffineSpace.h"
#include <vector>
#include <map>

namespace nert_renderer {
	enum class MatType { roughPlastic, DisneyBSDF, cookTorrance, mirror, simple_ggx };

	// Map ID = -1 means no texture
	struct floatEntry {
        float val;
		int map_id;
    };

    struct vec3Entry {
        gdt::vec3f val;
        int map_id;
    };

	class Material {
	public:
		const float CopperN[56] = {
		1.400313, 1.38,  1.358438, 1.34,  1.329063, 1.325, 1.3325,   1.34,
		1.334375, 1.325, 1.317812, 1.31,  1.300313, 1.29,  1.281563, 1.27,
		1.249062, 1.225, 1.2,      1.18,  1.174375, 1.175, 1.1775,   1.18,
		1.178125, 1.175, 1.172812, 1.17,  1.165312, 1.16,  1.155312, 1.15,
		1.142812, 1.135, 1.131562, 1.12,  1.092437, 1.04,  0.950375, 0.826,
		0.645875, 0.468, 0.35125,  0.272, 0.230813, 0.214, 0.20925,  0.213,
		0.21625,  0.223, 0.2365,   0.25,  0.254188, 0.26,  0.28,     0.3 };

		const float CopperK[56] = {
		1.662125, 1.687, 1.703313, 1.72,  1.744563, 1.77,  1.791625, 1.81,
		1.822125, 1.834, 1.85175,  1.872, 1.89425,  1.916, 1.931688, 1.95,
		1.972438, 2.015, 2.121562, 2.21,  2.177188, 2.13,  2.160063, 2.21,
		2.249938, 2.289, 2.326,    2.362, 2.397625, 2.433, 2.469187, 2.504,
		2.535875, 2.564, 2.589625, 2.605, 2.595562, 2.583, 2.5765,   2.599,
		2.678062, 2.809, 3.01075,  3.24,  3.458187, 3.67,  3.863125, 4.05,
		4.239563, 4.43,  4.619563, 4.817, 5.034125, 5.26,  5.485625, 5.717 };

		vec3Entry kd;
		vec3Entry ks;
		vec3Entry normalMap; // The val will be ignored

		floatEntry roughness;
		floatEntry specular_transmission;
		floatEntry metallic;
		floatEntry subsurface;
		floatEntry specular;
		floatEntry specular_tint;
		floatEntry anisotropic;
		floatEntry sheen;
		floatEntry sheen_tint;
		floatEntry clearcoat;
		floatEntry clearcoat_gloss;
		floatEntry eta;

		MatType type;

		Material(MatType type) : type(type) {
			init();
		}

		Material(std::string s_type) {
			if (s_type == "DisneyBSDF")
				type = MatType::DisneyBSDF;
			else if (s_type == "mirror")
				type = MatType::mirror;
			else if (s_type == "cookTorrance")
				type = MatType::cookTorrance;
			else if (s_type == "roughPlastic")
				type = MatType::roughPlastic;
			else if (s_type == "simple_ggx")
				type = MatType::simple_ggx;
			else
				throw std::runtime_error("Unknown material: " + s_type);
			init();
		}

		void init() {
			kd.val = gdt::vec3f(0.25, 0.25, 0.25);
			ks.val = gdt::vec3f(0.25, 0.25, 0.25);

			roughness.val = 0.5;
			specular_transmission.val = 0.f;
			metallic.val = 0.f;
			subsurface.val = 0.f;
			specular.val = 0.5f;
			specular_tint.val = 0.f;
			anisotropic.val = 0.f;
			sheen.val = 0.f;
			sheen_tint.val = 0.5f;
			clearcoat.val = 0.f;
			clearcoat_gloss.val = 1.f;

			float intIOR = 1.49;
			float extIOR = 1.000277;
			eta.val = intIOR / extIOR;

			kd.map_id = -1;
			ks.map_id = -1;
			normalMap.map_id = -1;

			roughness.map_id = -1;
			specular_transmission.map_id = -1;
			metallic.map_id = -1;
			subsurface.map_id = -1;
			specular.map_id = -1;
			specular_tint.map_id = -1;
			anisotropic.map_id = -1;
			sheen.map_id = -1;
			sheen_tint.map_id = -1;
			clearcoat.map_id = -1;
			clearcoat_gloss.map_id = -1;
			eta.map_id = -1;
		}
		void processMaterialPBRTFormat(std::vector<std::string>& commands, std::map<std::string, int> textureMap);
	};

	class roughPlasticMaterial : public Material {
	public:
		roughPlasticMaterial() : Material(MatType::roughPlastic) {
			kd.val = gdt::vec3f(0.5, 0.5, 0.5);
			ks.val = gdt::vec3f(1, 1, 1);
			roughness.val = 0.1f;
		}
	};

	class DisneyBSDFMaterial : public Material {
	public:
		DisneyBSDFMaterial() : Material(MatType::DisneyBSDF) {
			kd.val = gdt::vec3f(0.5, 0.5, 0.5);
			roughness.val = 0.5f;
		}
	};

	class cookTorranceMaterial : public Material {
	public:
		cookTorranceMaterial() : Material(MatType::cookTorrance) {
			kd.val = gdt::vec3f(0.5, 0.5, 0.5);
			ks.val = gdt::vec3f(0.5, 0.5, 0.5);
			roughness.val = 1.f;
		}
	};

	class mirrorMaterial : public Material {
	public:
		mirrorMaterial() : Material(MatType::mirror) {}
	};

	class simple_ggxMaterial : public Material {
	public:
		simple_ggxMaterial() : Material(MatType::simple_ggx) {
			kd.val = gdt::vec3f(0.5, 0.5, 0.5);
			roughness.val = 1.f;
		}
	};
}