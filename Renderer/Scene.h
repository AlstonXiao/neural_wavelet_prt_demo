#pragma once
#include<vector>
#include<map>
#include "gdt/math/AffineSpace.h"

#include "Model.h"
#include "EnvironmentMap.h"

namespace nert_renderer {

	struct Camera {
		/*! camera position - *from* where we are looking */
		gdt::vec3f from;
		/*! which point we are looking *at* */
		gdt::vec3f at;
		/*! general up-vector */
		gdt::vec3f up;
	};

	class Scene {

	public:
		Scene(bool debug = false) : m_debug(debug) {
			initialCamera = { /*from*/gdt::vec3f(0, 0, 1.f),
				/* at */gdt::vec3f(0, 0, 0),
				/* up */gdt::vec3f(0.f,1.f,0.f) };
			root = std::make_shared<instanceNode>();
		}

		// Geometry
		std::shared_ptr<instanceNode> root;
		std::vector<std::shared_ptr<Instance>> instance_list;

		// Apperances
		std::vector<std::shared_ptr<Material>> material_list;
		textureManager texture_manager;

		// Lighting
		std::vector<std::shared_ptr<Environment_Map>> EnvMap;

		gdt::box3f bounds;
		Camera initialCamera;

  		void loadPBRT(const std::string& pbrtFile);
		void computeBounds();

	private:
		bool m_debug;
	};
}