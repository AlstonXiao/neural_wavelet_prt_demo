#pragma once

#include <vector>
#include <filesystem>
#include "gdt/math/AffineSpace.h"

#include "Material.h"
#include "textures.h"

namespace nert_renderer {
    struct TriangleMesh {
        std::vector<gdt::vec3f> vertex;
        std::vector<gdt::vec3f> normal;
        std::vector<gdt::vec2f> texcoord;
        std::vector<gdt::vec3i> index;

        std::shared_ptr<Material> mat;
    };

    struct Instance {
        std::vector<std::shared_ptr<TriangleMesh>> meshes;
        gdt::box3f bounds; // Local to the mesh

        void addMesh(std::shared_ptr<TriangleMesh> mesh) {
            meshes.push_back(mesh);
            for (auto vtx : mesh->vertex)
                bounds.extend(vtx);
        }

        void addMeshFromFile(
            const std::string& type, 
            const std::filesystem::path& path, 
            std::shared_ptr<Material> mat, 
            textureManager& texture_manager
        );
    };

    class instanceNode {
    public:
        std::vector<std::shared_ptr<instanceNode>> childrenInstanceNode;
        std::vector<int> instanceIDs;
        gdt::AffineSpace3f transformation = gdt::AffineSpace3f(gdt::one);
    };
}
