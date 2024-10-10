#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

#define TINYPLY_IMPLEMENTATION
#include "3rdParty/tinyply.h"

#include <set>


namespace std {
    // This is needed for some reason.
    inline bool operator<(const tinyobj::index_t &a, const tinyobj::index_t &b) {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;
    
        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;
    
        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;
    
        return false;
    }
}

namespace nert_renderer {
    using namespace std;
    using namespace gdt;

    /*! find vertex with given position, normal, texcoord, and return
        its vertex ID, or, if it doesn't exit, add it to the mesh, and
        its just-created index */
    int addVertex(shared_ptr<TriangleMesh> mesh,
                    tinyobj::attrib_t &attributes,
                    const tinyobj::index_t &idx,
                    std::map<tinyobj::index_t,int> &knownVertices)
    {
        if (knownVertices.find(idx) != knownVertices.end())
        return knownVertices[idx];

        const vec3f *vertex_array   = (const vec3f*)attributes.vertices.data();
        const vec3f *normal_array   = (const vec3f*)attributes.normals.data();
        const vec2f *texcoord_array = (const vec2f*)attributes.texcoords.data();
        
        int newID = (int)mesh->vertex.size();
        knownVertices[idx] = newID;

        mesh->vertex.push_back(vertex_array[idx.vertex_index]);
        if (idx.normal_index >= 0) {
            while (mesh->normal.size() < mesh->vertex.size())
                mesh->normal.push_back(normal_array[idx.normal_index]);
        }
        if (idx.texcoord_index >= 0) {
            while (mesh->texcoord.size() < mesh->vertex.size())
                mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
        }

        // just for sanity's sake:
        if (mesh->texcoord.size() > 0)
        mesh->texcoord.resize(mesh->vertex.size());
        // just for sanity's sake:
        if (mesh->normal.size() > 0)
        mesh->normal.resize(mesh->vertex.size());
        
        return newID;
    }

    // May load multiple trimesh to an instance
    void loadOBJ(Instance* model, const std::filesystem::path &objFile, shared_ptr<Material> mat, textureManager& texture_manager)
    {
        const std::filesystem::path modelDir = objFile.parent_path();
        
        tinyobj::attrib_t attributes;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string err = "";

        bool readOK
        = tinyobj::LoadObj(&attributes,
                            &shapes,
                            &materials,
                            &err,
                            &err,
                            objFile.string().c_str(),
                            modelDir.string().c_str(),
                            /* triangulate */true);
        if (!readOK) {
            throw std::runtime_error("Could not read OBJ model from "+ objFile.string()+ " : "+err);
        }

        if (materials.empty() && mat == nullptr)
            throw std::runtime_error("could not parse materials ...");

        // std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
        // Select one shape first
        for (int shapeID=0;shapeID<(int)shapes.size();shapeID++) {
            tinyobj::shape_t &shape = shapes[shapeID];

            std::set<int> materialIDs;
            // Finding how many materials in one shape
            for (auto faceMatID : shape.mesh.material_ids)
                materialIDs.insert(faceMatID);
            
            std::map<tinyobj::index_t,int> knownVertices;
            
            // Add a mesh for each material
            for (int materialID : materialIDs) {
                auto mesh = shared_ptr<TriangleMesh>(new TriangleMesh());
                
                for (int faceID=0;faceID<shape.mesh.material_ids.size();faceID++) {
                    if (shape.mesh.material_ids[faceID] != materialID) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3*faceID+0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3*faceID+1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3*faceID+2];
                    
                    vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                                addVertex(mesh, attributes, idx1, knownVertices),
                                addVertex(mesh, attributes, idx2, knownVertices));
                    mesh->index.push_back(idx);

                    if (mat == nullptr){
                        mesh->mat = shared_ptr<cookTorranceMaterial>(new cookTorranceMaterial());

                        if (materialID >= 0) {
                            mesh->mat->kd.val = (const vec3f&)materials[materialID].diffuse;
                            mesh->mat->ks.val = (const vec3f&)materials[materialID].specular;
                            mesh->mat->roughness.val = pow((materials[materialID].shininess == 0) ? 0. : (1.f / materials[materialID].shininess), 2);

                            mesh->mat->kd.map_id = texture_manager.addTexture(modelDir / materials[materialID].diffuse_texname);
                            mesh->mat->ks.map_id = texture_manager.addTexture(modelDir / materials[materialID].specular_texname); 
                            mesh->mat->roughness.map_id = texture_manager.addTexture(modelDir / materials[materialID].specular_highlight_texname);  
                            mesh->mat->normalMap.map_id = texture_manager.addTexture(modelDir / materials[materialID].bump_texname);
                        }
                    } else {
                        mesh->mat = mat;
                    }
                }

                if (!mesh->vertex.empty()) {
                    model->meshes.push_back(mesh);
                    for (auto vtx : mesh->vertex)
                        model->bounds.extend(vtx);
                }
            }
        }
    }

    // load one trimesh, without materials
    shared_ptr<TriangleMesh> loadPly(const std::string& plyfile) {
        std::unique_ptr<std::istream> file_stream;
        std::vector<uint8_t> byte_buffer;
        using namespace tinyply;
        try {
            // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a 
            // stream is a net win for parsing speed, about 40% faster. 

            file_stream.reset(new std::ifstream(plyfile, std::ios::binary));
            if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + plyfile);

            file_stream->seekg(0, std::ios::end);
            const float size_mb = file_stream->tellg() * float(1e-6);
            file_stream->seekg(0, std::ios::beg);

            PlyFile file;
            file.parse_header(*file_stream);

            // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
            // See examples below on how to marry your own application-specific data structures with this one. 
            std::shared_ptr<PlyData> vertices, normals, texcoords, faces;

            // The header information can be used to programmatically extract properties on elements
            // known to exist in the header prior to reading the data. For brevity of this sample, properties 
            // like vertex position are hard-coded: 
            try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }, 3); }
            catch (const std::exception& e) {}//std::cerr << "tinyply exception: " << e.what() << std::endl; }

            try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }, 3); }
            catch (const std::exception& e) {}//std::cerr << "tinyply exception: " << e.what() << std::endl; }

            try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }, 2); }
            catch (const std::exception& e) {}//std::cerr << "tinyply exception: " << e.what() << std::endl; }

            // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
            // arbitrary ply files, it is best to leave this 0. 
            try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
            catch (const std::exception& e) {}//std::cerr << "tinyply exception: " << e.what() << std::endl; }

            file.read(*file_stream);

            auto mesh = shared_ptr<TriangleMesh>(new TriangleMesh());
            
            const size_t numVerticesBytes = vertices->buffer.size_bytes();
            std::vector<vec3f> verts(vertices->count);
            std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
            mesh->vertex = verts;

            if (normals) {
                const size_t numNormalBytes = normals->buffer.size_bytes();
                std::vector<vec3f> norms(normals->count);
                std::memcpy(norms.data(), normals->buffer.get(), numNormalBytes);
                mesh->normal = norms;
            } 

            if (texcoords) {
                const size_t numTextBytes = texcoords->buffer.size_bytes();
                std::vector<vec2f> texts(texcoords->count);
                std::memcpy(texts.data(), texcoords->buffer.get(), numTextBytes);
                mesh->texcoord = texts;
            }

            const size_t numFaceBytes = faces->buffer.size_bytes();
            std::vector<vec3i> facess(faces->count);
            std::memcpy(facess.data(), faces->buffer.get(), numFaceBytes);
            
            mesh->index = facess;
            return mesh;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
        }
        return nullptr;
    }

    void Instance::addMeshFromFile(const string& type, const std::filesystem::path& path, shared_ptr<Material> mat, textureManager& texture_manager){
        if (type == "plymesh") {
            if (mat == nullptr) {
                std::cerr << path.string() <<": ply mesh does not contain material information, using default cookTorranceMaterial" << std::endl;
                mat = std::shared_ptr<cookTorranceMaterial>(new cookTorranceMaterial());
            }
            shared_ptr<TriangleMesh> mesh = loadPly(path.string().c_str());
            mesh->mat = mat;
            meshes.push_back(mesh);
            for (auto vtx : mesh->vertex)
                bounds.extend(vtx);
        }
        else if (type == "objmesh") {
            loadOBJ(this, path.string(), mat, texture_manager);
        }
        else {
            std::cerr <<  path.string() <<": Format not supported" << std::endl;  
        }
    }

}
