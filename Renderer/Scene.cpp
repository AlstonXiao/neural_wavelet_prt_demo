
#include "Scene.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include "files.h"

namespace nert_renderer {
    using namespace std;
    using namespace gdt;
    namespace fs = std::filesystem;


    inline AffineSpace3f read_rotation(int startingIndex, std::vector<string>& commands) {
        AffineSpace3f a(one);
        a.l.vx.x = stof(commands[startingIndex + 0]);
        a.l.vx.y = stof(commands[startingIndex + 1]);
        a.l.vx.z = stof(commands[startingIndex + 2]);

        a.l.vy.x = stof(commands[startingIndex + 4]);
        a.l.vy.y = stof(commands[startingIndex + 5]);
        a.l.vy.z = stof(commands[startingIndex + 6]);

        a.l.vz.x = stof(commands[startingIndex + 8]);
        a.l.vz.y = stof(commands[startingIndex + 9]);
        a.l.vz.z = stof(commands[startingIndex + 10]);

        a.p.x = stof(commands[startingIndex + 12]);
        a.p.y = stof(commands[startingIndex + 13]);
        a.p.z = stof(commands[startingIndex + 14]);

        return a;
    }

    void Scene::loadPBRT(const std::string& pbrtFile) {
        std::map<std::string, std::shared_ptr<Material>> materialMap;
        std::map<std::string, int> textureMap;
        std::map<std::string, std::shared_ptr<instanceNode>> instanceMap;

        fs::path pbrtPath(pbrtFile);
        std::shared_ptr<Material> currentMat(new roughPlasticMaterial());
        fs::path modelDir = pbrtPath.parent_path();

        std::ifstream infile(pbrtPath.c_str());
        std::string line;
        bool skip = false;
        bool warningFlag = false;
        while (std::getline(infile, line)) {
            std::vector<std::string> commands;
            line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
            tokenize(line, ' ', commands);
            if (commands.size() == 0) continue;
            removeQuote(commands);

            if (commands[0] == "Camera") {
                initialCamera.from.x = stof(commands[4]);
                initialCamera.from.y = stof(commands[5]);
                initialCamera.from.z = stof(commands[6]);

                initialCamera.at.x = stof(commands[10]);
                initialCamera.at.y = stof(commands[11]);
                initialCamera.at.z = stof(commands[12]);
            }

            if (commands[0] == "WorldTransformation") {
                root->transformation = read_rotation(2, commands);
            }

            // This is for area light,
            // And we dont support area light as for now.
            // If you have attribute, please pre-process the document first
            if (commands[0] == "AttributeBegin") {
                std::cout << "WARNING! Contents in the AttributeBegin and AttributeEnd will be ignored. Please pre-process the document using python first!" << std::endl;
                skip = true;
            }
            if (commands[0] == "AttributeEnd") {
                skip = false;
            }
            if (skip) continue;

            // Reading Texture
            if (commands[0] == "Texture") {
                string name = commands[1];
                string type = commands[2];
                string file = commands[7];

                if (type != "spectrum") {
                    std::cout << "WARNING! Unsupported texture type: " << type << ". This texture will be ignored" << std::endl;
                    continue;
                }

                int texture_id = texture_manager.addTexture(modelDir / file);
                textureMap[name] = texture_id;
            }

            // Reading Material
            if (commands[0] == "MakeNamedMaterial") {
                string name = commands[1];
                string type = commands[5];

                std::shared_ptr<Material> mat(new Material(type));
                mat->processMaterialPBRTFormat(commands, textureMap);
                materialMap[name] = mat;
               
            }

            // select the current Material for all incoming shapes
            if (commands[0] == "NamedMaterial") {
                currentMat = materialMap[commands[1]];
            }

            // Create a new instance node with single instance belongs to it
            // And attach to root
            if (commands[0] == "Shape") {
                string type = commands[1];
                string file = commands[5];

                std::shared_ptr<Instance> newInstance(new Instance);
                std::shared_ptr<instanceNode> newNode(new instanceNode);
            
                newInstance->addMeshFromFile(type, modelDir / fs::path(file), currentMat, texture_manager);

                if (commands.size() > 7 && commands[7] == "Transform") {
                    newNode->transformation = read_rotation(8, commands);
                }
                newNode->instanceIDs.push_back(instance_list.size());
                instance_list.push_back(newInstance);

                root->childrenInstanceNode.push_back(newNode);
            }

            // Create a new instance node with single instance belongs to it
            // Store it, but not attach to root.
            if (commands[0] == "Object") {
                string type = commands[1];
                string file = commands[8];

                std::shared_ptr<Instance> newInstance(new Instance);
                std::shared_ptr<instanceNode> newNode(new instanceNode);

                newInstance->addMeshFromFile(type, modelDir / fs::path(file), currentMat, texture_manager);

                if (commands.size() > 10 && commands[10] == "Transform") {
                    newNode->transformation = read_rotation(11, commands);
                }

                newNode->instanceIDs.push_back(instance_list.size());
                instance_list.push_back(newInstance);

                string objectName = commands[4];
                instanceMap[objectName] = newNode;
            }

            // Using previously created instance. 
            // attach to root.
            // Note, we do not support instance tree more than 3 levels 
            // Root -> Instance Nodes -> Instance itself (but warped in a instance node).
            if (commands[0] == "ObjectInstance") {
                string objectName = commands[1];
                std::shared_ptr<instanceNode> newNode(new instanceNode);

                if (commands.size() > 2 && commands[2] == "Transform") {
                    newNode->transformation = read_rotation(3, commands);
                }
                newNode->childrenInstanceNode.push_back(instanceMap[objectName]);
                root->childrenInstanceNode.push_back(newNode);
            }
        }
        computeBounds();
        if (m_debug) std::cout << "Created a total of " << instance_list.size() << " instances" << std::endl;
        if (m_debug) std::cout << "The Bounds is " << bounds << std::endl;
    }

    box3f rotateBounds(box3f& original_bounds, AffineSpace3f& rotation) {
        vec3f bounds[8];
        bounds[0] = original_bounds.lower;
        bounds[1] = original_bounds.upper;

        bounds[2] = vec3f(bounds[0].x, bounds[1].y, bounds[0].z);
        bounds[3] = vec3f(bounds[0].x, bounds[0].y, bounds[1].z);
        bounds[4] = vec3f(bounds[0].x, bounds[1].y, bounds[1].z);

        bounds[5] = vec3f(bounds[1].x, bounds[1].y, bounds[0].z);
        bounds[6] = vec3f(bounds[1].x, bounds[0].y, bounds[1].z);
        bounds[7] = vec3f(bounds[1].x, bounds[0].y, bounds[0].z);

        box3f newBox;
        for (int i = 0; i < 8; i++)
            newBox.extend(xfmPoint(rotation, bounds[i]));
        return newBox;
    }

    void Scene::computeBounds() {
        std::vector<std::shared_ptr<instanceNode>> todoList;
        std::vector<AffineSpace3f> parentTransformation;

        todoList.push_back(root);
        AffineSpace3f baseSpace(one);
        parentTransformation.push_back(baseSpace);
        while (!todoList.empty()) {
            std::shared_ptr<instanceNode> currentInstance = todoList.back();
            todoList.pop_back();
            AffineSpace3f currentTransform = parentTransformation.back() * currentInstance->transformation;
            parentTransformation.pop_back();

            for (int childrenInstance = 0; childrenInstance < currentInstance->childrenInstanceNode.size(); childrenInstance++) {
                todoList.push_back(currentInstance->childrenInstanceNode[childrenInstance]);
                parentTransformation.push_back(currentTransform);
            }

            for (int instanceID : currentInstance->instanceIDs) {
                bounds.extend(rotateBounds(instance_list[instanceID]->bounds, currentTransform));
            }
        }
    }
}
