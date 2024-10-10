void Scene::randomizeOBJs(const std::string& texturePath, std::mt19937& rnd) {
   // load more texture
   std::filesystem::path p1(texturePath);
   std::map<std::string, int> knownTextures;
   for (const auto& entry : std::filesystem::directory_iterator(p1)) {
       loadTexture(texture_list, knownTextures, entry.path().filename().string(), entry.path().parent_path().string());
   }
   int textureCount = texture_list.size();
   AffineSpace3f a(one, vec3f(0));
   // randomize vertex
   
   for (auto& model : additional_model_list) {
       for (auto& mesh : model->meshes) {
           vec3f lower = bounds.lower;
           vec3f diff = bounds.upper - lower;
           vec3f pos = vec3f(random(rnd) * diff.x, random(rnd) * diff.y, random(rnd) * diff.z) + lower;

           vec3f scale = vec3f(random(rnd) * 1, random(rnd) * 1, random(rnd) * 1);
           vec3f rotation_axis = normalize(vec3f(random(rnd) * 2 - 1, random(rnd) * 2 - 1, random(rnd) * 2 - 1));
           float roation = random(rnd) * 2 * M_PI;
           for (int i = 0; i < mesh->vertex.size(); i++) {
               vec3f old = mesh->vertex[i];
               old = xfmPoint(a.rotate(rotation_axis, roation), old);
               old = xfmPoint(a.scale(scale), old);
               old = xfmPoint(a.translate(pos), old);
               mesh->vertex[i] = old;
           }
           // material
           mesh->mat->randomize(textureCount, rnd);
       }

   }

}
void Scene::addRandomizeLight(std::mt19937& rnd) {
    vec3f lower = bounds.lower;
    vec3f diff = bounds.upper- lower;
    vec3f pos = vec3f(random(rnd) * diff.x, random(rnd) * diff.y, random(rnd) * diff.z) + lower;
    vec3f power = vec3f(random(rnd) * light_list[0].power.x / 4);
    QuadLight Q;
    Q.origin = pos;
    Q.du = sampleSphere(rnd);
    Q.dv = sampleSphere(rnd);
    Q.power = power;
    light_list.push_back(Q);
}

void Scene::loadAdditionalScene(const std::string& modelPath, int numberOfObjects, std::mt19937& rnd) {
    std::filesystem::path p1(modelPath);
    std::vector<Instance*> modelList;
    for (const auto& entry : std::filesystem::directory_iterator(p1)) {
        modelList.push_back(loadOBJ(entry.path().string()+"/1.obj", texture_list, knownTextures));
    }
    
    for (int i = 0; i < numberOfObjects; i++) {
        std::uniform_int_distribution<> modelDist(0, modelList.size() - 1);
        int item = modelDist(rnd);
        additional_model_list.push_back(modelList[item]);
        modelList.erase(modelList.begin() + item);
    }

}

inline int number_of_files_in_directory(const std::string& path) {
    int count = 0;
    std::filesystem::path p1(path);
    for (auto& p : std::filesystem::directory_iterator(p1)) ++count;
    return count;

}

inline float random(std::mt19937& rnd) {
    std::uniform_real_distribution<> dist(0, 1);
    return dist(rnd);
}

inline vec3f sampleSphere(std::mt19937 rnd) {
    const float radius = sqrtf(random(rnd));
    const float theta = 2.f * M_PI * random(rnd);
    float x = radius * cosf(theta);
    float y = radius * sinf(theta);
    float z = sqrtf(fmax(0.f, 1.f - x * x - y * y));
    if (random(rnd) > 0.5) {
        z = -z;
    }
    return vec3f(x, y, z);
}


	
void Material::randomize(int textureCount, std::mt19937& rnd) {
	std::uniform_int_distribution<> textureDist(0, textureCount - 1);
	std::uniform_int_distribution<> matTypeDist(0, 5);
	type = MatType(matTypeDist(rnd));
	if (type == MatType::mirror && random(rnd) > 0.05) {
		type = MatType::uber;
	}
	if (type == MatType::mirror) return;

	roughness_square = pow(random(rnd), 2);
	std::uniform_int_distribution<> copperDist(0, 55);
	if (type == MatType::metal) {
		kd = vec3f(CopperN[copperDist(rnd)], CopperN[copperDist(rnd)], CopperN[copperDist(rnd)]);
		ks = vec3f(CopperK[copperDist(rnd)], CopperK[copperDist(rnd)], CopperK[copperDist(rnd)]);
	}
	else {
		if (has_kd_map || has_ks_map) {
			if (random(rnd) > 0.1) {
				kd_map_id = random(rnd) > 0.7 ? kd_map_id : textureDist(rnd);
			}
			else {
				has_kd_map = false;
				kd = vec3f(random(rnd), random(rnd), random(rnd));
			}
			if (random(rnd) > 0.1) {
				ks_map_id = random(rnd) > 0.7 ? ks_map_id : textureDist(rnd);
			}
			else {
				has_ks_map = false;
				ks = vec3f(random(rnd), random(rnd), random(rnd));
			}
		}
		else {
			if (random(rnd) > 0.9) {
				kd = vec3f(random(rnd), random(rnd), random(rnd));
				ks = vec3f(random(rnd), random(rnd), random(rnd));
			}
			else {
				kd_map_id =  textureDist(rnd);
				has_kd_map = true;
				ks_map_id = textureDist(rnd);
				has_ks_map = true;
			}

		}
	}
}