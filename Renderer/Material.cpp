#include "Material.h"

#define readEntry(name, field) else if (type == "float" && entry == name) \
{field.val = stof(commands[i + 3]);i = i + 5;} \
else if (type == "texture" && entry == name) \
{std::string textureName = commands[i + 3];field.map_id = textureMap[textureName];i = i + 5;}

namespace nert_renderer {
	using namespace gdt;

	void Material::processMaterialPBRTFormat(std::vector<std::string>& commands, std::map<std::string, int> textureMap){
		for (size_t i = 7; i < commands.size() - 1; ) {
			std::string type = (commands[i]);
			std::string entry = (commands[i + 1]);
			if (type == "rgb" && (entry == "Ks"|| entry == "ks")) {
				ks.val = vec3f(stof(commands[i + 3]), stof(commands[i + 4]), stof(commands[i + 5]));
				i = i + 7;
			}
			else if (type == "rgb" && (entry == "Kd" || entry == "kd" || entry == "base_color")) {
				kd.val = vec3f(stof(commands[i + 3]), stof(commands[i + 4]), stof(commands[i + 5]));
				i = i + 7;
			}
			else if (type == "texture" && (entry == "Ks"|| entry == "ks")) {
				std::string textureName = commands[i + 3];
				ks.map_id = textureMap[textureName];
				i = i + 5;
			}
			else if (type == "texture" && (entry == "Kd" || entry == "kd" || entry == "base_color")) {
				std::string textureName = commands[i + 3];
				kd.map_id= textureMap[textureName];
				i = i + 5;
			}
			else if (type == "texture" && (entry == "normal")) {
				std::string textureName = commands[i + 3];
				normalMap.map_id= textureMap[textureName];
				i = i + 5;
			}

			readEntry("roughness", roughness)
			readEntry("specular_transmission", specular_transmission)
			readEntry("metallic", metallic)
			readEntry("subsurface", subsurface)
			readEntry("specular", specular)
			readEntry("specular_tint", specular_tint)
			readEntry("anisotropic", anisotropic)
			readEntry("sheen", sheen)
			readEntry("sheen_tint", sheen_tint)
			readEntry("clearcoat", clearcoat)
			readEntry("clearcoat_gloss", clearcoat_gloss)
			readEntry("eta", eta)

			else {
				i = i + 1;
			}
		}
	}
}