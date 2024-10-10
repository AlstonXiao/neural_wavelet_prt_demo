#include "EnvironmentMap.h"
#include <filesystem>
#include "npy.h"

namespace nert_renderer {
	using namespace gdt;
	namespace fs = std::filesystem;

	
	Environment_Map::Environment_Map(std::string inFileName, bool sh_coeffs, bool debug) : m_debug(debug) {
		fs::path envFileName(inFileName);

		if (sh_coeffs)
			load_sh(envFileName);
		else
			loadCubeMap(envFileName);

	}

	Environment_Map::Environment_Map(std::string envMapFolder, std::string envMapName, bool debug) : m_debug(debug) {
		fs::path env_path(envMapFolder);
		if (m_debug) std::cout << "Loading Envmap texture from folder: " << envMapFolder.c_str() << std::endl;

		if (!fs::is_directory(env_path)) 
			throw std::runtime_error("Invalid Env map Folder Name, it should not be a file");

		fs::path cubeMapPath = env_path / "cube" / (envMapName + ".hdr");
		if (m_debug) std::cout << "Loading Envmap cubemap texture from folder: " << cubeMapPath.string() << std::endl;

		fs::path shPath = env_path / "sh" / (envMapName + ".npy");
		if (m_debug) std::cout << "Loading SH coefficients from folder: " << shPath.string() << std::endl;

		fs::path latLongMapPath = env_path / "latlong" / (envMapName + ".hdr");
		if (m_debug) std::cout << "Loading Envmap latLongMap texture from folder: " << latLongMapPath.string() << std::endl;

		fs::path waveletCoeffsPath = env_path / "waveletcoeffs" / (envMapName + ".npy");
		if (m_debug) std::cout << "Loading Envmap wavelet Coeffs texture from folder: " << waveletCoeffsPath.string() << std::endl;

		fs::path waveletStrengthPath = env_path / "waveletstrength" / (envMapName + ".npy");
		if (m_debug) std::cout << "Loading Envmap waveletstrength texture from folder: " << waveletStrengthPath.string() << std::endl;
		
		loadCubeMap(cubeMapPath);

		if (fs::exists(latLongMapPath)) {
			latlongMap = std::make_shared<HDRTexture>();
			latlongMap->loadTexture(latLongMapPath);
		}
		else {
			latlongMap = nullptr;
		}

		if (fs::exists(shPath)) {
			load_sh(shPath);
		}

		if (fs::exists(waveletCoeffsPath) && fs::exists(waveletStrengthPath)){
			std::vector<unsigned long> shape; bool fortran_order; std::vector<int> coeffData; std::vector<float> StrengthData;

			npy::LoadArrayFromNumpy(waveletCoeffsPath.string(), shape, fortran_order, coeffData);
			for (auto coef : coeffData) 
				topWaveletCoefficients.push_back(coef);
			
			npy::LoadArrayFromNumpy(waveletStrengthPath.string(), shape, fortran_order, StrengthData);
			assert(shape[1] == 3);
			for (float strength : StrengthData) 
				topWaveletStrength.push_back(strength);
		}
	}

	void Environment_Map::load_sh(fs::path sh_path) {
		std::vector<unsigned long> shape; bool fortran_order;

		npy::LoadArrayFromNumpy(sh_path.string(), shape, fortran_order, shCoefficients);
		assert(shape[1] == 3);
		int shTerm = sqrt(shape[0]) - 1;
		assert((shTerm + 1) * (shTerm + 1) == shape[0]);
	}


	void Environment_Map::loadCubeMap(std::filesystem::path envFileName) {
		if (!fs::exists(envFileName))
			throw std::runtime_error("Invalid Env map File Name");

		if (m_debug) std::cout << "Loading Envmap texture from" << envFileName.string() << std::endl;

		std::shared_ptr<HDRTexture> allFaces(new HDRTexture());
		if (envFileName.extension() == ".hdr") {
			allFaces->loadTexture(envFileName);
		}
		else if (envFileName.extension() == ".npy") {
			std::vector<unsigned long> shape; bool fortran_order; std::vector<float> data;
			npy::LoadArrayFromNumpy(envFileName.string(), shape, fortran_order, data);

			vec3f* vecData = reinterpret_cast<vec3f*>(data.data());
			std::vector<vec3f> reversedData(shape[0] * shape[1]);

			// Python is top left as 0, 0
			for (int i = 0; i < shape[0]; i++) {
				for (int j = 0; j < shape[1]; j++)
					reversedData[i * shape[1] + j] = vecData[(shape[0] - i - 1) * shape[1] + j];
			}

			assert(shape[2] == 3);
			allFaces = std::make_shared<HDRTexture>(reversedData, vec2i(shape[1], shape[0]));
		}
		else
			throw std::runtime_error("Unknown type " + envFileName.extension().string());

		initCubeFaces(allFaces);
	}

	void Environment_Map::initCubeFaces(std::shared_ptr<HDRTexture> allFaces) {
		int face_dim = allFaces->resolution[1] / 3;
		assert(face_dim == allFaces->resolution[0] / 4);

		faces.clear();
		faces.push_back(allFaces->get_submatrix(2 * face_dim, 1 * face_dim, face_dim, face_dim));
		faces.push_back(allFaces->get_submatrix(0 * face_dim, 1 * face_dim, face_dim, face_dim));
		faces.push_back(allFaces->get_submatrix(1 * face_dim, 2 * face_dim, face_dim, face_dim));
		faces.push_back(allFaces->get_submatrix(1 * face_dim, 0 * face_dim, face_dim, face_dim));
		faces.push_back(allFaces->get_submatrix(1 * face_dim, 1 * face_dim, face_dim, face_dim));
		faces.push_back(allFaces->get_submatrix(3 * face_dim, 1 * face_dim, face_dim, face_dim));

		// construct table dist
		init_sampling_dist(face_dim, face_dim);
	}

	static double AreaElement( double x, double y) {
		return atan2(x * y, sqrt(x * x + y * y + 1));
	}
	
	double TexelCoordSolidAngle(double a_U, double a_V, int a_Size) {
		//scale up to [-1, 1] range (inclusive), offset by 0.5 to point to texel center.
		double U = (2.0f * ((double)a_U + 0.5f) / (double)a_Size ) - 1.0f;
		double V = (2.0f * ((double)a_V + 0.5f) / (double)a_Size ) - 1.0f;
		
		double InvResolution = 1.0f / a_Size;
	
		// U and V are the -1..1 texture coordinate on the current face.
		// Get projected area for this texel
		double x0 = U - InvResolution;
		double y0 = V - InvResolution; double x1 = U + InvResolution;
		double y1 = V + InvResolution;
		double SolidAngle = AreaElement(x0, y0) - AreaElement(x0, y1) - AreaElement(x1, y0) + AreaElement(x1, y1);
	
		return SolidAngle;
	}

	double luminance(vec3f rgb) {
		return rgb.x * 0.212671f + rgb.y * 0.715160f + rgb.z * 0.072169f;
	}

	void Environment_Map::init_sampling_dist(int w, int h) {
		// envmap is an std::vector<Texture*>

		// Construct the 2D grid for each of the six faces. The other 
		// faces should follow depending on if we can put them into an array.
		for (int f_index = 0; f_index < 6; ++f_index) {
			std::vector<double> f(w * h);
			int i = 0;
			double total_weighted_luminance = 0.0f;
			for (int x = 0; x < w; x++) {
				// We shift the grids by 0.5 pixels because we are approximating
				// a piecewise bilinear distribution with a piecewise constant
				// distribution. This shifting is necessary to make the sampling
				// unbiased, as we can interpolate at a position of a black pixel
				// and get a non-zero contribution.
				double u = (x + double(0.5)) / double(w);
				for (int y = 0; y < h; y++) {
					double v = (y + double(0.5)) / double(h);
					double solid_angle_x = TexelCoordSolidAngle(u, v, w);
					vec3f solid_angle (solid_angle_x, solid_angle_x, solid_angle_x);
					total_weighted_luminance += luminance(faces[f_index]->pixel[x * w + y]); // * solid_angle);
					f[i++] = luminance(abs(faces[f_index]->pixel[x * w + y]) * solid_angle / (2.f * (float)M_PI / 3.f));
				}
			}
			sampling_dist.push_back(make_table_dist_2d(f, w, h));
			face_luminances.push_back(total_weighted_luminance);
		}

		double total_luminance = 0.0f;
		for (double n : face_luminances) total_luminance += n;

		// Construct the 1D table of faces
		std::vector<double> f(6);
		for (int i = 0; i < 6; ++i) {
			f[i] = face_luminances[i] / total_luminance;
		}
		face_dist = make_table_dist_1d(f);
		
	}

}
