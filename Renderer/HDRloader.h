/***********************************************************************************
	Created:	17:9:2002
	FileName: 	hdrloader.h
	Author:		Igor Kravtchenko
	
	Info:		Load HDR image and convert to a set of float32 RGB triplet.
************************************************************************************/
#include "gdt/math/AffineSpace.h"

namespace nert_renderer {
	using namespace gdt;
	class HDRLoaderResult {
	public:
		// Free up resources, the data should be owned by texture class, not here
		~HDRLoaderResult() { if (cols) delete[] cols; }
		int width, height;
		// each pixel takes 3 float32, each component can be of any value...
		float* cols;

		// Inverse to lower left first
		vec3f get(int u, int v) {
			return vec3f(cols[3 * (height - u - 1) * width + 3 * v],
				cols[3 * (height - u - 1) * width + 3 * v + 1],
				cols[3 * (height - u - 1) * width + 3 * v + 2]);
		}
	};

	class HDRLoader {
	public:
		static bool load(const char* fileName, HDRLoaderResult& res);
	};
}