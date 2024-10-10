

#include <cuda_runtime.h>
#include <optix.h>

#include "LaunchParams.h"
#include "perRayData.h"
#include "shaderUtils.h"

namespace nert_renderer {
    __forceinline__ __device__ float pdf_envmap(const TableDist2D_device& table, const float2& xy) {
        // Convert xy to integer rows & columns
        int w = table.width, h = table.height;
        int x = clamp(xy.x * w, float(0), float(w - 1));
        int y = clamp(xy.y * h, float(0), float(h - 1));
        // What's the PDF for sampling row y?
        float pdf_y = table.pdf_marginals[y];
        // What's the PDF for sampling row x?
        float pdf_x = table.pdf_rows[y * w + x];
        return pdf_y * pdf_x * w * h;
    }

    __forceinline__ __device__ float2 EnvDirecToUV(const float3& direc, int& f_index) {
        float x = direc.x;
        float y = direc.y;
        float z = direc.z;

        float absX = fabs(x);
        float absY = fabs(y);
        float absZ = fabs(z);

        int isXPositive = x > 0 ? 1 : 0;
        int isYPositive = y > 0 ? 1 : 0;
        int isZPositive = z > 0 ? 1 : 0;

        float maxAxis, uc, vc;

        // POSITIVE X
        if (isXPositive && absX >= absY && absX >= absZ) {
            // u (0 to 1) goes from +z to -z
            // v (0 to 1) goes from -y to +y
            maxAxis = absX;
            uc = z;
            vc = y;
            f_index = 0;
        }
        // NEGATIVE X
        if (!isXPositive && absX >= absY && absX >= absZ) {
            // u (0 to 1) goes from -z to +z
            // v (0 to 1) goes from -y to +y
            maxAxis = absX;
            uc = -z;
            vc = y;
            f_index = 1;
        }
        // POSITIVE Y
        if (isYPositive && absY >= absX && absY >= absZ) {
            // u (0 to 1) goes from -x to +x
            // v (0 to 1) goes from +z to -z
            maxAxis = absY;
            uc = x;
            vc = z;
            f_index = 2;
        }
        // NEGATIVE Y
        if (!isYPositive && absY >= absX && absY >= absZ) {
            // u (0 to 1) goes from -x to +x
            // v (0 to 1) goes from -z to +z
            maxAxis = absY;
            uc = x;
            vc = -z;
            f_index = 3;
        }
        // NEGATIVE Z
        if (!isZPositive && absZ >= absX && absZ >= absY) {
            // u (0 to 1) goes from -x to +x
            // v (0 to 1) goes from -y to +y
            maxAxis = absZ;
            uc = x;
            vc = y;
            f_index = 4;
        }
        // POSITIVE Z
        if (isZPositive && absZ >= absX && absZ >= absY) {
            // u (0 to 1) goes from +x to -x
            // v (0 to 1) goes from -y to +y
            maxAxis = absZ;
            uc = -x;
            vc = y;
            f_index = 5;
        }

        // Convert range from -1 to 1 to 0 to 1
        auto u = 0.5f * (uc / maxAxis + 1.0f);
        auto v = 0.5f * (vc / maxAxis + 1.0f);
        return make_float2(u, v);
    }

    __forceinline__ __device__ float3 EnvUVToDirec(float u, float v, int face_index) {
        // Turn uv coordinate into direction
        // convert range 0 to 1 to -1 to 1
        float uc = 2.0f * u - 1.0f;
        float vc = 2.0f * v - 1.0f;
        float x, y, z;
        switch (face_index) {
        case 0: x = 1.0f; y = vc; z = uc; break;	// POSITIVE X
        case 1: x = -1.0f; y = vc; z = -uc; break;	// NEGATIVE X
        case 2: x = uc; y = 1.0f; z = vc; break;	// POSITIVE Y
        case 3: x = uc; y = -1.0f; z = -vc; break;	// NEGATIVE Y
        case 4: x = uc; y = vc; z = -1.0f; break;	// NEGATIVE Z
        case 5: x = -uc; y = vc; z = 1.0f; break;	// POSITIVE Z
        }
        return normalize(make_float3(x, y, z));

    }
	
    __device__ __inline__ int sample_env(const TableDist1D_device& table, float rnd_param)
	{
		int size = table.size; // Size of PMF
		int ptr = size;
		for (int i = 0; i < size; i++) {
			if (table.cdf[i] > rnd_param) {
				ptr = i;
				break;
			}
		}
		int offset = clamp(ptr - 1, 0, size - 1);
		return offset;
		
	}

	__device__ __inline__ float2 sample_env_2d(const TableDist2D_device& table, float2 rnd_param_uv)
	{
		int w = table.width, h = table.height;
		// We first sample a row from the marginal distribution
		int y_offset = h;
		for (int i = 0; i < h; i++) {
			if (table.cdf_marginals[i] > rnd_param_uv.y) {
				y_offset = i;
				break;
			}
		}
		y_offset = clamp(y_offset - 1, 0, h - 1);

		// Uniformly remap rnd_param[1] to find the continuous offset 
		double dy = rnd_param_uv.y - table.cdf_marginals[y_offset];
		if ((table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]) > 0) {
			dy /= (table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]);
		}

		// Sample a column at the row y_offset
		const double* cdf = &table.cdf_rows[y_offset * (w + 1)];
		int x_offset = w;
		for (int i = 0; i < w; i++) {
			if (cdf[i] > rnd_param_uv.x) {
				x_offset = i;
				break;
			}
		}
		x_offset = clamp(x_offset - 1, 0, w - 1);

		// Uniformly remap rnd_param[0]
		double dx = rnd_param_uv.x - cdf[x_offset];
		if (cdf[x_offset + 1] - cdf[x_offset] > 0) {
			dx /= (cdf[x_offset + 1] - cdf[x_offset]);
		}
		return make_float2( (x_offset + dx) / w, (y_offset + dy) / h );
	}

	extern "C" __device__ float3 __direct_callable__sampleEnvironmapLight(const light_data& envmap, Random& random) {
		// Sample the face of the cubemap based on PMF
		int face_index = sample_env(envmap.cubeMap.face_dist, random());

		// Sample the uv coordinate of the selected face
		float2 uv = sample_env_2d(envmap.cubeMap.sampling_dist[face_index], make_float2(random(), random()));
		// Turn uv coordinate into direction
		return EnvUVToDirec(uv.x, uv.y, face_index);		
	}

    // Don't multiply by solid angle;
    // the PDF was already multiplied by solid angle during construction
    extern "C" __device__ float __direct_callable__pdf(const light_data & envmap, const float3 & direc) {
        int f_index;
        float2 uv = EnvDirecToUV(direc, f_index);
        float pdfChooseFace = envmap.cubeMap.face_dist.pmf[f_index];
        float pdfSolidEnv = pdf_envmap(envmap.cubeMap.sampling_dist[f_index], uv);
        return pdfChooseFace * pdfSolidEnv;
    }

    // Get the radiance
    extern "C" __device__ float3 __direct_callable__evaluate(const light_data & envmap, const float3 & direc) {
        int f_index;
        float2 uv = EnvDirecToUV(direc, f_index);
        return make_float3(tex2D<float4>(envmap.cubeMap.faces[f_index], uv.x, uv.y));
    }
}