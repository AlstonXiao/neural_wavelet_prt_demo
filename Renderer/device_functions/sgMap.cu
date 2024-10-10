#include <cuda_runtime.h>
#include <optix.h>

#include "LaunchParams.h"
#include "perRayData.h"
#include "shaderUtils.h"

namespace nert_renderer {
    extern "C" __device__ float3 __direct_callable__sampleEnvironmapLight(const light_data & envmap, Random & random) {
        float phi = random() * 2 * M_PIf;
        float cosTheta = random() * 2 - 1;
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
        return make_float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    }

    // Don't multiply by solid angle;
    // the PDF was already multiplied by solid angle during construction
    extern "C" __device__ float __direct_callable__pdf(const light_data & envmap, const float3 & direc) {
        return 1 / (4 * M_PIf);
    }

    // Get the radiance
    extern "C" __device__ float3 __direct_callable__evaluate(const light_data & envmap, const float3 & direc) {
        sgEnvMap sgs = envmap.sgMap;
        float3 radiance = make_float3(0., 0., 0.);

        for (int i = 0; i < sgs.sg_count; i++) {

            float3 location = sgs.sg_location[i];
            float3 color = sgs.sg_color[i];
            float roughness = sgs.sg_roughness[i];
            float3 z_direc = make_float3(direc.x, -direc.z, direc.y);
            float distance = expf(roughness * (dot(location, z_direc) - 1.));

            radiance += color * distance;
        }
        return radiance;

    }
}