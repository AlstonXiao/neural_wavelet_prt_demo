#include "RenderEngine.h"

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/common.h>

#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <random>
#include <filesystem>

#include "prt_network.h"
#include "npy.h"
#include "utils_kernels.h"

using namespace nert_renderer;
using namespace tcnn;

using precision_t = network_precision_t;

inline vec2i make_vec2i(int2 input) {
    return vec2i(input.x, input.y);
}

std::vector<precision_t> load_npy_to_precision(std::string fname) {
    std::vector<unsigned long> shape{};
    bool fortran_order;
    std::vector<float> data;

    npy::LoadArrayFromNumpy(fname, shape, fortran_order, data);
    std::vector<precision_t> ret(data.begin(), data.end());
    return ret;
}

namespace nert_renderer {
    std::shared_ptr< prt::PRT_Network<precision_t> > prt_net;
    int maxStream = 4;
    std::vector<CUstream> unSyncStreams(maxStream);

    CUDABuffer m_network_weights_buffer;
    GPUMatrixDynamic<float>* waveletInput[2] = { nullptr, nullptr };
    GPUMatrixDynamic<float>* waveletOutput = nullptr;

    void RenderEngine::initTCNN(const std::string modelPath) {
        namespace fs = std::filesystem;

        fs::path modelDir(modelPath);
        fs::path configPath = modelDir / "config_comp.json";
        std::ifstream ifs{ configPath.string()};
        auto config = json::parse(ifs);
        prt_net = std::make_shared<prt::PRT_Network<precision_t>>(config);

        fs::path out_prd_weights_fname = modelDir / "out_prd.npy";
        auto out_prd_weights = load_npy_to_precision(out_prd_weights_fname.string());
        fs::path cmp_net_weights_fname = modelDir / "cmp_net.npy";
        auto cmp_net_weights = load_npy_to_precision(cmp_net_weights_fname.string());
        fs::path enc_x_weights_fname = modelDir / "vtx_enc.npy";
        auto enc_x_weights = load_npy_to_precision(enc_x_weights_fname.string());
        fs::path enc_w_weights_fname = modelDir / "wve_enc.npy";
        auto enc_w_weights = load_npy_to_precision(enc_w_weights_fname.string());
           
        std::vector<precision_t> all_weights;
        all_weights.insert(all_weights.end(), enc_x_weights.begin(), enc_x_weights.end());
        all_weights.insert(all_weights.end(), out_prd_weights.begin(), out_prd_weights.end());
        all_weights.insert(all_weights.end(), cmp_net_weights.begin(), cmp_net_weights.end());
        m_network_weights_buffer.alloc_and_upload(all_weights);

        prt_net->set_params((precision_t*)m_network_weights_buffer.d_pointer(), (precision_t*)m_network_weights_buffer.d_pointer(), (precision_t*)m_network_weights_buffer.d_pointer());
        prt_net->set_wavelet_params(enc_w_weights);

        TCNNSetup = true;
        for (int i = 0; i < maxStream; i++) {
            cudaStreamCreate(&unSyncStreams[i]);
        }
        if (debug) std::cout << "TCNN weights loaded" << std::endl;
    }

    void RenderEngine::TCNNPass_p1(int frameID, CUstream stream) {
        vec2i fbSize = make_vec2i(launchParams.frame.size);
        vec2i blockSize = 32;
        vec2i numBlocks = divRoundUp(fbSize, blockSize);

        if (waveletInput[frameID] == nullptr) {
            waveletInput[frameID] = new GPUMatrixDynamic<float>(16, fbSize.x * fbSize.y * WAVELETNUM, stream);
            std::cout << "New GPU matrix generated for frame buffer " << frameID << std::endl;
        }
        else if (fbSize.x * fbSize.y * WAVELETNUM != waveletInput[frameID]->n()) {
            delete waveletInput[frameID];
            waveletInput[frameID] = new GPUMatrixDynamic<float>(16, fbSize.x * fbSize.y * WAVELETNUM, nullptr);
            std::cout << "New GPU matrix generated for frame buffer "<< frameID << std::endl;
        }
        prt::waveletDataCopy << <dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y), 0, stream >> >
            (
                launchParams.frame.size,
                (float*)fullframeCUDABuffer[frameID].fbFirstHitPos.d_pointer(),
                (float*)fullframeCUDABuffer[frameID].fbworldNormal.d_pointer(),
                (float*)fullframeCUDABuffer[frameID].fbFirstHitReflecDir.d_pointer(),
                (float*)fullframeCUDABuffer[frameID].fbRoughness.d_pointer(),
                (float*)fullframeCUDABuffer[frameID].fbFirstHitKd.d_pointer(),
                (float*)fullframeCUDABuffer[frameID].fbFirstHitKs.d_pointer(),
                (float*)waveletInput[frameID]->data()
                );
    }


    void RenderEngine::TCNNPass_p2(int envMapID, int frameID, renderConfig config) {
        vec2i fbSize = make_vec2i(launchParams.frame.size);
        vec2i blockSize = 32;
        vec2i numBlocks = divRoundUp(fbSize, blockSize);
        if (waveletOutput == nullptr) {
            waveletOutput = new GPUMatrixDynamic<float>(3, fbSize.x * fbSize.y * WAVELETNUM, nullptr);
            std::cout << "New GPU matrix generated for frame buffer output " << frameID << std::endl;
        }
        else if (fbSize.x * fbSize.y * WAVELETNUM != waveletOutput->n()) {
            delete waveletOutput;
            waveletOutput = new GPUMatrixDynamic<float>(3, fbSize.x * fbSize.y * WAVELETNUM, nullptr);
            std::cout << "New GPU matrix generated for frame buffer output " << frameID << std::endl;
        }
        prt_net->setWaveletCoefBuffer((int*)waveletCoefBuffer[envMapID].d_pointer());
        // If the waveletInput 1 has the previous shape, and p2 is called for the next shape, skip.
        assert(fbSize.x * fbSize.y * WAVELETNUM == waveletInput[frameID]->n());
        for (int i = 0; i < launchParams.frame.size.y; i++) {
            GPUMatrixDynamic<float> input = waveletInput[frameID]->slice_cols(launchParams.frame.size.x * WAVELETNUM * i, launchParams.frame.size.x * WAVELETNUM);
            GPUMatrixDynamic<float> inference_output = waveletOutput->slice_cols(launchParams.frame.size.x * WAVELETNUM * i, launchParams.frame.size.x * WAVELETNUM);
            prt_net->inference(unSyncStreams[i % maxStream], input, inference_output);
        }
        for (int i = 0; i < maxStream; i++) {
            cudaStreamSynchronize(unSyncStreams[i]);
        }
        prt::computeResponse << <dim3(numBlocks.x, numBlocks.y), dim3(blockSize.x, blockSize.y), 0, unSyncStreams[0] >> >
            (
                launchParams.frame.size,
                waveletOutput->data(),
                (float*)fullframeCUDABuffer[frameID].fbDirectDenoised.d_pointer(),
                (float*)waveletStrengthBuffer[envMapID].d_pointer(),
                (bool*)fullframeCUDABuffer[frameID].fbEnvironmentMapFlag.d_pointer(),
                (float*)fullframeCUDABuffer[frameID].fbPredictedIndirect.d_pointer(),
                (float*)fullframeCUDABuffer[frameID].fbPredictedIndirectWithDirect.d_pointer()
                );
        
 
        denoiseOrMoveData(frameID, config.finalDenoiser, fullframeCUDABuffer[frameID].fbPredictedIndirectWithDirect.d_pointer(),
            fullframeCUDABuffer[frameID].fbPredictedIndirectWithDirectDenoised.d_pointer(), 0, unSyncStreams[0]);

        denoiseOrMoveData(frameID, config.indirectDenoiser, fullframeCUDABuffer[frameID].fbPredictedIndirect.d_pointer(),
            fullframeCUDABuffer[frameID].fbPredictedIndirectDenoised.d_pointer(), 0, unSyncStreams[0]);
    }

    void RenderEngine::TCNNSync() {
        for (auto cudaStream : unSyncStreams) {
            cudaStreamSynchronize(cudaStream);
        }
    }

    bool RenderEngine::TCNNAvaliable() {
        return TCNNSetup && EnvmapSetup;
    }
} // ::nert_renderer
