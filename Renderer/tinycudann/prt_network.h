#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>

#include "prt_globals.h"
#include "utils_kernels.h"
#include "CUDABuffer.h"

namespace prt {
    template <typename T, typename PARAMS_T=T>
    class PRT_Network : public tcnn::DifferentiableObject<float, PARAMS_T, PARAMS_T> {
        private:
            std::unique_ptr<tcnn::Network<T>>  m_wavelet_vertex_outter_product;
            nert_renderer::CUDABuffer          m_wavelet_feature_vectors;
            int* waveletCoefsBuffer;

            std::unique_ptr<tcnn::Network<T>>  m_final_mlp;
            std::shared_ptr<tcnn::Encoding<T>> m_vtx_hash_grid;
            std::shared_ptr<tcnn::Encoding<T>> m_aux_buffer_encoding;

            uint32_t m_final_mlp_input_width;
            uint32_t m_final_mlp_alignment;
        public:
            /// <summary>
            /// Setup the PRT Network
            /// vtx_enc: vertex hash grid
            /// cmp_enc: aux buffer encoding scheme
            /// cmp_net: final decode mlp
            /// out_prd: outter product between the vertex feature and wavelet feature
            /// </summary>
            /// <param name="config"></param>
            PRT_Network(const nlohmann::json& config) {
                nlohmann::json vtx_enc = config.value("vtx_enc", nlohmann::json::object());
                nlohmann::json cmp_enc = config.value("cmp_enc", nlohmann::json::object());
                nlohmann::json cmp_net = config.value("cmp_net", nlohmann::json::object());
                nlohmann::json out_prd = config.value("out_prd", nlohmann::json::object());

                // Setup encodings for vectex and aux buffer
                m_vtx_hash_grid.reset(tcnn::create_encoding<T>(INPUT_VERTEX_SIZE, vtx_enc, 16u));
                m_aux_buffer_encoding.reset(tcnn::create_encoding<T>(INPUT_AUX_SIZE, cmp_enc, 16u));

                // setup outter product between vertex feature vector and wavelet feature vector
                out_prd["n_input_dims"] = m_vtx_hash_grid->padded_output_width();
                if (!out_prd.contains("n_output_dims")) {
                    out_prd["n_output_dims"] = 64;
                }
                m_wavelet_vertex_outter_product.reset(tcnn::create_network<T>(out_prd));

                // setup final mlp
                m_final_mlp_alignment = tcnn::minimum_alignment(cmp_net);
                m_final_mlp_input_width = tcnn::next_multiple(m_aux_buffer_encoding->output_width() + m_wavelet_vertex_outter_product->output_width(), m_final_mlp_alignment);
                cmp_net["n_input_dims"] = m_final_mlp_input_width;
                cmp_net["n_output_dims"] = 3;
                m_final_mlp.reset(tcnn::create_network<T>(cmp_net));
            }

            virtual ~PRT_Network() { }

            void set_wavelet_params(std::vector<T>& wavelet_weigths) {
                m_wavelet_feature_vectors.alloc_and_upload(wavelet_weigths);
           }

            void setWaveletCoefBuffer(int* coef) {
                waveletCoefsBuffer = coef;
            }

            void inference_mixed_precision_impl_1(
                cudaStream_t stream,
                const tcnn::GPUMatrixDynamic<float>& input,
                tcnn::GPUMatrixDynamic<T>& output,
                bool use_inference_params = true
            ) {
                uint32_t batch_size = input.n();

                int vtx_begin = 0;
                auto vtx_input = input.slice_rows(0, 3);

                tcnn::GPUMatrixDynamic<T> vtx_enc_out{ m_vtx_hash_grid->padded_output_width(), batch_size, stream, m_vtx_hash_grid->preferred_output_layout() };

                m_vtx_hash_grid->inference_mixed_precision(
                    stream,
                    vtx_input,
                    vtx_enc_out,
                    use_inference_params
                );

                /* multiplication */
                tcnn::linear_kernel(ptwise_mlt<T>, 0, stream,
                    vtx_enc_out.m() * vtx_enc_out.n(),
                    vtx_enc_out.n(),
                    vtx_enc_out.data(),
                    (T*)m_wavelet_feature_vectors.d_pointer(),
                    waveletCoefsBuffer,
                    output.data()
                );
            }

            void inference_mixed_precision_impl_2(
                cudaStream_t stream,
                const tcnn::GPUMatrixDynamic<float>& input,
                const tcnn::GPUMatrixDynamic<T>& input_piece,
                tcnn::GPUMatrixDynamic<T>& output,
                bool use_inference_params = true
            ) {
                uint32_t batch_size = input.n();
                auto gbf_input = input.slice_rows(3, m_aux_buffer_encoding->input_width());
                tcnn::GPUMatrixDynamic<T> cmp_net_inp{ m_final_mlp_input_width, batch_size, stream, m_aux_buffer_encoding->preferred_output_layout() };

                /* outer product context */
                auto out_prd_out = cmp_net_inp.slice_rows(0, m_wavelet_vertex_outter_product->padded_output_width());;
                m_wavelet_vertex_outter_product->inference_mixed_precision(
                    stream,
                    input_piece,
                    out_prd_out,
                    use_inference_params
                );

                /* aux buffer encoding context */
                auto gbf_out = cmp_net_inp.slice_rows(m_wavelet_vertex_outter_product->padded_output_width(), m_aux_buffer_encoding->padded_output_width());
                m_aux_buffer_encoding->inference_mixed_precision(
                    stream,
                    gbf_input,
                    gbf_out,
                    use_inference_params
                );

                /* final mlp context */
                m_final_mlp->inference_mixed_precision(
                    stream,
                    cmp_net_inp,
                    output,
                    use_inference_params
                );
            }

            void inference_mixed_precision_impl(
                cudaStream_t stream, 
                const tcnn::GPUMatrixDynamic<float>& input, 
                tcnn::GPUMatrixDynamic<T>& output, 
                bool use_inference_params = true
            ) override {
                uint32_t batch_size = input.n();
                tcnn::GPUMatrixDynamic<T> out_prd_inp{ m_vtx_hash_grid->padded_output_width(), batch_size, stream, m_vtx_hash_grid->preferred_output_layout() };

                inference_mixed_precision_impl_1(stream, input, out_prd_inp);
                inference_mixed_precision_impl_2(stream, input, out_prd_inp, output);
                return;
            }

            void set_params_impl(T* params, T* inference_params, T* gradients) override {
                size_t offset = 0;
                m_vtx_hash_grid->set_params(params + offset, inference_params + offset, gradients + offset);
                offset += m_vtx_hash_grid->n_params();

                m_wavelet_vertex_outter_product->set_params(params + offset, inference_params + offset, gradients + offset);
                offset += m_wavelet_vertex_outter_product->n_params();

                m_final_mlp->set_params(params + offset, inference_params + offset, gradients + offset);
                offset += m_final_mlp->n_params();

                m_aux_buffer_encoding->set_params(params + offset, inference_params + offset, gradients + offset);
                offset += m_aux_buffer_encoding->n_params();
            }

            ///////////////////////////////////////////
            /// Implemented but not used in TCNN
            ///////////////////////////////////////////
            uint32_t input_width() const override {
                return INPUT_BUFFER_SIZE;
            };

            uint32_t padded_output_width() const override {
                return tcnn::next_multiple(m_final_mlp->padded_output_width(), m_final_mlp_alignment);
            }

            uint32_t output_width() const override {
                return m_final_mlp->output_width();
            }

            uint32_t required_input_alignment() const override {
                return 1;
            }

            size_t n_params() const override {
                return m_vtx_hash_grid->n_params()
                    + m_aux_buffer_encoding->n_params()
                    + m_wavelet_vertex_outter_product->n_params()
                    + m_final_mlp->n_params();
            }

            tcnn::json hyperparams() const override {
                return {
                    {"otype", "prt_network"},
                    {"vertex_encoding", m_vtx_hash_grid->hyperparams()},
                    {"composite_encoding", m_aux_buffer_encoding->hyperparams()},
                    {"outer_product", m_wavelet_vertex_outter_product->hyperparams()},
                    {"composite_network", m_final_mlp->hyperparams()},
                };
            }
   
            std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
                auto out_prd_layers = m_wavelet_vertex_outter_product->layer_sizes();
                auto cmp_net_layers = m_final_mlp->layer_sizes();
                out_prd_layers.insert(
                    out_prd_layers.end(),
                    cmp_net_layers.begin(),
                    cmp_net_layers.end()
                );
                return out_prd_layers;
            }

 
            ///////////////////////////////////////////
            /// Not implemented and not used in TCNN
            ///////////////////////////////////////////
            std::unique_ptr<tcnn::Context> forward_impl(
                cudaStream_t stream, 
                const tcnn::GPUMatrixDynamic<float>& input, 
                tcnn::GPUMatrixDynamic<T>* output, 
                bool use_inference_params = false, 
                bool prepare_input_gradients = false
            ) override {
                auto forward = std::make_unique<tcnn::Context>();
                return forward;
            }

            void backward_impl(
                cudaStream_t stream,
                const tcnn::Context& ctx,
                const tcnn::GPUMatrixDynamic<float>& input,
                const tcnn::GPUMatrixDynamic<T>& output,
                const tcnn::GPUMatrixDynamic<T>& dL_doutput,
                tcnn::GPUMatrixDynamic<float>* dL_dinput = nullptr,
                bool use_inference_params = false,
                tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
            ) override {}

            void initialize_params(
                tcnn::pcg32& rnd,
                float* params_full_precision,
                float scale = 1
            ) override {}

        
        };
}
