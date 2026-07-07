/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include <functional>

#include "atb/atb_infer.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "loader/qwen3_decoder_loader.h"
#include "nlohmann/json.hpp"
#include "npu_base_layer.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "pytorch/adapter/utils/utils.h"
#include "xllm_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_atb_layers/core/include/atb_speed/base/model.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "xllm_atb_layers/models/qwen3/layer/decoder_layer.h"

namespace xllm {
namespace layer {

// Subclass that adds FlashComm1.0 tensor registration without modifying
// atb_layers.
class FlashCommQwenDecoderLayer : public atb_speed::qwen::QwenDecoderLayer {
 public:
  using atb_speed::qwen::QwenDecoderLayer::QwenDecoderLayer;

 protected:
  void ConstructInTensorMap() override {
    QwenDecoderLayer::ConstructInTensorMap();
    if (this->param.enableFlashComm)
      atb_speed::common::AddTensorToList(
          this->inTensorCandidates, "flash_comm", this->inTensorList);
  }
  std::map<unsigned int, std::vector<std::string>> GetAttentionIntensor()
      override {
    auto m = QwenDecoderLayer::GetAttentionIntensor();
    if (this->param.enableFlashComm)
      m[atb_speed::common::AttnInTensorCategory::ATTN_FC] = {"send_counts",
                                                             "sdispls",
                                                             "send_count",
                                                             "recv_counts",
                                                             "rdispls",
                                                             "recv_count",
                                                             "fake_rs_shape",
                                                             "fake_ag_shape"};
    return m;
  }
};

class NpuQwen3DecoderLayerImpl : public BaseLayer {
 public:
  explicit NpuQwen3DecoderLayerImpl(const ModelContext& context);

  ~NpuQwen3DecoderLayerImpl() override = default;

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);

  void set_layer_id(int32_t layer_id) override {
    prefill_param_.layerId = layer_id;
    decode_graph_param_.layerId = layer_id;
    decode_eager_param_.layerId = layer_id;
  }

 private:
  void param_from_args(atb_speed::qwen::QwenLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool isPrefill);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               ModelInputParams& input_params,
                               bool is_prefill,
                               int node_id,
                               bool use_graph_decode_input);

  void initialize_parallel_parameters(atb_speed::qwen::QwenLayerParam& param,
                                      const ParallelArgs& parallel_args);

  void initialize_quantization_parameters(
      atb_speed::qwen::QwenLayerParam& param);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::qwen::QwenLayerParam& param);

  int64_t init_attn_mask();

  void prepare_flash_comm_tensors(int64_t per_rank,
                                  int64_t hidden,
                                  int32_t ws,
                                  torch::Device dev);

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_graph_node_;
  atb_speed::Model::Node decode_eager_node_;
  std::string model_name_;
  atb_speed::qwen::QwenLayerParam prefill_param_;
  atb_speed::qwen::QwenLayerParam decode_graph_param_;
  atb_speed::qwen::QwenLayerParam decode_eager_param_;
  atb::Tensor internal_tensors_;
  atb::Tensor residual_tensors_;
  atb::Tensor placeholder_;

  // FlashComm1.0 tensor storage + host-side data
  atb::Tensor flash_send_counts_, flash_sdispls_, flash_send_count_;
  atb::Tensor flash_recv_counts_, flash_rdispls_, flash_recv_count_;
  atb::Tensor flash_fake_rs_shape_, flash_fake_ag_shape_;
  std::vector<int64_t> flash_send_counts_host_, flash_sdispls_host_,
      flash_send_count_host_ = {0};
  std::vector<int64_t> flash_recv_counts_host_, flash_rdispls_host_,
      flash_recv_count_host_ = {0};

  at::Tensor decode_attn_mask_;

  at::Tensor at_placeholder_;

  int device_id_;
  int32_t layer_id_;
  int rank_id_;
  int32_t num_hidden_layers_;
  std::vector<std::shared_ptr<at::Tensor>> prefill_tensor_storage_;
  std::vector<std::shared_ptr<at::Tensor>> decode_tensor_storage_;
  std::vector<std::shared_ptr<std::vector<int>>> prefill_vector_storage_;
  std::vector<std::shared_ptr<std::vector<int>>> decode_vector_storage_;
};
TORCH_MODULE(NpuQwen3DecoderLayer);

}  // namespace layer
}  // namespace xllm
