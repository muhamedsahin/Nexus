#pragma once

/**
 * @file hf.hpp
 * @brief FAZ 17.2: Hugging Face icin ice / disa aktarma — config.json -> lm::LmConfig, safetensors agirliklari ->
 *        lm::LanguageModel (ad + duzen esleme tek tabloda), Nexus -> HF dizini (config.json + model.safetensors).
 *
 * Sablonlar (surumlu; HF adlari transformers 4.40-4.5x LlamaForCausalLM / MistralForCausalLM / Qwen2ForCausalLM):
 *   llama/v1   : RMSNorm, SwiGLU (silu), RoPE (rotate_half), GQA; attention_bias / mlp_bias = false; rope_scaling
 *                none | linear | dynamic | yarn | llama3 (LLaMA-2/3, TinyLlama).
 *   mistral/v1 : llama/v1 + sliding_window -> kayan pencere.
 *   qwen2/v1   : llama/v1 + q/k/v bias (o bias yok); use_sliding_window desteklenmez (acik hata).
 * Ad / duzen eslemesi (HF Linear agirligi [cikis, giris]; Nexus lm::linear [giris, cikis]):
 *   model.embed_tokens.weight [V, d]                 -> wte [Vp, d] (dolgu satirlari 0)
 *   model.layers.L.input_layernorm.weight            -> hL.norm1.weight
 *   model.layers.L.self_attn.{q,k,v}_proj.weight^T   -> hL.attn.qkv.weight [d, (H + 2Hk) D] = [Wq^T | Wk^T | Wv^T]
 *   model.layers.L.self_attn.{q,k,v}_proj.bias       -> hL.attn.qkv.bias (qwen2)
 *   model.layers.L.self_attn.o_proj.weight^T         -> hL.attn.o.weight [H D, d]
 *   model.layers.L.post_attention_layernorm.weight   -> hL.norm2.weight
 *   model.layers.L.mlp.{gate,up}_proj.weight^T       -> hL.mlp.up.weight [d, 2F] = [gate^T | up^T]  (silu(gate) * up)
 *   model.layers.L.mlp.down_proj.weight^T            -> hL.mlp.down.weight [F, d]
 *   model.norm.weight                                -> norm_f.weight
 *   lm_head.weight [V, d]                            -> lm_head.weight [Vp, d] (bagli gommede yok)
 * RoPE: Nexus da yarim dondurme (i, i + D/2) kullanir: q/k permutasyonu gerekmez.
 * Eksik / fazla / sekli uyusmayan tensor acik hata (sessiz atlama yok); rotary_emb.inv_freq gibi tamponlar yok sayilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/json.hpp"
#include "nexus/lm/model.hpp"

#include <string>
#include <vector>

namespace nexus {
namespace interop {

struct HfTemplate {
  std::string architecture;  // "LlamaForCausalLM"
  std::string name;          // "llama/v1"
};
/// Desteklenen sablonlar (raporlama / dogrulama icin).
[[nodiscard]] const std::vector<HfTemplate>& hf_templates();

/// config.json -> LmConfig (sablon adi `used`'e yazilir). Desteklenmeyen alan acik hata.
[[nodiscard]] lm::LmConfig hf_config(const Json& config, std::string* used = nullptr);

struct HfLoadStats {
  std::string template_name;
  int tensors = 0;
  std::uint64_t bytes = 0;     // okunan safetensors bayti
  double ms = 0.0;             // toplam (acma + donusturme + kopyalama)
  double open_ms = 0.0;        // dosya acma + baslik dogrulama
  bool mmap = true;
  std::vector<std::string> files;
};

/// HF model dizini (config.json + model.safetensors ya da model.safetensors.index.json parcalari).
[[nodiscard]] lm::LanguageModel load_hf(const std::string& dir, bool mmap = true, HfLoadStats* stats = nullptr);

/// Nexus -> HF dizini (config.json + model.safetensors, F32 ya da F16). Yalniz sablonlarin kapsadigi yapilandirma.
void save_hf(const lm::LanguageModel& model, const std::string& dir, const std::string& dtype = "F32");

}  // namespace interop
}  // namespace nexus
