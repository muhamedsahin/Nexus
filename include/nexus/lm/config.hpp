#pragma once

/**
 * @file config.hpp
 * @brief FAZ 6.6: yapilandirma gudumlu dil modeli (GPT-2, LLaMA-2/3, Mistral, Qwen2 tarzi).
 *
 * Tek `LmConfig` hem modeli kurar (`nexus::lm::build`) hem de NXS arsivine JSON olarak
 * gomulur; yuklemede ayni model yeniden kurulur. `count_parameters` modeli ayirmadan
 * analitik sayar (8B on tanimlilar icin bellek gerekmez).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace nexus {
namespace lm {

enum class NormKind { LayerNorm, RmsNorm };
enum class MlpKind { Gelu, SwiGlu, GeGlu };
enum class PosKind { None, Sinusoidal, Learned, Rope, Alibi };
enum class InitKind { Gpt2, Llama, MuP };
/// FAZ 7.1: blok basina aktivasyon yeniden hesaplama (autograd::checkpoint).
/// every_k: l % k == 0 olan bloklar yeniden hesaplanir (L/k blok).
enum class RecomputeKind { None, Full, EveryK };

/// RoPE uzunluk olceklemesi (HF transformers `rope_scaling` alanlariyla ayni adlar).
struct RopeScaling {
  enum class Type { None, Linear, Ntk, DynamicNtk, Yarn, Llama3 };
  Type type = Type::None;
  float factor = 1.0f;
  int original_max_position = 0;  // 0: max_position
  float low_freq_factor = 1.0f;   // llama3
  float high_freq_factor = 4.0f;  // llama3
  float beta_fast = 32.0f;        // yarn
  float beta_slow = 1.0f;         // yarn
  float attention_factor = 0.0f;  // yarn; 0: 0.1 ln(factor) + 1
};

struct LmConfig {
  std::string name = "custom";
  int vocab_size = 0;
  int n_layer = 0;
  int d_model = 0;
  int n_head = 0;
  int n_kv_head = 0;  // 0: n_head (MHA)
  int head_dim = 0;   // 0: d_model / n_head
  int d_ff = 0;
  int max_position = 2048;
  NormKind norm = NormKind::RmsNorm;
  float norm_eps = 1.0e-5f;
  bool prenorm = true;  // false: son-normalizasyon (tur 33 CausalTransformerBlock tarifi)
  MlpKind mlp = MlpKind::SwiGlu;
  PosKind pos = PosKind::Rope;
  float rope_theta = 10000.0f;
  int rotary_dim = 0;  // 0: head_dim; < head_dim: kismi dondurme (GPT-NeoX)
  RopeScaling rope_scaling{};
  bool tie_embeddings = false;
  bool bias = false;      // tum Linear + LayerNorm bias'i (GPT-2)
  bool qkv_bias = false;  // yalniz QKV bias'i (Qwen2)
  bool qk_norm = false;   // Q ve K'ye bas boyutunda RMSNorm
  bool parallel_block = false;  // y = x + Attn(N(x)) + MLP(N(x))
  int window = 0;               // > 0: kayan pencere
  float attn_softcap = 0.0f;
  float resid_dropout = 0.0f;
  float attn_dropout = 0.0f;
  int vocab_pad_multiple = 1;  // 64/128: Tensor Core dolgusu; dolgu logitleri -inf
  InitKind init = InitKind::Gpt2;
  float init_std = 0.02f;
  bool init_truncated = false;  // N(0, s) +-2s'de kesik (yeniden ornekleme)
  int mup_base_width = 256;     // muP: genislik carpani m = d_model / mup_base_width
  std::uint64_t seed = 1234;
  std::string dtype = "float32";  // CPU yolu FP32; alan arsive yazilir
  RecomputeKind recompute = RecomputeKind::None;
  int recompute_every = 2;
  // ---- FAZ 14 MoE (nexus::moe; uzman = SwiGLU MLP) ----
  int moe_experts = 0;             // 0: yogun MLP
  int moe_top_k = 2;
  int moe_every = 1;               // her moe_every. blok MoE (blok l: l % moe_every == moe_every - 1)
  int moe_d_ff = 0;                // uzman basina F (0: d_ff)
  float moe_capacity_factor = 1.25f;
  int moe_min_capacity = 4;
  bool moe_reroute = false;        // false: tasan atama dusurulur
  int moe_groups = 1;              // GShard grup basina kapasite
  bool moe_noisy = false;          // gurultulu top-k (yalniz egitim)
  bool moe_shared_expert = false;  // yogun MLP paylasimli uzman olarak tutulur (DeepSeek)
  float moe_aux_alpha = 0.01f;
  float moe_z_alpha = 1.0e-3f;

  [[nodiscard]] int kv_heads() const noexcept { return n_kv_head > 0 ? n_kv_head : n_head; }
  [[nodiscard]] int hdim() const noexcept { return head_dim > 0 ? head_dim : (n_head > 0 ? d_model / n_head : 0); }
  [[nodiscard]] int rot_dim() const noexcept { return rotary_dim > 0 ? rotary_dim : hdim(); }
  [[nodiscard]] int padded_vocab() const noexcept {
    const int m = vocab_pad_multiple > 1 ? vocab_pad_multiple : 1;
    return (vocab_size + m - 1) / m * m;
  }
  [[nodiscard]] bool gated_mlp() const noexcept { return mlp != MlpKind::Gelu; }
  [[nodiscard]] bool is_moe_layer(int layer) const noexcept {
    return moe_experts > 0 && moe_every > 0 && layer % moe_every == moe_every - 1;
  }
  [[nodiscard]] int expert_ff() const noexcept { return moe_d_ff > 0 ? moe_d_ff : d_ff; }
};

/// Gecersiz alan icin TensorError (sessiz duzeltme yok).
void validate(const LmConfig& config);

/// Ayirmadan parametre sayisi (dolgulu sozluk dahil).
[[nodiscard]] std::int64_t count_parameters(const LmConfig& config);
/// FAZ 14: token basina etkin (aktif) parametre: MoE katmaninda yonlendirici + top_k uzman (+ paylasimli).
[[nodiscard]] std::int64_t count_active_parameters(const LmConfig& config);

/// gpt2-{124m,350m,774m,1.5b}, llama2-7b, llama3-8b, mistral-7b, qwen2-{0.5b,1.5b,7b}, tinyllama-1.1b.
[[nodiscard]] LmConfig preset(const std::string& name);
[[nodiscard]] std::vector<std::string> preset_names();

[[nodiscard]] std::string to_json(const LmConfig& config);
/// to_json'un yazdigi alanlar; bilinmeyen anahtar hata. Eksik anahtar varsayilan kalir.
[[nodiscard]] LmConfig from_json(const std::string& json);
/// `base` uzerine JSON nesnesindeki alanlari yazar (overrides).
[[nodiscard]] LmConfig apply_overrides(LmConfig base, const std::string& json);

[[nodiscard]] const char* to_string(NormKind kind) noexcept;
[[nodiscard]] const char* to_string(MlpKind kind) noexcept;
[[nodiscard]] const char* to_string(PosKind kind) noexcept;
[[nodiscard]] const char* to_string(InitKind kind) noexcept;
[[nodiscard]] const char* to_string(RopeScaling::Type type) noexcept;
[[nodiscard]] const char* to_string(RecomputeKind kind) noexcept;

}  // namespace lm
}  // namespace nexus
