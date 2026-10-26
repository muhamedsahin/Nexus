/**
 * @file config.cpp
 * @brief FAZ 6.6: LmConfig dogrulama, analitik parametre sayimi, on tanimlilar, JSON.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/config.hpp"

#include "nexus/core/error.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <sstream>

namespace nexus {
namespace lm {

const char* to_string(NormKind kind) noexcept { return kind == NormKind::LayerNorm ? "layernorm" : "rmsnorm"; }

const char* to_string(MlpKind kind) noexcept {
  switch (kind) {
    case MlpKind::Gelu:
      return "gelu";
    case MlpKind::SwiGlu:
      return "swiglu";
    case MlpKind::GeGlu:
      return "geglu";
  }
  return "unknown";
}

const char* to_string(PosKind kind) noexcept {
  switch (kind) {
    case PosKind::None:
      return "none";
    case PosKind::Sinusoidal:
      return "sinusoidal";
    case PosKind::Learned:
      return "learned";
    case PosKind::Rope:
      return "rope";
    case PosKind::Alibi:
      return "alibi";
  }
  return "unknown";
}

const char* to_string(InitKind kind) noexcept {
  switch (kind) {
    case InitKind::Gpt2:
      return "gpt2";
    case InitKind::Llama:
      return "llama";
    case InitKind::MuP:
      return "mup";
  }
  return "unknown";
}

const char* to_string(RopeScaling::Type type) noexcept {
  switch (type) {
    case RopeScaling::Type::None:
      return "none";
    case RopeScaling::Type::Linear:
      return "linear";
    case RopeScaling::Type::Ntk:
      return "ntk";
    case RopeScaling::Type::DynamicNtk:
      return "dynamic";
    case RopeScaling::Type::Yarn:
      return "yarn";
    case RopeScaling::Type::Llama3:
      return "llama3";
  }
  return "unknown";
}

const char* to_string(RecomputeKind kind) noexcept {
  switch (kind) {
    case RecomputeKind::None:
      return "none";
    case RecomputeKind::Full:
      return "full";
    case RecomputeKind::EveryK:
      return "every_k";
  }
  return "unknown";
}

namespace {

[[noreturn]] void fail(const std::string& what) { throw TensorError("Nexus lm: " + what); }

template <typename E>
E parse_enum(const std::string& s, std::initializer_list<E> all, const char* field) {
  for (const E e : all) {
    if (s == to_string(e)) {
      return e;
    }
  }
  fail(std::string("bilinmeyen ") + field + ": " + s);
}

}  // namespace

void validate(const LmConfig& c) {
  if (c.vocab_size < 2 || c.n_layer < 1 || c.d_model < 1 || c.n_head < 1 || c.d_ff < 1 || c.max_position < 1) {
    fail("vocab_size, n_layer, d_model, n_head, d_ff, max_position pozitif olmali");
  }
  if (c.head_dim == 0 && c.d_model % c.n_head != 0) {
    fail("d_model % n_head != 0 iken head_dim acikca verilmeli");
  }
  if (c.n_head % c.kv_heads() != 0) {
    fail("n_head, n_kv_head'in kati olmali (GQA)");
  }
  if (c.pos == PosKind::Rope && (c.rot_dim() % 2 != 0 || c.rot_dim() > c.hdim())) {
    fail("rotary_dim cift ve <= head_dim olmali");
  }
  if (c.vocab_pad_multiple < 1) {
    fail("vocab_pad_multiple >= 1 olmali");
  }
  if (c.resid_dropout < 0.0f || c.resid_dropout >= 1.0f || c.attn_dropout < 0.0f || c.attn_dropout >= 1.0f) {
    fail("dropout [0, 1) olmali");
  }
  if (c.init == InitKind::MuP && c.mup_base_width < 1) {
    fail("mup_base_width >= 1 olmali");
  }
  if (c.recompute == RecomputeKind::EveryK && c.recompute_every < 1) {
    fail("recompute_every >= 1 olmali");
  }
  if (c.dtype != "float32") {
    fail("CPU lm yolu yalniz float32 (dar tipler GPU yolunda)");
  }
  if (c.moe_experts < 0 || (c.moe_experts > 0 && (c.moe_top_k < 1 || c.moe_top_k > c.moe_experts || c.moe_every < 1 ||
                                                   c.moe_capacity_factor <= 0.0f || c.moe_groups < 1 || c.moe_min_capacity < 0))) {
    fail("MoE: 1 <= moe_top_k <= moe_experts, moe_every >= 1, moe_capacity_factor > 0, moe_groups >= 1");
  }
  if (c.moe_experts > 0 && (c.mlp != MlpKind::SwiGlu || c.bias || c.parallel_block || !c.prenorm)) {
    fail("MoE: uzman SwiGLU MLP ister; bias, paralel blok ve son-norm desteklenmez");
  }
  if (c.moe_experts > 0 && c.recompute != RecomputeKind::None) {
    fail("MoE: yeniden hesaplama (recompute) ile desteklenmez (yonlendirme yeniden kosulurdu)");
  }
}

std::int64_t count_parameters(const LmConfig& c) {
  validate(c);
  const std::int64_t d = c.d_model;
  const std::int64_t H = c.n_head;
  const std::int64_t Hk = c.kv_heads();
  const std::int64_t D = c.hdim();
  const std::int64_t F = c.d_ff;
  const std::int64_t V = c.padded_vocab();
  const bool ln_bias = c.norm == NormKind::LayerNorm && c.bias;
  const std::int64_t norm = d + (ln_bias ? d : 0);
  std::int64_t total = V * d;  // wte
  if (!c.tie_embeddings) {
    total += V * d;  // lm_head
  }
  if (c.pos == PosKind::Learned) {
    total += static_cast<std::int64_t>(c.max_position) * d;
  }
  std::int64_t layer = (c.parallel_block ? 1 : 2) * norm;
  layer += d * (H + 2 * Hk) * D;  // birlesik QKV
  if (c.bias || c.qkv_bias) {
    layer += (H + 2 * Hk) * D;
  }
  layer += H * D * d + (c.bias ? d : 0);  // cikis projeksiyonu
  if (c.qk_norm) {
    layer += 2 * D;
  }
  if (c.gated_mlp()) {
    layer += d * 2 * F + F * d + (c.bias ? 2 * F + d : 0);
  } else {
    layer += d * F + F * d + (c.bias ? F + d : 0);
  }
  total += c.n_layer * layer;
  total += norm;  // final norm (on-norm mimaride zorunlu; son-normda da tutulur)
  // FAZ 14: MoE katmanlari yogun MLP yerine (paylasimli uzman varsa ek olarak) yonlendirici + E uzman
  const std::int64_t mlp = c.gated_mlp() ? d * 2 * F + F * d + (c.bias ? 2 * F + d : 0) : d * F + F * d + (c.bias ? F + d : 0);
  const std::int64_t Fe = c.expert_ff();
  for (int l = 0; l < c.n_layer; ++l) {
    if (!c.is_moe_layer(l)) continue;
    if (!c.moe_shared_expert) total -= mlp;
    total += d * c.moe_experts * (c.moe_noisy ? 2 : 1) + static_cast<std::int64_t>(c.moe_experts) * 3 * d * Fe;
  }
  return total;
}

std::int64_t count_active_parameters(const LmConfig& c) {
  std::int64_t total = count_parameters(c);
  const std::int64_t d = c.d_model, Fe = c.expert_ff();
  for (int l = 0; l < c.n_layer; ++l) {
    if (c.is_moe_layer(l)) total -= static_cast<std::int64_t>(c.moe_experts - c.moe_top_k) * 3 * d * Fe;
  }
  return total;
}

namespace {

LmConfig gpt2(const char* name, int layers, int d, int heads) {
  LmConfig c;
  c.name = name;
  c.vocab_size = 50257;
  c.n_layer = layers;
  c.d_model = d;
  c.n_head = heads;
  c.d_ff = 4 * d;
  c.max_position = 1024;
  c.norm = NormKind::LayerNorm;
  c.norm_eps = 1.0e-5f;
  c.mlp = MlpKind::Gelu;
  c.pos = PosKind::Learned;
  c.tie_embeddings = true;
  c.bias = true;
  c.init = InitKind::Gpt2;
  return c;
}

LmConfig llama_like(const char* name, int vocab, int layers, int d, int heads, int kv, int ff, int max_pos,
                    float theta, float eps, bool tie) {
  LmConfig c;
  c.name = name;
  c.vocab_size = vocab;
  c.n_layer = layers;
  c.d_model = d;
  c.n_head = heads;
  c.n_kv_head = kv;
  c.d_ff = ff;
  c.max_position = max_pos;
  c.norm = NormKind::RmsNorm;
  c.norm_eps = eps;
  c.mlp = MlpKind::SwiGlu;
  c.pos = PosKind::Rope;
  c.rope_theta = theta;
  c.tie_embeddings = tie;
  c.init = InitKind::Llama;
  return c;
}

const std::map<std::string, std::function<LmConfig()>>& presets() {
  static const std::map<std::string, std::function<LmConfig()>> table = {
      {"gpt2-124m", [] { return gpt2("gpt2-124m", 12, 768, 12); }},
      {"gpt2-350m", [] { return gpt2("gpt2-350m", 24, 1024, 16); }},
      {"gpt2-774m", [] { return gpt2("gpt2-774m", 36, 1280, 20); }},
      {"gpt2-1.5b", [] { return gpt2("gpt2-1.5b", 48, 1600, 25); }},
      {"llama2-7b", [] { return llama_like("llama2-7b", 32000, 32, 4096, 32, 32, 11008, 4096, 10000.0f, 1e-5f, false); }},
      {"llama3-8b",
       [] {
         LmConfig c = llama_like("llama3-8b", 128256, 32, 4096, 32, 8, 14336, 8192, 500000.0f, 1e-5f, false);
         return c;
       }},
      {"mistral-7b",
       [] {
         LmConfig c = llama_like("mistral-7b", 32000, 32, 4096, 32, 8, 14336, 32768, 10000.0f, 1e-5f, false);
         c.window = 4096;  // v0.1 kayan pencere
         return c;
       }},
      {"qwen2-0.5b",
       [] {
         LmConfig c = llama_like("qwen2-0.5b", 151936, 24, 896, 14, 2, 4864, 32768, 1000000.0f, 1e-6f, true);
         c.qkv_bias = true;
         return c;
       }},
      {"qwen2-1.5b",
       [] {
         LmConfig c = llama_like("qwen2-1.5b", 151936, 28, 1536, 12, 2, 8960, 32768, 1000000.0f, 1e-6f, true);
         c.qkv_bias = true;
         return c;
       }},
      {"qwen2-7b",
       [] {
         LmConfig c = llama_like("qwen2-7b", 152064, 28, 3584, 28, 4, 18944, 32768, 1000000.0f, 1e-6f, false);
         c.qkv_bias = true;
         return c;
       }},
      {"tinyllama-1.1b",
       [] { return llama_like("tinyllama-1.1b", 32000, 22, 2048, 32, 4, 5632, 2048, 10000.0f, 1e-5f, false); }},
      // FAZ 14: kucuk MoE (8 uzman, k = 2, ~124M toplam / ~38M aktif; 8k sozluk). Mixtral sekli kucultulmus.
      {"moe-124m",
       [] {
         LmConfig c = llama_like("moe-124m", 8192, 5, 512, 8, 8, 1856, 1024, 10000.0f, 1e-5f, true);
         c.moe_experts = 8;
         c.moe_top_k = 2;
         return c;
       }},
  };
  return table;
}

// ---- Kucuk JSON (yalniz bu alanlar: sayi, dize, bool, bir duzey ic nesne) -----------------
struct JValue {
  enum class K { Num, Str, Bool, Obj } k = K::Num;
  double num = 0.0;
  std::string str;
  bool b = false;
  std::map<std::string, JValue> obj;
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}
  JValue parse() {
    JValue v = value();
    ws();
    if (i_ != s_.size()) {
      fail("JSON: fazladan karakter");
    }
    return v;
  }

 private:
  void ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) {
      ++i_;
    }
  }
  char peek() {
    ws();
    if (i_ >= s_.size()) {
      fail("JSON: beklenmeyen son");
    }
    return s_[i_];
  }
  void expect(char c) {
    if (peek() != c) {
      fail(std::string("JSON: '") + c + "' bekleniyordu");
    }
    ++i_;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
        ++i_;
      }
      out += s_[i_++];
    }
    if (i_ >= s_.size()) {
      fail("JSON: kapanmayan dize");
    }
    ++i_;
    return out;
  }
  JValue value() {
    const char c = peek();
    JValue v;
    if (c == '{') {
      v.k = JValue::K::Obj;
      ++i_;
      if (peek() == '}') {
        ++i_;
        return v;
      }
      while (true) {
        const std::string key = string();
        expect(':');
        v.obj[key] = value();
        if (peek() == ',') {
          ++i_;
          continue;
        }
        expect('}');
        return v;
      }
    }
    if (c == '"') {
      v.k = JValue::K::Str;
      v.str = string();
      return v;
    }
    if (s_.compare(i_, 4, "true") == 0) {
      v.k = JValue::K::Bool;
      v.b = true;
      i_ += 4;
      return v;
    }
    if (s_.compare(i_, 5, "false") == 0) {
      v.k = JValue::K::Bool;
      i_ += 5;
      return v;
    }
    char* end = nullptr;
    v.num = std::strtod(s_.c_str() + i_, &end);
    if (end == s_.c_str() + i_) {
      fail("JSON: deger cozulemedi");
    }
    i_ = static_cast<std::size_t>(end - s_.c_str());
    return v;
  }
  const std::string& s_;
  std::size_t i_ = 0;
};

double num(const JValue& v, const std::string& key) {
  if (v.k == JValue::K::Bool) {
    return v.b ? 1.0 : 0.0;
  }
  if (v.k != JValue::K::Num) {
    fail("JSON: " + key + " sayi olmali");
  }
  return v.num;
}
const std::string& str(const JValue& v, const std::string& key) {
  if (v.k != JValue::K::Str) {
    fail("JSON: " + key + " dize olmali");
  }
  return v.str;
}
bool boolean(const JValue& v, const std::string& key) {
  if (v.k != JValue::K::Bool) {
    fail("JSON: " + key + " bool olmali");
  }
  return v.b;
}

std::string fmt(float x) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(x));
  return buf;
}

}  // namespace

LmConfig preset(const std::string& name) {
  const auto& table = presets();
  const auto found = table.find(name);
  if (found == table.end()) {
    fail("bilinmeyen on tanimli: " + name);
  }
  return found->second();
}

std::vector<std::string> preset_names() {
  std::vector<std::string> out;
  for (const auto& [name, fn] : presets()) {
    (void)fn;
    out.push_back(name);
  }
  return out;
}

std::string to_json(const LmConfig& c) {
  std::ostringstream o;
  const RopeScaling& r = c.rope_scaling;
  o << "{\"name\":\"" << c.name << "\",\"vocab_size\":" << c.vocab_size << ",\"n_layer\":" << c.n_layer
    << ",\"d_model\":" << c.d_model << ",\"n_head\":" << c.n_head << ",\"n_kv_head\":" << c.n_kv_head
    << ",\"head_dim\":" << c.head_dim << ",\"d_ff\":" << c.d_ff << ",\"max_position\":" << c.max_position
    << ",\"norm\":\"" << to_string(c.norm) << "\",\"norm_eps\":" << fmt(c.norm_eps)
    << ",\"prenorm\":" << (c.prenorm ? "true" : "false") << ",\"mlp\":\"" << to_string(c.mlp) << "\",\"pos\":\""
    << to_string(c.pos) << "\",\"rope_theta\":" << fmt(c.rope_theta) << ",\"rotary_dim\":" << c.rotary_dim
    << ",\"rope_scaling\":{\"type\":\"" << to_string(r.type) << "\",\"factor\":" << fmt(r.factor)
    << ",\"original_max_position\":" << r.original_max_position << ",\"low_freq_factor\":" << fmt(r.low_freq_factor)
    << ",\"high_freq_factor\":" << fmt(r.high_freq_factor) << ",\"beta_fast\":" << fmt(r.beta_fast)
    << ",\"beta_slow\":" << fmt(r.beta_slow) << ",\"attention_factor\":" << fmt(r.attention_factor) << "}"
    << ",\"tie_embeddings\":" << (c.tie_embeddings ? "true" : "false") << ",\"bias\":" << (c.bias ? "true" : "false")
    << ",\"qkv_bias\":" << (c.qkv_bias ? "true" : "false") << ",\"qk_norm\":" << (c.qk_norm ? "true" : "false")
    << ",\"parallel_block\":" << (c.parallel_block ? "true" : "false") << ",\"window\":" << c.window
    << ",\"attn_softcap\":" << fmt(c.attn_softcap) << ",\"resid_dropout\":" << fmt(c.resid_dropout)
    << ",\"attn_dropout\":" << fmt(c.attn_dropout) << ",\"vocab_pad_multiple\":" << c.vocab_pad_multiple
    << ",\"init\":\"" << to_string(c.init) << "\",\"init_std\":" << fmt(c.init_std)
    << ",\"init_truncated\":" << (c.init_truncated ? "true" : "false") << ",\"mup_base_width\":" << c.mup_base_width
    << ",\"seed\":" << c.seed << ",\"dtype\":\"" << c.dtype << "\",\"recompute\":\""
    << to_string(c.recompute) << "\",\"recompute_every\":" << c.recompute_every << ",\"moe_experts\":" << c.moe_experts
    << ",\"moe_top_k\":" << c.moe_top_k << ",\"moe_every\":" << c.moe_every << ",\"moe_d_ff\":" << c.moe_d_ff
    << ",\"moe_capacity_factor\":" << fmt(c.moe_capacity_factor) << ",\"moe_min_capacity\":" << c.moe_min_capacity
    << ",\"moe_reroute\":" << (c.moe_reroute ? "true" : "false") << ",\"moe_groups\":" << c.moe_groups
    << ",\"moe_noisy\":" << (c.moe_noisy ? "true" : "false") << ",\"moe_shared_expert\":"
    << (c.moe_shared_expert ? "true" : "false") << ",\"moe_aux_alpha\":" << fmt(c.moe_aux_alpha)
    << ",\"moe_z_alpha\":" << fmt(c.moe_z_alpha) << "}";
  return o.str();
}

LmConfig apply_overrides(LmConfig c, const std::string& json) {
  const JValue root = Parser(json).parse();
  if (root.k != JValue::K::Obj) {
    fail("JSON: kok nesne olmali");
  }
  using NK = NormKind;
  using MK = MlpKind;
  using PK = PosKind;
  using IK = InitKind;
  using RT = RopeScaling::Type;
  for (const auto& [key, v] : root.obj) {
    if (key == "name") c.name = str(v, key);
    else if (key == "vocab_size") c.vocab_size = static_cast<int>(num(v, key));
    else if (key == "n_layer") c.n_layer = static_cast<int>(num(v, key));
    else if (key == "d_model") c.d_model = static_cast<int>(num(v, key));
    else if (key == "n_head") c.n_head = static_cast<int>(num(v, key));
    else if (key == "n_kv_head") c.n_kv_head = static_cast<int>(num(v, key));
    else if (key == "head_dim") c.head_dim = static_cast<int>(num(v, key));
    else if (key == "d_ff") c.d_ff = static_cast<int>(num(v, key));
    else if (key == "max_position") c.max_position = static_cast<int>(num(v, key));
    else if (key == "norm") c.norm = parse_enum<NK>(str(v, key), {NK::LayerNorm, NK::RmsNorm}, "norm");
    else if (key == "norm_eps") c.norm_eps = static_cast<float>(num(v, key));
    else if (key == "prenorm") c.prenorm = boolean(v, key);
    else if (key == "mlp") c.mlp = parse_enum<MK>(str(v, key), {MK::Gelu, MK::SwiGlu, MK::GeGlu}, "mlp");
    else if (key == "pos")
      c.pos = parse_enum<PK>(str(v, key), {PK::None, PK::Sinusoidal, PK::Learned, PK::Rope, PK::Alibi}, "pos");
    else if (key == "rope_theta") c.rope_theta = static_cast<float>(num(v, key));
    else if (key == "rotary_dim") c.rotary_dim = static_cast<int>(num(v, key));
    else if (key == "rope_scaling") {
      if (v.k != JValue::K::Obj) {
        fail("JSON: rope_scaling nesne olmali");
      }
      RopeScaling& r = c.rope_scaling;
      for (const auto& [rk, rv] : v.obj) {
        if (rk == "type")
          r.type = parse_enum<RT>(str(rv, rk), {RT::None, RT::Linear, RT::Ntk, RT::DynamicNtk, RT::Yarn, RT::Llama3},
                                  "rope_scaling.type");
        else if (rk == "factor") r.factor = static_cast<float>(num(rv, rk));
        else if (rk == "original_max_position") r.original_max_position = static_cast<int>(num(rv, rk));
        else if (rk == "low_freq_factor") r.low_freq_factor = static_cast<float>(num(rv, rk));
        else if (rk == "high_freq_factor") r.high_freq_factor = static_cast<float>(num(rv, rk));
        else if (rk == "beta_fast") r.beta_fast = static_cast<float>(num(rv, rk));
        else if (rk == "beta_slow") r.beta_slow = static_cast<float>(num(rv, rk));
        else if (rk == "attention_factor") r.attention_factor = static_cast<float>(num(rv, rk));
        else fail("JSON: bilinmeyen rope_scaling anahtari: " + rk);
      }
    }
    else if (key == "tie_embeddings") c.tie_embeddings = boolean(v, key);
    else if (key == "bias") c.bias = boolean(v, key);
    else if (key == "qkv_bias") c.qkv_bias = boolean(v, key);
    else if (key == "qk_norm") c.qk_norm = boolean(v, key);
    else if (key == "parallel_block") c.parallel_block = boolean(v, key);
    else if (key == "window") c.window = static_cast<int>(num(v, key));
    else if (key == "attn_softcap") c.attn_softcap = static_cast<float>(num(v, key));
    else if (key == "resid_dropout") c.resid_dropout = static_cast<float>(num(v, key));
    else if (key == "attn_dropout") c.attn_dropout = static_cast<float>(num(v, key));
    else if (key == "vocab_pad_multiple") c.vocab_pad_multiple = static_cast<int>(num(v, key));
    else if (key == "init") c.init = parse_enum<IK>(str(v, key), {IK::Gpt2, IK::Llama, IK::MuP}, "init");
    else if (key == "init_std") c.init_std = static_cast<float>(num(v, key));
    else if (key == "init_truncated") c.init_truncated = boolean(v, key);
    else if (key == "mup_base_width") c.mup_base_width = static_cast<int>(num(v, key));
    else if (key == "seed") c.seed = static_cast<std::uint64_t>(num(v, key));
    else if (key == "dtype") c.dtype = str(v, key);
    else if (key == "recompute")
      c.recompute = parse_enum<RecomputeKind>(str(v, key), {RecomputeKind::None, RecomputeKind::Full, RecomputeKind::EveryK},
                                              "recompute");
    else if (key == "recompute_every") c.recompute_every = static_cast<int>(num(v, key));
    else if (key == "moe_experts") c.moe_experts = static_cast<int>(num(v, key));
    else if (key == "moe_top_k") c.moe_top_k = static_cast<int>(num(v, key));
    else if (key == "moe_every") c.moe_every = static_cast<int>(num(v, key));
    else if (key == "moe_d_ff") c.moe_d_ff = static_cast<int>(num(v, key));
    else if (key == "moe_capacity_factor") c.moe_capacity_factor = static_cast<float>(num(v, key));
    else if (key == "moe_min_capacity") c.moe_min_capacity = static_cast<int>(num(v, key));
    else if (key == "moe_reroute") c.moe_reroute = boolean(v, key);
    else if (key == "moe_groups") c.moe_groups = static_cast<int>(num(v, key));
    else if (key == "moe_noisy") c.moe_noisy = boolean(v, key);
    else if (key == "moe_shared_expert") c.moe_shared_expert = boolean(v, key);
    else if (key == "moe_aux_alpha") c.moe_aux_alpha = static_cast<float>(num(v, key));
    else if (key == "moe_z_alpha") c.moe_z_alpha = static_cast<float>(num(v, key));
    else fail("JSON: bilinmeyen anahtar: " + key);
  }
  return c;
}

LmConfig from_json(const std::string& json) { return apply_overrides(LmConfig{}, json); }

}  // namespace lm
}  // namespace nexus
