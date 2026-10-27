/**
 * @file hf.cpp
 * @brief FAZ 17.2: Hugging Face config + safetensors ice / disa aktarma (bkz. nexus/interop/hf.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/hf.hpp"

#include "nexus/autograd/autograd.hpp"
#include "nexus/interop/safetensors.hpp"
#include "transpose.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;

namespace nexus {
namespace interop {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::string read_text(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw InteropError("acilamadi: " + p.string());
  std::ostringstream o;
  o << in.rdbuf();
  return o.str();
}

std::vector<float>& values_of(const lm::LanguageModel& m, const std::string& name) {
  const lm::NamedParam* p = m.find(name);
  if (p == nullptr) throw InteropError("Nexus modelinde parametre yok: " + name);
  return const_cast<autograd::Variable&>(p->value).mutable_values();
}

/// HF [out, in] -> Nexus [in, out] blogu: dst[r * dst_cols + col0 + c] = src[c * in + r].
void put_transposed(std::vector<float>& dst, std::int64_t dst_cols, std::int64_t col0, const std::vector<float>& src,
                    std::int64_t out, std::int64_t in) {
  detail::transpose_block(dst.data(), dst_cols, col0, src.data(), in, 0, out, in);
}

/// Nexus [in, out] blogu -> HF [out, in]: o[c * in + r] = src[r * src_cols + col0 + c].
std::vector<float> get_transposed(const std::vector<float>& src, std::int64_t src_cols, std::int64_t col0, std::int64_t out,
                                  std::int64_t in) {
  std::vector<float> o(static_cast<std::size_t>(out * in));
  detail::transpose_block(o.data(), in, 0, src.data(), src_cols, col0, in, out);
  return o;
}

}  // namespace

const std::vector<HfTemplate>& hf_templates() {
  static const std::vector<HfTemplate> t = {
      {"LlamaForCausalLM", "llama/v1"}, {"MistralForCausalLM", "mistral/v1"}, {"Qwen2ForCausalLM", "qwen2/v1"}};
  return t;
}

lm::LmConfig hf_config(const Json& j, std::string* used) {
  const std::string arch = j.at("architectures").arr.at(0).as_string();
  std::string tmpl;
  for (const HfTemplate& t : hf_templates())
    if (t.architecture == arch) tmpl = t.name;
  if (tmpl.empty()) throw InteropError("HF: desteklenmeyen mimari '" + arch + "' (llama/v1, mistral/v1, qwen2/v1)");
  if (used) *used = tmpl;
  auto num = [&](const char* k, double d) { const Json* v = j.get(k); return v && !v->is_null() ? v->as_double() : d; };
  auto flag = [&](const char* k, bool d) { const Json* v = j.get(k); return v && !v->is_null() ? v->as_bool() : d; };
  const std::string act = j.has("hidden_act") ? j.at("hidden_act").as_string() : "silu";
  if (act != "silu") throw InteropError("HF: hidden_act '" + act + "' desteklenmez (silu)");
  if (flag("mlp_bias", false)) throw InteropError("HF: mlp_bias = true desteklenmez");
  if (flag("attention_bias", false)) throw InteropError("HF: attention_bias = true (o_proj bias) desteklenmez");
  lm::LmConfig c;
  c.name = j.has("_name_or_path") ? j.at("_name_or_path").as_string() : tmpl;
  c.vocab_size = static_cast<int>(j.at("vocab_size").as_int());
  c.n_layer = static_cast<int>(j.at("num_hidden_layers").as_int());
  c.d_model = static_cast<int>(j.at("hidden_size").as_int());
  c.n_head = static_cast<int>(j.at("num_attention_heads").as_int());
  c.n_kv_head = static_cast<int>(num("num_key_value_heads", c.n_head));
  c.head_dim = j.has("head_dim") && !j.at("head_dim").is_null() ? static_cast<int>(j.at("head_dim").as_int()) : 0;
  c.d_ff = static_cast<int>(j.at("intermediate_size").as_int());
  c.max_position = static_cast<int>(num("max_position_embeddings", 2048));
  c.norm = lm::NormKind::RmsNorm;
  c.norm_eps = static_cast<float>(num("rms_norm_eps", 1e-6));
  c.mlp = lm::MlpKind::SwiGlu;
  c.pos = lm::PosKind::Rope;
  c.rope_theta = static_cast<float>(num("rope_theta", 10000.0));
  c.tie_embeddings = flag("tie_word_embeddings", tmpl == "qwen2/v1");
  c.init = lm::InitKind::Llama;
  if (tmpl == "qwen2/v1") {
    c.qkv_bias = true;
    if (flag("use_sliding_window", false)) throw InteropError("HF qwen2: use_sliding_window = true desteklenmez");
  }
  if (tmpl == "mistral/v1") {
    const Json* sw = j.get("sliding_window");
    if (sw && !sw->is_null()) c.window = static_cast<int>(sw->as_int());
  }
  if (const Json* rs = j.get("rope_scaling"); rs && !rs->is_null()) {
    const std::string type = rs->has("rope_type") ? rs->at("rope_type").as_string() : rs->at("type").as_string();
    using T = lm::RopeScaling::Type;
    c.rope_scaling.factor = static_cast<float>(rs->at("factor").as_double());
    if (rs->has("original_max_position_embeddings"))
      c.rope_scaling.original_max_position = static_cast<int>(rs->at("original_max_position_embeddings").as_int());
    if (type == "linear") c.rope_scaling.type = T::Linear;
    else if (type == "dynamic") c.rope_scaling.type = T::DynamicNtk;
    else if (type == "yarn") {
      c.rope_scaling.type = T::Yarn;
      if (rs->has("beta_fast")) c.rope_scaling.beta_fast = static_cast<float>(rs->at("beta_fast").as_double());
      if (rs->has("beta_slow")) c.rope_scaling.beta_slow = static_cast<float>(rs->at("beta_slow").as_double());
      if (rs->has("attention_factor") && !rs->at("attention_factor").is_null())
        c.rope_scaling.attention_factor = static_cast<float>(rs->at("attention_factor").as_double());
    } else if (type == "llama3") {
      c.rope_scaling.type = T::Llama3;
      c.rope_scaling.low_freq_factor = static_cast<float>(rs->at("low_freq_factor").as_double());
      c.rope_scaling.high_freq_factor = static_cast<float>(rs->at("high_freq_factor").as_double());
    } else if (type != "default") {
      throw InteropError("HF: rope_scaling '" + type + "' desteklenmez");
    }
  }
  return c;
}

lm::LanguageModel load_hf(const std::string& dir, bool mmap, HfLoadStats* stats) {
  const auto t0 = Clock::now();
  HfLoadStats st;
  st.mmap = mmap;
  const fs::path d(dir);
  const Json cfg = parse_json(read_text(d / "config.json"), (d / "config.json").string());
  lm::LmConfig c = hf_config(cfg, &st.template_name);
  // parcalar
  std::vector<std::string> files;
  if (fs::exists(d / "model.safetensors.index.json")) {
    const Json idx = parse_json(read_text(d / "model.safetensors.index.json"), "index");
    std::set<std::string> uniq;
    for (const auto& kv : idx.at("weight_map").obj) uniq.insert(kv.second.as_string());
    files.assign(uniq.begin(), uniq.end());
  } else {
    files.push_back("model.safetensors");
  }
  std::vector<SafeTensors> shards;
  std::map<std::string, std::pair<int, const StTensor*>> where;
  const auto to = Clock::now();
  for (const std::string& f : files) {
    shards.push_back(SafeTensors::open((d / f).string(), mmap));
    st.files.push_back(f);
  }
  for (std::size_t s = 0; s < shards.size(); ++s)
    for (const StTensor& t : shards[s].tensors()) where[t.name] = {static_cast<int>(s), &t};
  st.open_ms = ms_since(to);
  std::set<std::string> used;
  auto load = [&](const std::string& name, std::vector<std::int64_t> shape) {
    auto it = where.find(name);
    if (it == where.end()) throw InteropError("HF: agirlik yok '" + name + "' (" + st.template_name + ")");
    const StTensor& t = *it->second.second;
    if (t.shape != shape) {
      std::string got, want;
      for (auto v : t.shape) got += std::to_string(v) + " ";
      for (auto v : shape) want += std::to_string(v) + " ";
      throw InteropError("HF: " + name + " sekli [" + got + "], beklenen [" + want + "]");
    }
    used.insert(name);
    st.bytes += t.end - t.begin;
    st.tensors += 1;
    return shards[static_cast<std::size_t>(it->second.first)].to_f32(t);
  };
  lm::LanguageModel m(c, true);  // rastgele baslatma yok: her agirlik asagida yuklenir (eksik olan acik hata)
  const std::int64_t dm = c.d_model, H = c.n_head, Hk = c.kv_heads(), D = c.hdim(), F = c.d_ff, V = c.vocab_size,
                     Vp = c.padded_vocab();
  {
    const auto e = load("model.embed_tokens.weight", {V, dm});
    std::vector<float>& w = values_of(m, "wte");
    std::fill(w.begin(), w.end(), 0.0f);
    std::copy(e.begin(), e.end(), w.begin());
  }
  for (int l = 0; l < c.n_layer; ++l) {
    const std::string hp = "model.layers." + std::to_string(l) + ".", np = "h" + std::to_string(l) + ".";
    values_of(m, np + "norm1.weight") = load(hp + "input_layernorm.weight", {dm});
    values_of(m, np + "norm2.weight") = load(hp + "post_attention_layernorm.weight", {dm});
    std::vector<float>& qkv = values_of(m, np + "attn.qkv.weight");
    const std::int64_t cols = (H + 2 * Hk) * D;
    put_transposed(qkv, cols, 0, load(hp + "self_attn.q_proj.weight", {H * D, dm}), H * D, dm);
    put_transposed(qkv, cols, H * D, load(hp + "self_attn.k_proj.weight", {Hk * D, dm}), Hk * D, dm);
    put_transposed(qkv, cols, (H + Hk) * D, load(hp + "self_attn.v_proj.weight", {Hk * D, dm}), Hk * D, dm);
    if (c.qkv_bias) {
      std::vector<float>& b = values_of(m, np + "attn.qkv.bias");
      const auto bq = load(hp + "self_attn.q_proj.bias", {H * D}), bk = load(hp + "self_attn.k_proj.bias", {Hk * D}),
                 bv = load(hp + "self_attn.v_proj.bias", {Hk * D});
      std::copy(bq.begin(), bq.end(), b.begin());
      std::copy(bk.begin(), bk.end(), b.begin() + H * D);
      std::copy(bv.begin(), bv.end(), b.begin() + (H + Hk) * D);
    }
    std::vector<float>& o = values_of(m, np + "attn.o.weight");
    put_transposed(o, dm, 0, load(hp + "self_attn.o_proj.weight", {dm, H * D}), dm, H * D);
    std::vector<float>& up = values_of(m, np + "mlp.up.weight");
    put_transposed(up, 2 * F, 0, load(hp + "mlp.gate_proj.weight", {F, dm}), F, dm);
    put_transposed(up, 2 * F, F, load(hp + "mlp.up_proj.weight", {F, dm}), F, dm);
    std::vector<float>& down = values_of(m, np + "mlp.down.weight");
    put_transposed(down, dm, 0, load(hp + "mlp.down_proj.weight", {dm, F}), dm, F);
  }
  values_of(m, "norm_f.weight") = load("model.norm.weight", {dm});
  if (!c.tie_embeddings) {
    const auto h = load("lm_head.weight", {V, dm});
    std::vector<float>& w = values_of(m, "lm_head.weight");
    std::fill(w.begin(), w.end(), 0.0f);
    std::copy(h.begin(), h.end(), w.begin());
  }
  (void)Vp;
  for (const auto& kv : where) {
    if (used.count(kv.first) || kv.first.find("rotary_emb.inv_freq") != std::string::npos) continue;
    if (c.tie_embeddings && kv.first == "lm_head.weight") continue;  // bagli: kopya (bazi checkpoint'ler yazar)
    throw InteropError("HF: beklenmeyen agirlik '" + kv.first + "' (" + st.template_name + " sablonu esleyemedi)");
  }
  st.ms = ms_since(t0);
  if (stats) *stats = st;
  return m;
}

void save_hf(const lm::LanguageModel& m, const std::string& dir, const std::string& dtype) {
  const lm::LmConfig& c = m.config();
  if (c.norm != lm::NormKind::RmsNorm || c.mlp != lm::MlpKind::SwiGlu || c.pos != lm::PosKind::Rope || c.bias || !c.prenorm ||
      c.parallel_block || c.qk_norm || c.moe_experts > 0 || c.attn_softcap > 0.0f || c.rotary_dim != 0 ||
      c.init == lm::InitKind::MuP)
    throw InteropError("save_hf: yapilandirma HF llama/mistral/qwen2 sablonlarinin disinda");
  if (dtype != "F32" && dtype != "F16" && dtype != "BF16") throw InteropError("save_hf: dtype F32 | F16 | BF16");
  fs::create_directories(dir);
  const std::string arch = c.qkv_bias ? "Qwen2ForCausalLM" : (c.window > 0 ? "MistralForCausalLM" : "LlamaForCausalLM");
  const std::int64_t dm = c.d_model, H = c.n_head, Hk = c.kv_heads(), D = c.hdim(), F = c.d_ff, V = c.vocab_size;
  std::vector<std::pair<std::string, std::pair<std::vector<std::int64_t>, std::vector<float>>>> out;
  auto add = [&](const std::string& n, std::vector<std::int64_t> s, std::vector<float> v) { out.push_back({n, {std::move(s), std::move(v)}}); };
  auto vals = [&](const std::string& n) { return m.find(n)->value.values(); };
  {
    auto w = vals("wte");
    w.resize(static_cast<std::size_t>(V * dm));
    add("model.embed_tokens.weight", {V, dm}, w);
  }
  for (int l = 0; l < c.n_layer; ++l) {
    const std::string hp = "model.layers." + std::to_string(l) + ".", np = "h" + std::to_string(l) + ".";
    add(hp + "input_layernorm.weight", {dm}, vals(np + "norm1.weight"));
    add(hp + "post_attention_layernorm.weight", {dm}, vals(np + "norm2.weight"));
    const auto qkv = vals(np + "attn.qkv.weight");
    const std::int64_t cols = (H + 2 * Hk) * D;
    add(hp + "self_attn.q_proj.weight", {H * D, dm}, get_transposed(qkv, cols, 0, H * D, dm));
    add(hp + "self_attn.k_proj.weight", {Hk * D, dm}, get_transposed(qkv, cols, H * D, Hk * D, dm));
    add(hp + "self_attn.v_proj.weight", {Hk * D, dm}, get_transposed(qkv, cols, (H + Hk) * D, Hk * D, dm));
    if (c.qkv_bias) {
      const auto b = vals(np + "attn.qkv.bias");
      add(hp + "self_attn.q_proj.bias", {H * D}, std::vector<float>(b.begin(), b.begin() + H * D));
      add(hp + "self_attn.k_proj.bias", {Hk * D}, std::vector<float>(b.begin() + H * D, b.begin() + (H + Hk) * D));
      add(hp + "self_attn.v_proj.bias", {Hk * D}, std::vector<float>(b.begin() + (H + Hk) * D, b.end()));
    }
    add(hp + "self_attn.o_proj.weight", {dm, H * D}, get_transposed(vals(np + "attn.o.weight"), dm, 0, dm, H * D));
    const auto up = vals(np + "mlp.up.weight");
    add(hp + "mlp.gate_proj.weight", {F, dm}, get_transposed(up, 2 * F, 0, F, dm));
    add(hp + "mlp.up_proj.weight", {F, dm}, get_transposed(up, 2 * F, F, F, dm));
    add(hp + "mlp.down_proj.weight", {dm, F}, get_transposed(vals(np + "mlp.down.weight"), dm, 0, dm, F));
  }
  add("model.norm.weight", {dm}, vals("norm_f.weight"));
  if (!c.tie_embeddings) {
    auto w = vals("lm_head.weight");
    w.resize(static_cast<std::size_t>(V * dm));
    add("lm_head.weight", {V, dm}, w);
  }
  // sayi bicimi
  std::vector<std::vector<std::uint8_t>> blobs;
  std::vector<StWrite> sw;
  for (auto& [n, sv] : out) {
    const auto& v = sv.second;
    std::vector<std::uint8_t> b;
    if (dtype == "F32") {
      b.resize(v.size() * 4);
      std::memcpy(b.data(), v.data(), b.size());
    } else {
      b.resize(v.size() * 2);
      for (std::size_t i = 0; i < v.size(); ++i) {
        std::uint32_t u;
        std::memcpy(&u, &v[i], 4);
        std::uint16_t h;
        if (dtype == "BF16") {
          h = static_cast<std::uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
        } else {  // F16 (RNE, alt normal / tasma dahil)
          const std::uint32_t sign = (u >> 16) & 0x8000u;
          const int e = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
          std::uint32_t mant = u & 0x7FFFFFu;
          if (((u >> 23) & 0xFF) == 0xFF) {
            h = static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
          } else if (e >= 31) {
            h = static_cast<std::uint16_t>(sign | 0x7C00u);
          } else if (e <= 0) {
            if (e < -10) {
              h = static_cast<std::uint16_t>(sign);
            } else {
              mant |= 0x800000u;
              const int shift = 14 - e;
              std::uint32_t hm = mant >> shift;
              const std::uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
              if (rem > half || (rem == half && (hm & 1u))) ++hm;
              h = static_cast<std::uint16_t>(sign | hm);
            }
          } else {
            std::uint32_t hv = (static_cast<std::uint32_t>(e) << 10) | (mant >> 13);
            const std::uint32_t rem = mant & 0x1FFFu;
            if (rem > 0x1000u || (rem == 0x1000u && (hv & 1u))) ++hv;
            h = static_cast<std::uint16_t>(sign | hv);
          }
        }
        std::memcpy(b.data() + 2 * i, &h, 2);
      }
    }
    blobs.push_back(std::move(b));
  }
  for (std::size_t i = 0; i < out.size(); ++i)
    sw.push_back(StWrite{out[i].first, dtype, out[i].second.first, blobs[i].data(), blobs[i].size()});
  write_safetensors((fs::path(dir) / "model.safetensors").string(), sw, {{"format", "pt"}, {"producer", "nexus"}});
  std::ostringstream j;
  j.precision(17);
  j << "{\n  \"architectures\": [\"" << arch << "\"],\n  \"model_type\": \""
    << (c.qkv_bias ? "qwen2" : (c.window > 0 ? "mistral" : "llama")) << "\",\n  \"vocab_size\": " << V
    << ",\n  \"hidden_size\": " << dm << ",\n  \"intermediate_size\": " << F << ",\n  \"num_hidden_layers\": " << c.n_layer
    << ",\n  \"num_attention_heads\": " << H << ",\n  \"num_key_value_heads\": " << Hk << ",\n  \"head_dim\": " << D
    << ",\n  \"max_position_embeddings\": " << c.max_position << ",\n  \"rms_norm_eps\": " << c.norm_eps
    << ",\n  \"rope_theta\": " << c.rope_theta << ",\n  \"tie_word_embeddings\": " << (c.tie_embeddings ? "true" : "false")
    << ",\n  \"hidden_act\": \"silu\",\n  \"attention_bias\": false,\n  \"rope_scaling\": null";
  if (c.window > 0) j << ",\n  \"sliding_window\": " << c.window;
  j << ",\n  \"torch_dtype\": \"" << (dtype == "F32" ? "float32" : dtype == "F16" ? "float16" : "bfloat16") << "\"\n}\n";
  std::ofstream(fs::path(dir) / "config.json") << j.str();
}

}  // namespace interop
}  // namespace nexus
