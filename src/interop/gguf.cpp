/**
 * @file gguf.cpp
 * @brief FAZ 17.3: GGUF v3 yazma / okuma, Q8_0 / Q4_0 blok nicemleme (ggml referans formulleri).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/gguf.hpp"

#include "nexus/autograd/autograd.hpp"
#include "transpose.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace nexus {
namespace interop {

namespace {

using Clock = std::chrono::steady_clock;

std::uint16_t f2h(float f) {  // RNE
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const int e = static_cast<int>((u >> 23) & 0xFF) - 127 + 15;
  std::uint32_t mant = u & 0x7FFFFFu;
  if (((u >> 23) & 0xFF) == 0xFF) return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  if (e >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return static_cast<std::uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - e;
    std::uint32_t hm = mant >> shift;
    const std::uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (hm & 1u))) ++hm;
    return static_cast<std::uint16_t>(sign | hm);
  }
  std::uint32_t hv = (static_cast<std::uint32_t>(e) << 10) | (mant >> 13);
  const std::uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (hv & 1u))) ++hv;
  return static_cast<std::uint16_t>(sign | hv);
}

float h2f(std::uint16_t h) {
  const std::uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, f = h & 0x3FF;
  std::uint32_t o;
  if (e == 0) {
    if (f == 0) {
      o = s;
    } else {
      int ee = -1;
      std::uint32_t ff = f;
      do {
        ++ee;
        ff <<= 1;
      } while ((ff & 0x400) == 0);
      o = s | ((127 - 15 - ee) << 23) | ((ff & 0x3FF) << 13);
    }
  } else if (e == 31) {
    o = s | 0x7F800000u | (f << 13);
  } else {
    o = s | ((e + 127 - 15) << 23) | (f << 13);
  }
  float r;
  std::memcpy(&r, &o, 4);
  return r;
}

enum : std::uint32_t { T_U8 = 0, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

struct W {
  std::vector<std::uint8_t> b;
  template <typename T>
  void pod(T v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
  }
  void str(const std::string& s) {
    pod<std::uint64_t>(s.size());
    b.insert(b.end(), s.begin(), s.end());
  }
};

struct Kv {
  W w;
  int count = 0;
  void u32(const std::string& k, std::uint32_t v) {
    w.str(k);
    w.pod<std::uint32_t>(T_U32);
    w.pod(v);
    ++count;
  }
  void f32(const std::string& k, float v) {
    w.str(k);
    w.pod<std::uint32_t>(T_F32);
    w.pod(v);
    ++count;
  }
  void str(const std::string& k, const std::string& v) {
    w.str(k);
    w.pod<std::uint32_t>(T_STR);
    w.str(v);
    ++count;
  }
  void strs(const std::string& k, const std::vector<std::string>& v) {
    w.str(k);
    w.pod<std::uint32_t>(T_ARR);
    w.pod<std::uint32_t>(T_STR);
    w.pod<std::uint64_t>(v.size());
    for (const auto& s : v) w.str(s);
    ++count;
  }
  template <typename T>
  void arr(const std::string& k, std::uint32_t type, const std::vector<T>& v) {
    w.str(k);
    w.pod<std::uint32_t>(T_ARR);
    w.pod<std::uint32_t>(type);
    w.pod<std::uint64_t>(v.size());
    for (const T& x : v) w.pod(x);
    ++count;
  }
};

/// llama.cpp convert_hf_to_gguf.permute: [n_head, 2, rows / n_head / 2, cols] -> eksen 1 ve 2 takas.
std::vector<float> permute_qk(const std::vector<float>& w, std::int64_t rows, std::int64_t cols, std::int64_t heads) {
  std::vector<float> o(w.size());
  const std::int64_t per = rows / heads, half = per / 2;
  for (std::int64_t h = 0; h < heads; ++h)
    for (std::int64_t two = 0; two < 2; ++two)
      for (std::int64_t i = 0; i < half; ++i) {
        const std::int64_t src = h * per + two * half + i;  // [h, two, i]
        const std::int64_t dst = h * per + i * 2 + two;     // [h, i, two]
        std::copy_n(w.begin() + src * cols, cols, o.begin() + dst * cols);
      }
  return o;
}

/// Nexus [in, out] -> HF [out, in] (satir-ana): llama.cpp ne = [in, out].
std::vector<float> to_out_in(const std::vector<float>& src, std::int64_t src_cols, std::int64_t col0, std::int64_t out,
                             std::int64_t in) {
  std::vector<float> o(static_cast<std::size_t>(out * in));
  detail::transpose_block(o.data(), in, 0, src.data(), src_cols, col0, in, out);
  return o;
}

/// permute_qk'nin tersi: [h, i, two] -> [h, two, i].
std::vector<float> unpermute_qk(const std::vector<float>& w, std::int64_t rows, std::int64_t cols, std::int64_t heads) {
  std::vector<float> o(w.size());
  const std::int64_t per = rows / heads, half = per / 2;
  for (std::int64_t h = 0; h < heads; ++h)
    for (std::int64_t two = 0; two < 2; ++two)
      for (std::int64_t i = 0; i < half; ++i)
        std::copy_n(w.begin() + (h * per + i * 2 + two) * cols, cols, o.begin() + (h * per + two * half + i) * cols);
  return o;
}

}  // namespace

const char* to_string(GgufType t) noexcept {
  switch (t) {
    case GgufType::F32: return "F32";
    case GgufType::F16: return "F16";
    case GgufType::Q4_0: return "Q4_0";
    case GgufType::Q8_0: return "Q8_0";
  }
  return "?";
}

std::vector<std::uint8_t> quantize_q8_0(const float* x, std::size_t n) {
  if (n % 32 != 0) throw InteropError("Q8_0: eleman sayisi 32'nin kati olmali");
  std::vector<std::uint8_t> o(n / 32 * 34);
  for (std::size_t b = 0; b < n / 32; ++b) {
    const float* v = x + b * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(v[j]));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const std::uint16_t dh = f2h(d);
    std::memcpy(o.data() + b * 34, &dh, 2);
    for (int j = 0; j < 32; ++j) o[b * 34 + 2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::lround(v[j] * id)));
  }
  return o;
}

std::vector<std::uint8_t> quantize_q4_0(const float* x, std::size_t n) {
  if (n % 32 != 0) throw InteropError("Q4_0: eleman sayisi 32'nin kati olmali");
  std::vector<std::uint8_t> o(n / 32 * 18);
  for (std::size_t b = 0; b < n / 32; ++b) {
    const float* v = x + b * 32;
    float amax = 0.0f, mx = 0.0f;
    for (int j = 0; j < 32; ++j)
      if (std::fabs(v[j]) > amax) {
        amax = std::fabs(v[j]);
        mx = v[j];
      }
    const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const std::uint16_t dh = f2h(d);
    std::memcpy(o.data() + b * 18, &dh, 2);
    for (int j = 0; j < 16; ++j) {
      const int q0 = std::min(15, static_cast<int>(static_cast<std::int8_t>(v[j] * id + 8.5f)));
      const int q1 = std::min(15, static_cast<int>(static_cast<std::int8_t>(v[j + 16] * id + 8.5f)));
      o[b * 18 + 2 + j] = static_cast<std::uint8_t>(q0 | (q1 << 4));
    }
  }
  return o;
}

std::vector<float> dequantize(GgufType t, const std::uint8_t* p, std::size_t n) {
  std::vector<float> o(n);
  if (t == GgufType::F32) {
    std::memcpy(o.data(), p, n * 4);
  } else if (t == GgufType::F16) {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t h;
      std::memcpy(&h, p + 2 * i, 2);
      o[i] = h2f(h);
    }
  } else if (t == GgufType::Q8_0) {
    for (std::size_t b = 0; b < n / 32; ++b) {
      std::uint16_t dh;
      std::memcpy(&dh, p + b * 34, 2);
      const float d = h2f(dh);
      for (int j = 0; j < 32; ++j) o[b * 32 + j] = d * static_cast<float>(static_cast<std::int8_t>(p[b * 34 + 2 + j]));
    }
  } else {
    for (std::size_t b = 0; b < n / 32; ++b) {
      std::uint16_t dh;
      std::memcpy(&dh, p + b * 18, 2);
      const float d = h2f(dh);
      for (int j = 0; j < 16; ++j) {
        const std::uint8_t q = p[b * 18 + 2 + j];
        o[b * 32 + j] = d * static_cast<float>((q & 15) - 8);
        o[b * 32 + j + 16] = d * static_cast<float>((q >> 4) - 8);
      }
    }
  }
  return o;
}

void export_gguf(const lm::LanguageModel& m, const std::string& path, const GgufOptions& opt, GgufStats* stats) {
  const auto t0 = Clock::now();
  const lm::LmConfig& c = m.config();
  if (c.norm != lm::NormKind::RmsNorm || c.mlp != lm::MlpKind::SwiGlu || c.pos != lm::PosKind::Rope || c.bias || c.qkv_bias ||
      !c.prenorm || c.parallel_block || c.qk_norm || c.moe_experts > 0 || c.window > 0 || c.attn_softcap > 0.0f ||
      c.rotary_dim != 0 || c.rope_scaling.type != lm::RopeScaling::Type::None)
    throw InteropError("GGUF: yalniz llama mimarisi (RMSNorm + SwiGLU + RoPE, biassiz, olceklemesiz) disa aktarilir");
  const std::int64_t dm = c.d_model, H = c.n_head, Hk = c.kv_heads(), D = c.hdim(), F = c.d_ff, V = c.vocab_size;
  Kv kv;
  kv.str("general.architecture", "llama");
  kv.str("general.name", opt.name);
  kv.u32("general.alignment", 32);
  kv.u32("general.quantization_version", 2);
  kv.u32("general.file_type", opt.type == GgufType::F32 ? 0 : opt.type == GgufType::F16 ? 1 : opt.type == GgufType::Q4_0 ? 2 : 7);
  kv.u32("llama.context_length", static_cast<std::uint32_t>(c.max_position));
  kv.u32("llama.embedding_length", static_cast<std::uint32_t>(dm));
  kv.u32("llama.block_count", static_cast<std::uint32_t>(c.n_layer));
  kv.u32("llama.feed_forward_length", static_cast<std::uint32_t>(F));
  kv.u32("llama.rope.dimension_count", static_cast<std::uint32_t>(D));
  kv.u32("llama.attention.head_count", static_cast<std::uint32_t>(H));
  kv.u32("llama.attention.head_count_kv", static_cast<std::uint32_t>(Hk));
  kv.f32("llama.attention.layer_norm_rms_epsilon", c.norm_eps);
  kv.f32("llama.rope.freq_base", c.rope_theta);
  kv.u32("llama.vocab_size", static_cast<std::uint32_t>(V));
  if (!opt.tokenizer_json.empty()) {
    std::ifstream in(opt.tokenizer_json, std::ios::binary);
    if (!in) throw InteropError("GGUF: tokenizer.json acilamadi " + opt.tokenizer_json);
    std::ostringstream ss;
    ss << in.rdbuf();
    const Json tj = parse_json(ss.str(), opt.tokenizer_json);
    std::vector<std::string> tokens(static_cast<std::size_t>(V));
    std::vector<std::int32_t> types(static_cast<std::size_t>(V), 1);
    std::vector<float> scores(static_cast<std::size_t>(V), 0.0f);
    for (const auto& [tok, id] : tj.at("model").at("vocab").obj) {
      const std::int64_t i = id.as_int();
      if (i < 0 || i >= V) continue;
      // HF bayt geri dusus: "<0x0A>" -> tip 6 (bayt); SentencePiece bosluk isareti oldugu gibi
      tokens[static_cast<std::size_t>(i)] = tok;
      if (tok.size() == 6 && tok.rfind("<0x", 0) == 0 && tok.back() == '>') types[static_cast<std::size_t>(i)] = 6;
      scores[static_cast<std::size_t>(i)] = -static_cast<float>(i);  // SentencePiece skoru yok (tokenizer.model indirilmedi)
    }
    std::uint32_t bos = 1, eos = 2, unk = 0;
    for (const Json& a : tj.at("added_tokens").arr) {
      const std::int64_t i = a.at("id").as_int();
      if (i >= 0 && i < V) types[static_cast<std::size_t>(i)] = a.at("content").as_string() == "<unk>" ? 2 : 3;
      if (a.at("content").as_string() == "<s>") bos = static_cast<std::uint32_t>(i);
      if (a.at("content").as_string() == "</s>") eos = static_cast<std::uint32_t>(i);
      if (a.at("content").as_string() == "<unk>") unk = static_cast<std::uint32_t>(i);
    }
    kv.str("tokenizer.ggml.model", "llama");
    kv.strs("tokenizer.ggml.tokens", tokens);
    kv.arr("tokenizer.ggml.scores", T_F32, scores);
    kv.arr("tokenizer.ggml.token_type", T_I32, types);
    kv.u32("tokenizer.ggml.bos_token_id", bos);
    kv.u32("tokenizer.ggml.eos_token_id", eos);
    kv.u32("tokenizer.ggml.unknown_token_id", unk);
  }
  // tensorler (HF [out, in] satir-ana duzeninde)
  struct T {
    std::string name;
    std::vector<std::uint64_t> ne;
    GgufType type;
    std::vector<std::uint8_t> data;
  };
  std::vector<T> ts;
  auto vals = [&](const std::string& n) { return m.find(n)->value.values(); };
  auto add = [&](const std::string& name, const std::vector<float>& v, std::int64_t rows, std::int64_t cols) {
    T t;
    t.name = name;
    t.ne = rows == 1 ? std::vector<std::uint64_t>{static_cast<std::uint64_t>(cols)}
                     : std::vector<std::uint64_t>{static_cast<std::uint64_t>(cols), static_cast<std::uint64_t>(rows)};
    t.type = rows == 1 ? GgufType::F32 : opt.type;
    if (t.type != GgufType::F32 && t.type != GgufType::F16 && cols % 32 != 0) t.type = GgufType::F16;  // blok 32 yoksa
    const std::size_t n = static_cast<std::size_t>(rows * cols);
    if (t.type == GgufType::F32) {
      t.data.resize(n * 4);
      std::memcpy(t.data.data(), v.data(), n * 4);
    } else if (t.type == GgufType::F16) {
      t.data.resize(n * 2);
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint16_t h = f2h(v[i]);
        std::memcpy(t.data.data() + 2 * i, &h, 2);
      }
    } else if (t.type == GgufType::Q8_0) {
      t.data = quantize_q8_0(v.data(), n);
    } else {
      t.data = quantize_q4_0(v.data(), n);
    }
    ts.push_back(std::move(t));
  };
  {
    auto e = vals("wte");
    e.resize(static_cast<std::size_t>(V * dm));
    add("token_embd.weight", e, V, dm);
  }
  for (int l = 0; l < c.n_layer; ++l) {
    const std::string np = "h" + std::to_string(l) + ".", gp = "blk." + std::to_string(l) + ".";
    add(gp + "attn_norm.weight", vals(np + "norm1.weight"), 1, dm);
    const auto qkv = vals(np + "attn.qkv.weight");
    const std::int64_t cols = (H + 2 * Hk) * D;
    add(gp + "attn_q.weight", permute_qk(to_out_in(qkv, cols, 0, H * D, dm), H * D, dm, H), H * D, dm);
    add(gp + "attn_k.weight", permute_qk(to_out_in(qkv, cols, H * D, Hk * D, dm), Hk * D, dm, Hk), Hk * D, dm);
    add(gp + "attn_v.weight", to_out_in(qkv, cols, (H + Hk) * D, Hk * D, dm), Hk * D, dm);
    add(gp + "attn_output.weight", to_out_in(vals(np + "attn.o.weight"), dm, 0, dm, H * D), dm, H * D);
    add(gp + "ffn_norm.weight", vals(np + "norm2.weight"), 1, dm);
    const auto up = vals(np + "mlp.up.weight");
    add(gp + "ffn_gate.weight", to_out_in(up, 2 * F, 0, F, dm), F, dm);
    add(gp + "ffn_up.weight", to_out_in(up, 2 * F, F, F, dm), F, dm);
    add(gp + "ffn_down.weight", to_out_in(vals(np + "mlp.down.weight"), dm, 0, dm, F), dm, F);
  }
  add("output_norm.weight", vals("norm_f.weight"), 1, dm);
  {
    auto h = vals(c.tie_embeddings ? "wte" : "lm_head.weight");
    h.resize(static_cast<std::size_t>(V * dm));
    add("output.weight", h, V, dm);
  }
  W out;
  out.b.insert(out.b.end(), {'G', 'G', 'U', 'F'});
  out.pod<std::uint32_t>(3);
  out.pod<std::uint64_t>(ts.size());
  out.pod<std::uint64_t>(static_cast<std::uint64_t>(kv.count));
  out.b.insert(out.b.end(), kv.w.b.begin(), kv.w.b.end());
  std::uint64_t off = 0;
  for (const T& t : ts) {
    out.str(t.name);
    out.pod<std::uint32_t>(static_cast<std::uint32_t>(t.ne.size()));
    for (auto e : t.ne) out.pod(e);
    out.pod<std::uint32_t>(static_cast<std::uint32_t>(t.type));
    out.pod(off);
    off += (t.data.size() + 31) / 32 * 32;
  }
  while (out.b.size() % 32) out.b.push_back(0);
  std::ofstream f(path + ".part", std::ios::binary | std::ios::trunc);
  if (!f) throw InteropError("GGUF: yazilamadi " + path);
  f.write(reinterpret_cast<const char*>(out.b.data()), static_cast<std::streamsize>(out.b.size()));
  std::uint64_t total = out.b.size();
  GgufStats st;
  for (const T& t : ts) {
    f.write(reinterpret_cast<const char*>(t.data.data()), static_cast<std::streamsize>(t.data.size()));
    const std::size_t pad = (32 - t.data.size() % 32) % 32;
    static const char zeros[32] = {};
    f.write(zeros, static_cast<std::streamsize>(pad));
    total += t.data.size() + pad;
    st.types[to_string(t.type)] += 1;
  }
  f.close();
  if (!f) throw InteropError("GGUF: yazma hatasi " + path);
  std::remove(path.c_str());
  if (std::rename((path + ".part").c_str(), path.c_str()) != 0) throw InteropError("GGUF: yeniden adlandirilamadi " + path);
  st.bytes = total;
  st.tensors = static_cast<int>(ts.size());
  st.kv = kv.count;
  st.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  if (stats) *stats = st;
}

GgufFile read_gguf(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw InteropError("GGUF: acilamadi " + path);
  std::vector<std::uint8_t> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::size_t at = 0;
  auto need = [&](std::size_t n) {
    if (at + n > b.size()) throw InteropError("GGUF: kisa dosya (" + path + ")");
  };
  auto u32 = [&] {
    need(4);
    std::uint32_t v;
    std::memcpy(&v, b.data() + at, 4);
    at += 4;
    return v;
  };
  auto u64 = [&] {
    need(8);
    std::uint64_t v;
    std::memcpy(&v, b.data() + at, 8);
    at += 8;
    return v;
  };
  auto str = [&] {
    const std::uint64_t n = u64();
    need(static_cast<std::size_t>(n));
    std::string s(reinterpret_cast<const char*>(b.data() + at), static_cast<std::size_t>(n));
    at += static_cast<std::size_t>(n);
    return s;
  };
  need(4);
  if (std::memcmp(b.data(), "GGUF", 4) != 0) throw InteropError("GGUF: sihir yok");
  at = 4;
  GgufFile g;
  g.version = u32();
  if (g.version != 3) throw InteropError("GGUF: surum " + std::to_string(g.version) + " (beklenen 3)");
  const std::uint64_t nt = u64(), nkv = u64();
  std::uint32_t align = 32;
  auto scalar = [&](std::uint32_t t) -> std::string {
    switch (t) {
      case T_U8: case T_I8: case T_BOOL: { need(1); const int v = t == T_I8 ? static_cast<std::int8_t>(b[at]) : b[at]; at += 1; return std::to_string(v); }
      case T_U16: case T_I16: { need(2); std::uint16_t v; std::memcpy(&v, b.data() + at, 2); at += 2; return std::to_string(t == T_I16 ? static_cast<std::int16_t>(v) : v); }
      case T_U32: return std::to_string(u32());
      case T_I32: return std::to_string(static_cast<std::int32_t>(u32()));
      case T_F32: { need(4); float v; std::memcpy(&v, b.data() + at, 4); at += 4; std::ostringstream o; o << v; return o.str(); }
      case T_U64: return std::to_string(u64());
      case T_I64: return std::to_string(static_cast<std::int64_t>(u64()));
      case T_F64: { need(8); double v; std::memcpy(&v, b.data() + at, 8); at += 8; std::ostringstream o; o << v; return o.str(); }
      case T_STR: return str();
      default: throw InteropError("GGUF: bilinmeyen KV tipi " + std::to_string(t));
    }
  };
  for (std::uint64_t i = 0; i < nkv; ++i) {
    const std::string k = str();
    const std::uint32_t t = u32();
    if (t == T_ARR) {
      const std::uint32_t et = u32();
      const std::uint64_t n = u64();
      for (std::uint64_t j = 0; j < n; ++j) {
        std::string v = scalar(et);
        if (k == "tokenizer.ggml.tokens") g.tokens.push_back(std::move(v));
      }
      g.kv[k] = "[" + std::to_string(n) + "]";
    } else {
      g.kv[k] = scalar(t);
      if (k == "general.alignment") align = static_cast<std::uint32_t>(std::stoul(g.kv[k]));
    }
  }
  for (std::uint64_t i = 0; i < nt; ++i) {
    GgufTensor t;
    t.name = str();
    const std::uint32_t nd = u32();
    for (std::uint32_t d = 0; d < nd; ++d) t.ne.push_back(u64());
    t.type = static_cast<GgufType>(u32());
    t.offset = u64();
    g.tensors.push_back(std::move(t));
  }
  const std::size_t data0 = (at + align - 1) / align * align;
  for (GgufTensor& t : g.tensors) {
    std::size_t n = 1;
    for (auto e : t.ne) n *= static_cast<std::size_t>(e);
    std::size_t bytes = t.type == GgufType::F32 ? n * 4 : t.type == GgufType::F16 ? n * 2 : t.type == GgufType::Q8_0 ? n / 32 * 34 : n / 32 * 18;
    if (data0 + t.offset + bytes > b.size()) throw InteropError("GGUF: " + t.name + " veri disinda");
    t.data = dequantize(t.type, b.data() + data0 + t.offset, n);
  }
  return g;
}

lm::LanguageModel import_gguf(const std::string& path) {
  const GgufFile g = read_gguf(path);
  if (g.kv.count("general.architecture") == 0 || g.kv.at("general.architecture") != "llama")
    throw InteropError("GGUF ice aktarma: yalniz llama mimarisi");
  std::map<std::string, const GgufTensor*> t;
  for (const GgufTensor& x : g.tensors) t[x.name] = &x;
  auto get = [&](const std::string& n) -> const GgufTensor& {
    auto it = t.find(n);
    if (it == t.end()) throw InteropError("GGUF ice aktarma: tensor yok " + n);
    return *it->second;
  };
  auto u = [&](const char* k) { return static_cast<int>(std::stoll(g.kv.at(k))); };
  lm::LmConfig c;
  c.name = g.kv.count("general.name") ? g.kv.at("general.name") : "gguf";
  c.d_model = u("llama.embedding_length");
  c.n_layer = u("llama.block_count");
  c.d_ff = u("llama.feed_forward_length");
  c.n_head = u("llama.attention.head_count");
  c.n_kv_head = u("llama.attention.head_count_kv");
  c.max_position = u("llama.context_length");
  c.norm_eps = std::stof(g.kv.at("llama.attention.layer_norm_rms_epsilon"));
  c.rope_theta = std::stof(g.kv.at("llama.rope.freq_base"));
  c.vocab_size = static_cast<int>(get("token_embd.weight").ne.at(1));
  c.norm = lm::NormKind::RmsNorm;
  c.mlp = lm::MlpKind::SwiGlu;
  c.pos = lm::PosKind::Rope;
  c.init = lm::InitKind::Llama;
  c.tie_embeddings = t.count("output.weight") == 0;
  if (u("llama.rope.dimension_count") != c.d_model / c.n_head) c.head_dim = u("llama.rope.dimension_count");
  lm::LanguageModel m(c, true);
  auto vals = [&](const std::string& n) -> std::vector<float>& {
    return const_cast<autograd::Variable&>(m.find(n)->value).mutable_values();
  };
  const std::int64_t dm = c.d_model, H = c.n_head, Hk = c.kv_heads(), D = c.hdim(), F = c.d_ff, V = c.vocab_size;
  auto put = [&](std::vector<float>& dst, std::int64_t dst_cols, std::int64_t col0, const std::vector<float>& src, std::int64_t out,
                 std::int64_t in) { detail::transpose_block(dst.data(), dst_cols, col0, src.data(), in, 0, out, in); };
  {
    const auto& e = get("token_embd.weight").data;
    std::copy(e.begin(), e.end(), vals("wte").begin());
  }
  for (int l = 0; l < c.n_layer; ++l) {
    const std::string np = "h" + std::to_string(l) + ".", gp = "blk." + std::to_string(l) + ".";
    vals(np + "norm1.weight") = get(gp + "attn_norm.weight").data;
    vals(np + "norm2.weight") = get(gp + "ffn_norm.weight").data;
    std::vector<float>& qkv = vals(np + "attn.qkv.weight");
    const std::int64_t cols = (H + 2 * Hk) * D;
    put(qkv, cols, 0, unpermute_qk(get(gp + "attn_q.weight").data, H * D, dm, H), H * D, dm);
    put(qkv, cols, H * D, unpermute_qk(get(gp + "attn_k.weight").data, Hk * D, dm, Hk), Hk * D, dm);
    put(qkv, cols, (H + Hk) * D, get(gp + "attn_v.weight").data, Hk * D, dm);
    put(vals(np + "attn.o.weight"), dm, 0, get(gp + "attn_output.weight").data, dm, H * D);
    put(vals(np + "mlp.up.weight"), 2 * F, 0, get(gp + "ffn_gate.weight").data, F, dm);
    put(vals(np + "mlp.up.weight"), 2 * F, F, get(gp + "ffn_up.weight").data, F, dm);
    put(vals(np + "mlp.down.weight"), dm, 0, get(gp + "ffn_down.weight").data, dm, F);
  }
  vals("norm_f.weight") = get("output_norm.weight").data;
  if (!c.tie_embeddings) {
    const auto& h = get("output.weight").data;
    std::copy(h.begin(), h.end(), vals("lm_head.weight").begin());
  }
  (void)V;
  return m;
}

}  // namespace interop
}  // namespace nexus
