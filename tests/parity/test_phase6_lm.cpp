/**
 * @file test_phase6_lm.cpp
 * @brief FAZ 6 testleri: RoPE (HF formullerine karsi), olcekleme turleri, SwiGLU/GeGLU/RoPE/dikkat
 *        gradcheck, model parity (LLaMA/Qwen2/GPT-2 tarzi, tum gradyanlar), bagli gomme, sozluk
 *        dolgusu, parametre sayilari, NXS gidis-donus, muP carpanlari, derin yigin kararliligi.
 *
 * Python + PyTorch yoksa parity modlari 77 (SKIPPED) doner.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/model.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ag = nexus::autograd;
namespace lm = nexus::lm;
using ag::Shape;
using ag::Variable;

namespace {

constexpr int kSkip = 77;
bool g_profile = false;  // "deep N prof": ilk iki adimin ileri/geri suresi

int run_status(const std::string& command) {
#if defined(_WIN32)
  return std::system(("\"" + command + "\"").c_str());
#else
  return std::system(command.c_str());
#endif
}

bool have_python() {
  const std::string py = NEXUS_PARITY_PYTHON;
  return !py.empty() && std::filesystem::exists(py);
}

std::string python_cmd(const std::string& mode, const std::filesystem::path& dir) {
  return std::string("\"") + NEXUS_PARITY_PYTHON + "\" -I \"" + NEXUS_PARITY_SCRIPTS + "/reference_lm_torch.py\" " + mode +
         " \"" + dir.string() + "\"";
}

std::filesystem::path scratch(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / ("nexus_faz6_" + name);
  std::filesystem::create_directories(dir);
  return dir;
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& v) {
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(v.data()),
                                               static_cast<std::streamsize>(v.size() * 4));
}

std::vector<float> read_f32(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  in.seekg(0, std::ios::end);
  std::vector<float> v(static_cast<std::size_t>(in.tellg()) / 4);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
  return v;
}

double rel_l2(const float* a, const float* b, std::size_t n) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

void report(const std::string& name, double err, double tol) {
  std::printf("  info  %-58s hata=%.3e esik=%.1e\n", name.c_str(), err, tol);
  nexus_test::expect(err <= tol, name);
}

std::vector<float> randn(std::size_t n, std::uint64_t seed) {
  return Variable::randn(Shape{{static_cast<std::int64_t>(n)}}, seed, false).values();
}

std::vector<std::int32_t> tokens(int n, int vocab, std::uint64_t seed) {
  std::vector<std::int32_t> t(static_cast<std::size_t>(n));
  std::uint64_t s = seed;
  for (auto& x : t) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x = static_cast<std::int32_t>((s >> 33) % static_cast<std::uint64_t>(vocab));
  }
  return t;
}

// ---- RoPE ------------------------------------------------------------------------------
struct FreqCase {
  lm::RopeScaling s;
  int dim;
  float theta;
  int max_position;
  int seq_len;
};

std::string case_json(const FreqCase& c) {
  std::ostringstream o;
  o.precision(9);
  o << "{\"type\":\"" << lm::to_string(c.s.type) << "\",\"theta\":" << c.theta << ",\"dim\":" << c.dim
    << ",\"factor\":" << c.s.factor << ",\"max_position\":" << c.max_position
    << ",\"original_max_position\":" << c.s.original_max_position << ",\"low_freq_factor\":" << c.s.low_freq_factor
    << ",\"high_freq_factor\":" << c.s.high_freq_factor << ",\"beta_fast\":" << c.s.beta_fast
    << ",\"beta_slow\":" << c.s.beta_slow << ",\"attention_factor\":" << c.s.attention_factor
    << ",\"seq_len\":" << c.seq_len << "}";
  return o.str();
}

int test_rope() {
  nexus_test::section("test_rope_scaling_variants + test_rope_vs_reference");
  using T = lm::RopeScaling::Type;
  auto make = [](T t, float factor, int orig) {
    lm::RopeScaling s;
    s.type = t;
    s.factor = factor;
    s.original_max_position = orig;
    return s;
  };
  const std::vector<FreqCase> cases = {
      {make(T::None, 1, 0), 128, 10000.0f, 4096, 0},
      {make(T::None, 1, 0), 128, 500000.0f, 8192, 0},
      {make(T::Linear, 4, 0), 64, 10000.0f, 2048, 0},
      {make(T::Ntk, 8, 0), 128, 10000.0f, 4096, 0},
      {make(T::DynamicNtk, 4, 2048), 64, 10000.0f, 8192, 1000},
      {make(T::DynamicNtk, 4, 2048), 64, 10000.0f, 8192, 6000},
      {make(T::Yarn, 4, 4096), 128, 10000.0f, 16384, 0},
      {make(T::Yarn, 16, 2048), 64, 1000000.0f, 32768, 0},
      {make(T::Llama3, 8, 8192), 128, 500000.0f, 131072, 0},
      {make(T::Llama3, 32, 8192), 64, 500000.0f, 131072, 0},
  };
  // Bilinen degerler (formulden elle): llama3 en yuksek frekans aynen kalir, en dusuk /factor.
  {
    const auto base = lm::rope_inv_freq(128, 500000.0f, {}, 131072, 0);
    const auto l3 = lm::rope_inv_freq(128, 500000.0f, cases[8].s, 131072, 0);
    report("llama3: inv_freq[0] degismez", std::fabs(l3[0] - base[0]) / base[0], 0.0);
    report("llama3: inv_freq[63] = taban / 8", std::fabs(l3[63] - base[63] / 8.0f) / (base[63] / 8.0f), 1e-6);
    const auto lin = lm::rope_inv_freq(64, 10000.0f, cases[2].s, 2048, 0);
    report("linear: inv_freq[1] = 10000^(-2/64) / 4",
           std::fabs(lin[1] - static_cast<float>(std::pow(10000.0, -2.0 / 64.0) / 4.0)) / lin[1], 1e-6);
    report("yarn s=4: attention_factor = 0.1 ln 4 + 1 = 1.138629",
           std::fabs(lm::rope_attention_scale(cases[6].s) - 1.1386294f), 1e-6);
    const auto dyn_short = lm::rope_inv_freq(64, 10000.0f, cases[4].s, 8192, 1000);
    const auto plain = lm::rope_inv_freq(64, 10000.0f, {}, 8192, 0);
    report("dynamic NTK: L0 altinda frekanslar degismez", rel_l2(dyn_short.data(), plain.data(), plain.size()), 1e-7);
  }
  if (!have_python()) {
    std::printf("  SKIP  python yok: HF formul karsilastirmasi atlandi\n");
    return kSkip;
  }
  const auto dir = scratch("rope");
  // dondurme: N=7 satir, H=3, D=64, kismi rot=32 (GPT-NeoX) ve tam rot.
  int status = 0;
  for (const int rot : {64, 32}) {
    const int n = 7;
    const int heads = 3;
    const int hd = 64;
    FreqCase fc{make(T::Yarn, 4, 512), rot, 10000.0f, 2048, 0};
    std::ostringstream js;
    js << "{\"freq\":[";
    for (std::size_t i = 0; i < cases.size(); ++i) {
      js << (i ? "," : "") << case_json(cases[i]);
    }
    js << "],\"apply\":{\"positions\":[0,1,5,17,300,1023,2047],\"heads\":" << heads << ",\"head_dim\":" << hd
       << ",\"rot_dim\":" << rot << ",\"freq\":" << case_json(fc) << "}}";
    std::ofstream(dir / "rope_cases.json") << js.str();
    const std::vector<float> x = randn(static_cast<std::size_t>(n) * heads * hd, 9);
    write_f32(dir / "rope_x.bin", x);
    if (run_status(python_cmd("rope", dir)) != 0) {
      std::printf("  FAIL  python rope referansi calismadi\n");
      return 1;
    }
    const std::vector<float> ref = read_f32(dir / "rope_ref.bin");
    std::size_t off = 0;
    if (rot == 64) {
      for (const FreqCase& c : cases) {
        const auto inv = lm::rope_inv_freq(c.dim, c.theta, c.s, c.max_position, c.seq_len);
        report(std::string("inv_freq ") + lm::to_string(c.s.type) + " d=" + std::to_string(c.dim) + " s=" +
                   std::to_string(static_cast<int>(c.s.factor)) + " vs HF",
               rel_l2(inv.data(), ref.data() + off, inv.size()), 2e-7);
        off += inv.size();
        report(std::string("attention_factor ") + lm::to_string(c.s.type),
               std::fabs(lm::rope_attention_scale(c.s) - ref[off]), 1e-6);
        off += 1;
      }
    } else {
      for (const FreqCase& c : cases) {
        off += static_cast<std::size_t>(c.dim / 2) + 1;
      }
    }
    lm::RopeCache cache(rot, fc.theta, fc.s, fc.max_position);
    cache.ensure(2048);
    std::vector<float> y = x;
    const std::vector<int> pos = {0, 1, 5, 17, 300, 1023, 2047};
    lm::rope_apply(y.data(), n, heads, hd, pos.data(), cache, false);
    report("rope_apply (yarn, rot=" + std::to_string(rot) + ") vs HF apply_rotary_pos_emb",
           rel_l2(y.data(), ref.data() + off, y.size()), 1e-6);
    // transpoz = ters: geri dondurme x'i geri verir (a^2 olcegi ile).
    lm::rope_apply(y.data(), n, heads, hd, pos.data(), cache, true);
    const float a2 = lm::rope_attention_scale(fc.s) * lm::rope_attention_scale(fc.s);
    std::vector<float> xs = x;
    for (int r = 0; r < n; ++r) {
      for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < rot; ++i) {
          xs[(static_cast<std::size_t>(r) * heads + h) * hd + i] *= a2;
        }
      }
    }
    report("rope transpoz o rope = a^2 I (rot=" + std::to_string(rot) + ")", rel_l2(y.data(), xs.data(), y.size()),
           1e-6);
  }
  return status;
}

// ---- gradcheck ---------------------------------------------------------------------------
void test_gradcheck() {
  nexus_test::section("test_swiglu_gradcheck (+ GeGLU, RoPE, GQA dikkat)");
  {
    const Variable x = Variable::randn(Shape{{5, 12}}, 3, true);
    const auto r = ag::gradcheck([](const std::vector<Variable>& in) { return ag::sum(lm::swiglu(in[0])); }, {x});
    std::printf("  info  swiglu gradcheck max_rel=%.3e\n", r.max_rel_error);
    nexus_test::expect(r.ok, "swiglu gradcheck");
    const auto g = ag::gradcheck([](const std::vector<Variable>& in) { return ag::sum(lm::geglu(in[0])); }, {x});
    std::printf("  info  geglu gradcheck max_rel=%.3e\n", g.max_rel_error);
    nexus_test::expect(g.ok, "geglu gradcheck");
  }
  {
    const Variable x = Variable::randn(Shape{{4, 2 * 8}}, 5, true);
    auto cache = std::make_shared<lm::RopeCache>(8, 10000.0f, lm::RopeScaling{}, 64);
    cache->ensure(64);
    auto pos = std::make_shared<std::vector<int>>(std::vector<int>{0, 3, 9, 40});
    const Variable w = Variable::randn(Shape{{4, 16}}, 6, false);
    const auto r = ag::gradcheck(
        [&](const std::vector<Variable>& in) { return ag::sum(ag::mul(lm::rope(in[0], 2, 8, pos, cache), w)); }, {x});
    std::printf("  info  rope gradcheck max_rel=%.3e\n", r.max_rel_error);
    nexus_test::expect(r.ok, "rope gradcheck");
  }
  {
    lm::AttentionSpec s;
    s.batch = 2;
    s.time = 5;
    s.heads = 4;
    s.kv_heads = 2;
    s.head_dim = 8;
    const Variable q = Variable::randn(Shape{{10, 32}}, 7, true);
    const Variable k = Variable::randn(Shape{{10, 16}}, 8, true);
    const Variable v = Variable::randn(Shape{{10, 16}}, 9, true);
    const Variable w = Variable::randn(Shape{{10, 32}}, 10, false);
    // Sonlu fark FP32 flash'ta gurultulu (toplam kayip ~10, eps 1e-3): kahin olarak skoru
    // acikca yazan autograd dikkati (narrow/matmul/softmax) kullanilir; gradyanlar analitik-analitik.
    auto explicit_attn = [&](const Variable& qq, const Variable& kk, const Variable& vv) {
      std::vector<float> mask(25, 0.0f);
      for (int i = 0; i < 5; ++i) {
        for (int j = i + 1; j < 5; ++j) {
          mask[static_cast<std::size_t>(i * 5 + j)] = -1.0e30f;
        }
      }
      const Variable m = Variable::make(std::move(mask), Shape{{5, 5}}, false);
      std::vector<Variable> rows;
      for (int b = 0; b < 2; ++b) {
        std::vector<Variable> heads;
        for (int h = 0; h < 4; ++h) {
          const Variable qh = ag::narrow(ag::narrow(qq, 0, b * 5, 5), 1, h * 8, 8);
          const Variable kh = ag::narrow(ag::narrow(kk, 0, b * 5, 5), 1, (h / 2) * 8, 8);
          const Variable vh = ag::narrow(ag::narrow(vv, 0, b * 5, 5), 1, (h / 2) * 8, 8);
          const Variable sc = ag::add(ag::mul_scalar(ag::matmul(qh, ag::transpose(kh)), 1.0f / std::sqrt(8.0f)), m);
          heads.push_back(ag::matmul(ag::softmax(sc), vh));
        }
        rows.push_back(ag::cat(heads, 1));
      }
      return ag::cat(rows, 0);
    };
    const Variable qa = Variable::make(q.values(), q.shape(), true), ka = Variable::make(k.values(), k.shape(), true),
                   va = Variable::make(v.values(), v.shape(), true);
    const Variable qb = Variable::make(q.values(), q.shape(), true), kb = Variable::make(k.values(), k.shape(), true),
                   vb = Variable::make(v.values(), v.shape(), true);
    const Variable fa = lm::attention(qa, ka, va, s);
    const Variable ea = explicit_attn(qb, kb, vb);
    ag::sum(ag::mul(fa, w)).backward();
    ag::sum(ag::mul(ea, w)).backward();
    report("GQA flash dikkat O vs acik dikkat", rel_l2(fa.values().data(), ea.values().data(), fa.values().size()), 1e-6);
    double worst = 0.0;
    for (const auto& [a, b] : {std::pair{qa, qb}, std::pair{ka, kb}, std::pair{va, vb}}) {
      worst = std::max(worst, rel_l2(a.grad().values().data(), b.grad().values().data(), a.grad().values().size()));
    }
    report("GQA flash dikkat dQ/dK/dV vs acik dikkat", worst, 1e-5);
  }
}

// ---- Model parity -----------------------------------------------------------------------
lm::LmConfig small(const std::string& kind) {
  lm::LmConfig c;
  c.name = kind;
  c.vocab_size = 97;
  c.n_layer = 2;
  c.d_model = 32;
  c.n_head = 4;
  c.d_ff = 48;
  c.max_position = 64;
  c.init_std = 0.2f;  // sinyal tasinsin (parity icin; 0.02 neredeyse dogrusal)
  if (kind == "llama") {
    c.n_kv_head = 2;
    c.init = lm::InitKind::Llama;
  } else if (kind == "qwen2") {
    c.n_kv_head = 1;
    c.qkv_bias = true;
    c.tie_embeddings = true;
    c.qk_norm = true;
    c.rotary_dim = 4;
    c.rope_scaling.type = lm::RopeScaling::Type::Yarn;
    c.rope_scaling.factor = 4.0f;
    c.rope_scaling.original_max_position = 16;
    c.norm_eps = 1e-6f;
  } else if (kind == "gpt2") {
    c.norm = lm::NormKind::LayerNorm;
    c.mlp = lm::MlpKind::Gelu;
    c.pos = lm::PosKind::Learned;
    c.tie_embeddings = true;
    c.bias = true;
    c.vocab_pad_multiple = 64;
  } else if (kind == "parallel") {
    c.parallel_block = true;
    c.mlp = lm::MlpKind::GeGlu;
    c.rope_scaling.type = lm::RopeScaling::Type::Llama3;
    c.rope_scaling.factor = 8.0f;
    c.rope_scaling.original_max_position = 32;
    c.rope_theta = 500000.0f;
  } else if (kind == "postnorm") {
    c.prenorm = false;
    c.norm = lm::NormKind::LayerNorm;
    c.bias = true;
  }
  return c;
}

int parity_case(const std::string& kind) {
  const lm::LmConfig c = small(kind);
  const lm::LanguageModel model(c);
  const int B = 2;
  const int T = 24;
  const auto toks = tokens(B * T, c.vocab_size, 77);
  const Variable loss = model.loss(toks, B, T);
  loss.backward();
  const auto dir = scratch("model_" + kind);
  {
    std::ofstream w(dir / "weights.bin", std::ios::binary);
    std::ostringstream m;
    m << "{\"config\":" << lm::to_json(c) << ",\"batch\":" << B << ",\"time\":" << T << ",\"params\":[";
    bool first = true;
    for (const lm::NamedParam& p : model.parameters()) {
      const auto& v = p.value.values();
      w.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
      m << (first ? "" : ",") << "{\"name\":\"" << p.name << "\",\"shape\":[";
      for (std::size_t i = 0; i < p.value.shape().dims.size(); ++i) {
        m << (i ? "," : "") << p.value.shape().dims[i];
      }
      m << "]}";
      first = false;
    }
    m << "]}";
    std::ofstream(dir / "manifest.json") << m.str();
    std::ofstream(dir / "tokens.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(toks.data()), static_cast<std::streamsize>(toks.size() * 4));
  }
  if (run_status(python_cmd("model", dir)) != 0) {
    std::printf("  FAIL  python model referansi calismadi (%s)\n", kind.c_str());
    return 1;
  }
  const float ref_loss = read_f32(dir / "ref_loss.bin")[0];
  const std::vector<float> ref = read_f32(dir / "ref_grads.bin");
  report(kind + ": kayip vs PyTorch", std::fabs(loss.values()[0] - ref_loss) / std::fabs(ref_loss), 1e-5);
  std::size_t off = 0;
  double worst = 0.0;
  std::string worst_name;
  for (const lm::NamedParam& p : model.parameters()) {
    const std::size_t n = p.value.values().size();
    std::vector<float> g(n, 0.0f);
    if (p.value.meta() && p.value.meta()->grad) {
      g = p.value.grad().values();
    }
    const double e = rel_l2(g.data(), ref.data() + off, n);
    if (e > worst) {
      worst = e;
      worst_name = p.name;
    }
    off += n;
  }
  report(kind + ": tum gradyanlar vs PyTorch (en kotu: " + worst_name + ")", worst, 1e-4);
  return 0;
}

int test_parity() {
  nexus_test::section("test_decoder_block_parity: LLaMA/Qwen2/GPT-2/paralel/son-norm vs PyTorch (HF formulleri)");
  if (!have_python()) {
    std::printf("  SKIP  python yok\n");
    return kSkip;
  }
  int rc = 0;
  for (const char* kind : {"llama", "qwen2", "gpt2", "parallel", "postnorm"}) {
    rc |= parity_case(kind);
  }
  return rc;
}

// ---- Bagli gomme, dolgu, sayilar, NXS, muP ----------------------------------------------
void test_tied() {
  nexus_test::section("test_tied_embeddings_single_copy + sozluk dolgusu");
  lm::LmConfig c = small("llama");
  c.tie_embeddings = true;
  const lm::LanguageModel tied(c);
  nexus_test::expect(tied.find("lm_head.weight") == nullptr, "bagli modelde lm_head parametresi yok");
  lm::LmConfig cu = c;
  cu.tie_embeddings = false;
  const lm::LanguageModel untied(cu);
  // Ayni agirliklar: lm_head = wte.
  for (const lm::NamedParam& p : tied.parameters()) {
    Variable dst = untied.find(p.name)->value;
    dst.mutable_values() = p.value.values();
  }
  {
    Variable head = untied.find("lm_head.weight")->value;
    head.mutable_values() = tied.find("wte")->value.values();
  }
  const auto toks = tokens(2 * 16, c.vocab_size, 5);
  const Variable lt = tied.loss(toks, 2, 16);
  const Variable lu = untied.loss(toks, 2, 16);
  lt.backward();
  lu.backward();
  report("bagli kayip = bagsiz (lm_head = wte)", std::fabs(lt.values()[0] - lu.values()[0]), 1e-6);
  std::vector<float> sum = untied.find("wte")->value.grad().values();
  const auto& gh = untied.find("lm_head.weight")->value.grad().values();
  for (std::size_t i = 0; i < sum.size(); ++i) {
    sum[i] += gh[i];
  }
  const auto& gt = tied.find("wte")->value.grad().values();
  report("bagli dwte = dwte(gomme) + dW(lm_head)", rel_l2(gt.data(), sum.data(), sum.size()), 1e-6);
  const auto path = scratch("tied") / "tied.nxs";
  tied.save_nxs(path.string());
  const nexus::nxs::Archive ar = nexus::nxs::load_archive(path.string());
  int wte_copies = 0;
  for (const auto& w : ar.weights) {
    wte_copies += (w.name == "wte" || w.name == "lm_head.weight") ? 1 : 0;
  }
  nexus_test::expect(wte_copies == 1, "NXS'de gomme tek kopya");

  // Dolgu: Vp = 128 > V = 97; dolgu logitleri -inf, kayip dolgusuz modelle ayni.
  lm::LmConfig cp = c;
  cp.vocab_pad_multiple = 64;
  const lm::LanguageModel padded(cp);
  for (const lm::NamedParam& p : tied.parameters()) {
    Variable dst = padded.find(p.name)->value;
    if (p.name == "wte") {
      std::vector<float> v = dst.values();
      std::copy(p.value.values().begin(), p.value.values().end(), v.begin());
      dst.mutable_values() = v;
    } else {
      dst.mutable_values() = p.value.values();
    }
  }
  const Variable lg = padded.logits(toks, 2, 16);
  bool all_inf = true;
  for (std::int64_t r = 0; r < lg.shape().dims[0]; ++r) {
    for (int j = c.vocab_size; j < cp.padded_vocab(); ++j) {
      all_inf = all_inf && std::isinf(lg.values()[static_cast<std::size_t>(r * cp.padded_vocab() + j)]);
    }
  }
  nexus_test::expect(cp.padded_vocab() == 128 && all_inf, "dolgu logitleri -inf (V=97 -> 128)");
  report("dolgulu kayip = dolgusuz", std::fabs(padded.loss(toks, 2, 16).values()[0] - lt.values()[0]), 1e-6);
}

void test_counts() {
  nexus_test::section("test_build_param_counts: resmi degerler (HF model kartlari)");
  const std::vector<std::pair<std::string, std::int64_t>> official = {
      {"gpt2-124m", 124439808},     {"gpt2-350m", 354823168},    {"gpt2-774m", 774030080},
      {"gpt2-1.5b", 1557611200},    {"llama2-7b", 6738415616},   {"llama3-8b", 8030261248},
      {"mistral-7b", 7241732096},   {"qwen2-0.5b", 494032768},   {"qwen2-1.5b", 1543714304},
      {"qwen2-7b", 7615616512},     {"tinyllama-1.1b", 1100048384}};
  for (const auto& [name, n] : official) {
    const std::int64_t got = lm::count_parameters(lm::preset(name));
    std::printf("  info  %-16s nexus=%lld resmi=%lld\n", name.c_str(), static_cast<long long>(got),
                static_cast<long long>(n));
    nexus_test::expect(got == n, name + " parametre sayisi");
  }
  // Analitik sayim = gercekten kurulan model (tum secenek birlesimleri, kucuk sekil).
  for (const char* kind : {"llama", "qwen2", "gpt2", "parallel", "postnorm"}) {
    const lm::LmConfig c = small(kind);
    const lm::LanguageModel m(c);
    nexus_test::expect(m.num_parameters() == lm::count_parameters(c), std::string("analitik = kurulan: ") + kind);
  }
  const lm::LanguageModel g = lm::build("gpt2-124m");
  nexus_test::expect(g.num_parameters() == 124439808, "gpt2-124m gercekten kurulur: 124 439 808");
  const lm::LmConfig o = lm::apply_overrides(lm::preset("llama3-8b"), "{\"n_layer\":2,\"vocab_size\":1000}");
  nexus_test::expect(o.n_layer == 2 && o.vocab_size == 1000 && o.n_kv_head == 8, "overrides");
  nexus_test::expect(lm::from_json(lm::to_json(lm::preset("qwen2-7b"))).qkv_bias, "JSON gidis-donus");
}

void test_nxs() {
  nexus_test::section("NXS: tur 35 + tur 34, config gomulu, gidis-donus");
  const lm::LmConfig c = small("qwen2");
  const lm::LanguageModel m(c);
  const auto path = scratch("nxs") / "qwen.nxs";
  m.save_nxs(path.string());
  const lm::LanguageModel back = lm::LanguageModel::load_nxs(path.string());
  const auto toks = tokens(2 * 12, c.vocab_size, 9);
  const auto a = m.logits(toks, 2, 12).values();
  const auto b = back.logits(toks, 2, 12).values();
  nexus_test::expect(a == b, "yuklenen model ayni logitleri verir (bit-bit)");
  const nexus::nxs::Archive ar = nexus::nxs::load_archive(path.string());
  nexus_test::expect(ar.layers[0].type == 35 && ar.layers[0].children.size() == 2 && ar.layers[0].children[0].type == 34,
                     "arsiv: kok tur 35, 2 x DecoderBlock tur 34");
}

void test_mup() {
  nexus_test::section("muP: genislik carpanlari");
  auto cfg = [](int d) {
    lm::LmConfig c = small("llama");
    c.d_model = d;
    c.n_head = d / 8;
    c.n_kv_head = c.n_head;
    c.d_ff = 2 * d;
    c.init = lm::InitKind::MuP;
    c.mup_base_width = 32;
    c.init_std = 0.02f;
    return c;
  };
  const lm::LanguageModel a(cfg(32));
  const lm::LanguageModel b(cfg(128));
  auto stdev = [](const lm::NamedParam* p) {
    double s = 0.0;
    for (const float x : p->value.values()) {
      s += static_cast<double>(x) * x;
    }
    return std::sqrt(s / static_cast<double>(p->value.values().size()));
  };
  const double ra = stdev(a.find("h0.attn.qkv.weight"));
  const double rb = stdev(b.find("h0.attn.qkv.weight"));
  report("gizli matris std: m=4'te 1/sqrt(4) = 0.5 kat", std::fabs(rb / ra - 0.5), 0.03);
  nexus_test::expect(b.find("h0.mlp.up.weight")->lr_mult == 0.25f && b.find("wte")->lr_mult == 1.0f &&
                         b.find("h0.norm1.weight")->lr_mult == 1.0f,
                     "Adam LR carpani: gizli 1/m, gomme/norm 1");
}

// ---- Derin yigin kararliligi --------------------------------------------------------------
struct DeepResult {
  std::vector<float> loss;
  std::vector<double> first_norms, last_norms;
  bool finite = true;
  double seconds = 0.0;
};

DeepResult train_deep(int layers, bool prenorm, int steps, double lr) {
  lm::LmConfig c;
  c.name = prenorm ? "deep_pre" : "deep_post";
  c.vocab_size = 128;
  c.n_layer = layers;
  c.d_model = 64;
  c.n_head = 4;
  c.d_ff = 172;
  c.max_position = 64;
  c.prenorm = prenorm;
  c.init = lm::InitKind::Gpt2;
  lm::LanguageModel model(c);
  nexus::fused::AdamWConfig ac;
  ac.lr = lr;  // isinma yok (son-norm'un zayif noktasi)
  ac.beta2 = 0.95;
  ac.weight_decay = 0.1;
  ac.max_grad_norm = 1.0;
  lm::AdamW opt(model, ac);
  DeepResult r;
  const int B = 4;
  const int T = 32;
  // Stokastik Markov verisi: her token icin 4 ardil (.50/.25/.15/.10), taban 1.208 nat.
  std::vector<std::array<int, 4>> succ(128);
  std::uint64_t st = 0x243F6A8885A308D3ULL;
  auto next = [&st]() {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return st >> 33;
  };
  for (auto& row : succ) {
    for (int& x : row) {
      x = static_cast<int>(next() % 128);
    }
  }
  const auto start = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s) {
    std::vector<std::int32_t> toks(static_cast<std::size_t>(B) * T);
    for (int b = 0; b < B; ++b) {
      int x = static_cast<int>(next() % 128);
      for (int t = 0; t < T; ++t) {
        toks[static_cast<std::size_t>(b) * T + t] = x;
        const std::uint64_t u = next() % 100;
        x = succ[static_cast<std::size_t>(x)][u < 50 ? 0 : (u < 75 ? 1 : (u < 90 ? 2 : 3))];
      }
    }
    model.zero_grad();
    const auto t0 = std::chrono::steady_clock::now();
    const Variable loss = model.loss(toks, B, T);
    const auto t1 = std::chrono::steady_clock::now();
    loss.backward();
    const auto t2 = std::chrono::steady_clock::now();
    if (s < 2 && g_profile) {
      std::printf("  prof  ileri=%.1f ms geri=%.1f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count(),
                  std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    r.loss.push_back(loss.values()[0]);
    if (!std::isfinite(loss.values()[0])) {
      r.finite = false;
      break;
    }
    if (s == 0) {
      r.first_norms = model.layer_grad_norms();
    }
    if (s == steps - 1) {
      r.last_norms = model.layer_grad_norms();
    }
    (void)opt.step();
  }
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return r;
}

// learn_gate = false: ASan kosusu (kisa, yalniz bellek + NaN denetimi); ogrenme olcutu acikca SKIP yazilir.
void test_deep(int steps, bool learn_gate) {
  nexus_test::section("test_deep_stack_stability: 32/48 katman, on-norm vs son-norm, Markov (taban 1.208)");
  for (const double lr : {1e-3, 3e-3}) {
    for (const int layers : {32, 48}) {
      for (const bool pre : {true, false}) {
        const DeepResult r = train_deep(layers, pre, steps, lr);
        const auto mean = [&](int end) {
          double a = 0.0;
          const int n = std::min(end, static_cast<int>(r.loss.size()));
          const int w = std::min(20, n);
          for (int i = n - w; i < n; ++i) {
            a += r.loss[static_cast<std::size_t>(i)];
          }
          return a / std::max(w, 1);
        };
        auto spread = [](const std::vector<double>& v) {
          if (v.empty()) return 0.0;
          const auto [lo, hi] = std::minmax_element(v.begin(), v.end());
          return *hi / std::max(*lo, 1e-30);
        };
        std::string curve;
        for (int at = 100; at <= steps; at += 100) {
          char buf[32];
          std::snprintf(buf, sizeof(buf), " %d:%.3f", at, mean(at));
          curve += buf;
        }
        std::printf("  info  lr=%.0e L=%d %s: kayip[0]=%.3f egri(son20 ort.)%s sonlu=%s | katman grad normu max/min "
                    "ilk=%.2f son=%.2f | %.0fs\n",
                    lr, layers, pre ? "on-norm " : "son-norm", r.loss.empty() ? 0.0 : r.loss[0], curve.c_str(),
                    r.finite ? "evet" : "HAYIR", spread(r.first_norms), spread(r.last_norms), r.seconds);
        if (pre) {
          const std::string tag = " (lr=" + std::to_string(lr) + ", L=" + std::to_string(layers) + ")";
          nexus_test::expect(r.finite && static_cast<int>(r.loss.size()) == steps, "on-norm NaN'siz" + tag);
          if (learn_gate) {
            nexus_test::expect(mean(steps) < 0.6 * r.loss[0], "on-norm ogreniyor" + tag);
          } else {
            std::printf("  SKIP  on-norm ogreniyor%s: ASan kosusu %d adim; olcut Release'te 500 adimla\n", tag.c_str(), steps);
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "all";
  int rc = 0;
  try {
    if (mode == "all" || mode == "rope") rc |= test_rope() == 1 ? 1 : 0;
    if (mode == "all" || mode == "gradcheck") test_gradcheck();
    if (mode == "all" || mode == "parity") {
      const int p = test_parity();
      if (p == kSkip && mode == "parity") return kSkip;
      rc |= p == 1 ? 1 : 0;
    }
    if (mode == "all" || mode == "tied") test_tied();
    if (mode == "all" || mode == "counts") test_counts();
    if (mode == "all" || mode == "nxs") test_nxs();
    if (mode == "all" || mode == "mup") test_mup();
    g_profile = argc > 3 && std::string(argv[3]) == "prof";
    if (mode == "deep") test_deep(argc > 2 ? std::atoi(argv[2]) : 500, !(argc > 3 && std::string(argv[3]) == "asan"));
  } catch (const std::exception& e) {
    std::printf("  FAIL  %s\n", e.what());
    return 1;
  }
  const int f = nexus_test::finish("faz6-lm");
  return f != 0 ? f : rc;
}
