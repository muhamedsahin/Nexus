/**
 * @file test_phase17_interop.cpp
 * @brief FAZ 17: birlikte calisabilirlik — safetensors, HF ice / disa aktarma, GGUF, ONNX.
 *
 *   test_phase17_interop safetensors : test_safetensors_roundtrip — yaz/oku (F32/F16/BF16/I64), mmap = kopya, bozuk
 *                                      dosyalar (boyut, aralik, bosluk, tasma, dtype) acik hata
 *   test_phase17_interop hf_tiny     : llama/v1, mistral/v1, qwen2/v1 sablonlari: rastgele kucuk HF dizini -> Nexus
 *                                      logitleri == bagimsiz torch referansi (FP64) <= 1e-4; save_hf -> load_hf ayni
 *   test_phase17_interop hf_tinyllama: test_hf_import_logits — TinyLlama-1.1B (gercek checkpoint, BF16) yukle, logitler
 *                                      bagimsiz FP64 referansla <= 1e-4; yukleme GB/s (mmap vs tam kopya)
 *   test_phase17_interop gguf        : test_gguf_roundtrip (dosya duzeyi) — Nexus -> GGUF (F32/F16/Q8_0/Q4_0) -> bagimsiz
 *                                      Python GGUF okuyucu + ileri gecis logitleri (llama.cpp YOK: indirilmedi)
 *   test_phase17_interop onnx        : test_onnx_roundtrip — FAZ 13 IR -> ONNX (opset 17/18, dinamik B/T, dis veri) ->
 *                                      bagimsiz Python protobuf cozucu + yorumlayici logitleri (onnxruntime YOK)
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/interop/gguf.hpp"
#include "nexus/interop/hf.hpp"
#include "nexus/interop/onnx.hpp"
#include "nexus/interop/safetensors.hpp"
#include "nexus/lm/model.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ix = nexus::interop;
namespace lm = nexus::lm;
namespace ag = nexus::autograd;

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int run_status(const std::string& command) {
#if defined(_WIN32)
  return std::system(("\"" + command + "\"").c_str());
#else
  return std::system(command.c_str());
#endif
}

fs::path scratch(const std::string& n) {
  const fs::path d = fs::temp_directory_path() / ("nexus_faz17_" + n);
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d);
  return d;
}

std::string py() { return NEXUS_PARITY_PYTHON; }
std::string script() { return NEXUS_FAZ17_SCRIPT; }
bool have_py() { return !py().empty() && fs::exists(py()); }
std::string q(const fs::path& p) { return "\"" + p.string() + "\""; }

std::vector<float> read_floats(const fs::path& p) {
  std::ifstream in(p, std::ios::binary);
  in.seekg(0, std::ios::end);
  std::vector<float> v(static_cast<std::size_t>(in.tellg()) / 4);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
  return v;
}

void write_tokens(const fs::path& p, const std::vector<std::int32_t>& t) {
  std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(t.data()), static_cast<std::streamsize>(t.size() * 4));
}

/// Nexus logitleri [T, V] (dolgu sutunlari atilir).
std::vector<float> nexus_logits(const lm::LanguageModel& m, const std::vector<std::int32_t>& t) {
  ag::NoGradGuard ng;
  const std::vector<float> full = m.logits(t, 1, static_cast<int>(t.size())).values();
  const std::size_t Vp = full.size() / t.size(), V = static_cast<std::size_t>(m.config().vocab_size);
  std::vector<float> o(t.size() * V);
  for (std::size_t r = 0; r < t.size(); ++r) std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(r * Vp), V, o.begin() + static_cast<std::ptrdiff_t>(r * V));
  return o;
}

double rel(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  double m = 0, s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    s = std::max(s, std::fabs(static_cast<double>(b[i])));
  }
  return m / std::max(s, 1e-30);
}

std::vector<float> py_logits(const fs::path& dir, const std::vector<std::int32_t>& t, const char* dtype = "float64",
                             fs::path work = {}) {
  if (work.empty()) work = dir;
  fs::create_directories(work);
  const fs::path tf = work / "tok.bin", of = work / "ref_logits.bin";
  write_tokens(tf, t);
  const int rc = run_status("\"" + py() + "\" \"" + script() + "\" logits " + q(dir) + " " + q(tf) + " " + q(of) + " " + dtype);
  if (rc != 0) throw std::runtime_error("python referansi calismadi");
  return read_floats(of);
}

// ==========================================================================================

void test_safetensors() {
  const fs::path d = scratch("st");
  std::vector<float> a(3 * 5);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) * 0.5f - 3.0f;
  std::vector<std::uint16_t> h = {0x3C00, 0xC000, 0x3555, 0x0001};   // F16: 1, -2, ~1/3, en kucuk alt normal
  std::vector<std::uint16_t> bf = {0x3F80, 0xC040};                   // BF16: 1, -3
  std::vector<std::int64_t> ids = {7, -1, 1LL << 40};
  ix::write_safetensors((d / "a.safetensors").string(),
                        {{"w", "F32", {3, 5}, a.data(), a.size() * 4},
                         {"h", "F16", {4}, h.data(), h.size() * 2},
                         {"b", "BF16", {2}, bf.data(), bf.size() * 2},
                         {"ids", "I64", {3}, ids.data(), ids.size() * 8},
                         {"empty", "F32", {0, 4}, nullptr, 0}},
                        {{"format", "pt"}});
  for (bool mm : {true, false}) {
    const ix::SafeTensors st = ix::SafeTensors::open((d / "a.safetensors").string(), mm);
    const auto w = st.to_f32(st.at("w"));
    const auto hv = st.to_f32(st.at("h"));
    const auto bv = st.to_f32(st.at("b"));
    std::vector<std::int64_t> iv(3);
    std::memcpy(iv.data(), st.bytes(st.at("ids")).data(), 24);
    const bool ok = w == a && hv[0] == 1.0f && hv[1] == -2.0f && std::fabs(hv[2] - 1.0f / 3) < 1e-3 && hv[3] == std::ldexp(1.0f, -24) &&
                    bv[0] == 1.0f && bv[1] == -3.0f && iv == ids && st.at("empty").numel() == 0 && st.metadata().at("format") == "pt" &&
                    st.mapped() == mm;
    nexus_test::expect(ok, std::string("test_safetensors_roundtrip (") + (mm ? "mmap" : "tam kopya") + "): F32/F16/BF16/I64/bos, metadata");
  }
  // bozuk dosyalar
  const std::vector<std::uint8_t> good = [&] {
    std::ifstream in(d / "a.safetensors", std::ios::binary);
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }();
  std::uint64_t hl;
  std::memcpy(&hl, good.data(), 8);
  const std::string hdr(good.begin() + 8, good.begin() + 8 + static_cast<std::ptrdiff_t>(hl));
  auto attempt = [&](const std::string& name, std::vector<std::uint8_t> bytes) {
    const fs::path p = d / (name + ".safetensors");
    std::ofstream(p, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    std::string err;
    try {
      (void)ix::SafeTensors::open(p.string(), true);
    } catch (const std::exception& e) {
      err = e.what();
    }
    std::printf("  info  %-22s -> %s\n", name.c_str(), err.empty() ? "KABUL EDILDI" : err.substr(0, 110).c_str());
    return !err.empty();
  };
  auto with_header = [&](std::string nh) {
    while (nh.size() % 8) nh += ' ';
    std::vector<std::uint8_t> b(8);
    const std::uint64_t n = nh.size();
    std::memcpy(b.data(), &n, 8);
    b.insert(b.end(), nh.begin(), nh.end());
    b.insert(b.end(), good.begin() + 8 + static_cast<std::ptrdiff_t>(hl), good.end());
    return b;
  };
  auto replace = [&](std::string s, const std::string& from, const std::string& to) {
    const auto at = s.find(from);
    if (at != std::string::npos) s.replace(at, from.size(), to);
    return s;
  };
  bool all = true;
  {
    auto b = good;
    const std::uint64_t huge = good.size();
    std::memcpy(b.data(), &huge, 8);
    all &= attempt("baslik_boyu", b);
  }
  all &= attempt("kesik_dosya", std::vector<std::uint8_t>(good.begin(), good.end() - 9));
  all &= attempt("dtype_bilinmeyen", with_header(replace(hdr, "\"F16\"", "\"Q4\"")));
  all &= attempt("sekil_uyusmaz", with_header(replace(hdr, "[3,5]", "[3,6]")));
  all &= attempt("negatif_boyut", with_header(replace(hdr, "[3,5]", "[-3,5]")));
  all &= attempt("tasma", with_header(replace(hdr, "[3,5]", "[4294967296,4294967296]")));
  all &= attempt("json_bozuk", with_header(hdr.substr(0, hdr.size() / 2)));
  {
    // ust uste aralik: ilk iki tensoru ayni yere isaret ettir
    std::string s = hdr;
    const auto p = s.find("\"data_offsets\":[60,");
    if (p != std::string::npos) s.replace(p, 19, "\"data_offsets\":[52,");
    all &= attempt("ust_uste_aralik", with_header(s));
  }
  nexus_test::expect(all, "bozuk / kotu niyetli safetensors (8 durum) acik hatayla reddedilir");
}

void test_hf_tiny() {
  if (!have_py()) {
    std::printf("  SKIP  torch'lu python yok\n");
    std::exit(77);
  }
  const std::vector<std::int32_t> t = {3, 17, 200, 5, 9, 9, 120, 44, 1, 0, 77, 150, 31, 2, 99, 64};
  for (const char* kind : {"llama", "mistral", "qwen2"}) {
    const fs::path d = scratch(std::string("tiny_") + kind);
    if (run_status("\"" + py() + "\" \"" + script() + "\" tiny " + q(d) + " " + kind + " 7") != 0) throw std::runtime_error("tiny uretilemedi");
    ix::HfLoadStats st;
    const lm::LanguageModel m = ix::load_hf(d.string(), true, &st);
    const auto ref = py_logits(d, t);
    const auto got = nexus_logits(m, t);
    const double e = rel(got, ref);
    // geri yazma: Nexus -> HF -> Nexus
    ix::save_hf(m, (d / "back").string(), "F32");
    const lm::LanguageModel m2 = ix::load_hf((d / "back").string());
    const bool same = nexus_logits(m2, t) == got;
    std::printf("  info  %-8s sablon %s, %d tensor; Nexus vs torch FP64 goreli %.1e; save_hf -> load_hf %s (pencere %d, qkv_bias %d, bagli %d)\n",
                kind, st.template_name.c_str(), st.tensors, e, same ? "bit bit" : "FARKLI", m.config().window, m.config().qkv_bias,
                m.config().tie_embeddings);
    nexus_test::expect(e <= 1e-4 && same, std::string("HF ") + st.template_name + ": logitler <= 1e-4, gidis-donus bit bit");
  }
  // desteklenmeyen alanlar acik hata
  const fs::path d = scratch("tiny_bad");
  (void)run_status("\"" + py() + "\" \"" + script() + "\" tiny " + q(d) + " llama 1");
  std::string cfg;
  {
    std::ifstream in(d / "config.json");
    cfg.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }
  auto expect_error = [&](const std::string& from, const std::string& to, const char* what) {
    std::string c = cfg;
    const auto at = c.find(from);
    c.replace(at, from.size(), to);
    std::ofstream(d / "config.json") << c;
    std::string err;
    try {
      (void)ix::load_hf(d.string());
    } catch (const std::exception& e) {
      err = e.what();
    }
    std::ofstream(d / "config.json") << cfg;
    std::printf("  info  %-26s -> %s\n", what, err.substr(0, 100).c_str());
    return !err.empty();
  };
  const bool errs = expect_error("\"silu\"", "\"gelu\"", "hidden_act gelu") &&
                    expect_error("LlamaForCausalLM", "GPTNeoXForCausalLM", "mimari GPT-NeoX") &&
                    expect_error("\"intermediate_size\": 128", "\"intermediate_size\": 96", "sekil uyusmazligi");
  nexus_test::expect(errs, "desteklenmeyen mimari / aktivasyon / sekil acik hata");
}

void test_hf_tinyllama() {
  const fs::path d = std::getenv("NEXUS_TINYLLAMA") ? std::getenv("NEXUS_TINYLLAMA") : "C:/Users/muham/nexus-data/tinyllama";
  if (!fs::exists(d / "model.safetensors") || !have_py()) {
    std::printf("  SKIP  TinyLlama ya da python yok (%s)\n", d.string().c_str());
    std::exit(77);
  }
  // 1-2-3: "<s> Once upon a time" benzeri tokenlar (sozluk kimlikleri; tokenizer gerekmez)
  const std::vector<std::int32_t> t = {1, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257, 365, 2354, 29889, 2296};
  for (bool mm : {true, false}) {
    ix::HfLoadStats st;
    const auto t0 = Clock::now();
    const lm::LanguageModel m = ix::load_hf(d.string(), mm, &st);
    const double ms = ms_since(t0);
    std::printf("  info  %s: %d tensor, %.2f GB (BF16), acma+dogrulama %.0f ms, toplam %.0f ms (%.2f GB/s okunan), %.1fM parametre\n",
                mm ? "mmap" : "tam kopya", st.tensors, st.bytes / 1e9, st.open_ms, ms, st.bytes / (ms * 1e6), m.num_parameters() / 1e6);
    std::printf("{\"mode\": \"bench_hf_load\", \"mmap\": %s, \"bytes\": %llu, \"open_ms\": %.1f, \"total_ms\": %.1f, \"gb_s\": %.3f}\n",
                mm ? "true" : "false", static_cast<unsigned long long>(st.bytes), st.open_ms, ms, st.bytes / (ms * 1e6));
    if (!mm) continue;
    const auto tl = Clock::now();
    const auto got = nexus_logits(m, t);
    const double fwd = ms_since(tl);
    const auto ref = py_logits(d, t, "float64", scratch("tinyllama_ref"));
    const double e = rel(got, ref);
    int same = 0;
    const std::size_t V = static_cast<std::size_t>(m.config().vocab_size);
    for (std::size_t r = 0; r < t.size(); ++r) {
      const auto a = std::max_element(got.begin() + static_cast<std::ptrdiff_t>(r * V), got.begin() + static_cast<std::ptrdiff_t>((r + 1) * V));
      const auto b = std::max_element(ref.begin() + static_cast<std::ptrdiff_t>(r * V), ref.begin() + static_cast<std::ptrdiff_t>((r + 1) * V));
      same += (a - got.begin()) - static_cast<std::ptrdiff_t>(r * V) == (b - ref.begin()) - static_cast<std::ptrdiff_t>(r * V) ? 1 : 0;
    }
    std::printf("  info  TinyLlama-1.1B (%s, %d katman, GQA %d/%d): %zu token ileri %.0f ms; Nexus FP32 vs bagimsiz torch FP64 goreli %.2e; "
                "argmax %d/%zu\n",
                st.template_name.c_str(), m.config().n_layer, m.config().n_head, m.config().kv_heads(), t.size(), fwd, e, same, t.size());
    nexus_test::expect(e <= 1e-4 && same == static_cast<int>(t.size()),
                       "test_hf_import_logits: hazir LLaMA (TinyLlama-1.1B) logitleri referansla <= 1e-4");
  }
}

std::vector<float> py_gguf_logits(const fs::path& gguf, const fs::path& work, const std::vector<std::int32_t>& t, const char* dtype) {
  const fs::path tf = work / "tok.bin", of = work / "gguf_logits.bin";
  write_tokens(tf, t);
  if (run_status("\"" + py() + "\" \"" + script() + "\" gguf_logits " + q(gguf) + " " + q(tf) + " " + q(of) + " " + dtype) != 0)
    throw std::runtime_error("python GGUF okuyucu calismadi");
  return read_floats(of);
}

void test_gguf() {
  if (!have_py()) {
    std::printf("  SKIP  torch'lu python yok\n");
    std::exit(77);
  }
  const std::vector<std::int32_t> t = {3, 17, 200, 5, 9, 9, 120, 44, 1, 0, 77, 150, 31, 2, 99, 64};
  const fs::path d = scratch("gguf");
  (void)run_status("\"" + py() + "\" \"" + script() + "\" tiny " + q(d / "hf") + " llama 11");
  const lm::LanguageModel m = ix::load_hf((d / "hf").string());
  const auto ref = nexus_logits(m, t);
  struct Case {
    ix::GgufType type;
    double limit;
  };
  for (const Case cs : {Case{ix::GgufType::F32, 1e-5}, Case{ix::GgufType::F16, 5e-3}, Case{ix::GgufType::Q8_0, 3e-2},
                        Case{ix::GgufType::Q4_0, 0.3}}) {
    const fs::path f = d / (std::string("tiny_") + ix::to_string(cs.type) + ".gguf");
    ix::GgufOptions o;
    o.type = cs.type;
    ix::GgufStats st;
    ix::export_gguf(m, f.string(), o, &st);
    const ix::GgufFile g = ix::read_gguf(f.string());
    const auto py_lg = py_gguf_logits(f, d, t, "float64");
    const lm::LanguageModel back = ix::import_gguf(f.string());
    const double fmt = rel(nexus_logits(back, t), py_lg);  // ayni acilmis agirliklar: bicim / permutasyon dogrulugu
    const double e = rel(py_lg, ref);                      // nicemleme hatasi (orijinal FP32'ye gore)
    std::printf("  info  %-5s %7llu bayt, %d tensor, %d KV; C++ okuyucu: surum %u, mimari %s, %zu tensor; bagimsiz Python okuyucu + "
                "ileri gecis vs GGUF'tan ice aktarilan Nexus %.1e; nicemleme hatasi (orijinale gore) %.2e\n",
                ix::to_string(cs.type), static_cast<unsigned long long>(st.bytes), st.tensors, st.kv, g.version,
                g.kv.at("general.architecture").c_str(), g.tensors.size(), fmt, e);
    (void)cs.limit;
    nexus_test::expect(g.tensors.size() == static_cast<std::size_t>(st.tensors) && fmt <= 1e-4 &&
                           (cs.type != ix::GgufType::F32 || e <= 1e-5),
                       std::string("test_gguf_roundtrip (dosya duzeyi) ") + ix::to_string(cs.type) +
                           ": Nexus -> GGUF -> bagimsiz okuyucu == Nexus ice aktarma (<= 1e-4)");
  }
  // nicemleme cekirdekleri: bilinen degerler
  std::vector<float> x(64);
  for (int i = 0; i < 64; ++i) x[static_cast<std::size_t>(i)] = std::sin(0.37f * i) * (i % 7 == 0 ? 4.0f : 1.0f);
  const auto q8 = ix::dequantize(ix::GgufType::Q8_0, ix::quantize_q8_0(x.data(), x.size()).data(), x.size());
  const auto q4 = ix::dequantize(ix::GgufType::Q4_0, ix::quantize_q4_0(x.data(), x.size()).data(), x.size());
  double e8 = 0, e4 = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    e8 = std::max(e8, static_cast<double>(std::fabs(q8[i] - x[i])));
    e4 = std::max(e4, static_cast<double>(std::fabs(q4[i] - x[i])));
  }
  std::printf("  info  blok nicemleme en buyuk hata: Q8_0 %.4f (amax/254 = %.4f), Q4_0 %.4f (asimetrik: amax/8 = %.4f)\n", e8, 4.0 / 254, e4, 4.0 / 8);
  nexus_test::expect(e8 <= 4.0 / 254 + 1e-3 && e4 <= 4.0 / 8 + 1e-3,
                     "Q8_0 hatasi <= amax/254; Q4_0 (d = max/-8, seviyeler -8..7) <= amax/8");
  // TinyLlama: boyut + sure; Q8_0 / Q4_0 logit sapmasi (FP32 referans)
  const fs::path tl = std::getenv("NEXUS_TINYLLAMA") ? std::getenv("NEXUS_TINYLLAMA") : "C:/Users/muham/nexus-data/tinyllama";
  if (!fs::exists(tl / "model.safetensors")) {
    std::printf("  info  TinyLlama yok: buyuk model olcumu atlandi\n");
    return;
  }
  const lm::LanguageModel big = ix::load_hf(tl.string());
  const std::vector<std::int32_t> bt = {1, 9038, 2501, 263, 931, 29892, 727, 471};
  const auto bref = nexus_logits(big, bt);
  for (ix::GgufType ty : {ix::GgufType::F16, ix::GgufType::Q8_0, ix::GgufType::Q4_0}) {
    const fs::path f = d / (std::string("tinyllama_") + ix::to_string(ty) + ".gguf");
    ix::GgufOptions o;
    o.type = ty;
    o.tokenizer_json = (tl / "tokenizer.json").string();
    o.name = "TinyLlama-1.1B-Chat-v1.0";
    ix::GgufStats st;
    ix::export_gguf(big, f.string(), o, &st);
    std::string tps;
    for (const auto& [k, v] : st.types) tps += k + "x" + std::to_string(v) + " ";
    double e = -1;
    int same = 0;
    double fmt = -1;
    if (ty != ix::GgufType::F16) {
      const auto lg = py_gguf_logits(f, d, bt, "float32");
      e = rel(lg, bref);
      if (ty == ix::GgufType::Q4_0) {
        const lm::LanguageModel back = ix::import_gguf(f.string());
        fmt = rel(nexus_logits(back, bt), lg);
        nexus_test::expect(fmt <= 1e-4, "TinyLlama Q4_0 GGUF: bagimsiz okuyucu == Nexus ice aktarma (<= 1e-4)");
      }
      const std::size_t V = 32000;
      for (std::size_t r = 0; r < bt.size(); ++r)
        same += std::max_element(lg.begin() + static_cast<std::ptrdiff_t>(r * V), lg.begin() + static_cast<std::ptrdiff_t>((r + 1) * V)) - lg.begin() ==
                        std::max_element(bref.begin() + static_cast<std::ptrdiff_t>(r * V), bref.begin() + static_cast<std::ptrdiff_t>((r + 1) * V)) - bref.begin()
                    ? 1
                    : 0;
    }
    std::printf("{\"mode\": \"bench_gguf_export\", \"model\": \"TinyLlama-1.1B\", \"type\": \"%s\", \"bytes\": %llu, \"ms\": %.0f, "
                "\"kv\": %d, \"tensors\": \"%s\", \"logit_rel_vs_fp32\": %.4f, \"argmax_same\": %d, \"format_rel\": %.1e}\n",
                ix::to_string(ty), static_cast<unsigned long long>(st.bytes), st.ms, st.kv, tps.c_str(), e, same, fmt);
  }
}

std::vector<float> py_onnx_logits(const fs::path& onnx, const fs::path& work, const std::vector<std::int32_t>& t, const char* dtype,
                                  int batch) {
  const fs::path tf = work / "tok.bin", of = work / "onnx_logits.bin";
  write_tokens(tf, t);
  if (run_status("\"" + py() + "\" \"" + script() + "\" onnx_logits " + q(onnx) + " " + q(tf) + " " + q(of) + " " + dtype + " " +
                 std::to_string(batch)) != 0)
    throw std::runtime_error("python ONNX yorumlayicisi calismadi");
  return read_floats(of);
}

/// Nexus logitleri [B*T, V] (B satir yigini).
std::vector<float> nexus_logits_batch(const lm::LanguageModel& m, const std::vector<std::int32_t>& t, int B) {
  ag::NoGradGuard ng;
  const int T = static_cast<int>(t.size()) / B;
  const std::vector<float> full = m.logits(t, B, T).values();
  const std::size_t Vp = full.size() / t.size(), V = static_cast<std::size_t>(m.config().vocab_size);
  std::vector<float> o(t.size() * V);
  for (std::size_t r = 0; r < t.size(); ++r) std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(r * Vp), V, o.begin() + static_cast<std::ptrdiff_t>(r * V));
  return o;
}

void test_onnx() {
  if (!have_py()) {
    std::printf("  SKIP  torch'lu python yok\n");
    std::exit(77);
  }
  const fs::path d = scratch("onnx");
  struct Case {
    std::string name;
    lm::LmConfig c;
  };
  std::vector<Case> cases;
  {
    lm::LmConfig c = lm::preset("gpt2-124m");
    c.n_layer = 2;
    c.d_model = 64;
    c.n_head = 4;
    c.d_ff = 256;
    c.vocab_size = 97;
    c.max_position = 64;
    c.init_std = 0.2f;
    cases.push_back({"gpt2 tipi (ogrenilen konum, LayerNorm + bias, GELU, bagli)", c});
  }
  {
    lm::LmConfig c = lm::preset("llama2-7b");
    c.n_layer = 2;
    c.d_model = 64;
    c.n_head = 4;
    c.n_kv_head = 2;
    c.d_ff = 160;
    c.vocab_size = 97;
    c.max_position = 64;
    c.init_std = 0.2f;
    cases.push_back({"llama tipi (RoPE, RMSNorm, SwiGLU, GQA 4/2)", c});
  }
  for (const Case& cs : cases) {
    const lm::LanguageModel m(cs.c);
    for (int opset : {17, 18}) {
      for (bool ext : {false, true}) {
        const fs::path f = d / ("m" + std::to_string(opset) + (ext ? "_ext" : "") + ".onnx");
        ix::OnnxOptions o;
        o.opset = opset;
        o.external_data = ext;
        ix::OnnxStats st;
        ix::export_onnx(m, f.string(), o, &st);
        double worst = 0.0;
        for (auto [B, T] : {std::pair<int, int>{1, 9}, std::pair<int, int>{2, 16}}) {  // dinamik eksenler
          std::vector<std::int32_t> t(static_cast<std::size_t>(B * T));
          for (std::size_t i = 0; i < t.size(); ++i) t[i] = static_cast<std::int32_t>((i * 37 + 5) % 97);
          worst = std::max(worst, rel(py_onnx_logits(f, d, t, "float64", B), nexus_logits_batch(m, t, B)));
        }
        std::printf("  info  %s, opset %d%s: %d ONNX dugumu (%d IR dugumu), %d baslatici, model %llu + veri %llu bayt; "
                    "yorumlayici vs Nexus (B,T) = (1,9) ve (2,16) goreli %.1e\n",
                    cs.name.substr(0, 10).c_str(), opset, ext ? " dis veri" : "", st.nodes, st.ir_nodes, st.initializers,
                    static_cast<unsigned long long>(st.model_bytes), static_cast<unsigned long long>(st.data_bytes), worst);
        nexus_test::expect(worst <= 1e-4 && (!ext || st.data_bytes > 0),
                           "test_onnx_roundtrip: " + cs.name + " opset " + std::to_string(opset) + (ext ? " (dis veri)" : "") +
                               " <= 1e-4 (dinamik batch/seq)");
      }
    }
  }
  // TinyLlama: 4.4 GB FP32 baslatici -> protobuf 2 GB siniri: dis veri zorunlu
  const fs::path tl = std::getenv("NEXUS_TINYLLAMA") ? std::getenv("NEXUS_TINYLLAMA") : "C:/Users/muham/nexus-data/tinyllama";
  if (!fs::exists(tl / "model.safetensors")) return;
  const lm::LanguageModel big = ix::load_hf(tl.string());
  ix::OnnxOptions o;
  o.external_data = true;
  ix::OnnxStats st;
  const fs::path f = d / "tinyllama.onnx";
  ix::export_onnx(big, f.string(), o, &st);
  const std::vector<std::int32_t> bt = {1, 9038, 2501, 263, 931, 29892, 727, 471};
  const double e = rel(py_onnx_logits(f, d, bt, "float32", 1), nexus_logits(big, bt));
  std::printf("{\"mode\": \"bench_onnx_export\", \"model\": \"TinyLlama-1.1B\", \"nodes\": %d, \"model_bytes\": %llu, \"data_bytes\": %llu, "
              "\"ms\": %.0f, \"rel_vs_nexus\": %.2e}\n",
              st.nodes, static_cast<unsigned long long>(st.model_bytes), static_cast<unsigned long long>(st.data_bytes), st.ms, e);
  nexus_test::expect(e <= 1e-4, "TinyLlama -> ONNX (dis veri, FP32 yorumlayici) == Nexus (<= 1e-4)");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "safetensors";
  try {
    if (mode == "safetensors") test_safetensors();
    else if (mode == "hf_tiny") test_hf_tiny();
    else if (mode == "hf_tinyllama") test_hf_tinyllama();
    else if (mode == "gguf") test_gguf();
    else if (mode == "onnx") test_onnx();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase17_interop");
}
