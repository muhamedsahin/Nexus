/**
 * @file test_phase7_memory_cuda.cpp
 * @brief FAZ 7 GPU: yeniden hesaplama (bit-bit), parcali CE, ZeRO-Offload = GPU AdamW, 8-bit AdamW
 *        yakinsamasi ve bellek muhasebesi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/gpt.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
using nexus::DType;

namespace {

nc::Stream g_stream;

// Markov verisi (FAZ 4 tiny ile ayni kural): 4 ardil, taban 1.208 nat.
struct Markov {
  int vocab;
  std::vector<std::array<std::int32_t, 4>> succ;
  std::uint64_t st = 0x243F6A8885A308D3ULL;
  explicit Markov(int v) : vocab(v), succ(static_cast<std::size_t>(v)) {
    for (auto& row : succ) {
      for (auto& t : row) {
        t = static_cast<std::int32_t>(next() % static_cast<std::uint64_t>(vocab));
      }
    }
  }
  std::uint64_t next() {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    return st >> 33;
  }
  std::vector<std::int32_t> batch(int b, int t, int step) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(b) * t);
    std::uint64_t s = 0x9E3779B97F4A7C15ULL ^ static_cast<std::uint64_t>(step) * 0xD1B54A32D192ED03ULL;
    for (int i = 0; i < b; ++i) {
      s = s * 6364136223846793005ULL + 1442695040888963407ULL;
      std::int32_t x = static_cast<std::int32_t>((s >> 33) % static_cast<std::uint64_t>(vocab));
      for (int j = 0; j < t; ++j) {
        out[static_cast<std::size_t>(i) * t + j] = x;
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const std::uint64_t u = (s >> 33) % 100;
        x = succ[static_cast<std::size_t>(x)][u < 50 ? 0 : (u < 75 ? 1 : (u < 90 ? 2 : 3))];
      }
    }
    return out;
  }
};

nc::GptConfig small(DType t) {
  nc::GptConfig c;
  c.vocab = 512;
  c.max_seq = 128;
  c.layers = 4;
  c.heads = 4;
  c.dim = 256;  // D = 64: flash
  c.compute_dtype = t;
  c.deterministic = true;
  c.deterministic_attention = true;  // atomiksiz dQ: bit-bit karsilastirma icin
  return c;
}

struct Result {
  float loss = 0.0f;
  std::vector<float> grads;
  std::size_t activation = 0;
};

Result one_step(const nc::GptConfig& c, int B, int T) {
  nc::GptModel m(c, B, T, 11, g_stream);
  Markov data(c.vocab);
  const auto tok = data.batch(B, T, 0);
  nc::DeviceBuffer d(tok.size() * 4, g_stream);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
  m.forward_backward(d.as<std::int32_t>());
  Result r;
  r.loss = m.read_loss();
  r.grads = m.download_gradients();
  r.activation = m.activation_bytes();
  return r;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

void report(const std::string& name, double err, double tol) {
  std::printf("  info  %-60s hata=%.3e esik=%.1e\n", name.c_str(), err, tol);
  nexus_test::expect(err <= tol, name);
}

void test_recompute() {
  nexus_test::section("test_checkpoint_grad_equal (GPU): none / full / every_k / selective");
  for (const DType t : {DType::BF16, DType::F32}) {
    const nc::GptConfig base = small(t);
    const Result none = one_step(base, 2, 128);
    const char* tag = t == DType::BF16 ? "bf16 flash" : "fp32 math";
    for (const nc::Recompute rc : {nc::Recompute::Full, nc::Recompute::EveryK}) {
      nc::GptConfig c = base;
      c.recompute = rc;
      const Result r = one_step(c, 2, 128);
      std::printf("  info  %s %-8s aktivasyon %.1f MiB (none %.1f MiB)\n", tag, nc::to_string(rc), r.activation / 1048576.0,
                  none.activation / 1048576.0);
      nexus_test::expect(r.loss == none.loss && r.grads == none.grads,
                         std::string(tag) + " " + nc::to_string(rc) + ": kayip ve tum gradyanlar bit-bit ayni");
      nexus_test::expect(r.activation < none.activation, std::string(tag) + " " + nc::to_string(rc) + ": daha az aktivasyon");
    }
    if (t == DType::BF16) {
      nc::GptConfig c = base;
      c.recompute = nc::Recompute::Selective;
      const Result r = one_step(c, 2, 128);
      const Result r2 = one_step(c, 2, 128);
      std::printf("  info  %s selective aktivasyon %.1f MiB (none %.1f MiB)\n", tag, r.activation / 1048576.0,
                  none.activation / 1048576.0);
      nexus_test::expect(r.grads == r2.grads, "selective belirlenimci (iki kosu bit-bit)");
      // Selective GELU'yu epilog yerine ayri cekirdekle uretir: none ile ulp duzeyi fark.
      report("selective vs none kayip (GELU cekirdegi farki)", std::fabs(r.loss - none.loss) / none.loss, 1e-3);
      report("selective vs none tum gradyanlar", rel_l2(r.grads, none.grads), 2e-2);
      nexus_test::expect(r.activation < none.activation, "selective daha az aktivasyon");
    }
  }
}

void test_chunked_ce() {
  nexus_test::section("test_chunked_ce_equals_full: lm_head + CE parcali (37 satir) vs tam");
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);  // FP32 tam (TF32 farkli M'de farkli doseme)
  for (const DType t : {DType::F32, DType::BF16}) {
    const nc::GptConfig base = small(t);
    nc::GptConfig c = base;
    c.ce_chunk_rows = 37;  // 256 satira bolunmeyen parca
    const Result full = one_step(base, 2, 128);
    const Result ch = one_step(c, 2, 128);
    const std::string tag = t == DType::F32 ? " fp32" : " bf16";
    report("kayip" + tag, std::fabs(ch.loss - full.loss) / full.loss, 1e-6);
    report("tum gradyanlar" + tag, rel_l2(ch.grads, full.grads), t == DType::F32 ? 1e-5 : 1e-2);
    std::printf("  info  aktivasyon tam %.2f MiB, parcali %.2f MiB\n", full.activation / 1048576.0,
                ch.activation / 1048576.0);
    nexus_test::expect(ch.activation < full.activation, "parcali logit tamponu kucuk" + tag);
  }
}

std::vector<float> train(const nc::GptConfig& c, int B, int T, int steps, std::vector<float>* losses,
                         double* cpu_ms = nullptr, std::uint64_t seed = 11) {
  nc::GptModel m(c, B, T, seed, g_stream);
  Markov data(c.vocab);
  nc::DeviceAdamWConfig adam;
  adam.lr = 1e-3;
  adam.beta2 = 0.95;
  adam.weight_decay = 0.1;
  adam.max_grad_norm = 1.0;
  nc::DeviceBuffer d(static_cast<std::size_t>(B) * T * 4, g_stream);
  for (int s = 0; s < steps; ++s) {
    const auto tok = data.batch(B, T, s);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
    m.forward_backward(d.as<std::int32_t>());
    if (losses != nullptr) {
      losses->push_back(m.read_loss());
    }
    m.optimizer_step(adam);
    if (cpu_ms != nullptr) {
      *cpu_ms += m.last_cpu_update_ms();
    }
  }
  g_stream.synchronize();
  return m.download_parameters();
}

void test_zero_offload() {
  nexus_test::section("test_zero_offload_equals_gpu_adam: 200 adim, BF16, FP32 gradyan D2H");
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  nc::GptConfig gpu = small(DType::BF16);
  gpu.master = nc::MasterWeights::Fp32Master;
  nc::GptConfig off = gpu;
  off.optimizer = nc::OptimizerKind::ZeroOffload;
  off.offload_grad_dtype = DType::F32;
  std::vector<float> lg, lo;
  double cpu_ms = 0.0;
  const std::vector<float> pg = train(gpu, 2, 128, 200, &lg);
  const std::vector<float> po = train(off, 2, 128, 200, &lo, &cpu_ms);
  report("parametreler 200 adim sonra (DPU kapali)", rel_l2(po, pg), 1e-5);
  report("kayip egrisi son adim", std::fabs(lo.back() - lg.back()) / lg.back(), 1e-5);
  std::printf("  info  kayip[0]=%.4f kayip[199] gpu=%.5f offload=%.5f | CPU AdamW ort. %.2f ms/adim\n", lg[0], lg.back(),
              lo.back(), cpu_ms / 200.0);
  // BF16 gradyan D2H (bellek modu): ayni egri, olculen fark raporlanir.
  nc::GptConfig off16 = off;
  off16.offload_grad_dtype = DType::BF16;
  std::vector<float> l16;
  (void)train(off16, 2, 128, 200, &l16);
  std::printf("  info  BF16 gradyanli offload kayip[199]=%.5f (FP32 gradyanli %.5f, bagil %+.2e)\n", l16.back(), lo.back(),
              (l16.back() - lo.back()) / lo.back());
  nexus_test::expect(std::isfinite(l16.back()) && l16.back() < 0.8 * l16.front(), "BF16 gradyanli offload ogreniyor");
  nc::GptModel m(off, 2, 128, 11, g_stream);
  nc::GptModel g(gpu, 2, 128, 11, g_stream);
  std::printf("  info  aygit durum bayti: gpu AdamW %.1f MiB, offload %.1f MiB (+ host %.1f MiB)\n",
              g.device_state_bytes() / 1048576.0, m.device_state_bytes() / 1048576.0, m.host_state_bytes() / 1048576.0);
  nexus_test::expect(m.device_state_bytes() < g.device_state_bytes() / 2, "offload aygit durumunu yarinin altina indirir");
}

double window(const std::vector<float>& v, int end, int w = 50) {
  double a = 0.0;
  for (int i = end - w; i < end; ++i) {
    a += v[static_cast<std::size_t>(i)];
  }
  return a / w;
}

void test_8bit() {
  nexus_test::section("test_8bit_adam_converges: tiny (L6 d384 T256 V8192 B4), 500 adim, 3 tohum ortalamasi");
  nc::GptConfig c;
  c.vocab = 8192;
  c.max_seq = 256;
  c.layers = 6;
  c.heads = 6;
  c.dim = 384;
  c.compute_dtype = DType::BF16;
  c.deterministic_attention = true;
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  nc::GptConfig c8 = c;
  c8.optimizer = nc::OptimizerKind::AdamW8bit;
  // Tek kosu bu gorevde anlamsiz: yalniz dQ toplama sirasini degistirmek egriyi 500. adimda %11
  // oynatti (plato cikis zamani). Ayni tohumlar (baslatma) iki optimizerla; ortalama egri karsilastirilir.
  const std::uint64_t seeds[] = {11, 12, 13};
  std::vector<double> m32(5, 0.0), m8(5, 0.0), s32(5, 0.0);
  std::vector<std::vector<double>> runs32;
  for (const std::uint64_t seed : seeds) {
    std::vector<float> l32, l8;
    (void)train(c, 4, 256, 500, &l32, nullptr, seed);
    (void)train(c8, 4, 256, 500, &l8, nullptr, seed);
    std::vector<double> r;
    for (int k = 0; k < 5; ++k) {
      m32[static_cast<std::size_t>(k)] += window(l32, 100 * (k + 1)) / 3.0;
      m8[static_cast<std::size_t>(k)] += window(l8, 100 * (k + 1)) / 3.0;
      r.push_back(window(l32, 100 * (k + 1)));
    }
    runs32.push_back(r);
    std::printf("  info  tohum %llu @500: 32-bit=%.4f 8-bit=%.4f bagil=%+.3e\n", static_cast<unsigned long long>(seed),
                window(l32, 500), window(l8, 500), (window(l8, 500) - window(l32, 500)) / window(l32, 500));
  }
  for (int k = 0; k < 5; ++k) {
    double v = 0.0;
    for (const auto& r : runs32) {
      v += (r[static_cast<std::size_t>(k)] - m32[static_cast<std::size_t>(k)]) *
           (r[static_cast<std::size_t>(k)] - m32[static_cast<std::size_t>(k)]);
    }
    s32[static_cast<std::size_t>(k)] = std::sqrt(v / 2.0);  // tohumlar arasi std (n-1)
    std::printf("  info  ortalama egri %3d: 32-bit=%.4f 8-bit=%.4f bagil=%+.3e | 32-bit tohum std %.3e\n", 100 * (k + 1),
                m32[static_cast<std::size_t>(k)], m8[static_cast<std::size_t>(k)],
                (m8[static_cast<std::size_t>(k)] - m32[static_cast<std::size_t>(k)]) / m32[static_cast<std::size_t>(k)],
                s32[static_cast<std::size_t>(k)] / m32[static_cast<std::size_t>(k)]);
  }
  const double rel = std::fabs(m8[4] - m32[4]) / m32[4];
  const double noise = s32[4] / m32[4];
  std::printf("  info  @500 ortalama: 8-bit bagil fark %.3e, spec %%2: %s, tohum std %.3e\n", rel,
              rel <= 0.02 ? "gecer" : "GECMEZ", noise);
  nexus_test::expect(rel <= std::max(0.02, 2.0 * noise), "8-bit AdamW ortalama kaybi 32-bit'in max(%2, 2 x tohum std) bandinda");
  nc::GptModel a(c, 4, 256, 1, g_stream);
  nc::GptModel b(c8, 4, 256, 1, g_stream);
  std::printf("  info  aygit durum bayti: AdamW32 %.1f MiB, AdamW8 %.1f MiB\n", a.device_state_bytes() / 1048576.0,
              b.device_state_bytes() / 1048576.0);
  nexus_test::expect(b.device_state_bytes() < a.device_state_bytes(), "8-bit daha az durum bellegi");
}

void test_fused() {
  nexus_test::section("test_fused_8bit_update: geri geciste katman katman 8-bit AdamW vs ayri adim (C modu)");
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  nc::GptConfig c = small(DType::BF16);
  c.master = nc::MasterWeights::NoMaster;
  c.optimizer = nc::OptimizerKind::AdamW8bit;
  c.adam8_embedding_fp32 = false;
  nc::GptConfig f = c;
  f.optimizer = nc::OptimizerKind::AdamW8bitFused;
  f.recompute = nc::Recompute::Full;
  // Kirpma kapali: gecikmeli kirpma farki olmasin; kalan fark stokastik yuvarlama tohumlari.
  auto run = [&](const nc::GptConfig& cfg, std::vector<float>* losses) {
    nc::GptModel m(cfg, 2, 128, 11, g_stream);
    Markov data(cfg.vocab);
    nc::DeviceAdamWConfig adam;
    adam.lr = 1e-3;
    adam.beta2 = 0.95;
    adam.weight_decay = 0.1;
    if (cfg.optimizer == nc::OptimizerKind::AdamW8bitFused) {
      m.set_fused_optimizer(adam);
    }
    nc::DeviceBuffer d(2 * 128 * 4, g_stream);
    for (int s = 0; s < 200; ++s) {
      const auto tok = data.batch(2, 128, s);
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
      m.forward_backward(d.as<std::int32_t>());
      losses->push_back(m.read_loss());
      m.optimizer_step(adam);
    }
    return m.device_state_bytes();
  };
  std::vector<float> ls, lf;
  const std::size_t bs = run(c, &ls);
  const std::size_t bf = run(f, &lf);
  const double a = window(ls, 200), b = window(lf, 200);
  std::printf("  info  kayip[0]=%.4f | son 50 ort.: ayri adim %.4f, fused %.4f (bagil %+.2e) | durum %.2f -> %.2f MiB\n", ls[0], a,
              b, (b - a) / a, bs / 1048576.0, bf / 1048576.0);
  nexus_test::expect(lf[0] == ls[0], "ilk kayip ayni (ayni baslatma)");
  report("fused 200 adim egrisi ayri adima yakin", std::fabs(b - a) / a, 2e-2);
  nexus_test::expect(b < 0.8 * lf[0], "fused ogreniyor");
  nexus_test::expect(bf < bs, "fused: tam gradyan tamponu yok, daha az aygit durumu");
}

// Sanitizer icin kisa kosu: her optimizer/recompute/parcali CE yolu 3 adim.
void test_smoke() {
  nexus_test::section("smoke: 8-bit, fused, offload, selective, parcali CE (3 adim)");
  for (const nc::OptimizerKind opt : {nc::OptimizerKind::AdamW8bit, nc::OptimizerKind::AdamW8bitFused,
                                      nc::OptimizerKind::ZeroOffload}) {
    nc::GptConfig c = small(DType::BF16);
    c.layers = 2;
    c.optimizer = opt;
    c.recompute = opt == nc::OptimizerKind::ZeroOffload ? nc::Recompute::Selective : nc::Recompute::Full;
    c.ce_chunk_rows = 100;
    if (opt == nc::OptimizerKind::AdamW8bitFused) {
      c.master = nc::MasterWeights::NoMaster;
      c.adam8_embedding_fp32 = false;
    }
    std::vector<float> losses;
    nc::GptModel m(c, 2, 128, 3, g_stream);
    Markov data(c.vocab);
    nc::DeviceAdamWConfig adam;
    adam.max_grad_norm = 1.0;
    if (opt == nc::OptimizerKind::AdamW8bitFused) {
      m.set_fused_optimizer(adam);
    }
    nc::DeviceBuffer d(2 * 128 * 4, g_stream);
    for (int s = 0; s < 3; ++s) {
      const auto tok = data.batch(2, 128, s);
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
      m.forward_backward(d.as<std::int32_t>());
      losses.push_back(m.read_loss());
      m.optimizer_step(adam);
    }
    g_stream.synchronize();
    nexus_test::expect(std::isfinite(losses.back()), std::string("smoke ") + nc::to_string(opt) + ": sonlu kayip");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return 77;
  }
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    g_stream = nc::Stream::create(0);
    nc::StreamGuard guard(g_stream);
    if (mode == "all" || mode == "recompute") test_recompute();
    if (mode == "all" || mode == "chunked") test_chunked_ce();
    if (mode == "all" || mode == "offload") test_zero_offload();
    if (mode == "all" || mode == "adam8") test_8bit();
    if (mode == "all" || mode == "fused") test_fused();
    if (mode == "smoke") test_smoke();
    g_stream.synchronize();
    g_stream.destroy();
  } catch (const std::exception& e) {
    std::printf("  FAIL  %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("faz7-memory-cuda");
}
