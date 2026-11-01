/**
 * @file dalga3_gemm.cpp
 * @brief Dalga 3.1: GPT-2 124M egitim adimindaki her GEMM sekli, Nexus cuBLASLt (sezgisel / autotune).
 *
 * Kullanim: dalga3_gemm --bt 1024 --out sonuc.json [--epilogue 1]
 *   --epilogue 0: tum sekiller BF16 giris -> BF16 cikis, epilogsuz (PyTorch torch.matmul ile ayni is;
 *                 karsilastirma benchmarks/runner/dalga3_gemm_torch.py).
 *   --epilogue 1: Nexus GptModel'in gercek cagrisi (cikis tipi + epilog: bias, bias+gelu+aux, dgelu, bgrad).
 * Girisler sabit tohumlu N(0, 0.02) (sabit deger GPU guc/saatini gercek disi etkiler). Her sekil:
 * 10 isinma, sonra 7 tekrar x 50 cagri, CUDA event; cagri basina medyan ms raporlanir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/cuda/runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
using nexus::DType;

namespace {

struct Shape {
  const char* name;
  bool ta, tb;
  int m, n, k;
  DType out;           // --epilogue 1 cikis tipi
  nc::Epilogue epi;    // --epilogue 1 epilogu
};

std::vector<Shape> shapes(int bt) {
  constexpr int C = 768, V = 50304;
  using E = nc::Epilogue;
  return {
      {"fwd.qkv", false, false, bt, 3 * C, C, DType::BF16, E::Bias},
      {"fwd.attn_proj", false, false, bt, C, C, DType::F32, E::Bias},
      {"fwd.fc", false, false, bt, 4 * C, C, DType::BF16, E::BiasGelu},
      {"fwd.fc_proj", false, false, bt, C, 4 * C, DType::F32, E::Bias},
      {"fwd.lm_head", false, true, bt, V, C, DType::F32, E::None},
      {"bwd.lm_head.dx", false, false, bt, C, V, DType::F32, E::None},
      {"bwd.lm_head.dw", true, false, V, C, bt, DType::F32, E::None},
      {"bwd.fc_proj.dx", false, true, bt, 4 * C, C, DType::BF16, E::DGelu},
      {"bwd.fc_proj.dw", true, false, 4 * C, C, bt, DType::F32, E::BGradB},
      {"bwd.fc.dx", false, true, bt, C, 4 * C, DType::F32, E::None},
      {"bwd.fc.dw", true, false, C, 4 * C, bt, DType::F32, E::BGradB},
      {"bwd.attn_proj.dx", false, true, bt, C, C, DType::BF16, E::None},
      {"bwd.attn_proj.dw", true, false, C, C, bt, DType::F32, E::BGradB},
      {"bwd.qkv.dx", false, true, bt, C, 3 * C, DType::F32, E::None},
      {"bwd.qkv.dw", true, false, C, 3 * C, bt, DType::F32, E::BGradB},
  };
}

void random_bf16(void* dst, std::size_t n, unsigned seed, const nc::Stream& s) {
  std::mt19937 g(seed);
  std::normal_distribution<float> d(0.0f, 0.02f);
  std::vector<float> h(n);
  for (auto& x : h) x = d(g);
  nc::DeviceBuffer f(n * 4, s);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(f.data(), h.data(), n * 4, cudaMemcpyHostToDevice, s.handle()));
  nc::cast(dst, DType::BF16, f.data(), DType::F32, n, s.handle());
  s.synchronize();
}

double time_ms(const nc::Stream& s, const std::function<void()>& fn) {
  for (int i = 0; i < 10; ++i) fn();
  s.synchronize();
  nc::Event a, b;
  std::vector<double> v;
  for (int r = 0; r < 7; ++r) {
    a.record(s);
    for (int i = 0; i < 50; ++i) fn();
    b.record(s);
    b.synchronize();
    v.push_back(a.elapsed_ms(b) / 50.0);
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  int bt = 1024;
  bool epilogue = false;
  std::string out, only;
  bool tune = true;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--bt")) bt = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--out")) out = argv[i + 1];
    else if (!std::strcmp(argv[i], "--epilogue")) epilogue = std::atoi(argv[i + 1]) != 0;
    else if (!std::strcmp(argv[i], "--only")) only = argv[i + 1];  // tek sekil (profil icin)
    else if (!std::strcmp(argv[i], "--tune")) tune = std::atoi(argv[i + 1]) != 0;
  }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    std::printf("SKIP: CUDA aygiti yok\n");
    return 77;
  }
  nc::Stream s;
  std::string json = "{\"bt\":" + std::to_string(bt) + ",\"epilogue\":" + (epilogue ? "true" : "false") + ",\"shapes\":[";
  std::printf("%-18s %6s %6s %6s %-10s %10s %10s %8s\n", "sekil", "m", "n", "k", "epilog", "sezgi_ms", "autot_ms", "TFLOP/s");
  bool first = true;
  for (const Shape& sh : shapes(bt)) {
    if (!only.empty() && only != sh.name) continue;
    const std::size_t na = static_cast<std::size_t>(sh.m) * sh.k, nb = static_cast<std::size_t>(sh.k) * sh.n,
                      ncn = static_cast<std::size_t>(sh.m) * sh.n;
    const DType ct = epilogue ? sh.out : DType::BF16;
    const nc::Epilogue epi = epilogue ? sh.epi : nc::Epilogue::None;
    nc::DeviceBuffer a(na * 2, s), b(nb * 2, s), c(ncn * nexus::size_of(ct), s), aux(ncn * nexus::size_of(ct), s),
        bias(static_cast<std::size_t>(std::max(sh.m, sh.n)) * 4, s);
    random_bf16(a.data(), na, 1u, s);
    random_bf16(b.data(), nb, 2u, s);
    if (epi == nc::Epilogue::DGelu) {
      if (ct == DType::BF16) random_bf16(aux.data(), ncn, 3u, s);
      else nc::fill(aux.as<float>(), 0.1f, ncn, s.handle());
    }
    NEXUS_CUDA_CHECK(cudaMemsetAsync(bias.data(), 0, static_cast<std::size_t>(std::max(sh.m, sh.n)) * 4, s.handle()));
    nc::GemmArgs g;
    g.trans_a = sh.ta;
    g.trans_b = sh.tb;
    g.m = sh.m;
    g.n = sh.n;
    g.k = sh.k;
    g.a = a.data();
    g.lda = sh.ta ? sh.m : sh.k;
    g.b = b.data();
    g.ldb = sh.tb ? sh.k : sh.n;
    g.c = c.data();
    g.ldc = sh.n;
    g.a_type = g.b_type = DType::BF16;
    g.c_type = ct;
    g.bias_type = ct;
    g.epilogue = epi;
    if (epi == nc::Epilogue::Bias || epi == nc::Epilogue::BiasGelu) g.bias = bias.data();
    if (epi == nc::Epilogue::BiasGelu || epi == nc::Epilogue::DGelu) g.aux = aux.data();
    if (epi == nc::Epilogue::BGradB) g.bias_grad = bias.data();
    double ms[2];
    for (int t = 0; t < 2; ++t) {
      nc::clear_gemm_cache();
      nc::set_gemm_autotune(t == 1);
      ms[t] = t == 1 && !tune ? 0.0 : time_ms(s, [&] { nc::gemm(g, s.handle()); });
    }
    nc::set_gemm_autotune(false);
    // BiasGelu karsiligi ayri yol (GptModel selective): Bias epiloglu GEMM -> on-aktivasyon (aux tamponu),
    // ardindan ayri GELU cekirdegi. Ayni iki cikti (aktivasyon + on-aktivasyon), iki baslatma.
    double split_ms = 0.0;
    if (epi == nc::Epilogue::BiasGelu) {
      nc::GemmArgs gb = g;
      gb.epilogue = nc::Epilogue::Bias;
      gb.aux = nullptr;
      gb.c = aux.data();
      nc::clear_gemm_cache();
      split_ms = time_ms(s, [&] {
        nc::gemm(gb, s.handle());
        nc::gelu_forward(c.data(), aux.data(), ct, ncn, s.handle());
      });
      std::printf("%-18s ayri yol (Bias GEMM + GELU cekirdegi) %.4f ms\n", sh.name, split_ms);
    }
    const double flop = 2.0 * sh.m * sh.n * sh.k;
    const double best = std::min(ms[0], ms[1]);
    std::printf("%-18s %6d %6d %6d %-10s %10.4f %10.4f %8.1f\n", sh.name, sh.m, sh.n, sh.k, nc::to_string(epi), ms[0],
                ms[1], flop / best / 1e9);
    char row[512];
    std::snprintf(row, sizeof row,
                  "%s{\"name\":\"%s\",\"ta\":%s,\"tb\":%s,\"m\":%d,\"n\":%d,\"k\":%d,\"out\":\"%s\",\"epilogue\":\"%s\","
                  "\"heuristic_ms\":%.6f,\"autotune_ms\":%.6f,\"split_ms\":%.6f}",
                  first ? "" : ",", sh.name, sh.ta ? "true" : "false", sh.tb ? "true" : "false", sh.m, sh.n, sh.k,
                  std::string(nexus::to_string(ct)).c_str(), nc::to_string(epi), ms[0], ms[1], split_ms);
    json += row;
    first = false;
  }
  json += "]}\n";
  if (!out.empty()) std::ofstream(out) << json;
  return 0;
}
