/** FAZ 5.G benchmark: CPU FlashAttention (FP32) ileri / ileri+geri; PyTorch CPU SDPA ile ayni sekiller
 *  (benchmarks/runner/bench_sdpa_torch_cpu.py). */
#include "nexus/attention/cpu.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
namespace {
double median_ms(const std::function<void()>& fn, int warmup = 1, int reps = 5) {
  for (int i = 0; i < warmup; ++i) fn();
  std::vector<double> v;
  for (int i = 0; i < reps; ++i) {
    const auto a = std::chrono::steady_clock::now();
    fn();
    v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  constexpr int B = 1, H = 8, D = 64;
  std::ostringstream out;
  out.precision(7);
  out << "{\n  \"phase\":5,\"backend\":\"nexus_cpu_flash_fp32\",\"threads\":" << std::thread::hardware_concurrency()
      << ",\"shape\":{\"batch\":" << B << ",\"heads\":" << H << ",\"head_dim\":" << D << "},\"cases\":[\n";
  bool first = true;
  for (const int T : {512, 1024, 2048, 4096}) {
    for (const bool causal : {false, true}) {
      nexus::AttentionParams p;
      p.batch = B; p.heads = H; p.kv_heads = H; p.tq = T; p.tk = T; p.head_dim = D;
      p.dtype = nexus::DType::F32; p.causal = causal;
      p.set_contiguous_bthd();
      const std::size_t n = static_cast<std::size_t>(B) * T * H * D;
      std::vector<float> q(n), k(n), v(n), o(n), dout(n), dq(n), dk(n), dv(n), lse(static_cast<std::size_t>(B) * H * T);
      std::mt19937 rng(3);
      std::normal_distribution<float> normal(0.f, 1.f);
      for (auto* x : {&q, &k, &v, &dout}) for (float& e : *x) e = normal(rng);
      auto fwd = [&] { nexus::attention::flash_attention_forward_cpu(o.data(), lse.data(), q.data(), k.data(), v.data(), p); };
      auto bwd = [&] {
        nexus::attention::flash_attention_backward_cpu(dq.data(), dk.data(), dv.data(), dout.data(), q.data(), k.data(),
                                                       v.data(), o.data(), lse.data(), p);
      };
      const double tf = median_ms(fwd);
      const double tfb = median_ms([&] { fwd(); bwd(); });
      std::printf("  info  T=%5d %-6s fwd %9.3f ms  fwd+bwd %9.3f ms\n", T, causal ? "causal" : "full", tf, tfb);
      out << (first ? "" : ",\n") << "    {\"T\":" << T << ",\"causal\":" << (causal ? "true" : "false")
          << ",\"fwd_ms\":" << tf << ",\"fwd_bwd_ms\":" << tfb << "}";
      first = false;
    }
  }
  out << "\n  ],\"protocol\":{\"warmup\":1,\"repeats\":5,\"stat\":\"median\",\"timing\":\"steady_clock\"}\n}\n";
  const std::filesystem::path dir(NEXUS_FAZ5_DOCS_DIR);
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "bench_attention_cpu.json", std::ios::binary) << out.str();
  std::printf("  OK    CPU dikkat benchmark JSON yazildi\n");
  return 0;
}
