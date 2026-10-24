/** FAZ 5 benchmark: FlashAttention-2 cekirdek TFLOPS / bellek ve GPT-2 124M flash vs math. */
#include "nexus/cuda/attention.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/gpt.hpp"
#include "nexus/cuda/kernels.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <vector>
namespace nc = nexus::cuda;
namespace {
struct Stats { double median = 0, min = 0, max = 0, std = 0; int n = 0; };
Stats summarize(std::vector<double> v) {
  Stats s; s.n = static_cast<int>(v.size()); std::sort(v.begin(), v.end());
  s.median = v[v.size() / 2]; s.min = v.front(); s.max = v.back();
  double mean = 0; for (double x : v) mean += x; mean /= v.size();
  for (double x : v) s.std += (x - mean) * (x - mean); s.std = std::sqrt(s.std / v.size());
  return s;
}
Stats time_gpu(const nc::Stream& stream, const std::function<void()>& fn, int warmup = 2, int reps = 7) {
  for (int i = 0; i < warmup; ++i) fn(); stream.synchronize();
  nc::Event a, b; std::vector<double> samples;
  for (int i = 0; i < reps; ++i) {
    a.record(stream); fn(); b.record(stream); b.synchronize(); samples.push_back(a.elapsed_ms(b));
  }
  return summarize(samples);
}
std::string json(const Stats& s, const char* unit) {
  std::ostringstream o; o.precision(7);
  o << "{\"median\":" << s.median << ",\"min\":" << s.min << ",\"max\":" << s.max
    << ",\"std\":" << s.std << ",\"repeats\":" << s.n << ",\"unit\":\"" << unit << "\"}";
  return o.str();
}

// FA2 makalesi duzeni: toplam 16k token, gizli boyut 2048 (H = 2048 / D), BF16.
void kernels(std::ostringstream& out, const nc::Stream& stream) {
  auto& pool = nc::caching_device_allocator();
  constexpr int kTokens = 16384, kHidden = 2048;
  const std::size_t n = static_cast<std::size_t>(kTokens) * kHidden;
  std::vector<float> host(n);
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.f, 1.f);
  for (float& x : host) x = normal(rng);
  nc::DeviceBuffer f32(n * 4, stream);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(f32.data(), host.data(), n * 4, cudaMemcpyHostToDevice, stream.handle()));
  out << "  \"kernel\":[\n";
  bool first = true;
  for (const int D : {64, 128}) {
    for (const int T : {512, 1024, 2048, 4096, 8192, 16384}) {
      for (const bool causal : {false, true}) {
        const int B = kTokens / T, H = kHidden / D;
        nexus::AttentionParams p;
        p.batch = B; p.heads = H; p.kv_heads = H; p.tq = T; p.tk = T; p.head_dim = D;
        p.dtype = nexus::DType::BF16; p.causal = causal;
        p.set_contiguous_bthd();
        pool.empty_cache(); pool.reset_peak_stats();
        const std::size_t base = pool.stats(0).allocated_bytes;
        nc::DeviceBuffer q(n * 2, stream), k(n * 2, stream), v(n * 2, stream), o(n * 2, stream), dout(n * 2, stream),
            dq(n * 2, stream), dk(n * 2, stream), dv(n * 2, stream);
        nc::DeviceBuffer lse(static_cast<std::size_t>(B) * H * T * 4, stream);
        nc::DeviceBuffer scratch(nc::flash_attention_backward_scratch_bytes(p), stream);
        for (nc::DeviceBuffer* b : {&q, &k, &v, &dout}) {
          nc::cast(b->data(), nexus::DType::BF16, f32.data(), nexus::DType::F32, n, stream.handle());
        }
        auto fwd = [&] { nc::flash_attention_forward(o.data(), lse.as<float>(), q.data(), k.data(), v.data(), p, stream.handle()); };
        auto bwd = [&] {
          nc::flash_attention_backward(dq.data(), dk.data(), dv.data(), dout.data(), q.data(), k.data(), v.data(),
                                       o.data(), lse.as<float>(), p, scratch.data(), stream.handle());
        };
        const Stats sf = time_gpu(stream, fwd, 2, 7);
        const Stats sb = time_gpu(stream, bwd, 2, 7);
        const Stats sfb = time_gpu(stream, [&] { fwd(); bwd(); }, 2, 7);
        const std::size_t peak = pool.stats(0).peak_allocated_bytes - base;
        // FLOP (FA2 makalesi): ileri 4 B H T^2 D (causal / 2), geri 2.5 x ileri.
        const double ff = 4.0 * B * H * static_cast<double>(T) * T * D / (causal ? 2.0 : 1.0);
        const double tf = ff / (sf.median * 1e9), tb = 2.5 * ff / (sb.median * 1e9), tfb = 3.5 * ff / (sfb.median * 1e9);
        // Math yolunun yalniz skor/olasilik tamponu (BF16 [B,H,T,T]); olculmez, karsilastirma icin yazilir.
        const double math_scores = 2.0 * B * H * static_cast<double>(T) * T;
        std::printf("  info  d=%3d T=%5d %-6s fwd %6.2f TF (%8.3f ms) bwd %6.2f TF fwd+bwd %6.2f TF peak=%6.1f MiB\n", D, T,
                    causal ? "causal" : "full", tf, sf.median, tb, tfb, peak / 1048576.0);
        out << (first ? "" : ",\n") << "    {\"d\":" << D << ",\"T\":" << T << ",\"batch\":" << B << ",\"heads\":" << H
            << ",\"causal\":" << (causal ? "true" : "false") << ",\"fwd\":" << json(sf, "ms") << ",\"bwd\":" << json(sb, "ms")
            << ",\"fwd_bwd\":" << json(sfb, "ms") << ",\"fwd_tflops\":" << tf << ",\"bwd_tflops\":" << tb
            << ",\"fwd_bwd_tflops\":" << tfb << ",\"peak_bytes\":" << peak << ",\"math_score_buffer_bytes\":" << math_scores
            << "}";
        first = false;
      }
    }
  }
  out << "\n  ],\n";
}

void gpt(std::ostringstream& out, const nc::Stream& stream) {
  auto& pool = nc::caching_device_allocator();
  nc::set_float32_matmul_precision(nc::MatmulPrecision::High);
  for (const int T : {1024, 2048, 4096}) {
    for (const nc::AttentionBackend backend : {nc::AttentionBackend::Math, nc::AttentionBackend::Flash}) {
      const std::string name = std::string("gpt2s_bf16_T") + std::to_string(T) + "_" + nc::to_string(backend);
      pool.empty_cache(); pool.reset_peak_stats();
      try {
        nc::GptConfig config;
        config.compute_dtype = nexus::DType::BF16;
        config.max_seq = T;
        config.attention = backend;
        constexpr int batch = 1;
        nc::GptModel model(config, batch, T, 1, stream);
        nc::DeviceAdamWConfig adam; adam.lr = 6e-4; adam.weight_decay = 0.1; adam.max_grad_norm = 1.0; adam.beta2 = 0.95;
        nc::GptTrainer trainer(model, adam, 10);
        std::vector<std::int32_t> tokens(static_cast<std::size_t>(batch) * T);
        for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<std::int32_t>((i * 2654435761u) % 50257u);
        const Stats st = time_gpu(stream, [&] { (void)trainer.step(tokens.data()); }, 2, 5);
        (void)trainer.synchronize();
        const double tok_s = batch * T / (st.median * 1e-3);
        const auto peak = pool.stats(0).peak_allocated_bytes;
        std::printf("  info  %-28s medyan=%9.3f ms %8.0f token/s peak=%6.0f MiB akt=%6.0f MiB\n", name.c_str(), st.median,
                    tok_s, peak / 1048576.0, model.activation_bytes() / 1048576.0);
        out << "  \"" << name << "\":" << json(st, "ms/step") << ",\"" << name << "_tokens_per_s\":" << tok_s << ",\""
            << name << "_peak_bytes\":" << peak << ",\"" << name << "_activation_bytes\":" << model.activation_bytes()
            << ",\n";
      } catch (const std::exception& e) {
        std::printf("  info  %s calismadi: %s\n", name.c_str(), e.what());
        std::string what = e.what();
        std::replace(what.begin(), what.end(), '"', '\'');
        out << "  \"" << name << "_error\":\"" << what << "\",\n";
      }
      pool.empty_cache();
    }
  }
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
}
}  // namespace
int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) { std::printf("  SKIP  CUDA aygiti yok\n"); return 77; }
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    nc::Stream stream = nc::Stream::create(0); nc::StreamGuard guard(stream);
    cudaDeviceProp prop{}; NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::ostringstream out; out.precision(7);
    out << "{\n  \"phase\":5,\"device\":\"" << prop.name << "\",\"compute_capability\":\"" << prop.major << "." << prop.minor
        << "\",\"protocol\":{\"warmup\":2,\"repeats\":\"7 (kernel) / 5 (gpt)\",\"timing\":\"CUDA events\"},\n";
    if (mode == "all" || mode == "kernel") kernels(out, stream);
    if (mode == "all" || mode == "gpt") gpt(out, stream);
    out << "  \"notes\":\"PyTorch CUDA kurulu degil: flash-attn/PyTorch SDPA ile GPU karsilastirmasi yok. "
           "math_score_buffer_bytes hesaplanmistir (olculmez).\"\n}\n";
    if (mode == "all") {
      const std::filesystem::path dir(NEXUS_FAZ5_DOCS_DIR); std::filesystem::create_directories(dir);
      std::ofstream(dir / "bench_cuda_phase5.json", std::ios::binary) << out.str();
      std::printf("  OK    FAZ 5 benchmark JSON yazildi\n");
    }
    stream.destroy(); return 0;
  } catch (const std::exception& e) { std::printf("  FAIL  %s\n", e.what()); return 1; }
}
