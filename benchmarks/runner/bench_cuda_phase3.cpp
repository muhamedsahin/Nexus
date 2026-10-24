/**
 * @file bench_cuda_phase3.cpp
 * @brief FAZ 1 + FAZ 3 GPU benchmarklari (RTX 3070 Laptop), BOLUM G protokolu:
 *        isinma + 7 tekrar, medyan/min/max/std, CUDA olay zamanlamasi, makine kimligi.
 *
 * Olculenler (promt.txt BENCHMARK'LAR):
 *  FAZ 1: onbellek isabetinde ayir/birak gecikmesi vs cudaMalloc/cudaFree;
 *         eleman eleman add (2^24 FP32) ve sum indirgeme bant genisligi.
 *  FAZ 3: Linear fwd+bwd (4096x4096, B*T=8192) TFLOPS, FP32 ve TF32, cuBLAS'a gore
 *         ek yuk; LayerNorm/RMSNorm fwd+bwd GB/s; CE (B*T=8192, V=50304) sure ve
 *         tepe bellek; AdamW 124M ms/adim; GPT-2 124M uctan uca token/s ve tepe bellek.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/gpt.hpp"
#include "nexus/cuda/kernels.hpp"

#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace nc = nexus::cuda;

namespace {

struct Stats {
  double median = 0, min = 0, max = 0, std = 0;
  int n = 0;
};

Stats summarize(std::vector<double> v) {
  Stats s;
  s.n = static_cast<int>(v.size());
  std::sort(v.begin(), v.end());
  s.median = v[v.size() / 2];
  s.min = v.front();
  s.max = v.back();
  double mean = 0;
  for (double x : v) {
    mean += x;
  }
  mean /= static_cast<double>(v.size());
  for (double x : v) {
    s.std += (x - mean) * (x - mean);
  }
  s.std = std::sqrt(s.std / static_cast<double>(v.size()));
  return s;
}

// GPU suresi (ms), CUDA olaylariyla; fn kuyruga koyar.
Stats time_gpu(const nc::Stream& stream, const std::function<void()>& fn, int warmup = 3, int reps = 7, int inner = 1) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  stream.synchronize();
  std::vector<double> ms;
  nc::Event a;
  nc::Event b;
  for (int r = 0; r < reps; ++r) {
    a.record(stream);
    for (int i = 0; i < inner; ++i) {
      fn();
    }
    b.record(stream);
    b.synchronize();
    ms.push_back(static_cast<double>(a.elapsed_ms(b)) / inner);
  }
  return summarize(ms);
}

std::string json(const Stats& s, const char* unit = "ms") {
  std::ostringstream o;
  o.precision(6);
  o << "{\"median\": " << s.median << ", \"min\": " << s.min << ", \"max\": " << s.max << ", \"std\": " << s.std
    << ", \"repeats\": " << s.n << ", \"unit\": \"" << unit << "\"}";
  return o.str();
}

void line(const char* name, const Stats& s, const char* unit, const std::string& extra = "") {
  std::printf("  info  %-40s medyan=%10.4f %s  min=%.4f max=%.4f std=%.4f  %s\n", name, s.median, unit, s.min, s.max,
              s.std, extra.c_str());
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return 77;
  }
  cudaDeviceProp prop{};
  NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int mem_clock_khz = 0;
  int bus_width = 0;
  NEXUS_CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
  NEXUS_CUDA_CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0));
  const double peak_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width / 8.0) / 1e9;
  int driver = 0;
  int runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  std::printf("== faz1+faz3 GPU benchmarki: %s, sm_%d%d, %.0f GB/s teorik ==\n", prop.name, prop.major, prop.minor,
              peak_gbs);
  nc::Stream stream = nc::Stream::create(0);
  nc::StreamGuard guard(stream);
  cudaStream_t s = stream.handle();
  std::ostringstream out;
  out << "{\n  \"bench\": \"bench_cuda_phase3\",\n  \"machine\": {\"gpu\": \"" << prop.name << "\", \"sm\": \""
      << prop.major << prop.minor << "\", \"driver\": " << driver << ", \"runtime\": " << runtime
      << ", \"peak_mem_gbs\": " << peak_gbs << ", \"compiler\": \"MSVC " << _MSC_VER << "\", \"git_commit\": \""
      << NEXUS_GIT_COMMIT << "\"},\n  \"protocol\": \"isinma 3 + 7 tekrar, CUDA olaylari, medyan/min/max/std\",\n"
      << "  \"results\": {\n";

  // ---- FAZ 1: ayirici gecikmesi ----------------------------------------------------
  {
    auto& pool = nc::caching_device_allocator();
    constexpr int kIters = 100000;
    std::vector<double> hit_ns;
    for (int r = 0; r < 7; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kIters; ++i) {
        void* p = pool.allocate(1 << 20, stream);
        pool.deallocate(p);
      }
      hit_ns.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / kIters);
    }
    std::vector<double> raw_ns;
    for (int r = 0; r < 7; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 200; ++i) {
        void* p = nullptr;
        NEXUS_CUDA_CHECK(cudaMalloc(&p, 1 << 20));
        NEXUS_CUDA_CHECK(cudaFree(p));
      }
      raw_ns.push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / 200);
    }
    const Stats hit = summarize(hit_ns);
    const Stats raw = summarize(raw_ns);
    line("ayirici onbellek isabeti (1 MiB al+birak)", hit, "ns");
    line("cudaMalloc+cudaFree (1 MiB)", raw, "ns", "hizlanma=" + std::to_string(raw.median / hit.median) + "x");
    out << "    \"allocator_cache_hit_ns\": " << json(hit, "ns") << ",\n    \"cuda_malloc_free_ns\": " << json(raw, "ns")
        << ",\n";
  }

  // ---- FAZ 1: eleman eleman add ve sum ------------------------------------------------
  {
    const std::size_t n = 1u << 24;
    nc::DeviceBuffer a(n * 4), b(n * 4), c(n * 4), r(4), scratch(nc::sum_scratch_bytes(n));
    nc::fill(a.as<float>(), 1.0f, n, s);
    nc::fill(b.as<float>(), 2.0f, n, s);
    const Stats add = time_gpu(stream, [&] { nc::add(c.as<float>(), a.as<float>(), b.as<float>(), n, s); }, 3, 7, 10);
    const Stats sum = time_gpu(stream, [&] { nc::sum(r.as<float>(), a.as<float>(), n, scratch.data(), s); }, 3, 7, 10);
    const double add_gbs = 3.0 * n * 4 / (add.median * 1e6);
    const double sum_gbs = 1.0 * n * 4 / (sum.median * 1e6);
    line("add 2^24 FP32", add, "ms", std::to_string(add_gbs) + " GB/s (" + std::to_string(100 * add_gbs / peak_gbs) + "% tepe)");
    line("sum 2^24 FP32", sum, "ms", std::to_string(sum_gbs) + " GB/s (" + std::to_string(100 * sum_gbs / peak_gbs) + "% tepe)");
    // Pratik tavan: suruculun aygit-aygit kopyasi (okuma + yazma = 2 * n * 4 bayt).
    const Stats copy = time_gpu(stream, [&] {
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(c.data(), a.data(), n * 4, cudaMemcpyDeviceToDevice, s));
    }, 3, 7, 10);
    const double copy_gbs = 2.0 * n * 4 / (copy.median * 1e6);
    line("cudaMemcpy D2D 2^24 FP32 (pratik tavan)", copy, "ms",
         std::to_string(copy_gbs) + " GB/s (" + std::to_string(100 * copy_gbs / peak_gbs) + "% tepe)");
    out << "    \"memcpy_d2d_2p24_f32\": " << json(copy) << ", \"memcpy_d2d_gbs\": " << copy_gbs << ",\n";
    out << "    \"add_2p24_f32\": " << json(add) << ", \"add_gbs\": " << add_gbs << ",\n";
    out << "    \"sum_2p24_f32\": " << json(sum) << ", \"sum_gbs\": " << sum_gbs << ",\n";
  }

  // ---- FAZ 3.1: Linear fwd+bwd 4096x4096, B*T=8192 -------------------------------------
  {
    const int M = 8192, K = 4096, N = 4096;
    nc::DeviceBuffer x(static_cast<std::size_t>(M) * K * 4), w(static_cast<std::size_t>(K) * N * 4),
        y(static_cast<std::size_t>(M) * N * 4), dy(static_cast<std::size_t>(M) * N * 4),
        dx(static_cast<std::size_t>(M) * K * 4), dw(static_cast<std::size_t>(K) * N * 4), bias(N * 4), db(N * 4);
    nc::fill(x.as<float>(), 0.01f, static_cast<std::size_t>(M) * K, s);
    nc::fill(w.as<float>(), 0.01f, static_cast<std::size_t>(K) * N, s);
    nc::fill(dy.as<float>(), 0.01f, static_cast<std::size_t>(M) * N, s);
    nc::fill(bias.as<float>(), 0.0f, N, s);
    auto linear = [&] {
      nc::GemmArgs f;
      f.m = M; f.n = N; f.k = K; f.a = x.as<float>(); f.lda = K; f.b = w.as<float>(); f.ldb = N; f.c = y.as<float>();
      f.ldc = N; f.epilogue = nc::Epilogue::Bias; f.bias = bias.as<float>();
      nc::gemm(f, s);
      nc::GemmArgs gx;  // dX = dY W^T
      gx.trans_b = true; gx.m = M; gx.n = K; gx.k = N; gx.a = dy.as<float>(); gx.lda = N; gx.b = w.as<float>();
      gx.ldb = N; gx.c = dx.as<float>(); gx.ldc = K;
      nc::gemm(gx, s);
      nc::GemmArgs gw;  // dW = X^T dY, db = sum dY
      gw.trans_a = true; gw.m = K; gw.n = N; gw.k = M; gw.a = x.as<float>(); gw.lda = K; gw.b = dy.as<float>();
      gw.ldb = N; gw.c = dw.as<float>(); gw.ldc = N; gw.epilogue = nc::Epilogue::BGradB; gw.bias_grad = db.as<float>();
      nc::gemm(gw, s);
    };
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    cublasSetStream(handle, s);
    auto raw = [&] {  // ayni uc GEMM, cuBLAS'a dogrudan (sutun-ana esleme), epilogsuz
      const float one = 1.0f, zero = 0.0f;
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, w.as<float>(), N, x.as<float>(), K, &zero,
                  y.as<float>(), N);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &one, w.as<float>(), N, dy.as<float>(), N, &zero,
                  dx.as<float>(), K);
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &one, dy.as<float>(), N, x.as<float>(), K, &zero,
                  dw.as<float>(), N);
    };
    const double flops = 3.0 * 2.0 * M * N * static_cast<double>(K);
    for (const bool tf32 : {false, true}) {
      nc::set_float32_matmul_precision(tf32 ? nc::MatmulPrecision::High : nc::MatmulPrecision::Highest);
      cublasSetMathMode(handle, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
      const Stats ours = time_gpu(stream, linear, 2, 7);
      const Stats theirs = time_gpu(stream, raw, 2, 7);
      const double tflops = flops / (ours.median * 1e9);
      const double raw_tflops = flops / (theirs.median * 1e9);
      const char* tag = tf32 ? "TF32" : "FP32";
      line((std::string("Linear fwd+bwd 8192x4096x4096 ") + tag).c_str(), ours, "ms",
           std::to_string(tflops) + " TFLOPS");
      line((std::string("cuBLAS dogrudan ayni 3 GEMM ") + tag).c_str(), theirs, "ms",
           std::to_string(raw_tflops) + " TFLOPS, Nexus ek yuk=" +
               std::to_string(100.0 * (ours.median - theirs.median) / theirs.median) + "%");
      out << "    \"linear_fwd_bwd_" << (tf32 ? "tf32" : "fp32") << "\": " << json(ours) << ", \"linear_tflops_"
          << (tf32 ? "tf32" : "fp32") << "\": " << tflops << ",\n    \"cublas_direct_" << (tf32 ? "tf32" : "fp32")
          << "\": " << json(theirs) << ", \"cublas_tflops_" << (tf32 ? "tf32" : "fp32") << "\": " << raw_tflops
          << ",\n";
    }
    nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
    cublasDestroy(handle);
  }

  // ---- FAZ 3.2: LayerNorm / RMSNorm fwd+bwd -------------------------------------------
  {
    const int rows = 8192, cols = 768;
    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    nc::DeviceBuffer x(n * 4), y(n * 4), dy(n * 4), dx(n * 4), w(cols * 4), b(cols * 4), dw(cols * 4), dbb(cols * 4),
        mean(rows * 4), rstd(rows * 4), scratch(nc::layernorm_backward_scratch_bytes(rows, cols));
    nc::fill(x.as<float>(), 0.5f, n, s);
    nc::fill(dy.as<float>(), 0.1f, n, s);
    nc::fill(w.as<float>(), 1.0f, cols, s);
    nc::fill(b.as<float>(), 0.0f, cols, s);
    const Stats ln = time_gpu(stream, [&] {
      nc::layernorm_forward(y.as<float>(), mean.as<float>(), rstd.as<float>(), x.as<float>(), w.as<float>(),
                            b.as<float>(), rows, cols, 1e-5f, s);
      nc::layernorm_backward(dx.as<float>(), dw.as<float>(), dbb.as<float>(), dy.as<float>(), x.as<float>(),
                             w.as<float>(), mean.as<float>(), rstd.as<float>(), rows, cols, false,
                             scratch.as<float>(), s);
    });
    const Stats rms = time_gpu(stream, [&] {
      nc::rmsnorm_forward(y.as<float>(), rstd.as<float>(), x.as<float>(), w.as<float>(), rows, cols, 1e-6f, s);
      nc::rmsnorm_backward(dx.as<float>(), dw.as<float>(), dy.as<float>(), x.as<float>(), w.as<float>(),
                           rstd.as<float>(), rows, cols, false, scratch.as<float>(), s);
    });
    // Asgari trafik: ileri x oku + y yaz, geri dy + x oku + dx yaz = 5 * n * 4 bayt.
    const double bytes = 5.0 * n * 4;
    line("LayerNorm fwd+bwd 8192x768", ln, "ms",
         std::to_string(bytes / (ln.median * 1e6)) + " GB/s (" + std::to_string(100 * bytes / (ln.median * 1e6) / peak_gbs) + "% tepe)");
    line("RMSNorm fwd+bwd 8192x768", rms, "ms",
         std::to_string(bytes / (rms.median * 1e6)) + " GB/s (" + std::to_string(100 * bytes / (rms.median * 1e6) / peak_gbs) + "% tepe)");
    out << "    \"layernorm_fwd_bwd_8192x768\": " << json(ln) << ", \"layernorm_gbs\": " << bytes / (ln.median * 1e6)
        << ",\n    \"rmsnorm_fwd_bwd_8192x768\": " << json(rms) << ", \"rmsnorm_gbs\": " << bytes / (rms.median * 1e6)
        << ",\n";
  }

  // ---- FAZ 3.5: CE, B*T=8192, V=50304 ------------------------------------------------------
  {
    const int batch = 8, time = 1024, vocab = 50304;
    const std::size_t n = static_cast<std::size_t>(batch) * time * vocab;
    auto& pool = nc::caching_device_allocator();
    nc::DeviceBuffer logits(n * 4), loss(4), tokens(static_cast<std::size_t>(batch) * time * 4),
        scratch(nc::cross_entropy_scratch_bytes(batch, time));
    nc::fill(logits.as<float>(), 0.0f, n, s);
    NEXUS_CUDA_CHECK(cudaMemsetAsync(tokens.data(), 0, static_cast<std::size_t>(batch) * time * 4, s));
    stream.synchronize();
    pool.reset_peak_stats();
    const auto before = pool.stats(0).allocated_bytes;
    const Stats ce = time_gpu(stream, [&] {
      nc::cross_entropy_shifted(logits.as<float>(), loss.as<float>(), nullptr, logits.as<float>(),
                                tokens.as<std::int32_t>(), batch, time, vocab, {}, 1.0f, scratch.data(), s);
    });
    const auto peak_extra = pool.stats(0).peak_allocated_bytes - before;
    line("CE kaydirilmis yerinde B*T=8192 V=50304", ce, "ms",
         "logit=" + std::to_string(n * 4 / 1048576) + " MiB, ek tepe=" + std::to_string(peak_extra) + " B");
    out << "    \"ce_8192x50304\": " << json(ce) << ", \"ce_logit_bytes\": " << n * 4
        << ", \"ce_extra_peak_bytes\": " << peak_extra << ",\n";
  }

  // ---- FAZ 3.6: AdamW 124M -----------------------------------------------------------------
  {
    const std::size_t n = 124439808;
    nc::DeviceBuffer p(n * 4), g(n * 4), m(n * 4), v(n * 4), norm(4), scratch(nc::adamw_scratch_bytes(1, n));
    nc::fill(p.as<float>(), 0.01f, n, s);
    nc::fill(g.as<float>(), 0.001f, n, s);
    nc::fill(m.as<float>(), 0.0f, n, s);
    nc::fill(v.as<float>(), 0.0f, n, s);
    nc::DeviceParamSlot slot;
    slot.param = p.as<float>();
    slot.grad = g.as<float>();
    slot.exp_avg = m.as<float>();
    slot.exp_avg_sq = v.as<float>();
    slot.numel = n;
    nc::DeviceAdamWConfig cfg;
    cfg.lr = 6e-4;
    cfg.weight_decay = 0.1;
    cfg.max_grad_norm = 1.0;
    std::int64_t step = 0;
    const Stats adam = time_gpu(stream, [&] {
      nc::adamw_multi_tensor(&slot, 1, cfg, ++step, norm.as<float>(), scratch.data(), s);
    });
    // Trafik: norm geçisi g oku (4n) + guncelleme p,g,m,v oku + p,m,v yaz (28n).
    const double bytes = 32.0 * n;
    line("AdamW 124M (+global norm kirpma)", adam, "ms",
         std::to_string(bytes / (adam.median * 1e6)) + " GB/s (" + std::to_string(100 * bytes / (adam.median * 1e6) / peak_gbs) + "% tepe)");
    out << "    \"adamw_124m\": " << json(adam) << ", \"adamw_gbs\": " << bytes / (adam.median * 1e6) << ",\n";
  }

  // ---- FAZ 3.I: GPT-2 124M uctan uca -----------------------------------------------------------
  {
    auto& pool = nc::caching_device_allocator();
    pool.empty_cache();
    for (const bool tf32 : {false, true}) {
      nc::set_float32_matmul_precision(tf32 ? nc::MatmulPrecision::High : nc::MatmulPrecision::Highest);
      for (const int batch : {1, 2}) {
        pool.reset_peak_stats();
        try {
          nc::GptConfig config;
          const int time = 1024;
          nc::GptModel model(config, batch, time, 1, stream);
          nc::DeviceAdamWConfig adam;
          adam.lr = 6e-4;
          adam.weight_decay = 0.1;
          adam.max_grad_norm = 1.0;
          adam.beta2 = 0.95;
          nc::GptTrainer trainer(model, adam, 10);
          std::vector<std::int32_t> tokens(static_cast<std::size_t>(batch) * time);
          for (std::size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] = static_cast<std::int32_t>((i * 2654435761u) % 50257u);
          }
          const Stats st = time_gpu(stream, [&] { (void)trainer.step(tokens.data()); }, 2, 7);
          (void)trainer.synchronize();
          const double tok_s = batch * time / (st.median * 1e-3);
          const auto peak = pool.stats(0).peak_allocated_bytes;
          const std::string tag = std::string("GPT-2 124M B=") + std::to_string(batch) + " T=1024 " + (tf32 ? "TF32" : "FP32");
          line(tag.c_str(), st, "ms/adim",
               std::to_string(static_cast<long long>(tok_s)) + " token/s, tepe=" + std::to_string(peak / 1048576) + " MiB");
          out << "    \"gpt2s_b" << batch << "_" << (tf32 ? "tf32" : "fp32") << "\": " << json(st, "ms/step")
              << ", \"gpt2s_b" << batch << "_" << (tf32 ? "tf32" : "fp32") << "_tokens_per_s\": " << tok_s
              << ", \"gpt2s_b" << batch << "_" << (tf32 ? "tf32" : "fp32") << "_peak_bytes\": " << peak << ",\n";
        } catch (const std::exception& error) {
          std::printf("  info  GPT-2 124M B=%d %s sigmadi: %s\n", batch, tf32 ? "TF32" : "FP32", error.what());
          out << "    \"gpt2s_b" << batch << "_" << (tf32 ? "tf32" : "fp32") << "_error\": \"OOM\",\n";
        }
        pool.empty_cache();
      }
    }
    nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  }

  out << "    \"notes\": \"PyTorch CUDA bu kurulumda yok (.bench-venv torch 2.14.1+cpu); GPU karsilastirmasi cuBLAS dogrudan cagriya karsidir.\"\n  }\n}\n";
  const std::filesystem::path dir(NEXUS_FAZ3_DOCS_DIR);
  std::filesystem::create_directories(dir);
  std::ofstream(dir / "bench_cuda_phase3.json", std::ios::binary) << out.str();
  std::printf("  OK    bench-cuda-faz3 tamamlandi -> %s\n", (dir / "bench_cuda_phase3.json").string().c_str());
  stream.destroy();
  return 0;
}
