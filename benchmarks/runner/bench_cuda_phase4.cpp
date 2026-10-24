/** FAZ 4 benchmark: dar Linear fwd+bwd ve GPT-2 124M AMP modlari. */
#include "nexus/cuda/gpt.hpp"
#include "nexus/cuda/gemm.hpp"
#include <algorithm>
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
void report(const std::string& name, const Stats& s, const std::string& extra) {
  std::printf("  info  %-34s medyan=%9.3f ms %s\n", name.c_str(), s.median, extra.c_str());
}
void linear(std::ostringstream& out, const nc::Stream& stream) {
  constexpr int M = 8192, N = 4096, K = 4096;
  const std::size_t nx = static_cast<std::size_t>(M) * K, nw = static_cast<std::size_t>(K) * N, ny = static_cast<std::size_t>(M) * N;
  nc::DeviceBuffer xf(nx * 4, stream), wf(nw * 4, stream), dyf(ny * 4, stream);
  nc::fill(xf.as<float>(), 0.01f, nx, stream.handle()); nc::fill(wf.as<float>(), 0.01f, nw, stream.handle());
  nc::fill(dyf.as<float>(), 0.01f, ny, stream.handle());
  for (const nexus::DType type : {nexus::DType::BF16, nexus::DType::F16}) {
    const std::size_t e = nexus::size_of(type);
    nc::DeviceBuffer x(nx * e, stream), w(nw * e, stream), dy(ny * e, stream), y(ny * e, stream), dx(nx * e, stream),
        dw(nw * 4, stream), db(N * 4, stream);
    nc::cast(x.data(), type, xf.data(), nexus::DType::F32, nx, stream.handle());
    nc::cast(w.data(), type, wf.data(), nexus::DType::F32, nw, stream.handle());
    nc::cast(dy.data(), type, dyf.data(), nexus::DType::F32, ny, stream.handle());
    auto run = [&] {
      nc::GemmArgs g; g.m = M; g.n = N; g.k = K; g.a = x.data(); g.lda = K; g.b = w.data(); g.ldb = N;
      g.c = y.data(); g.ldc = N; g.a_type = type; g.b_type = type; g.c_type = type; nc::gemm(g, stream.handle());
      g = nc::GemmArgs{}; g.trans_b = true; g.m = M; g.n = K; g.k = N; g.a = dy.data(); g.lda = N;
      g.b = w.data(); g.ldb = N; g.c = dx.data(); g.ldc = K; g.a_type = type; g.b_type = type; g.c_type = type;
      nc::gemm(g, stream.handle());
      g = nc::GemmArgs{}; g.trans_a = true; g.m = K; g.n = N; g.k = M; g.a = x.data(); g.lda = K;
      g.b = dy.data(); g.ldb = N; g.c = dw.data(); g.ldc = N; g.a_type = type; g.b_type = type;
      g.c_type = nexus::DType::F32; g.bias_type = nexus::DType::F32; g.epilogue = nc::Epilogue::BGradB;
      g.bias_grad = db.data(); nc::gemm(g, stream.handle());
    };
    const Stats st = time_gpu(stream, run, 3, 15);
    const double tflops = 3.0 * 2.0 * M * N * static_cast<double>(K) / (st.median * 1e9);
    // Ampere GA10x dense BF16/FP16 (FP32 biriktirici): SM basina 512 FLOP/saat; azami saat hizinda.
    cudaDeviceProp prop{};
    NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    NEXUS_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    const double peak = prop.multiProcessorCount * 512.0 * clock_khz * 1e3 / 1e12;
    const std::string tag = type == nexus::DType::BF16 ? "bf16" : "fp16";
    report("Linear fwd+bwd " + tag, st,
           std::to_string(tflops) + " TFLOPS (" + std::to_string(100.0 * tflops / peak) + "% / tepe " +
               std::to_string(peak) + ")");
    out << "  \"linear_" << tag << "\":" << json(st, "ms") << ",\"linear_" << tag << "_tflops\":" << tflops
        << ",\"linear_" << tag << "_dense_peak_tflops\":" << peak << ",\n";
  }
}
void gpt(std::ostringstream& out, const nc::Stream& stream) {
  struct Case { const char* name; nexus::DType type; nc::MasterWeights master; bool tf32; };
  const std::vector<Case> cases = {{"fp32", nexus::DType::F32, nc::MasterWeights::Fp32Master, false},
                                   {"tf32", nexus::DType::F32, nc::MasterWeights::Fp32Master, true},
                                   {"bf16_B", nexus::DType::BF16, nc::MasterWeights::Fp32Master, true},
                                   {"fp16_B", nexus::DType::F16, nc::MasterWeights::Fp32Master, true},
                                   {"bf16_A", nexus::DType::BF16, nc::MasterWeights::CastPerForward, true},
                                   {"bf16_C", nexus::DType::BF16, nc::MasterWeights::NoMaster, true}};
  auto& pool = nc::caching_device_allocator();
  for (const auto& c : cases) {
    pool.empty_cache(); pool.reset_peak_stats();
    nc::set_float32_matmul_precision(c.tf32 ? nc::MatmulPrecision::High : nc::MatmulPrecision::Highest);
    try {
      nc::GptConfig config; config.compute_dtype = c.type; config.master = c.master;
      constexpr int batch = 1, time = 1024;
      nc::GptModel model(config, batch, time, 1, stream);
      nc::DeviceAdamWConfig adam; adam.lr = 6e-4; adam.weight_decay = 0.1; adam.max_grad_norm = 1.0; adam.beta2 = 0.95;
      nc::GptTrainer trainer(model, adam, 10);
      std::vector<std::int32_t> tokens(batch * time);
      for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<std::int32_t>((i * 2654435761u) % 50257u);
      const Stats st = time_gpu(stream, [&] { (void)trainer.step(tokens.data()); });
      (void)trainer.synchronize();
      const double tok_s = batch * time / (st.median * 1e-3);
      const auto peak = pool.stats(0).peak_allocated_bytes;
      report(std::string("GPT-2 124M ") + c.name, st,
             std::to_string(static_cast<long long>(tok_s)) + " token/s, peak=" + std::to_string(peak / 1048576) + " MiB");
      out << "  \"gpt2s_" << c.name << "\":" << json(st, "ms/step") << ",\"gpt2s_" << c.name << "_tokens_per_s\":"
          << tok_s << ",\"gpt2s_" << c.name << "_peak_bytes\":" << peak << ",\"gpt2s_" << c.name
          << "_state_bytes\":" << model.state_bytes() << ",\"gpt2s_" << c.name << "_activation_bytes\":"
          << model.activation_bytes() << ",\n";
    } catch (const std::exception& e) {
      std::printf("  info  GPT-2 %s calismadi: %s\n", c.name, e.what());
      out << "  \"gpt2s_" << c.name << "_error\":\"" << e.what() << "\",\n";
    }
    pool.empty_cache();
  }
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
}
}  // namespace
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) { std::printf("  SKIP  CUDA aygiti yok\n"); return 77; }
  try {
    nc::Stream stream = nc::Stream::create(0); nc::StreamGuard guard(stream);
    cudaDeviceProp prop{}; NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::ostringstream out; out.precision(7);
    out << "{\n  \"phase\":4,\"device\":\"" << prop.name << "\",\"compute_capability\":\"" << prop.major << "." << prop.minor
        << "\",\"protocol\":{\"warmup\":2,\"repeats\":7,\"timing\":\"CUDA events\"},\n";
    linear(out, stream);
    gpt(out, stream);
    out << "  \"notes\":\"Yerel FP8 Tensor Core yok; FP8 hiz iddiasi yapilmaz. PyTorch karsilastirmasi ayri calistirilmalidir.\"\n}\n";
    const std::filesystem::path dir(NEXUS_FAZ4_DOCS_DIR); std::filesystem::create_directories(dir);
    std::ofstream(dir / "bench_cuda_phase4.json", std::ios::binary) << out.str();
    std::printf("  OK    FAZ 4 benchmark JSON yazildi\n");
    stream.destroy(); return 0;
  } catch (const std::exception& e) { std::printf("  FAIL  %s\n", e.what()); return 1; }
}

