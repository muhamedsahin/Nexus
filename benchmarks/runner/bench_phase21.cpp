/** Phase 21: synchronized, correctness-gated raw kernel measurements.
 * Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
 * öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
 */
#include "nexus/backend/backend.hpp"
#include "nexus/cpu/threads.hpp"
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>
#if defined(NEXUS_HAS_CUDA_KERNELS)
#include "nexus/cuda/attention.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/cuda/infer.hpp"
#endif
namespace {
double distance(double actual, double expected) {
  if (!std::isfinite(actual)) throw std::runtime_error("nonfinite kernel output");
  return std::abs(actual-expected);
}
void emit(const char* name, const char* device, const char* dtype, const char* shape,
          const std::function<void()>& fn, const std::function<void()>& sync,
          const std::function<double()>& check) {
  for (int i = 0; i < 2; ++i) fn();
  sync();
  const double error = check();
  if (!std::isfinite(error) || error > 0.02) throw std::runtime_error("kernel correctness gate failed");
#if defined(NEXUS_HAS_CUDA_KERNELS)
  if (std::string(device)=="cuda") nexus::cuda::caching_device_allocator().reset_peak_stats();
#endif
  std::vector<double> samples;
  for (int i = 0; i < 7; ++i) {
    const auto start = std::chrono::steady_clock::now();
    fn(); sync();
    samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
  }
  std::cout << std::setprecision(17) << "{\"case\":\"" << name << "\",\"device\":\"" << device
            << "\",\"dtype\":\"" << dtype << "\",\"shape\":" << shape
            << ",\"warmup\":2,\"seed\":1337,\"timing\":\"host_synchronized\",\"max_abs\":" << error << ",\"samples_ms\":[";
  for (std::size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
  std::cout << "]";
  if (std::string(device)=="cpu") std::cout << ",\"cpu_threads\":" << nexus::cpu::Pool::instance().size();
#if defined(NEXUS_HAS_CUDA_KERNELS)
  if (std::string(device)=="cuda") {
    const auto memory=nexus::cuda::caching_device_allocator().stats(0);
    std::size_t free=0,total=0;
    NEXUS_CUDA_CHECK(cudaMemGetInfo(&free,&total));
    std::cout << ",\"peak_allocated_bytes\":" << memory.peak_allocated_bytes
              << ",\"peak_reserved_bytes\":" << memory.peak_reserved_bytes
              << ",\"driver_free_bytes\":" << free << ",\"driver_total_bytes\":" << total
              << ",\"cuda_build\":" << CUDART_VERSION;
  }
#endif
  std::cout << "}" << std::endl;
}

void cpu() {
  constexpr int n = 2048;
  std::vector<float> a(n*n, 0.03125f), b(n*n, 0.015625f), c(n*n);
  nexus::backend::MatMulArgs args;
  args.m = args.n = args.k = args.lda = args.ldb = args.ldc = n;
  args.a = a.data(); args.b = b.data(); args.c = c.data();
  auto& driver = nexus::backend::Registry::instance().require(nexus::backend::Kind::Cpu);
  emit("gemm-2048", "cpu", "f32", "[2048,2048,2048]", [&] {driver.matmul(nexus::DeviceRef::cpu(), args);}, [] {}, [&] {
    double error = 0;
    for (float x : c) error = std::max(error, distance(x, 1.0));
    return error;
  });
}
#if defined(NEXUS_HAS_CUDA_KERNELS)
namespace nc = nexus::cuda;
void gpu() {
  if (!nc::device_count()) {
    std::cout << "{\"case\":\"cuda\",\"status\":\"unavailable\",\"reason\":\"no CUDA device\"}\n";
    return;
  }
  const nc::Stream stream;
  auto sync = [&] {stream.synchronize();};
  auto init = [&](nc::DeviceBuffer& dst, std::size_t size, float value) {
    nc::DeviceBuffer tmp(size*4, stream);
    nc::fill(tmp.as<float>(), value, size, stream.handle());
    nc::cast(dst.data(), nexus::DType::BF16, tmp.data(), nexus::DType::F32, size, stream.handle());
    sync();
  };
  auto read = [&](nc::DeviceBuffer& src, std::size_t count) {
    nc::DeviceBuffer tmp(count*4, stream);
    nc::cast(tmp.data(), nexus::DType::F32, src.data(), nexus::DType::BF16, count, stream.handle());
    std::vector<float> host(count);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(host.data(), tmp.data(), count*4, cudaMemcpyDeviceToHost, stream.handle()));
    sync(); return host;
  };
  {
    constexpr int n = 2048;
    nc::DeviceBuffer a(n*n*2, stream), b(n*n*2, stream), c(n*n*2, stream);
    init(a, n*n, 0.03125f); init(b, n*n, 0.015625f);
    nc::GemmArgs args;
    args.m = args.n = args.k = args.lda = args.ldb = args.ldc = n;
    args.a = a.data(); args.b = b.data(); args.c = c.data();
    args.a_type = args.b_type = args.c_type = nexus::DType::BF16;
    emit("gemm-2048", "cuda", "bf16", "[2048,2048,2048]", [&] {nc::gemm(args, stream.handle());}, sync, [&] {
      double error = 0; for (float x : read(c, n*n)) error = std::max(error, distance(x,1)); return error;
    });
  }
  {
    constexpr int t = 2048, h = 8, d = 64;
    constexpr std::size_t count = t*h*d;
    nc::AttentionParams p;
    p.batch = 1; p.heads = p.kv_heads = h; p.tq = p.tk = t; p.head_dim = d;
    p.dtype = nexus::DType::BF16; p.causal = true; p.set_contiguous_bthd();
    nc::DeviceBuffer q(count*2, stream), k(count*2, stream), v(count*2, stream), o(count*2, stream),
        dout(count*2, stream), dq(count*2, stream), dk(count*2, stream), dv(count*2, stream), lse(h*t*4, stream),
        scratch(nc::flash_attention_backward_scratch_bytes(p), stream);
    init(q, count, 0); init(k, count, 0); init(v, count, 0.25f); init(dout, count, 0.03125f);
    auto fn = [&] {
      nc::flash_attention_forward(o.data(), lse.as<float>(), q.data(), k.data(), v.data(), p, stream.handle());
      nc::flash_attention_backward(dq.data(), dk.data(), dv.data(), dout.data(), q.data(), k.data(), v.data(),
                                   o.data(), lse.as<float>(), p, scratch.data(), stream.handle());
    };
    emit("flash-fwd-bwd", "cuda", "bf16", "[1,8,2048,64]", fn, sync, [&] {
      double error = 0;
      for (float x : read(o, count)) error = std::max(error, distance(x,0.25));
      for (nc::DeviceBuffer* buffer : {&dq, &dk})
        for (float x : read(*buffer, count)) error = std::max(error, distance(x,0));
      const auto grad = read(dv, count);
      double harmonic = 0;
      for (int j = t-1; j >= 0; --j) {
        harmonic += 1.0 / (j+1);
        for (int i = 0; i < h*d; ++i)
          error = std::max(error, distance(grad[static_cast<std::size_t>(j)*h*d+i],harmonic/32));
      }
      return error;
    });
  }
}
void inference(const char* path) {
  namespace ni = nexus::cuda::infer;
  if (!nc::device_count()) throw std::runtime_error("CUDA device required for inference");
  const auto weights = ni::load_gpt2(path);
  ni::EngineConfig config;
  config.max_seqs = 1; config.max_tokens = 1024; config.kv_bytes = std::size_t{128} << 20;
  ni::Engine engine(weights,config);
  ni::SamplingParams sampling; sampling.temperature = 0;
  std::vector<std::int32_t> prompt(32);
  for (int i = 0; i < 32; ++i) prompt[i] = (i*1337+7)%50257;
  std::vector<std::int32_t> expected;
  std::vector<double> ttft, tpot;
  for (int i = 0; i < 9; ++i) {
    ni::GenerateStats s;
    const auto tokens = ni::generate(engine,prompt,32,sampling,&s);
    if (i == 0) expected = tokens;
    if (tokens != expected || tokens.size()!=32 || !std::isfinite(s.ttft_ms) || !std::isfinite(s.tpot_ms))
      throw std::runtime_error("inference determinism/length/finite gate failed");
    if (i >= 2) {ttft.push_back(s.ttft_ms); tpot.push_back(s.tpot_ms);}
  }
  std::cout << std::setprecision(17) << "{\"model\":\"gpt2-124m\",\"dtype\":\"f16\",\"prompt_tokens\":32,\"generated_tokens\":32,\"warmup\":2,\"ttft_ms\":[";
  for (std::size_t i=0;i<ttft.size();++i) std::cout << (i ? "," : "") << ttft[i];
  std::cout << "],\"tpot_ms\":[";
  for (std::size_t i=0;i<tpot.size();++i) std::cout << (i ? "," : "") << tpot[i];
  std::cout << "],\"tokens\":[";
  for (std::size_t i=0;i<expected.size();++i) std::cout << (i ? "," : "") << expected[i];
  std::cout << "]}" << std::endl;
}
#endif
}
int main(int argc, char** argv) {
  (void)argv;
  try {
#if defined(NEXUS_HAS_CUDA_KERNELS)
    if (argc == 3 && std::string(argv[1]) == "--infer") {inference(argv[2]); return 0;}
#endif
    if (argc != 1) throw std::runtime_error("usage: bench_phase21 [--infer GPT2.nxg]");
    cpu();
#if defined(NEXUS_HAS_CUDA_KERNELS)
    gpu();
#else
    std::cout << "{\"case\":\"cuda\",\"status\":\"unavailable\",\"reason\":\"CPU-only build\"}\n";
#endif
  } catch (const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
  return 0;
}
