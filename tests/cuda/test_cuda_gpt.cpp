/**
 * @file test_cuda_gpt.cpp
 * @brief FAZ 3.I: GPU GPT-2 egitim yolu vs PyTorch (FP32, ayni agirlik/token/AdamW).
 *
 * promt.txt FAZ 3 TESTLER: test_gpt2s_parity_fp32 — PyTorch ile ilk 10 adim kayip
 * farki <= 1e-4. Ayrica ilk adimin tum parametre gradyanlari karsilastirilir.
 *
 * Kullanim: test_cuda_gpt <small|gpt2s> [adim]
 *   small: L=2, C=64, H=4, V=257 (tek), T=32, B=2  — her ctest'te
 *   gpt2s: GPT-2 124M sekli, T=1024, B=1           — test_gpt2s_parity_fp32
 *
 * PyTorch yoksa 77 (SKIPPED). Referans CPU PyTorch'tur: bu kurulumda CUDA'li PyTorch yok.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/gpt.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace nc = nexus::cuda;

namespace {

constexpr int kSkip = 77;

int run_status(const std::string& command) { return std::system(("\"" + command + "\"").c_str()); }

int run(int argc, char** argv) {
  const std::string mode = argc >= 2 ? argv[1] : "small";
  const int steps = argc >= 3 ? std::atoi(argv[2]) : 10;
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return kSkip;
  }
  const std::string python = NEXUS_PARITY_PYTHON;
  if (python.empty() || !std::filesystem::exists(python)) {
    std::printf("  SKIP  PyTorch'lu python yok\n");
    return kSkip;
  }
  nc::GptConfig config;
  int batch = 2;
  int time = 32;
  if (mode == "small") {
    config.vocab = 257;
    config.max_seq = 64;
    config.layers = 2;
    config.heads = 4;
    config.dim = 64;
  } else if (mode == "gpt2s") {
    batch = 1;
    time = 1024;
  } else {
    std::printf("kullanim: test_cuda_gpt <small|gpt2s> [adim]\n");
    return 2;
  }
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);  // FP32 parity: TF32 kapali
  nc::Stream stream = nc::Stream::create(0);
  nc::StreamGuard guard(stream);
  nc::GptModel model(config, batch, time, 20261024, stream);
  std::printf("gpt %s: L=%d C=%d H=%d V=%d B=%d T=%d param=%zu aktivasyon=%.1f MiB\n", mode.c_str(), config.layers,
              config.dim, config.heads, config.vocab, batch, time, model.num_parameters(),
              static_cast<double>(model.activation_bytes()) / 1048576.0);

  const std::vector<float> initial = model.download_parameters();
  const std::size_t bt = static_cast<std::size_t>(batch) * time;
  std::vector<std::int32_t> tokens(bt * static_cast<std::size_t>(steps));
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (auto& t : tokens) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    t = static_cast<std::int32_t>((state >> 33) % static_cast<std::uint64_t>(config.vocab));
  }
  nc::DeviceAdamWConfig adam;
  adam.lr = mode == "small" ? 3e-3 : 6e-4;
  adam.beta1 = 0.9;
  adam.beta2 = 0.95;
  adam.eps = 1e-8;
  adam.weight_decay = 0.1;
  adam.max_grad_norm = 1.0;

  nc::DeviceBuffer d_tokens(bt * sizeof(std::int32_t), stream);
  std::vector<float> losses;
  std::vector<float> grads0;
  const auto start = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s) {
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(d_tokens.data(), tokens.data() + s * bt, bt * 4, cudaMemcpyHostToDevice,
                                     stream.handle()));
    model.forward_backward(d_tokens.as<std::int32_t>());
    losses.push_back(model.read_loss());
    if (s == 0) {
      grads0 = model.download_gradients();
    }
    model.optimizer_step(adam);
  }
  stream.synchronize();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("  info  nexus kayip: %.6f -> %.6f (%d adim, %.2f s)\n", losses.front(), losses.back(), steps, seconds);

  const std::filesystem::path dir = std::filesystem::temp_directory_path() / ("nexus_gpt_parity_" + mode);
  std::filesystem::create_directories(dir);
  {
    std::ofstream(dir / "weights.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(initial.data()), static_cast<std::streamsize>(initial.size() * 4));
    std::ofstream(dir / "tokens.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * 4));
    std::ofstream(dir / "nexus_grads0.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(grads0.data()), static_cast<std::streamsize>(grads0.size() * 4));
    std::ofstream m(dir / "manifest.json", std::ios::binary);
    char buf[64];
    m << "{\n  \"vocab\": " << config.vocab << ", \"max_seq\": " << config.max_seq << ", \"layers\": " << config.layers
      << ", \"heads\": " << config.heads << ", \"dim\": " << config.dim << ",\n  \"batch\": " << batch
      << ", \"time\": " << time << ", \"steps\": " << steps << ", \"ln_eps\": 1e-5,\n";
    m << "  \"adam\": {\"lr\": " << adam.lr << ", \"beta1\": " << adam.beta1 << ", \"beta2\": " << adam.beta2
      << ", \"eps\": " << adam.eps << ", \"weight_decay\": " << adam.weight_decay
      << ", \"max_grad_norm\": " << adam.max_grad_norm << "},\n  \"nexus_losses\": [";
    for (std::size_t i = 0; i < losses.size(); ++i) {
      std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(losses[i]));
      m << (i ? ", " : "") << buf;
    }
    m << "],\n  \"params\": [\n";
    const auto& info = model.parameter_info();
    for (std::size_t i = 0; i < info.size(); ++i) {
      m << "    {\"name\": \"" << info[i].name << "\", \"offset\": " << info[i].offset << ", \"numel\": " << info[i].numel
        << ", \"decay\": " << (info[i].decay ? "true" : "false") << ", \"shape\": [";
      for (std::size_t d = 0; d < info[i].shape.size(); ++d) {
        m << (d ? "," : "") << info[i].shape[d];
      }
      m << "]}" << (i + 1 < info.size() ? "," : "") << "\n";
    }
    m << "  ]\n}\n";
  }
  const std::string q = "\"";
  const int status = run_status(q + python + q + " -I " + q + NEXUS_GPT_REFERENCE + q + " " + q + dir.string() + q);
  if (status == kSkip) {
    std::printf("  SKIP  PyTorch kurulu degil\n");
    return kSkip;
  }
  std::printf("  %s  Nexus GPU GPT-2 vs PyTorch FP32 (%s, %d adim)\n", status == 0 ? "PASS" : "FAIL", mode.c_str(),
              steps);
  return status == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::printf("  FAIL  beklenmeyen istisna: %s\n", error.what());
    return 1;
  }
}
