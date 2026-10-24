/** FAZ 7 benchmark: GPT-2 350M/1.5B x {none, selective, full} x {AdamW32, AdamW8, ZeRO-Offload}
 *  tepe bellek + adim suresi (beklenen sutun: nexus_estimate), ve 8 GB'ta 1.5B egitim kosusu. */
#include "estimate.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/gpt.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#endif
namespace nc = nexus::cuda;
namespace ne = nexus::estimate;
using nexus::DType;
namespace {

int g_time = 1024;  // train kipinde argv[3]

struct Shape {
  const char* name;
  int layers, dim, heads;
};

// Ogrenilebilir Markov verisi (her token 4 ardil, .50/.25/.15/.10; taban 1.208 nat).
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
  std::vector<std::int32_t> batch(int t) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(t));
    std::int32_t x = static_cast<std::int32_t>(next() % static_cast<std::uint64_t>(vocab));
    for (int j = 0; j < t; ++j) {
      out[static_cast<std::size_t>(j)] = x;
      const std::uint64_t u = next() % 100;
      x = succ[static_cast<std::size_t>(x)][u < 50 ? 0 : (u < 75 ? 1 : (u < 90 ? 2 : 3))];
    }
    return out;
  }
};

nc::GptConfig config(const Shape& sh, nc::Recompute rc, nc::OptimizerKind opt) {
  nc::GptConfig c;
  if (opt == nc::OptimizerKind::AdamW8bitFused) {
    c.master = nc::MasterWeights::NoMaster;  // fused: BF16 + stokastik yuvarlama
    c.adam8_embedding_fp32 = false;
  }
  c.layers = sh.layers;
  c.dim = sh.dim;
  c.heads = sh.heads;
  c.compute_dtype = DType::BF16;
  c.recompute = rc;
  c.optimizer = opt;
  c.ce_chunk_rows = 256;
  c.offload_grad_dtype = DType::BF16;
  return c;
}

// Bos fiziksel host bellegi (GiB). ZeroOffload'un host durumu buna sigmazsa sayfalama
// olcumu anlamsizlastirir (ilk denemede 1.3B: 53 s/adim, ardindan surec dustu).
double available_host_gib() {
#if defined(_WIN32)
  MEMORYSTATUSEX m{};
  m.dwLength = sizeof(m);
  if (GlobalMemoryStatusEx(&m) != 0) {
    return static_cast<double>(std::min(m.ullAvailPhys, m.ullAvailPageFile)) / (1ull << 30);
  }
#endif
  return 1e9;
}

std::int64_t gpt_params(const Shape& sh) {
  ne::ModelShape m;
  m.n_layer = sh.layers;
  m.d_model = sh.dim;
  m.n_head = sh.heads;
  m.n_kv_head = sh.heads;
  m.d_ff = 4 * sh.dim;
  m.vocab = 50257;
  m.seq_len = 1024;
  m.learned_pos = 1024;
  return ne::count_parameters(m).total;
}

// nexus_estimate: aygit basina beklenen bayt (param BF16 golge + master/grad/optim plana gore + aktivasyon).
double expected_gib(const Shape& sh, nc::Recompute rc, nc::OptimizerKind opt) {
  ne::ModelShape m;
  m.n_layer = sh.layers;
  m.d_model = sh.dim;
  m.n_head = sh.heads;
  m.n_kv_head = sh.heads;
  m.d_ff = 4 * sh.dim;
  m.vocab = 50257;
  m.seq_len = 1024;
  m.learned_pos = 1024;
  ne::DTypePlan d;
  d.param_bytes = 2;
  d.activation_elem_bytes = 2;
  if (opt == nc::OptimizerKind::AdamW) {
    d.grad_bytes = 4;
    d.optim_bytes = 8;
    d.master_bytes = 4;
  } else if (opt == nc::OptimizerKind::AdamW8bit) {
    d.grad_bytes = 4;
    d.optim_bytes = 2;
    d.master_bytes = 4;
  } else if (opt == nc::OptimizerKind::AdamW8bitFused) {
    d.grad_bytes = 0;  // tek katmanlik tampon
    d.optim_bytes = 2;
    d.master_bytes = 0;
  } else {
    d.grad_bytes = 0;  // katman katman host'a akar
    d.optim_bytes = 0;
    d.master_bytes = 0;
  }
  ne::Plan p;
  p.full_recompute = rc == nc::Recompute::Full;
  const ne::MemoryEstimate e = ne::estimate_memory(m, d, p);
  return static_cast<double>(e.static_sharded + e.activation_bytes) / (1ull << 30);
}

void table(std::ostringstream& out, const nc::Stream& stream) {
  auto& pool = nc::caching_device_allocator();
  const Shape shapes[] = {{"gpt2-350m", 24, 1024, 16}, {"gpt3-1.3b", 24, 2048, 16}, {"gpt2-1.5b", 48, 1600, 25}};
  out << "  \"table\":[\n";
  bool first = true;
  for (const Shape& sh : shapes) {
    for (const nc::OptimizerKind opt : {nc::OptimizerKind::AdamW, nc::OptimizerKind::AdamW8bit, nc::OptimizerKind::ZeroOffload,
                                        nc::OptimizerKind::AdamW8bitFused}) {
      for (const nc::Recompute rc : {nc::Recompute::None, nc::Recompute::Selective, nc::Recompute::Full}) {
        const double expect = expected_gib(sh, rc, opt);
        pool.empty_cache();
        pool.reset_peak_stats();
        std::string status = "ok";
        double ms = 0.0, cpu_ms = 0.0, peak = 0.0;
        try {
          if (expect > 9.0) {
            throw std::runtime_error("beklenen > 9 GiB: denenmedi");
          }
          if (opt == nc::OptimizerKind::AdamW8bitFused && rc == nc::Recompute::None) {
            throw std::runtime_error("denenmedi (fused yalniz recompute ile anlamli)");
          }
          if (opt == nc::OptimizerKind::ZeroOffload) {
            const double need = static_cast<double>(gpt_params(sh)) * 14.0 / (1ull << 30);  // master+m+v FP32 + BF16 pinned
            const double have = available_host_gib();
            if (need > 0.9 * have) {
              char buf[160];
              std::snprintf(buf, sizeof(buf), "host bellegi yetersiz: gerek %.1f GiB, bos %.1f GiB (denenmedi)", need, have);
              throw std::runtime_error(buf);
            }
          }
          nc::GptModel model(config(sh, rc, opt), 1, 1024, 1, stream);
          // WDDM 8 GB'i asan ayirmayi sistem bellegine tasir (FAZ 5: T=4096 math 10 GB "calisti",
          // 24 s/adim). Bu yuzden ayrilan > 7.3 GiB ise "sigmaz" sayilir ve olculmez.
          if (pool.stats(0).reserved_bytes > (73ull << 30) / 10) {
            throw std::runtime_error("aygit ayirmasi 7.3 GiB'i asti (WDDM tasmasi)");
          }
          Markov data(50257);
          nc::DeviceBuffer d(1024 * 4, stream);
          nc::DeviceAdamWConfig adam;
          adam.lr = 1e-4;
          adam.weight_decay = 0.1;
          adam.max_grad_norm = 1.0;
          adam.beta2 = 0.95;
          if (opt == nc::OptimizerKind::AdamW8bitFused) {
            model.set_fused_optimizer(adam);
          }
          const int warm = 1, reps = opt == nc::OptimizerKind::ZeroOffload ? 2 : 3;
          std::vector<double> t;
          for (int s = 0; s < warm + reps; ++s) {
            const auto tok = data.batch(1024);
            NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, stream.handle()));
            stream.synchronize();
            const auto a = std::chrono::steady_clock::now();
            model.forward_backward(d.as<std::int32_t>());
            model.optimizer_step(adam);
            stream.synchronize();
            if (s >= warm) {
              t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
              cpu_ms += model.last_cpu_update_ms() / reps;
            }
          }
          std::sort(t.begin(), t.end());
          ms = t[t.size() / 2];
          peak = static_cast<double>(pool.stats(0).peak_allocated_bytes) / (1ull << 30);
          if (peak > 7.3) {
            status = "WDDM tasmasi (peak > 7.3 GiB)";
          }
        } catch (const std::exception& e) {
          status = e.what();
          std::replace(status.begin(), status.end(), '"', '\'');
          std::replace(status.begin(), status.end(), '\n', ' ');
        }
        pool.empty_cache();
        std::printf("  info  %-10s %-9s %-12s beklenen=%5.2f GiB  tepe=%5.2f GiB  %9.1f ms/adim (CPU %7.1f ms)  %s\n", sh.name,
                    nc::to_string(rc), nc::to_string(opt), expect, peak, ms, cpu_ms, status.c_str());
        out << (first ? "" : ",\n") << "    {\"model\":\"" << sh.name << "\",\"recompute\":\"" << nc::to_string(rc)
            << "\",\"optimizer\":\"" << nc::to_string(opt) << "\",\"expected_gib\":" << expect << ",\"peak_gib\":" << peak
            << ",\"ms_per_step\":" << ms << ",\"cpu_update_ms\":" << cpu_ms << ",\"status\":\"" << status << "\"}";
        first = false;
      }
    }
  }
  out << "\n  ],\n";
}

// Kabul: >= 1.3B parametre 8 GB'ta BF16 + full recompute + ZeRO-Offload ile egitilir, kayip duser.
void train_big(std::ostringstream& out, const nc::Stream& stream, const Shape& sh, nc::OptimizerKind opt, int steps) {
  auto& pool = nc::caching_device_allocator();
  pool.empty_cache();
  pool.reset_peak_stats();
  nc::GptConfig c = config(sh, nc::Recompute::Full, opt);
  const auto build0 = std::chrono::steady_clock::now();
  nc::GptModel model(c, 1, g_time, 7, stream);
  const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - build0).count();
  std::printf("  info  %s / %s: %zu parametre, aygit durumu %.2f GiB, host durumu %.2f GiB, aktivasyon %.2f GiB, kurulum %.0f s\n",
              sh.name, nc::to_string(opt), model.num_parameters(), model.device_state_bytes() / 1073741824.0, model.host_state_bytes() / 1073741824.0,
              model.activation_bytes() / 1073741824.0, build_s);
  std::size_t free_b = 0, total_b = 0;
  NEXUS_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  std::printf("  info  ayirici: ayrilan %.2f GiB, rezerve %.2f GiB | cudaMemGetInfo bos %.2f / %.2f GiB\n",
              pool.stats(0).allocated_bytes / 1073741824.0, pool.stats(0).reserved_bytes / 1073741824.0,
              free_b / 1073741824.0, total_b / 1073741824.0);
  // Veri: sozlugun ilk 1024 token'i uzerinde Markov zinciri (taban 1.208 nat). 50257 tokenli zincir
  // 60 adimda (61k token) ogrenilemez; kayip ln(50257) civarinda kalir (ilk deneme).
  Markov data(1024);
  nc::DeviceBuffer d(static_cast<std::size_t>(g_time) * 4, stream);
  nc::DeviceAdamWConfig adam;
  adam.lr = 3e-4;
  adam.weight_decay = 0.1;
  adam.max_grad_norm = 1.0;
  adam.beta2 = 0.95;
  if (opt == nc::OptimizerKind::AdamW8bitFused) {
    model.set_fused_optimizer(adam);
  }
  std::vector<float> losses;
  std::vector<double> gpu_ms, cpu_ms, total_ms;
  for (int s = 0; s < steps; ++s) {
    const auto tok = data.batch(g_time);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, stream.handle()));
    stream.synchronize();
    const auto a = std::chrono::steady_clock::now();
    model.forward_backward(d.as<std::int32_t>());
    const float loss = model.read_loss();  // senkron: ileri+geri biter
    const auto b = std::chrono::steady_clock::now();
    model.optimizer_step(adam);
    stream.synchronize();
    const auto e = std::chrono::steady_clock::now();
    losses.push_back(loss);
    gpu_ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    cpu_ms.push_back(model.last_cpu_update_ms());
    total_ms.push_back(std::chrono::duration<double, std::milli>(e - a).count());
    if (s % 5 == 0 || s + 1 == steps) {
      std::printf("  info  adim %3d kayip %.4f | ileri+geri %.0f ms, CPU AdamW %.0f ms, toplam %.0f ms\n", s, loss,
                  gpu_ms.back(), cpu_ms.back(), total_ms.back());
    }
  }
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double peak = static_cast<double>(pool.stats(0).peak_allocated_bytes) / (1ull << 30);
  // MFU/HFU: model FLOP = token * (6N + 12 L d s) (PaLM Ek B); tam yeniden hesaplama +1 ileri = 8N + 16 L d s.
  const double N = static_cast<double>(model.num_parameters());
  const double attn = 12.0 * sh.layers * sh.dim * static_cast<double>(g_time);
  const double model_flops = static_cast<double>(g_time) * (6.0 * N + attn);
  const double hw_flops = static_cast<double>(g_time) * (8.0 * N + attn * 4.0 / 3.0);
  const double peak_tf = 40.0 * 512.0 * 2.1e9;  // 40 SM x 512 FLOP/saat x 2100 MHz (FAZ 4)
  const double t = median(total_ms) * 1e-3;
  const double tg = median(gpu_ms) * 1e-3;
  std::printf("  info  tepe aygit bellegi %.2f GiB | medyan adim %.0f ms (ileri+geri %.0f, CPU AdamW %.0f) | %.0f token/s\n",
              peak, median(total_ms), median(gpu_ms), median(cpu_ms), g_time / t);
  std::printf("  info  MFU %.1f%% (adim) / %.1f%% (yalniz GPU) | HFU %.1f%% (yalniz GPU)\n", 100.0 * model_flops / t / peak_tf,
              100.0 * model_flops / tg / peak_tf, 100.0 * hw_flops / tg / peak_tf);
  double first10 = 0.0, last10 = 0.0;
  for (int i = 0; i < 10; ++i) {
    first10 += losses[static_cast<std::size_t>(i)] / 10.0;
    last10 += losses[losses.size() - 1 - static_cast<std::size_t>(i)] / 10.0;
  }
  std::printf("  info  kayip: ilk 10 ort. %.4f -> son 10 ort. %.4f (%s)\n", first10, last10, last10 < first10 ? "duser" : "DUSMEZ");
  out << "  \"train_" << sh.name << "_" << nc::to_string(opt) << "\":{\"time\":" << g_time << ",\"params\":" << model.num_parameters() << ",\"steps\":" << steps << ",\"peak_gib\":" << peak
      << ",\"device_state_gib\":" << model.device_state_bytes() / 1073741824.0
      << ",\"host_state_gib\":" << model.host_state_bytes() / 1073741824.0 << ",\"median_step_ms\":" << median(total_ms)
      << ",\"median_gpu_ms\":" << median(gpu_ms) << ",\"median_cpu_ms\":" << median(cpu_ms)
      << ",\"mfu_step\":" << model_flops / t / peak_tf << ",\"mfu_gpu\":" << model_flops / tg / peak_tf
      << ",\"hfu_gpu\":" << hw_flops / tg / peak_tf << ",\"loss_first10\":" << first10 << ",\"loss_last10\":" << last10
      << ",\"losses\":[";
  for (std::size_t i = 0; i < losses.size(); ++i) {
    out << (i ? "," : "") << losses[i];
  }
  out << "]},\n";
}
// Selective'in sure maliyeti: none ve selective modelleri ayni anda kurulur, adimlar sirayla
// birbirine karistirilarak (saat/isi kaymasi her ikisini esit etkiler) olculur.
void selective_cost(std::ostringstream& out, const nc::Stream& stream, int reps) {
  // Modeller tek tek kurulur (ayni anda uc model 8 GB'i asar); turlar halinde sirayla olculur,
  // boylece paylasilan GPU yuku ve saat kaymasi uc modu esit etkiler.
  const Shape sh{"gpt2-124m", 12, 768, 12};
  const nc::Recompute modes[] = {nc::Recompute::None, nc::Recompute::Selective, nc::Recompute::Full};
  std::vector<double> times[3];
  std::size_t act[3] = {0, 0, 0};
  Markov data(1024);
  nc::DeviceBuffer d(1024 * 4, stream);
  for (int round = 0; round < 3; ++round) {
    for (int k = 0; k < 3; ++k) {
      nc::GptModel m(config(sh, modes[k], nc::OptimizerKind::AdamW8bit), 1, 1024, 1, stream);
      act[k] = m.activation_bytes();
      for (int i = 0; i < 3 + reps; ++i) {
        const auto tok = data.batch(1024);
        NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), tok.data(), tok.size() * 4, cudaMemcpyHostToDevice, stream.handle()));
        stream.synchronize();
        nc::Event a, b;
        a.record(stream);
        m.forward_backward(d.as<std::int32_t>());
        b.record(stream);
        b.synchronize();
        if (i >= 3) {
          times[k].push_back(a.elapsed_ms(b));
        }
      }
      nc::caching_device_allocator().empty_cache();
    }
  }
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double n = med(times[0]), se = med(times[1]), f = med(times[2]);
  std::printf("  info  gpt2-124m ileri+geri (CUDA olay, 3 tur x %d): none %.1f ms, selective %.1f ms (%+.1f%%), full %.1f ms (%+.1f%%)\n",
              reps, n, se, 100.0 * (se / n - 1.0), f, 100.0 * (f / n - 1.0));
  std::printf("  info  aktivasyon: none %.0f MiB, selective %.0f MiB, full %.0f MiB\n", act[0] / 1048576.0,
              act[1] / 1048576.0, act[2] / 1048576.0);
  out << "  \"selective_cost\":{\"model\":\"gpt2-124m\",\"none_ms\":" << n << ",\"selective_ms\":" << se
      << ",\"full_ms\":" << f << ",\"none_act\":" << act[0] << ",\"selective_act\":" << act[1]
      << ",\"full_act\":" << act[2] << "},\n";
}
}  // namespace
int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) { std::printf("  SKIP  CUDA aygiti yok\n"); return 77; }
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    nc::Stream stream = nc::Stream::create(0); nc::StreamGuard guard(stream);
    nc::set_float32_matmul_precision(nc::MatmulPrecision::High);
    cudaDeviceProp prop{}; NEXUS_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::ostringstream out; out.precision(7);
    out << "{\n  \"phase\":7,\"device\":\"" << prop.name << "\",\"batch\":1,\"time\":1024,\"dtype\":\"bf16\",\"ce_chunk_rows\":256,\n";
    const int steps = argc > 2 ? std::atoi(argv[2]) : 60;
    if (argc > 3) {
      g_time = std::atoi(argv[3]);
    }
    if (mode == "all" || mode == "train") {
      train_big(out, stream, {"gpt3-1.3b", 24, 2048, 16}, nc::OptimizerKind::AdamW8bitFused, steps);
    }
    if (mode == "selective") {
      selective_cost(out, stream, steps);
    }
    if (mode == "train15") {
      train_big(out, stream, {"gpt2-1.5b", 48, 1600, 25}, nc::OptimizerKind::AdamW8bitFused, steps);
    }
    if (mode == "train_offload") {
      train_big(out, stream, {"gpt2-1.5b", 48, 1600, 25}, nc::OptimizerKind::ZeroOffload, steps);
    }
    if (mode == "all" || mode == "table") table(out, stream);
    out << "  \"notes\":\"PyTorch CUDA, bitsandbytes ve DeepSpeed kurulu degil: karsilastirma yok.\"\n}\n";
    {
      const std::filesystem::path dir(NEXUS_FAZ7_DOCS_DIR); std::filesystem::create_directories(dir);
      std::ofstream(dir / ("bench_cuda_phase7_" + mode + ".json"), std::ios::binary) << out.str();
      std::printf("  OK    FAZ 7 benchmark JSON yazildi (%s)\n", mode.c_str());
    }
    stream.destroy(); return 0;
  } catch (const std::exception& e) { std::printf("  FAIL  %s\n", e.what()); return 1; }
}
