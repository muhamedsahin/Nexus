/**
 * @file test_phase13_compile_cuda.cpp
 * @brief FAZ 13 (GPU): trainer.compile — GEMM dolgusu + tum adim CUDA Graph; NVRTC fuzyon cekirdekleri.
 *
 *   test_phase13_compile_cuda graph  : test_cuda_graph_replay_equal — ayni dolgulu eager ile 12 adim kayip, norm ve tum
 *                                      parametreler BIT BIT ayni (biriktirme 3, lr zamanlamasi degisken); dolgusuz eager ile
 *                                      kayip farki kucuk (farkli GEMM sekli)
 *   test_phase13_compile_cuda nvrtc  : IR fuzyon kumeleri -> CUDA kaynagi -> NVRTC -> cubin (disk onbellegi) -> calisma;
 *                                      CPU JIT ile ayni sonuc; ikinci derleme NVRTC cagirmaz
 *   test_phase13_compile_cuda bench  : gpt2s token/s (eager, +dolgu, compile, max-autotune), adim basina host suresi,
 *                                      yakalama suresi; kucuk yigin (host sinirli) durumu
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/codegen.hpp"
#include "nexus/compile/compile.hpp"
#include "nexus/compile/passes.hpp"
#include "nexus/compile/trace.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/lm_backend.hpp"
#include "nexus/cuda/nvrtc_jit.hpp"
#include "nexus/lm/model.hpp"
#include "nexus/train/lm_trainer.hpp"
#include "nexus_test.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace nt = nexus::train;
namespace ir = nexus::ir;
namespace fs = std::filesystem;

namespace {

std::vector<std::int32_t> lcg_tokens(std::size_t n, std::uint32_t seed, int vocab) {
  std::vector<std::int32_t> v(n);
  std::uint32_t s = seed;
  for (std::int32_t& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<std::int32_t>((s >> 8) % static_cast<std::uint32_t>(vocab));
  }
  return v;
}

class CountingSource final : public nt::TokenBatchSource {
 public:
  explicit CountingSource(int vocab) : vocab_(vocab) {}
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
    for (int r = 0; r < rows; ++r) {
      const std::uint64_t id = index_++;
      const auto row = lcg_tokens(static_cast<std::size_t>(time), static_cast<std::uint32_t>(id * 2654435761u + 17u), vocab_);
      std::memcpy(dst + static_cast<std::size_t>(r) * time, row.data(), row.size() * 4);
      if (ids != nullptr) ids[r] = id;
    }
  }
  std::string state() const override { return std::to_string(index_); }
  void restore(const std::string& s) override { index_ = std::stoull(s); }

 private:
  int vocab_;
  std::uint64_t index_ = 0;
};

nc::GptConfig tiny() {
  nc::GptConfig c;
  c.vocab = 509;  // 64'un kati degil: dolgu devrede
  c.max_seq = 64;
  c.layers = 2;
  c.heads = 4;
  c.dim = 64;
  return c;
}

nc::GptConfig gpt2s(int T) {
  nc::GptConfig c;  // 124M, V 50257
  c.max_seq = std::max(1024, T);
  return c;
}

struct Run {
  std::vector<double> losses, norms;
  std::vector<float> params;
  std::size_t graph_nodes = 0;
  std::int64_t replays = 0;
  double tok_s = 0.0, mfu = 0.0, host_ms = 0.0, first_ms = 0.0, second_ms = 0.0;
  double peak_gb = 0.0;
};

Run train(nc::GptConfig config, bool compile, const char* mode, int B, int T, int accum, int steps, bool timing,
          nt::RecomputePolicy recompute = nt::RecomputePolicy::Selective) {
  CountingSource src(config.vocab);
  nt::SpikeOptions ns;
  ns.enabled = false;
  nt::LanguageTrainerBuilder b(nc::gpt_backend(config));
  b.execution_policy(nexus::ExecutionPolicy::cuda(0))
      .precision(nt::Precision::BF16)
      .recompute(recompute)
      .deterministic(!timing)
      .data(src)
      .optimizer(nt::AdamW{6.0e-4, 0.9, 0.95, 1.0e-8, 0.1})
      .schedule(nt::WarmupCosine{3, steps + 10, 0.1})
      .micro_batch(B)
      .seq_len(T)
      .global_batch_tokens(static_cast<std::int64_t>(B) * T * accum)
      .grad_clip(1.0)
      .spike_detector(ns)
      .log_console(false);
  if (compile) b.compile(true, mode);
  nexus::cuda::device_allocator().reset_peak_stats();
  auto t = b.build();
  Run r;
  auto& be = dynamic_cast<nc::GptLmBackend&>(t->backend());
  auto t0 = std::chrono::steady_clock::now();
  (void)t->train(1);  // isinma (eager)
  r.first_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  t0 = std::chrono::steady_clock::now();
  (void)t->train(2);  // compile'da yakalama + ilk yeniden oynatma
  r.second_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  (void)t->train(steps);
  for (const auto& h : t->history()) {
    r.losses.push_back(h.loss);
    r.norms.push_back(h.grad_norm);
  }
  if (timing) {
    // ayni adimlardan sonuncular: ortalama token/s ve host suresi (adim - GPU senkron beklemesi yerine dogrudan olcum)
    double tps = 0.0, mfu = 0.0;
    int n = 0;
    for (std::size_t i = 3; i < t->history().size(); ++i) {
      tps += t->history()[i].tokens_per_s;
      mfu += t->history()[i].mfu;
      ++n;
    }
    r.tok_s = n ? tps / n : 0.0;
    r.mfu = n ? mfu / n : 0.0;
    // host suresi: bir adimin micro_step + optimizer_step cagrilari (GPU beklemeden)
    std::vector<std::int32_t> tok(static_cast<std::size_t>(B) * T);
    src.next(tok.data(), B, T, nullptr);
    double host = 0.0;
    const int reps = 5;
    for (int k = 0; k < reps; ++k) {
      (void)be.finish_step();
      const auto h0 = std::chrono::steady_clock::now();
      for (int i = 0; i < accum; ++i) be.micro_step(tok.data(), 1.0f / accum, i, accum);
      be.optimizer_step(nt::AdamW{6.0e-4, 0.9, 0.95, 1.0e-8, 0.1}, 1e-4, 1.0);
      host += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    }
    (void)be.finish_step();
    r.host_ms = host / reps;
  }
  r.params = be.model().download_parameters();
  r.graph_nodes = be.graph_nodes();
  r.replays = be.graph_replays();
  r.peak_gb = nexus::cuda::device_allocator().stats(0).peak_reserved_bytes / 1e9;
  return r;
}

void test_graph() {
  nc::GptConfig padded = tiny();
  padded.vocab_pad_multiple = 64;
  padded.fuse_ce_cast = true;  // compile ile ayni cekirdekler (graph disinda)
  padded.fast_ce = true;
  const auto none = nt::RecomputePolicy::None;
  const Run eager = train(padded, false, "default", 4, 64, 3, 12, false, none);
  const Run graph = train(tiny(), true, "default", 4, 64, 3, 12, false, none);  // compile dolguyu kendisi acar
  const Run plain = train(tiny(), false, "default", 4, 64, 3, 12, false, none);
  bool same = eager.losses == graph.losses && eager.norms == graph.norms && eager.params == graph.params;
  double dl = 0.0;
  for (std::size_t i = 0; i < plain.losses.size(); ++i) dl = std::max(dl, std::fabs(plain.losses[i] - graph.losses[i]));
  std::printf("  info  CUDA Graph: %zu dugum, %lld yeniden oynatma; 12 adim kayip %.5f -> %.5f; dolgulu eager ile %s; "
              "dolgusuz eager ile en buyuk kayip farki %.2e\n",
              graph.graph_nodes, static_cast<long long>(graph.replays), graph.losses.front(), graph.losses.back(),
              same ? "BIT BIT ayni" : "FARKLI", dl);
  nexus_test::expect(graph.graph_nodes > 50 && graph.replays >= 11, "tum adim tek CUDA Graph olarak yakalandi ve yeniden oynatildi");
  nexus_test::expect(same, "test_cuda_graph_replay_equal: yeniden oynatma == ayni cekirdeklerle eager (kayip, norm, parametre)");
  nexus_test::expect(dl < 2e-3, "GEMM dolgusu kaybi degistirmez (dolgusuz eager ile kucuk fark)");
}

void test_nvrtc() {
  // Yalniz SDPA desenli GPT benzeri ileri+geri grafik: eleman eleman kumeler (bias, gelu, geri zincirleri).
  nexus::lm::LmConfig c = nexus::lm::preset("gpt2-124m");
  c.n_layer = 2;
  c.d_model = 64;
  c.n_head = 4;
  c.d_ff = 128;
  c.vocab_size = 101;
  c.max_position = 64;
  const ir::TracedLm tr = ir::trace_lm(c);
  const ir::Joint j = ir::build_joint(ir::rewrite_patterns(tr.graph, nullptr, 1024, ir::kPatAttention));
  ir::FusionStats fs;
  const ir::Graph fused = ir::fuse_elementwise(j.graph, &fs);
  std::vector<const ir::FusedOp*> ks;
  std::vector<std::string> names;
  for (const ir::Node& n : fused.nodes) {
    if (n.op != ir::Op::Fused) continue;
    const std::string nm = ir::kernel_name(*n.fused);
    if (std::find(names.begin(), names.end(), nm) != names.end()) continue;
    ks.push_back(n.fused.get());
    names.push_back(nm);
  }
  const fs::path dir = fs::temp_directory_path() / "nexus_faz13_nvrtc";
  fs::remove_all(dir);
  nc::NvrtcJit jit(dir.string());
  const auto t0 = std::chrono::steady_clock::now();
  jit.load(ir::cuda_source(ks), names);
  const double cold = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  nc::NvrtcJit jit2(dir.string());
  const auto t1 = std::chrono::steady_clock::now();
  (void)jit2.load(ir::cuda_source(ks), names);
  const double warm = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
  // CPU JIT ile karsilastir: her kume rastgele girdilerle
  ir::CpuJit cpu((dir / "cpu").string());
  std::vector<std::string> cnames;
  for (const auto& n : names) {
    cnames.push_back(n);
    cnames.push_back(n + "_s");
  }
  auto cmod = cpu.load(ir::cpu_source(ks), cnames);
  double worst = 0.0;
  const long long rows = 37, cols = 128;
  for (std::size_t k = 0; k < ks.size(); ++k) {
    const ir::FusedOp& f = *ks[k];
    std::vector<std::vector<float>> in(f.input_is_row.size()), outc(f.outputs.size()), outg(f.outputs.size());
    std::uint32_t s = 7u + static_cast<std::uint32_t>(k);
    for (std::size_t e = 0; e < in.size(); ++e) {
      in[e].resize(static_cast<std::size_t>(f.input_is_row[e] ? cols : rows * cols));
      for (float& x : in[e]) {
        s = s * 1664525u + 1013904223u;
        x = static_cast<float>((s >> 8) % 2001) / 1000.0f - 1.0f;
      }
    }
    std::vector<const float*> ip;
    for (auto& v : in) ip.push_back(v.data());
    std::vector<float*> op;
    for (auto& v : outc) {
      v.assign(static_cast<std::size_t>(rows * cols), 0.0f);
      op.push_back(v.data());
    }
    cmod.at(names[k])(ip.data(), op.data(), rows, cols);
    for (auto& v : outg) v.assign(static_cast<std::size_t>(rows * cols), 0.0f);
    jit.run(names[k], in, outg, rows, cols);
    for (std::size_t o = 0; o < outc.size(); ++o) {
      for (std::size_t e = 0; e < outc[o].size(); ++e) {
        worst = std::max(worst, static_cast<double>(std::fabs(outc[o][e] - outg[o][e])) / std::max(1.0f, std::fabs(outc[o][e])));
      }
    }
  }
  std::printf("  info  NVRTC: %zu kume (%d dugum) -> %zu cekirdek; soguk %.0f ms (%d nvrtc), sicak %.1f ms (%d nvrtc, %d isabet); "
              "CPU JIT'e gore en buyuk goreli fark %.1e\n",
              ks.size(), fs.fused_nodes, names.size(), cold, jit.stats().compiler_invocations, warm, jit2.stats().compiler_invocations,
              jit2.stats().cache_hits, worst);
  nexus_test::expect(!ks.empty() && worst < 1e-5, "NVRTC fuzyon cekirdekleri == CPU JIT (AVX2/SVML) cekirdekleri");
  nexus_test::expect(jit.stats().compiler_invocations == 1 && jit2.stats().compiler_invocations == 0 && jit2.stats().cache_hits >= 1,
                     "cubin disk onbellegi: ikinci yukleme NVRTC cagirmaz");
}

int bench(int B, int T, int accum, int steps, const std::string& only) {
  struct Case {
    const char* name;
    bool pad, compile;
    const char* mode;
  };
  for (const Case cs : {Case{"eager", false, false, "default"}, Case{"eager+dolgu", true, false, "default"},
                        Case{"compile", false, true, "default"}, Case{"compile max-autotune", false, true, "max-autotune"}}) {
    if (!only.empty() && only != cs.name) continue;  // olcumler ayri sureclerde (ayirici onbellegi tasmasin)
    nc::GptConfig c = gpt2s(T);
    if (cs.pad) c.vocab_pad_multiple = 64;
    nexus::cuda::clear_gemm_cache();
    nexus::cuda::set_gemm_autotune(false);
    const Run r = train(c, cs.compile, cs.mode, B, T, accum, steps, true);
    nexus::cuda::device_allocator().empty_cache();
    std::printf("{\"mode\": \"bench_gpu\", \"case\": \"%s\", \"B\": %d, \"T\": %d, \"accum\": %d, \"tokens_per_s\": %.0f, \"mfu\": %.4f, "
                "\"host_ms_per_step\": %.3f, \"first_step_ms\": %.0f, \"capture_plus_step_ms\": %.0f, \"graph_nodes\": %zu, "
                "\"peak_gb\": %.2f}\n",
                cs.name, B, T, accum, r.tok_s, r.mfu, r.host_ms, r.first_ms, r.second_ms, r.graph_nodes, r.peak_gb);
  }
  nexus::cuda::set_gemm_autotune(false);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "graph";
  try {
    if (mode == "graph") test_graph();
    else if (mode == "nvrtc") test_nvrtc();
    else if (mode == "bench" && argc > 5)
      return bench(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), argc > 6 ? argv[6] : "");
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase13_compile_cuda");
}
