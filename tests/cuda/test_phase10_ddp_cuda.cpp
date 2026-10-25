/**
 * @file test_phase10_ddp_cuda.cpp
 * @brief FAZ 10: GPU DDP (cuda::gpt_backend + tcp ProcessGroup), tek GPU'yu paylasan surecler.
 *
 *   nexus-run -n 2 -- test_phase10_ddp_cuda equiv   : dp=2 x micro 2 == dp=1 x micro 4 (FP32, TF32 kapali,
 *                                                     deterministik), 100 adim; rank 0 referansi da kosar
 *   nexus-run -n 2 -- test_phase10_ddp_cuda overlap : kova iletisimi geri gecis bitmeden baslar (CUDA olaylari)
 *   nexus-run -n N -- test_phase10_ddp_cuda step [STEPS] : adim suresi (kucuk GPT, B=8 T=256 rank basina)
 * Uyari: tum rank'ler AYNI GPU'yu paylasir — dogruluk ve ortusme kaniti icindir, hiz olceklemesi icin degil.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/lm_backend.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus/train/lm_trainer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace nt = nexus::train;
namespace nd = nexus::dist;

namespace {

constexpr int kSkip = 77;
int g_fail = 0;

void check(bool ok, int rank, const std::string& what) {
  std::printf("  %s  rank %d: %s\n", ok ? "PASS" : "FAIL", rank, what.c_str());
  g_fail += ok ? 0 : 1;
}

class GlobalRows final : public nt::TokenBatchSource {
 public:
  GlobalRows(int rank, int per_rank, int global, int vocab) : rank_(rank), per_rank_(per_rank), global_(global), vocab_(vocab) {}
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
    for (int j = 0; j < rows; ++j) {
      const std::uint64_t id = calls_ * static_cast<std::uint64_t>(global_) + static_cast<std::uint64_t>(rank_ * per_rank_ + j);
      std::uint32_t s = static_cast<std::uint32_t>(id * 2654435761u + 17u);
      for (int t = 0; t < time; ++t) {
        s = s * 1664525u + 1013904223u;
        dst[j * time + t] = static_cast<std::int32_t>((s >> 8) % static_cast<std::uint32_t>(vocab_));
      }
      if (ids) ids[j] = id;
    }
    ++calls_;
  }
  std::string state() const override { return std::to_string(calls_); }
  void restore(const std::string& v) override { calls_ = std::stoull(v); }

 private:
  int rank_, per_rank_, global_, vocab_;
  std::uint64_t calls_ = 0;
};

nc::GptConfig tiny() {
  nc::GptConfig c;
  c.vocab = 512;
  c.max_seq = 64;
  c.layers = 2;
  c.heads = 4;
  c.dim = 64;
  return c;
}

nc::GptModel& gpt(nt::LmBackend& b) { return dynamic_cast<nc::GptLmBackend&>(b).model(); }

std::unique_ptr<nt::LanguageModelTrainer> make(const nc::GptConfig& cfg, nt::Precision prec, nt::TokenBatchSource& src,
                                               int micro, int global_rows, int T, std::shared_ptr<nd::ProcessGroup> g,
                                               nd::DdpOptions opt, int steps,
                                               std::function<void(nt::LmBackend&, std::int64_t)> hook = {}) {
  nt::SpikeOptions no_spike;
  no_spike.enabled = false;
  nt::LanguageTrainerBuilder b(nc::gpt_backend(cfg));
  b.execution_policy(nexus::ExecutionPolicy::cuda(0))
      .precision(prec)
      .deterministic(true)
      .seed(11)
      .data(src)
      .optimizer(nt::AdamW{1.0e-3, 0.9, 0.95, 1.0e-8, 0.1})
      .schedule(nt::WarmupCosine{std::min(10, steps - 1), steps, 0.1})
      .micro_batch(micro)
      .seq_len(T)
      .global_batch_tokens(static_cast<std::int64_t>(global_rows) * T)
      .grad_clip(1.0)
      .spike_detector(no_spike)
      .log_console(false);
  if (hook) b.after_backward(std::move(hook));
  if (g) b.process_group(std::move(g), opt);
  return b.build();
}

int equiv(nd::GroupPtr g) {
  const nc::MatmulPrecision prev = nc::float32_matmul_precision();
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  const int dp = g->size();
  const int T = 32;
  const int steps = 100;
  const nc::GptConfig cfg = tiny();
  GlobalRows src(g->rank(), 2, 2 * dp, cfg.vocab);
  auto t = make(cfg, nt::Precision::F32, src, 2, 2 * dp, T, g, {}, steps);
  t->train(steps);
  std::vector<float> w = gpt(t->backend()).download_parameters();
  std::vector<float> mx = w, mn = w;
  g->all_reduce(mx.data(), mx.size(), nexus::DType::F32, nd::ReduceOp::Max)->wait();
  g->all_reduce(mn.data(), mn.size(), nexus::DType::F32, nd::ReduceOp::Min)->wait();
  check(std::memcmp(mx.data(), mn.data(), mx.size() * 4) == 0, g->rank(), "tum rank'lerin parametreleri bit bit ayni");
  if (g->rank() == 0) {
    // Referans: dp = 1, micro = 2 * dp, ayni satirlar. Oge basina en buyuk |gradyan| kaydedilir.
    std::vector<float> gmax;
    GlobalRows rsrc(0, 2 * dp, 2 * dp, cfg.vocab);
    auto r = make(cfg, nt::Precision::F32, rsrc, 2 * dp, 2 * dp, T, nullptr, {}, steps,
                  [&gmax](nt::LmBackend& b, std::int64_t) {
                    const std::vector<float> gr = gpt(b).download_gradients();
                    gmax.resize(gr.size(), 0.0f);
                    for (std::size_t i = 0; i < gr.size(); ++i) gmax[i] = std::max(gmax[i], std::fabs(gr[i]));
                  });
    r->train(steps);
    const std::vector<float> ref = gpt(r->backend()).download_parameters();
    float top = 0.0f;
    for (float v : gmax) top = std::max(top, v);
    double worst = 0.0;
    double noise_worst = 0.0;
    std::size_t noise = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      const double d = std::fabs(static_cast<double>(w[i]) - ref[i]);
      if (gmax[i] < 1.0e-6f * top) {  // gercek gradyani sifir (anahtar bias'i) ya da dolgu: yalniz gurultu
        noise += 1;
        noise_worst = std::max(noise_worst, d);
      } else {
        worst = std::max(worst, d);
      }
    }
    double lmax = 0.0;
    for (std::size_t i = 0; i < t->history().size(); ++i) lmax = std::max(lmax, std::fabs(t->history()[i].loss - r->history()[i].loss));
    std::printf("  info  dp=%d, %zu parametre; gurultu ogesi %zu (max fark %.2e)\n", dp, ref.size(), noise, noise_worst);
    std::printf("  info  agirlik max fark (100 adim) %.3e, kayip max fark %.3e, kayip %.4f -> %.4f\n", worst, lmax,
                r->history().front().loss, r->history().back().loss);
    check(worst <= 1.0e-6, 0, "dp=" + std::to_string(dp) + " x micro 2 == dp=1 x micro " + std::to_string(2 * dp) +
                                  ": 100 adim sonra agirlik farki <= 1e-6");
    check(lmax <= 1.0e-5, 0, "adim kayiplari ayni (<= 1e-5)");
  }
  nc::set_float32_matmul_precision(prev);
  return g_fail;
}

int overlap(nd::GroupPtr g, int steps) {
  nc::GptConfig cfg;
  cfg.vocab = 8192;
  cfg.max_seq = 256;
  cfg.layers = 6;
  cfg.heads = 6;
  cfg.dim = 384;
  GlobalRows src(g->rank(), 4, 4 * g->size(), cfg.vocab);
  nd::DdpOptions opt;
  opt.bucket_bytes = 4 << 20;
  auto t = make(cfg, nt::Precision::BF16, src, 4, 4 * g->size(), 256, g, opt, steps);
  t->train(steps);
  if (g->rank() == 0) {
    const nd::DdpStepStats& s = t->history().back().ddp;
    double ms = 0.0;
    for (std::size_t i = 2; i < t->history().size(); ++i) ms += t->history()[i].step_ms;
    std::printf("  info  %d rank (ayni GPU), %d kova, geri gecis %.2f ms, ilk iletisim %.2f ms, son bitis %.2f ms, "
                "mesgul %.2f ms, ortusen %.2f ms, adim %.1f ms\n",
                g->size(), s.buckets, s.backward_ms, s.comm_first_start_ms, s.comm_last_end_ms, s.comm_busy_ms,
                s.comm_overlapped_ms, ms / static_cast<double>(t->history().size() - 2));
    check(s.buckets > 1 && s.comm_first_start_ms >= 0.0 && s.comm_first_start_ms < s.backward_ms, 0,
          "ilk kova iletisimi geri gecis bitmeden basladi (CUDA olay zamanlari)");
    check(s.comm_overlapped_ms > 0.0, 0, "iletisimin bir kismi geri gecisle ortusuyor");
  }
  return g_fail;
}

int step_time(nd::GroupPtr g, int steps) {
  nc::GptConfig cfg;
  cfg.vocab = 8192;
  cfg.max_seq = 256;
  cfg.layers = 6;
  cfg.heads = 6;
  cfg.dim = 384;
  const int dp = g ? g->size() : 1;
  const int rank = g ? g->rank() : 0;
  GlobalRows src(rank, 8, 8 * dp, cfg.vocab);
  auto t = make(cfg, nt::Precision::BF16, src, 8, 8 * dp, 256, dp > 1 ? g : nullptr, {}, steps + 3);
  t->train(steps + 3);
  double ms = 0.0;
  for (std::size_t i = 3; i < t->history().size(); ++i) ms += t->history()[i].step_ms;
  ms /= steps;
  if (rank == 0) {
    const nd::DdpStepStats& s = t->history().back().ddp;
    std::printf("{\"framework\":\"nexus\",\"ranks\":%d,\"params\":%lld,\"rows_per_rank\":8,\"T\":256,\"ms_per_step\":%.2f,"
                "\"tokens_per_s\":%.0f,\"buckets\":%d,\"comm_busy_ms\":%.2f,\"overlapped_ms\":%.2f}\n",
                dp, static_cast<long long>(t->backend().shape().params), ms, t->tokens_per_step() / (ms / 1e3), s.buckets,
                s.comm_busy_ms, s.comm_overlapped_ms);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return kSkip;
  }
  const std::string mode = argc > 1 ? argv[1] : "equiv";
  try {
    nd::GroupPtr g = nd::init_process_group("tcp");
    if (mode == "equiv") return equiv(g) == 0 ? 0 : 1;
    if (mode == "overlap") return overlap(g, argc > 2 ? std::stoi(argv[2]) : 6) == 0 ? 0 : 1;
    if (mode == "step") return step_time(g, argc > 2 ? std::stoi(argv[2]) : 20);
    std::printf("bilinmeyen mod: %s\n", mode.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
}
