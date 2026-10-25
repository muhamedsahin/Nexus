/**
 * @file test_phase11_fsdp.cpp
 * @brief FAZ 11: ZeRO-0/1/2/3, FSDP, HSDP, onceden getirme, NVMe takasi, parcali kayit (CPU, thread + tcp).
 *
 *   test_phase11_fsdp stages      : ZeRO-0/1/2/3 (dp=4) 100 adim, agirliklar ve m/v BIT BIT ayni; DDP ile de
 *   test_phase11_fsdp memory      : dp=1,2,4,8 rank basina kalici bellek (ZeRO-1/2/3), ZeRO-3 ~ 1/dp
 *   test_phase11_fsdp prefetch    : kaydedilen sira, AG(k+1) birim k hesabindan once, sinirlayici <= N
 *   test_phase11_fsdp nvme        : NVMe gidis-donus (hizali, asenkron) + ZeRO-3 NVMe takasli egitim == RAM'de
 *   test_phase11_fsdp reshard     : dp=4'te parcali kayit -> dp=2'de yukleme: tam durum bit bit ayni
 *   test_phase11_fsdp hybrid      : HSDP (2 dugum x 2) == FULL_SHARD dp=4 (agirlik) ; BF16 iletisim kaybi azalir
 *   nexus-run -n 4 -- test_phase11_fsdp stages_tcp : ZeRO-3 surecler arasi == ZeRO-0
 *   test_phase11_fsdp bench_mem N ZERO : rank basina kalici bellek ve adim suresi (rapor)
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/fsdp.hpp"
#include "nexus/dist/nvme.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/train/lm_trainer.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nd = nexus::dist;
namespace nt = nexus::train;
using nexus::DType;

namespace {

constexpr int kVocab = 11;
std::mutex g_mu;
std::atomic<int> g_fail{0};

void check(bool ok, int rank, const std::string& what) {
  if (!ok) {
    g_fail.fetch_add(1);
    std::lock_guard<std::mutex> lock(g_mu);
    std::printf("  FAIL  rank %d: %s\n", rank, what.c_str());
  }
}

nexus::nn::Sequential make_model(int blocks, int embed) {
  nexus::nn::Sequential m;
  m.add(std::make_shared<nexus::nn::Embedding>(kVocab, embed, -1));
  m.add(std::make_shared<nexus::nn::PositionalEncoding>(embed, 64, false));
  for (int i = 0; i < blocks; ++i) m.add(std::make_shared<nexus::nn::CausalTransformerBlock>(embed, 2, 2 * embed));
  m.add(std::make_shared<nexus::nn::Linear>(static_cast<std::size_t>(embed), static_cast<std::size_t>(kVocab), true));
  return m;
}

void copy_weights(const nexus::nn::Module& from, nexus::nn::Module& to) {
  const auto a = from.parameters();
  const auto b = to.parameters();
  for (std::size_t i = 0; i < a.size(); ++i) std::memcpy(b[i]->data.data(), a[i]->data.data(), a[i]->data.numel() * 4);
}

class GlobalRows final : public nt::TokenBatchSource {
 public:
  GlobalRows(int rank, int per_rank, int global) : rank_(rank), per_rank_(per_rank), global_(global) {}
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
    for (int j = 0; j < rows; ++j) {
      const std::uint64_t id = calls_ * static_cast<std::uint64_t>(global_) + static_cast<std::uint64_t>(rank_ * per_rank_ + j);
      std::uint32_t s = static_cast<std::uint32_t>(id * 2654435761u + 17u);
      for (int t = 0; t < time; ++t) {
        s = s * 1664525u + 1013904223u;
        dst[j * time + t] = static_cast<std::int32_t>((s >> 8) % kVocab);
      }
      if (ids) ids[j] = id;
    }
    ++calls_;
  }
  std::string state() const override { return std::to_string(calls_); }
  void restore(const std::string& v) override { calls_ = std::stoull(v); }

 private:
  int rank_, per_rank_, global_;
  std::uint64_t calls_ = 0;
};

struct Run {
  int zero = 3;  // -1: DDP
  nd::FsdpOptions fsdp{};
  int blocks = 1;
  int embed = 16;
  int per_rank = 2;
  int T = 12;
  int steps = 100;
  int accum = 1;
};

struct Result {
  std::vector<float> w, m, v;
  std::vector<double> losses;
  nd::FsdpMemory mem;
  double mean_step_ms = 0.0;
  std::vector<std::string> trace;
  std::vector<int> order;
  int max_inflight = 0;
  std::vector<unsigned char> shard;
  std::size_t units = 0;
};

std::unique_ptr<nt::LanguageModelTrainer> make_trainer(nexus::nn::Sequential& model, nt::TokenBatchSource& src,
                                                       const Run& r, nd::GroupPtr g) {
  nt::SpikeOptions no_spike;
  no_spike.enabled = false;
  nt::LanguageTrainerBuilder b(model);
  b.data(src)
      .optimizer(nt::AdamW{1.0e-3, 0.9, 0.95, 1.0e-8, 0.1})
      .schedule(nt::WarmupCosine{std::min(10, r.steps - 1), r.steps, 0.1})
      .micro_batch(r.per_rank)
      .seq_len(r.T)
      .global_batch_tokens(static_cast<std::int64_t>(r.per_rank) * r.T * r.accum * g->size())
      .grad_clip(1.0)
      .spike_detector(no_spike)
      .log_console(false);
  nt::LanguageTrainerBuilder::Parallel p;
  p.zero = r.zero;
  p.fsdp = r.fsdp;
  if (r.zero == 4) {  // HSDP
    p.zero = 3;
    p.fsdp.sharding = nd::ShardingStrategy::Hybrid;
    b.fsdp(g, p.fsdp);
  } else {
    b.parallel(g, p);
  }
  return b.build();
}

/// rank'te calisir; rank 0 sonucu doldurur.
void rank_run(nd::ProcessGroup& g, const nexus::nn::Sequential& init, const Run& r, Result* out) {
  nexus::nn::Sequential model = make_model(r.blocks, r.embed);
  if (g.rank() == 0) copy_weights(init, model);
  GlobalRows src(g.rank(), r.per_rank * r.accum, r.per_rank * r.accum * g.size());
  nd::GroupPtr sp(&g, [](nd::ProcessGroup*) {});
  auto t = make_trainer(model, src, r, sp);
  t->train(r.steps);
  nd::FsdpEngine* e = t->backend().fsdp();
  Result res;
  if (e != nullptr) {
    res.w = e->full_parameters();
    res.m = e->full_optimizer_state(0);
    res.v = e->full_optimizer_state(1);
    res.mem = e->persistent_memory();
    res.trace = e->trace();
    res.order = e->recorded_order();
    res.max_inflight = e->max_inflight();
    res.shard = e->save_shard();
    res.units = e->units().size();
  } else {
    for (auto* p : model.parameters()) res.w.insert(res.w.end(), p->data.data(), p->data.data() + p->data.numel());
  }
  double ms = 0.0;
  for (const auto& h : t->history()) {
    res.losses.push_back(h.loss);
    ms += h.step_ms;
  }
  res.mean_step_ms = ms / static_cast<double>(t->history().size());
  if (out != nullptr) *out = std::move(res);
}

std::vector<Result> run_world(int n, const nexus::nn::Sequential& init, const Run& r) {
  nd::ThreadWorld world(n);
  std::vector<Result> res(static_cast<std::size_t>(n));
  std::vector<std::thread> ts;
  for (int k = 0; k < n; ++k) {
    ts.emplace_back([&, k] {
      try {
        nd::GroupPtr g = world.create_group(k);
        rank_run(*g, init, r, &res[static_cast<std::size_t>(k)]);
        g->synchronize();
      } catch (const std::exception& e) {
        check(false, k, std::string("istisna: ") + e.what());
      }
    });
  }
  for (auto& t : ts) t.join();
  return res;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  double w = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) w = std::max(w, std::fabs(static_cast<double>(a[i]) - b[i]));
  return w;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// ------------------------------------------------------------------------------------------
int test_stages() {
  nexus_test::section("test_zero_stages_equivalence: dp=4, ZeRO-0/1/2/3, 100 adim, FP32");
  const nexus::nn::Sequential init = make_model(1, 16);
  Run r;
  std::vector<Result> base;
  for (int z = 0; z <= 3; ++z) {
    r.zero = z;
    std::vector<Result> got = run_world(4, init, r);
    std::printf("  info  %-24s kayip %.5f -> %.5f, rank 0 kalici %.1f KB (parametre %.1f, gradyan %.1f, opt %.1f)\n",
                nd::to_string(nd::zero_stage(z)), got[0].losses.front(), got[0].losses.back(), got[0].mem.total() / 1e3,
                got[0].mem.param_bytes / 1e3, got[0].mem.grad_bytes / 1e3, got[0].mem.optimizer_bytes / 1e3);
    if (z == 0) {
      base = got;
      continue;
    }
    const double dw = max_diff(got[0].w, base[0].w);
    nexus_test::info(std::string("zero") + std::to_string(z) + "_agirlik_max_fark", dw);
    nexus_test::expect(same_bits(got[0].w, base[0].w) && same_bits(got[0].m, base[0].m) && same_bits(got[0].v, base[0].v),
                       "ZeRO-" + std::to_string(z) + " == ZeRO-0: agirlik, m, v bit bit ayni (100 adim)");
    bool losses = got[0].losses == base[0].losses;
    nexus_test::expect(losses, "ZeRO-" + std::to_string(z) + " adim kayiplari ayni");
  }
  for (int k = 1; k < 4; ++k) {
    nexus_test::expect(same_bits(base[static_cast<std::size_t>(k)].w, base[0].w), "rank'ler arasi tam parametre ayni");
  }
  // DDP (FAZ 10) ile: farkli indirgeme deseni (halka all-reduce) -> bit degil, <= 1e-6 (gurultu parametresi haric degil:
  // burada ZeRO-0 da RS+AG kullandigi icin anahtar bias'i farki gorulur; raporlanir).
  r.zero = -1;
  std::vector<Result> ddp = run_world(4, init, r);
  nexus_test::info("ddp_ile_max_fark_(anahtar_biasi_dahil)", max_diff(ddp[0].w, base[0].w));
  double lmax = 0.0;
  for (std::size_t i = 0; i < ddp[0].losses.size(); ++i) lmax = std::max(lmax, std::fabs(ddp[0].losses[i] - base[0].losses[i]));
  nexus_test::info("ddp_ile_kayip_max_fark", lmax);
  nexus_test::expect(lmax <= 1e-5 && g_fail.load() == 0, "FSDP ZeRO-0 ile DDP kayiplari ayni (<= 1e-5)");
  return 0;
}

int test_memory() {
  nexus_test::section("test_fsdp_memory_scaling: rank basina kalici bellek, dp=1,2,4,8");
  const nexus::nn::Sequential init = make_model(4, 64);
  Run r;
  r.blocks = 4;
  r.embed = 64;
  r.steps = 3;
  for (int z = 1; z <= 3; ++z) {
    r.zero = z;
    double base = 0.0;
    for (const int n : {1, 2, 4, 8}) {
      std::vector<Result> got = run_world(n, init, r);
      const double total = static_cast<double>(got[0].mem.total());
      if (n == 1) base = total;
      std::printf("  info  ZeRO-%d dp=%d: %.1f KB (parametre %.1f, gradyan %.1f, opt %.1f) = dp1'in %.3f'u\n", z, n,
                  total / 1e3, got[0].mem.param_bytes / 1e3, got[0].mem.grad_bytes / 1e3, got[0].mem.optimizer_bytes / 1e3,
                  total / base);
      if (z == 3 && n > 1) {
        nexus_test::expect(std::fabs(total / base - 1.0 / n) < 0.02, "ZeRO-3 dp=" + std::to_string(n) + ": kalici bellek ~ 1/dp");
      }
    }
  }
  return 0;
}

int test_prefetch() {
  nexus_test::section("test_fsdp_prefetch_order: ZeRO-3 dp=2, 4 blok, limit 2");
  const nexus::nn::Sequential init = make_model(4, 32);
  Run r;
  r.zero = 3;
  r.blocks = 4;
  r.embed = 32;
  r.steps = 2;
  r.fsdp.limit_all_gathers = 2;
  std::vector<Result> got = run_world(2, init, r);
  const Result& a = got[0];
  std::string order;
  for (int u : a.order) order += std::to_string(u) + " ";
  std::printf("  info  kaydedilen ileri sira: %s(%zu birim), en cok ucusta AG: %d\n", order.c_str(), a.units, a.max_inflight);
  bool ascending = a.order.size() == a.units;
  for (std::size_t i = 0; i < a.order.size(); ++i) ascending = ascending && a.order[i] == static_cast<int>(i);
  nexus_test::expect(ascending, "ileri yurutme sirasi kaydedildi (birim 0..n-1)");
  // 2. adim: ikinci "F <ilk birim>" olayindan geriye, o adimin ilk AG'sine kadar olan bolum.
  auto find = [&](const std::string& ev, std::size_t from) {
    for (std::size_t i = from; i < a.trace.size(); ++i)
      if (a.trace[i] == ev) return i;
    return a.trace.size();
  };
  std::vector<std::string> names;  // birim adlari ileri sirayla (ilk adimin F olaylarindan)
  for (const std::string& ev : a.trace) {
    if (ev.rfind("F ", 0) == 0) {
      if (!names.empty() && ev.substr(2) == names.front()) break;
      names.push_back(ev.substr(2));
    }
  }
  const std::size_t first_f = find("F " + names.front(), 0);
  const std::size_t second_f = find("F " + names.front(), first_f + 1);
  std::size_t step2 = second_f;  // adim 2'nin ilk AG'si (ilk birim icin)
  while (step2 > 0 && a.trace[step2 - 1].rfind("AG ", 0) == 0) --step2;
  bool prefetched = second_f < a.trace.size() && names.size() == a.units;
  for (std::size_t k = 0; prefetched && k + 1 < names.size(); ++k) {
    const std::size_t f = find("F " + names[k], step2);
    const std::size_t ag = find("AG " + names[k + 1], step2);
    prefetched = ag < f;
  }
  nexus_test::expect(prefetched, "adim 2'de her birim k icin AG(k+1), birim k'nin ileri hesabindan once verildi (ortusme)");
  // geri gecis (adim 2): B(k) baslamadan once AG(k-1) verilmis olmali.
  const std::size_t bstart = find("B " + names.back(), second_f);
  bool back = bstart < a.trace.size();
  for (std::size_t k = names.size() - 1; back && k > 0; --k) {
    const std::size_t b = find("B " + names[k], second_f);
    std::size_t ag = a.trace.size();
    for (std::size_t i = find("F " + names.back(), second_f); i < b; ++i)
      if (a.trace[i] == "AG " + names[k - 1]) ag = i;
    back = ag < b;
  }
  nexus_test::expect(a.max_inflight == 2, "sinirlayici: ucusta en cok 2 all-gather");
  nexus_test::expect(back, "geri geciste her birim k icin AG(k-1), B(k)'den once verildi");
  std::printf("  info  adim 2 izi: ");
  for (std::size_t i = step2; i < a.trace.size(); ++i) std::printf("%s | ", a.trace[i].c_str());
  std::printf("\n");
  return 0;
}

int test_nvme() {
  nexus_test::section("test_nvme_offload_roundtrip");
  const std::string dir = (std::filesystem::temp_directory_path() / "nexus_faz11_nvme").string();
  {
    nd::NvmeStore store(dir);
    for (const std::size_t bytes : {std::size_t{1}, std::size_t{4096}, std::size_t{10'000'001}}) {
      nd::AlignedBuffer w(bytes), rd(bytes);
      for (std::size_t i = 0; i < w.capacity(); ++i) w.as<unsigned char>()[i] = static_cast<unsigned char>((i * 131 + bytes) & 0xFF);
      const auto t0 = std::chrono::steady_clock::now();
      store.write("k" + std::to_string(bytes), w)->wait();
      store.read("k" + std::to_string(bytes), rd)->wait();
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      nexus_test::expect(std::memcmp(w.data(), rd.data(), w.capacity()) == 0, "NVMe gidis-donus " + std::to_string(bytes) + " bayt");
      if (bytes > 1'000'000) nexus_test::info("yaz+oku_GB_s", 2.0 * w.capacity() / s / 1e9);
    }
  }
  const nexus::nn::Sequential init = make_model(2, 32);
  Run r;
  r.zero = 3;
  r.blocks = 2;
  r.embed = 32;
  r.steps = 20;
  std::vector<Result> ram = run_world(2, init, r);
  r.fsdp.offload.optimizer_nvme = true;
  r.fsdp.offload.params_nvme = true;
  r.fsdp.offload.directory = dir;
  std::vector<Result> disk = run_world(2, init, r);
  std::printf("  info  RAM: kalici %.1f KB | NVMe takasli: RAM %.1f KB (opt %.1f) + NVMe %.1f KB, adim %.2f ms -> %.2f ms\n",
              ram[0].mem.total() / 1e3, disk[0].mem.total() / 1e3, disk[0].mem.optimizer_bytes / 1e3, disk[0].mem.nvme_bytes / 1e3,
              ram[0].mean_step_ms, disk[0].mean_step_ms);
  nexus_test::expect(same_bits(ram[0].w, disk[0].w) && same_bits(ram[0].m, disk[0].m), "NVMe takasli ZeRO-3 == RAM'de (bit bit, 20 adim)");
  nexus_test::expect(disk[0].mem.optimizer_bytes == 0 && disk[0].mem.nvme_bytes > 0, "optimizer durumu ve parametre parcasi RAM'de degil");
  std::filesystem::remove_all(dir);
  return 0;
}

int test_reshard() {
  nexus_test::section("test_sharded_checkpoint_reshard: dp=4 kayit -> dp=2 yukleme");
  const nexus::nn::Sequential init = make_model(2, 32);
  Run r;
  r.zero = 3;
  r.blocks = 2;
  r.embed = 32;
  r.steps = 20;
  std::vector<Result> four = run_world(4, init, r);
  std::vector<std::vector<unsigned char>> blobs;
  for (const Result& x : four) blobs.push_back(x.shard);
  // dp = 2 dunyada bos motor kur ve 4 parcadan yeniden parcala.
  nd::ThreadWorld world(2);
  std::vector<std::vector<float>> w(2), m(2);
  std::vector<std::thread> ts;
  for (int k = 0; k < 2; ++k) {
    ts.emplace_back([&, k] {
      try {
        nd::GroupPtr g = world.create_group(k);
        nexus::nn::Sequential model = make_model(2, 32);
        nd::FsdpOptions o;
        nd::FsdpEngine e(model, g, o);
        e.initialize();
        e.load_resharded(blobs);
        w[static_cast<std::size_t>(k)] = e.full_parameters();
        m[static_cast<std::size_t>(k)] = e.full_optimizer_state(0);
        // yeniden parcalanan durumla bir ileri gecis calisir
        nexus_model::Tensor in = nexus_model::Tensor::zeros({2, 12});
        (void)e.forward(in);
      } catch (const std::exception& ex) {
        check(false, k, ex.what());
      }
    });
  }
  for (auto& t : ts) t.join();
  nexus_test::expect(g_fail.load() == 0, "yeniden parcalama hatasiz");
  nexus_test::expect(same_bits(w[0], four[0].w) && same_bits(w[1], four[0].w), "dp=2'de tam parametreler dp=4'tekiyle bit bit ayni");
  nexus_test::expect(same_bits(m[0], four[0].m), "dp=2'de tam m durumu ayni");
  return 0;
}

int test_hybrid() {
  nexus_test::section("HSDP (2 dugum x 2 rank) ve BF16 iletisim");
  const nexus::nn::Sequential init = make_model(1, 16);
  Run r;
  r.steps = 50;
  r.zero = 3;
  std::vector<Result> full = run_world(4, init, r);
  r.zero = 4;  // hybrid
  r.fsdp.ranks_per_node = 2;
  std::vector<Result> hyb = run_world(4, init, r);
  const double dw = max_diff(hyb[0].w, full[0].w);
  double lmax = 0.0;
  for (std::size_t i = 0; i < full[0].losses.size(); ++i) lmax = std::max(lmax, std::fabs(hyb[0].losses[i] - full[0].losses[i]));
  nexus_test::info("hsdp_fullshard_agirlik_max_fark", dw);
  nexus_test::info("hsdp_fullshard_kayip_max_fark", lmax);
  std::printf("  info  HSDP rank 0 kalici %.1f KB (dugum ici parca), FULL_SHARD %.1f KB\n", hyb[0].mem.total() / 1e3, full[0].mem.total() / 1e3);
  nexus_test::expect(lmax <= 1e-5, "HSDP kayiplari FULL_SHARD ile ayni (<= 1e-5)");
  nexus_test::expect(hyb[0].mem.total() > full[0].mem.total(), "HSDP dugum ici parcalar: kalici bellek FULL_SHARD'dan buyuk");
  r.zero = 3;
  r.fsdp = nd::FsdpOptions{};
  r.fsdp.mixed_precision.param = DType::BF16;
  r.fsdp.mixed_precision.reduce = DType::BF16;
  std::vector<Result> bf = run_world(4, init, r);
  nexus_test::info("bf16_kayip_ilk", bf[0].losses.front());
  nexus_test::info("bf16_kayip_son", bf[0].losses.back());
  nexus_test::info("fp32_kayip_son", full[0].losses.back());
  nexus_test::expect(bf[0].losses.back() < bf[0].losses.front() && std::fabs(bf[0].losses.back() - full[0].losses.back()) < 0.05,
                     "BF16 all-gather/reduce-scatter ile kayip duser, FP32'den < 0.05 farkli");
  return 0;
}

int stages_tcp() {
  nd::GroupPtr g = nd::init_process_group("tcp");
  const nexus::nn::Sequential init = make_model(1, 16);
  Run r;
  r.zero = 3;
  r.steps = 50;
  Result z3;
  rank_run(*g, init, r, g->rank() == 0 ? &z3 : nullptr);
  r.zero = 0;
  Result z0;
  rank_run(*g, init, r, g->rank() == 0 ? &z0 : nullptr);
  if (g->rank() == 0) {
    const bool ok = same_bits(z3.w, z0.w) && z3.losses == z0.losses;
    std::printf("  %s  ZeRO-3 == ZeRO-0 (%d surec, tcp, 50 adim, bit bit)\n", ok ? "PASS" : "FAIL", g->size());
    return ok ? 0 : 1;
  }
  return 0;
}

int bench_mem(int n, int zero) {
  const nexus::nn::Sequential init = make_model(6, 256);
  Run r;
  r.zero = zero;
  r.blocks = 6;
  r.embed = 256;
  r.per_rank = 4;
  r.T = 64;
  r.steps = 6;
  std::vector<Result> got = run_world(n, init, r);
  std::printf("{\"dp\":%d,\"zero\":%d,\"params\":%zu,\"persistent_mb\":%.3f,\"param_mb\":%.3f,\"grad_mb\":%.3f,\"opt_mb\":%.3f,"
              "\"mean_step_ms\":%.2f}\n",
              n, zero, got[0].w.size(), got[0].mem.total() / 1e6, got[0].mem.param_bytes / 1e6, got[0].mem.grad_bytes / 1e6,
              got[0].mem.optimizer_bytes / 1e6, got[0].mean_step_ms);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "stages";
  try {
    if (mode == "stages") test_stages();
    else if (mode == "memory") test_memory();
    else if (mode == "prefetch") test_prefetch();
    else if (mode == "nvme") test_nvme();
    else if (mode == "reshard") test_reshard();
    else if (mode == "hybrid") test_hybrid();
    else if (mode == "stages_tcp") return stages_tcp();
    else if (mode == "bench_mem" && argc > 3) return bench_mem(std::stoi(argv[2]), std::stoi(argv[3]));
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  nexus_test::expect(g_fail.load() == 0, "rank ici denetimler");
  return nexus_test::finish("test_phase11_fsdp");
}
