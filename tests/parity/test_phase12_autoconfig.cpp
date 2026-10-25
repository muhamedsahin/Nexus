/**
 * @file test_phase12_autoconfig.cpp
 * @brief FAZ 12.G: otomatik paralellik onericisi (nexus_estimate auto_parallel) — tutarlilik denetimleri.
 *
 * Tahmin modelinin kendisi olcum degildir; burada yalniz ic tutarlilik ve kisitlar denetlenir:
 * FAZ 0 bellek formuluyle ayni statik bellek, SP/TP/CP'nin bellegi azaltmasi, siralama, dugum ici TP,
 * sigmayan durumda acik "sigmiyor" notu.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "autoparallel.hpp"
#include "estimate.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstdio>

namespace ne = nexus::estimate;

namespace {

ne::ModelShape llama(std::int64_t layers, std::int64_t d, std::int64_t heads, std::int64_t kv, std::int64_t ff, std::int64_t seq) {
  ne::ModelShape s;
  s.n_layer = layers;
  s.d_model = d;
  s.n_head = heads;
  s.n_kv_head = kv;
  s.d_ff = ff;
  s.vocab = 32000;
  s.seq_len = seq;
  s.micro_batch = 1;
  s.learned_pos = 0;
  s.tie_head = false;
  s.ffn_swiglu = true;
  s.bias = false;
  s.rmsnorm = true;
  return s;
}

ne::DTypePlan mixed() {
  ne::DTypePlan d;
  d.param_bytes = 2;
  d.grad_bytes = 4;
  d.optim_bytes = 8;
  d.master_bytes = 4;
  d.activation_elem_bytes = 2;
  return d;
}

void print(const char* title, const ne::AutoResult& r) {
  std::printf("  info  %s: %lld denendi, %lld sigdi\n", title, static_cast<long long>(r.evaluated), static_cast<long long>(r.feasible));
  int i = 1;
  for (const ne::Candidate& c : r.best) {
    std::printf("  info    #%d dp=%d tp=%d pp=%d cp=%d zero=%d mb=%d recompute=%d bellek %.1f GiB adim %.2f s MFU %.1f%%\n", i++,
                c.dp, c.tp, c.pp, c.cp, c.zero, c.micro_batch, c.recompute ? 1 : 0, c.memory_gib, c.step_s, 100.0 * c.mfu);
  }
  if (!r.note.empty()) std::printf("  info    not: %s\n", r.note.c_str());
}

}  // namespace

int main() {
  const ne::DTypePlan dt = mixed();
  const ne::ModelShape m7 = llama(32, 4096, 32, 32, 11008, 2048);
  ne::Cluster dgx;  // 8 x 80 GB, 312 TFLOPS (A100 BF16)
  ne::AutoOptions opt;
  opt.global_batch_tokens = 1 << 22;

  // FAZ 0 ile ayni statik bellek (tp=pp=cp=1, zero 0)
  {
    ne::Plan p;
    const ne::MemoryEstimate f0 = ne::estimate_memory(m7, dt, p);
    const ne::Candidate c = ne::evaluate_plan(m7, dt, dgx, opt, 8, 1, 1, 1, 0, 1, false);
    const double f0_gib = static_cast<double>(f0.static_sharded) / (1024.0 * 1024.0 * 1024.0);
    std::printf("  info  statik bellek: FAZ 0 %.3f GiB, onerici %.3f GiB\n", f0_gib, c.static_gib);
    nexus_test::expect(std::fabs(f0_gib - c.static_gib) < 1e-6, "statik bellek FAZ 0 estimate_memory ile ayni (tp=pp=cp=1, zero 0)");
  }
  // bellek monotonlugu
  {
    const ne::Candidate t1 = ne::evaluate_plan(m7, dt, dgx, opt, 8, 1, 1, 1, 0, 2, false);
    const ne::Candidate t2 = ne::evaluate_plan(m7, dt, dgx, opt, 4, 2, 1, 1, 0, 2, false);
    const ne::Candidate c2 = ne::evaluate_plan(m7, dt, dgx, opt, 4, 1, 1, 2, 0, 2, false);
    const ne::Candidate rc = ne::evaluate_plan(m7, dt, dgx, opt, 8, 1, 1, 1, 0, 2, true);
    ne::AutoOptions nosp = opt;
    nosp.sequence_parallel = false;
    const ne::Candidate t2n = ne::evaluate_plan(m7, dt, dgx, nosp, 4, 2, 1, 1, 0, 2, false);
    nexus_test::expect(t2.memory_gib < t1.memory_gib, "tp=2 bellegi tp=1'den az");
    nexus_test::expect(t2.activation_gib < t2n.activation_gib, "SP aktivasyon bellegi SP'siz TP'den az");
    nexus_test::expect(c2.activation_gib < t1.activation_gib, "cp=2 aktivasyon bellegi azaltir");
    nexus_test::expect(rc.activation_gib < t1.activation_gib && rc.compute_s > t1.compute_s, "recompute: az bellek, fazla hesap");
  }
  // 7B, 8 x 80 GB
  {
    const ne::AutoResult r = ne::auto_parallel(m7, dt, dgx, opt);
    print("7B @ 8 x 80 GB", r);
    bool sorted = true, fits = true;
    for (std::size_t i = 0; i < r.best.size(); ++i) {
      fits = fits && r.best[i].fits && r.best[i].memory_gib <= dgx.mem_gib;
      if (i > 0) sorted = sorted && r.best[i - 1].step_s <= r.best[i].step_s;
    }
    nexus_test::expect(r.best.size() == 3 && sorted && fits, "7B @ 8x80GB: en iyi 3 oneri, sigan ve adim suresine gore sirali");
  }
  // 70B, 64 x 80 GB (8 dugum): TP dugum disina tasmamali
  {
    const ne::ModelShape m70 = llama(80, 8192, 64, 8, 28672, 4096);
    ne::Cluster c = dgx;
    c.gpus = 64;
    const ne::AutoResult r = ne::auto_parallel(m70, dt, c, opt);
    print("70B @ 64 x 80 GB", r);
    bool intra = !r.best.empty();
    for (const ne::Candidate& x : r.best) intra = intra && x.tp <= c.gpus_per_node;
    nexus_test::expect(intra, "70B @ 64x80GB: oneriler var ve TP dugum ici (<= 8)");
  }
  // 7B, 1 x 8 GB (bu makine): sigmaz -> acik not
  {
    ne::Cluster laptop;
    laptop.gpus = 1;
    laptop.gpus_per_node = 1;
    laptop.mem_gib = 8.0;
    laptop.peak_tflops = 43.0;
    const ne::AutoResult r = ne::auto_parallel(m7, dt, laptop, opt);
    print("7B @ 1 x 8 GB", r);
    nexus_test::expect(r.best.empty() && !r.note.empty(), "7B @ 1x8GB: sigan yok, acik not (sessiz oneri yok)");
  }
  return nexus_test::finish("test_phase12_autoconfig");
}
