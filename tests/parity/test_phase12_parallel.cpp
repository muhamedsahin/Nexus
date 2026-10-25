/**
 * @file test_phase12_parallel.cpp
 * @brief FAZ 12: DeviceMesh/DTensor, TP (+ sozluk-parcali CE), SP, async TP, PP zamanlamalari, CP (ring/Ulysses),
 *        ZeRO-1, 3B/4B birlesik esdegerlik (CPU, thread arka ucu).
 *
 *   test_phase12_parallel dtensor : test_dtensor_redistribute_all_pairs (1B mesh 4, 2B mesh 2x2) + op yayilimi
 *   test_phase12_parallel tp      : test_tp_equivalence — tp=1 (== lm::LanguageModel otograd) vs tp=2/4, SP, async TP
 *   test_phase12_parallel vocab   : test_vocab_parallel_ce_equals_full — dolgulu sozluk, tp=1/2/4 kayip + gomme gradyani
 *   test_phase12_parallel pp      : test_pp_schedules_equivalence — GPipe/1F1B/interleaved/ZB-H1 ayni gradyan (bit bit)
 *   test_phase12_parallel ring    : test_ring_attention_equals_flash — ring (ardisik/zigzag) + Ulysses, ileri/geri
 *   test_phase12_parallel zero    : ZeRO-1 == ZeRO-0 (bit bit, 5 adim)
 *   test_phase12_parallel 3d      : 12.H (dp=2, tp=2, pp=2) 8 rank, 3 adim == tek rank
 *   test_phase12_parallel 4d      : test_4d_equivalence (dp=2, pp=2, cp=2, tp=2) 16 rank, 3 adim == tek rank
 *   test_phase12_parallel trainer : 12.8 API trainer.parallel({dp, tp, pp, sp, pp_schedule}) == tek rank egitmen
 *   nexus-run -n 4 -- test_phase12_parallel tcp : dp=2, tp=2+SP+async surecler arasi == tek rank
 *   test_phase12_parallel bench_tp TP SP ASYNC STEPS | bench_pp SCHED V M | bench_cp T CP ALG LAYOUT : olcum (JSON)
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/attention/cpu.hpp"
#include "nexus/autograd/autograd.hpp"
#include "nexus/dist/context_parallel.hpp"
#include "nexus/dist/dtensor.hpp"
#include "nexus/dist/parallel_lm.hpp"
#include "nexus/dist/pipeline.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus/lm/model.hpp"
#include "nexus/train/lm_trainer.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nd = nexus::dist;
namespace ag = nexus::autograd;
namespace lm = nexus::lm;
namespace nt = nexus::train;
using nexus::DType;

namespace {

std::mutex g_mu;
std::atomic<int> g_fail{0};

void check(bool ok, const std::string& what) {
  if (!ok) {
    g_fail.fetch_add(1);
    std::lock_guard<std::mutex> lock(g_mu);
    std::printf("  FAIL  %s\n", what.c_str());
  }
}

/// n rank'lik thread dunyasi; fn(rank, world) her rank'te.
void run_world(int n, const std::function<void(int, nd::GroupPtr)>& fn) {
  nd::ThreadWorld world(n);
  std::vector<std::thread> threads;
  for (int r = 0; r < n; ++r) {
    threads.emplace_back([&, r] {
      try {
        nd::GroupOptions o;
        o.timeout = std::chrono::seconds(120);
        nd::GroupPtr g = world.create_group(r, o);
        fn(r, g);
      } catch (const std::exception& e) {
        check(false, "rank " + std::to_string(r) + " istisna: " + e.what());
      }
    });
  }
  for (auto& t : threads) t.join();
}

// ==========================================================================================
// 12.A DTensor
// ==========================================================================================

std::vector<nd::Placement> all_placements(int ndim_tensor) {
  std::vector<nd::Placement> p{nd::Placement::replicate(), nd::Placement::partial()};
  for (int d = 0; d < ndim_tensor; ++d) p.push_back(nd::Placement::shard(d));
  return p;
}

bool valid(const nd::Placements& p) {
  std::vector<int> seen;
  for (const auto& x : p) {
    if (x.is_shard()) {
      if (std::find(seen.begin(), seen.end(), x.dim) != seen.end()) return false;
      seen.push_back(x.dim);
    }
  }
  return true;
}

/// Tam degerden beklenen yerel parca (Partial: koordinat 0 tam, digerleri 0 — ama toplama gore test edilir).
std::vector<float> expected_local(const std::vector<float>& full, const std::vector<std::int64_t>& shape, nd::DeviceMesh& mesh,
                                  const nd::Placements& p) {
  return nd::DTensor::distribute(full, shape, mesh, p).local();
}

void test_dtensor() {
  const std::vector<std::int64_t> shape{6, 10};  // 4'e esit bolunmez (6 -> 2,2,2,0; 10 -> 3,3,3,1)
  std::vector<float> full(60);
  for (int i = 0; i < 60; ++i) full[static_cast<std::size_t>(i)] = static_cast<float>((i * 7) % 23 - 11);  // tamsayi: toplam tam
  for (int ndim : {1, 2}) {
    const int n = 4;
    std::atomic<int> pairs{0}, ok_pairs{0};
    run_world(n, [&](int rank, nd::GroupPtr world) {
      nd::DeviceMesh mesh(world, ndim == 1 ? std::vector<int>{4} : std::vector<int>{2, 2},
                          ndim == 1 ? std::vector<std::string>{"x"} : std::vector<std::string>{"x", "y"});
      const auto pl = all_placements(2);
      std::vector<nd::Placements> combos;
      if (ndim == 1) {
        for (const auto& a : pl) combos.push_back({a});
      } else {
        for (const auto& a : pl) {
          for (const auto& b : pl) {
            if (valid({a, b})) combos.push_back({a, b});
          }
        }
      }
      for (const auto& src : combos) {
        for (const auto& dst : combos) {
          // Partial kaynak: rank'e ozel paylar (toplam = full): pay_r = full * (r + 1) / (toplam agirlik) yerine tamsayi bolusum.
          nd::DTensor a = nd::DTensor::distribute(full, shape, mesh, src);
          bool partial = false;
          for (const auto& x : src) partial = partial || x.is_partial();
          if (partial) {
            // Paylari degistir: koordinat 0'dan rank'lere tamsayi tasi (toplam korunur).
            nd::Placements rep = src;
            for (auto& x : rep) x = x.is_partial() ? nd::Placement::replicate() : x;
            const std::vector<float> base = nd::DTensor::distribute(full, shape, mesh, rep).local();
            int coords_sum = 0, parts = 1;
            for (int d = 0; d < mesh.ndim(); ++d) {
              if (src[static_cast<std::size_t>(d)].is_partial()) {
                coords_sum = coords_sum * mesh.size(d) + mesh.coordinate(d);
                parts *= mesh.size(d);
              }
            }
            std::vector<float> share(base.size());
            for (std::size_t e = 0; e < base.size(); ++e) {
              const float delta = static_cast<float>((static_cast<int>(e) % 5) - 2);  // tamsayi
              // pay = base / parts degil: rank 0 base - (parts-1) delta, digerleri delta
              share[e] = coords_sum == 0 ? base[e] - static_cast<float>(parts - 1) * delta : delta;
            }
            a = nd::DTensor::from_local(share, shape, mesh, src);
          }
          const nd::DTensor b = a.redistribute(dst);
          const std::vector<float> got_full = b.full_tensor();
          bool good = got_full == full;
          bool dst_partial = false;
          for (const auto& x : dst) dst_partial = dst_partial || x.is_partial();
          if (!dst_partial) good = good && b.local() == expected_local(full, shape, mesh, dst);
          good = good && b.local_shape() == nd::DTensor::local_shape_of(shape, mesh, dst);
          if (rank == 0) {
            pairs.fetch_add(1);
            if (good) ok_pairs.fetch_add(1);
          }
          if (!good) check(false, "redistribute " + nd::to_string(src) + " -> " + nd::to_string(dst) + " rank " + std::to_string(rank));
        }
      }
      // ---- op yayilimi: her op icin tum (gecerli) giris yerlesimleri, sonuc tam hesapla ayni.
      const std::vector<std::int64_t> as{6, 8}, bs{8, 5};
      std::vector<float> A(48), Bm(40), E(48), gamma(8), beta(8);
      for (int i = 0; i < 48; ++i) A[static_cast<std::size_t>(i)] = static_cast<float>((i * 5) % 7 - 3) * 0.25f;
      for (int i = 0; i < 48; ++i) E[static_cast<std::size_t>(i)] = static_cast<float>((i * 3) % 11 - 5) * 0.125f;
      for (int i = 0; i < 40; ++i) Bm[static_cast<std::size_t>(i)] = static_cast<float>((i * 3) % 5 - 2) * 0.5f;
      for (int i = 0; i < 8; ++i) {
        gamma[static_cast<std::size_t>(i)] = 1.0f + 0.1f * i;
        beta[static_cast<std::size_t>(i)] = 0.05f * i;
      }
      std::vector<float> C(30, 0.0f);
      for (int i = 0; i < 6; ++i)
        for (int k = 0; k < 8; ++k)
          for (int j = 0; j < 5; ++j) C[static_cast<std::size_t>(i * 5 + j)] += A[static_cast<std::size_t>(i * 8 + k)] * Bm[static_cast<std::size_t>(k * 5 + j)];
      auto close = [](const std::vector<float>& x, const std::vector<float>& y, float tol) {
        if (x.size() != y.size()) return false;
        for (std::size_t i = 0; i < x.size(); ++i) {
          if (std::fabs(x[i] - y[i]) > tol) return false;
        }
        return true;
      };
      int ops = 0;
      for (const auto& pa : combos) {
        for (const auto& pb : combos) {
          const nd::DTensor a = nd::DTensor::distribute(A, as, mesh, pa);
          const nd::DTensor b = nd::DTensor::distribute(Bm, bs, mesh, pb);
          const nd::DTensor c = nd::dt_matmul(a, b);
          check(close(c.full_tensor(), C, 1e-5f), "dt_matmul " + nd::to_string(pa) + " x " + nd::to_string(pb));
          const nd::ShardingDecision dec = nd::propagate(nd::DOp::Matmul, {pa, pb});
          check(c.placements() == dec.output, "dt_matmul cikis yerlesimi kural ile ayni");
          const nd::DTensor e = nd::DTensor::distribute(E, as, mesh, pb.size() ? pb : pa);
          std::vector<float> sum(48), prod(48);
          for (int i = 0; i < 48; ++i) {
            sum[static_cast<std::size_t>(i)] = A[static_cast<std::size_t>(i)] + E[static_cast<std::size_t>(i)];
            prod[static_cast<std::size_t>(i)] = A[static_cast<std::size_t>(i)] * E[static_cast<std::size_t>(i)];
          }
          check(close(nd::dt_add(a, e).full_tensor(), sum, 1e-6f), "dt_add " + nd::to_string(pa) + " + " + nd::to_string(pb));
          check(close(nd::dt_mul(a, e).full_tensor(), prod, 1e-6f), "dt_mul " + nd::to_string(pa) + " * " + nd::to_string(pb));
          ops += 3;
        }
        const nd::DTensor a = nd::DTensor::distribute(A, as, mesh, pa);
        std::vector<float> ln(48), sm(48);
        for (int r = 0; r < 6; ++r) {
          double mu = 0, var = 0, mx = -1e30, se = 0;
          for (int i = 0; i < 8; ++i) mu += A[static_cast<std::size_t>(r * 8 + i)];
          mu /= 8;
          for (int i = 0; i < 8; ++i) var += (A[static_cast<std::size_t>(r * 8 + i)] - mu) * (A[static_cast<std::size_t>(r * 8 + i)] - mu);
          var /= 8;
          for (int i = 0; i < 8; ++i) mx = std::max(mx, static_cast<double>(A[static_cast<std::size_t>(r * 8 + i)]));
          for (int i = 0; i < 8; ++i) se += std::exp(A[static_cast<std::size_t>(r * 8 + i)] - mx);
          for (int i = 0; i < 8; ++i) {
            const std::size_t at = static_cast<std::size_t>(r * 8 + i);
            ln[at] = static_cast<float>((A[at] - mu) / std::sqrt(var + 1e-5)) * gamma[static_cast<std::size_t>(i)] + beta[static_cast<std::size_t>(i)];
            sm[at] = static_cast<float>(std::exp(A[at] - mx) / se);
          }
        }
        check(close(nd::dt_layer_norm(a, gamma, beta, 1e-5f).full_tensor(), ln, 1e-5f), "dt_layer_norm " + nd::to_string(pa));
        check(close(nd::dt_softmax(a, 1).full_tensor(), sm, 1e-6f), "dt_softmax " + nd::to_string(pa));
        // embedding: tablo A [6, 8] (sozluk 6), kimlikler
        const std::vector<std::int32_t> ids{5, 0, 3, 3, 1, 4, 2};
        std::vector<float> emb;
        for (std::int32_t id : ids) emb.insert(emb.end(), A.begin() + id * 8, A.begin() + id * 8 + 8);
        check(close(nd::dt_embedding(a, ids).full_tensor(), emb, 0.0f), "dt_embedding " + nd::to_string(pa));
        ops += 3;
      }
      if (rank == 0) std::printf("  info  %dB mesh: %d op yayilim denetimi\n", ndim, ops);
    });
    std::printf("  info  %dB mesh: %d / %d redistribute cifti dogru\n", ndim, ok_pairs.load(), pairs.load());
    nexus_test::expect(pairs.load() > 0 && ok_pairs.load() == pairs.load(),
                       std::to_string(ndim) + "B mesh: tum (kaynak, hedef) yerlesim ciftleri dogru");
  }
  // kural adlari (rapor)
  const nd::ShardingDecision d1 = nd::propagate(nd::DOp::Matmul, {{nd::Placement::shard(1)}, {nd::Placement::shard(0)}});
  nexus_test::expect(d1.output[0].is_partial(), "matmul S1 x S0 -> Partial (" + d1.rules[0] + ")");
  const nd::ShardingDecision d2 = nd::propagate(nd::DOp::Embedding, {{nd::Placement::shard(0)}, {nd::Placement::replicate()}});
  nexus_test::expect(d2.output[0].is_partial(), "embedding tablo S0 -> Partial (" + d2.rules[0] + ")");
  bool threw = false;
  try {
    run_world(2, [&](int, nd::GroupPtr world) {
      nd::DeviceMesh mesh(world, {2, 1}, {"a", "b"});
      try {
        (void)nd::DTensor::distribute(std::vector<float>(4), {2, 2}, mesh, {nd::Placement::shard(0), nd::Placement::shard(0)});
      } catch (const nd::DistError&) {
        threw = true;
      }
    });
  } catch (...) {
  }
  nexus_test::expect(threw, "ic ice parcalama (iki mesh boyutu ayni tensor boyutu) acik hata");
}

// ==========================================================================================
// Model yardimcilari
// ==========================================================================================

lm::LmConfig cfg_gpt(int layers = 4) {
  lm::LmConfig c;
  c.name = "faz12-gpt";
  c.vocab_size = 61;
  c.vocab_pad_multiple = 8;  // 64: tp'ye bolunur, dolgu sutunlari -inf
  c.n_layer = layers;
  c.d_model = 32;
  c.n_head = 4;
  c.n_kv_head = 4;
  c.d_ff = 64;
  c.max_position = 64;
  c.norm = lm::NormKind::LayerNorm;
  c.mlp = lm::MlpKind::Gelu;
  c.pos = lm::PosKind::Learned;
  c.bias = true;
  c.tie_embeddings = true;
  c.init = lm::InitKind::Gpt2;
  c.seed = 7;
  return c;
}

lm::LmConfig cfg_llama(int layers = 4, int kv = 2) {
  lm::LmConfig c;
  c.name = "faz12-llama";
  c.vocab_size = 64;
  c.n_layer = layers;
  c.d_model = 32;
  c.n_head = 4;
  c.n_kv_head = kv;
  c.d_ff = 48;
  c.max_position = 128;
  c.norm = lm::NormKind::RmsNorm;
  c.mlp = lm::MlpKind::SwiGlu;
  c.pos = lm::PosKind::Rope;
  c.bias = false;
  c.tie_embeddings = false;
  c.init = lm::InitKind::Llama;
  c.seed = 11;
  return c;
}

std::vector<std::int32_t> make_tokens(int batch, int seq, int vocab, std::uint32_t seed) {
  std::vector<std::int32_t> t(static_cast<std::size_t>(batch) * seq);
  std::uint32_t s = seed * 2654435761u + 12345u;
  for (auto& x : t) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<std::int32_t>((s >> 9) % static_cast<std::uint32_t>(vocab));
  }
  return t;
}

struct RefResult {
  std::vector<float> losses;
  std::map<std::string, std::vector<float>> grads;
};

/// lm::LanguageModel otograd: sum_i w_i loss_i gradyani.
RefResult reference(const lm::LanguageModel& model, const std::vector<std::vector<std::int32_t>>& mbs, int batch, int seq) {
  RefResult r;
  const_cast<lm::LanguageModel&>(model).zero_grad();
  const float w = 1.0f / static_cast<float>(mbs.size());
  for (const auto& t : mbs) {
    const ag::Variable loss = model.loss(t, batch, seq);
    r.losses.push_back(loss.values()[0]);
    ag::mul_scalar(loss, w).backward();
  }
  for (const auto& p : model.parameters()) r.grads[p.name] = p.value.grad().values();
  const_cast<lm::LanguageModel&>(model).zero_grad();
  return r;
}

double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double mx = 0.0, ref = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, static_cast<double>(std::fabs(a[i] - b[i])));
    ref = std::max(ref, static_cast<double>(std::fabs(b[i])));
  }
  return mx / std::max(ref, 1e-12);
}

struct RunOut {
  std::vector<std::vector<float>> losses;  // adim basina mikro kayiplar
  std::map<std::string, std::vector<float>> grads;  // ilk adimin indirgenmis gradyani
  std::map<std::string, std::vector<float>> weights;  // son agirliklar
  double replica_diff = 0.0;         // ilk adim gradyan kopyalari (zero 0)
  double weight_replica_diff = 0.0;  // son agirlik kopyalari
  std::vector<nd::ParallelStepStats> stats;  // ilk adim, rank basina
  std::vector<std::string> layout;
};

/// Paralel calistirma: global mikro yiginlar mbs (dp kopyasi r: [r M, (r + 1) M)); steps adim AdamW.
RunOut run_parallel(const lm::LanguageModel& model, nd::ParallelConfig cfg, const std::vector<std::vector<std::int32_t>>& mbs,
                    int batch, int seq, int steps) {
  const int world_n = cfg.dp * cfg.tp * cfg.pp * cfg.cp;
  RunOut out;
  out.stats.resize(static_cast<std::size_t>(world_n));
  out.layout.resize(static_cast<std::size_t>(world_n));
  // Gradyan/agirlik toplama: bariyerde tum rank nesneleri canli, rank 0 toplar (ayni surec).
  std::vector<std::unique_ptr<nd::ParallelLm>> holders(static_cast<std::size_t>(world_n));
  run_world(world_n, [&](int rank, nd::GroupPtr world) {
    holders[static_cast<std::size_t>(rank)] = std::make_unique<nd::ParallelLm>(model, world, cfg);
    nd::ParallelLm& pl = *holders[static_cast<std::size_t>(rank)];
    const int dpi = pl.mesh().coordinate("dp");
    const int M = static_cast<int>(mbs.size()) / cfg.dp;
    std::vector<const std::int32_t*> micro;
    for (int i = 0; i < M; ++i) micro.push_back(mbs[static_cast<std::size_t>(dpi * M + i)].data());
    const std::vector<float> w(static_cast<std::size_t>(M), 1.0f / static_cast<float>(mbs.size()));
    for (int s = 0; s < steps; ++s) {
      std::vector<float> l = pl.forward_backward(micro, w, batch, seq);
      // dp kopyalarinin kayiplarini birlestir (dp sirasiyla)
      std::vector<float> all(static_cast<std::size_t>(M) * cfg.dp);
      pl.data_parallel_group()->all_gather(l.data(), all.data(), l.size(), DType::F32)->wait();
      if (s == 0) {
        out.stats[static_cast<std::size_t>(rank)] = pl.last_stats();
        out.layout[static_cast<std::size_t>(rank)] = pl.layout_summary();
        world->barrier()->wait();
        if (rank == 0) {
          std::vector<const nd::ParallelLm*> ranks;
          for (auto& h : holders) ranks.push_back(h.get());
          double rd = 0.0;
          for (const auto& p : model.parameters()) {
            double d = 0.0;
            out.grads[p.name] = nd::assemble_full(ranks, p.name, 1, &d);
            rd = std::max(rd, d);
          }
          out.replica_diff = rd;
        }
        world->barrier()->wait();
      }
      if (rank == 0) out.losses.push_back(all);
      (void)pl.step(nd::FsdpAdamW{1e-2, 0.9, 0.95, 1e-8, 0.1}, 1e-2, 1.0);
    }
    world->barrier()->wait();
    if (rank == 0) {
      std::vector<const nd::ParallelLm*> ranks;
      for (auto& h : holders) ranks.push_back(h.get());
      for (const auto& p : model.parameters()) {
        double d = 0.0;
        out.weights[p.name] = nd::assemble_full(ranks, p.name, 0, &d);
        out.weight_replica_diff = std::max(out.weight_replica_diff, d);
      }
    }
    world->barrier()->wait();
    holders[static_cast<std::size_t>(rank)].reset();  // gruplar ThreadWorld'den once kapansin
  });
  return out;
}

double compare_grads(const RunOut& a, const std::map<std::string, std::vector<float>>& ref, std::string* worst) {
  double mx = 0.0;
  for (const auto& kv : ref) {
    auto it = a.grads.find(kv.first);
    if (it == a.grads.end()) {
      if (worst) *worst = kv.first + " (yok)";
      return 1e9;
    }
    const double d = rel_diff(it->second, kv.second);
    if (d > mx) {
      mx = d;
      if (worst) *worst = kv.first;
    }
  }
  return mx;
}

double compare_maps_abs(const std::map<std::string, std::vector<float>>& a, const std::map<std::string, std::vector<float>>& b) {
  double mx = 0.0;
  for (const auto& kv : b) {
    const auto& x = a.at(kv.first);
    for (std::size_t i = 0; i < x.size(); ++i) mx = std::max(mx, static_cast<double>(std::fabs(x[i] - kv.second[i])));
  }
  return mx;
}

double max_loss_diff(const RunOut& a, const RunOut& b) {
  double mx = 0.0;
  for (std::size_t s = 0; s < a.losses.size(); ++s) {
    for (std::size_t i = 0; i < a.losses[s].size(); ++i) {
      mx = std::max(mx, static_cast<double>(std::fabs(a.losses[s][i] - b.losses[s][i])));
    }
  }
  return mx;
}

std::vector<std::vector<std::int32_t>> micro_batches(int count, int batch, int seq, int vocab, std::uint32_t seed) {
  std::vector<std::vector<std::int32_t>> m;
  for (int i = 0; i < count; ++i) m.push_back(make_tokens(batch, seq, vocab, seed + static_cast<std::uint32_t>(i)));
  return m;
}

// ==========================================================================================
// 12.B/C/D: TP, SP, async TP
// ==========================================================================================

void test_tp() {
  struct Case {
    const char* name;
    lm::LmConfig cfg;
  };
  const std::vector<Case> cases = {{"gpt (LN, GELU, ogrenilen konum, bias, bagli)", cfg_gpt()},
                                   {"llama (RMS, SwiGLU, RoPE, GQA kv=2, ayrik)", cfg_llama()},
                                   {"llama kv=1 (tp>Hkv: KV cogaltma)", cfg_llama(2, 1)}};
  const int B = 2, T = 16;
  for (const Case& cs : cases) {
    const lm::LanguageModel model(cs.cfg);
    const auto mbs = micro_batches(2, B, T, cs.cfg.vocab_size, 3);
    const RefResult ref = reference(model, mbs, B, T);
    nd::ParallelConfig base;
    const RunOut r1 = run_parallel(model, base, mbs, B, T, 1);
    std::string worst;
    const double g1 = compare_grads(r1, ref.grads, &worst);
    double l1 = 0.0;
    for (std::size_t i = 0; i < ref.losses.size(); ++i) l1 = std::max(l1, static_cast<double>(std::fabs(r1.losses[0][i] - ref.losses[i])));
    std::printf("  info  %s: tp=1 vs otograd kayip |fark| %.2e, gradyan goreli %.2e (%s)\n", cs.name, l1, g1, worst.c_str());
    nexus_test::expect(l1 < 1e-5 && g1 < 1e-4, std::string(cs.name) + ": tp=1 == lm::LanguageModel otograd");
    struct Variant {
      int tp;
      bool sp, async;
    };
    for (const Variant v : {Variant{2, false, false}, Variant{4, false, false}, Variant{2, true, false}, Variant{4, true, false},
                            Variant{2, true, true}, Variant{4, true, true}}) {
      nd::ParallelConfig c;
      c.tp = v.tp;
      c.sp = v.sp;
      c.async_tp = v.async;
      const RunOut r = run_parallel(model, c, mbs, B, T, 1);
      const double g = compare_grads(r, ref.grads, &worst);
      const double l = max_loss_diff(r, r1);
      std::printf("  info  %s: tp=%d sp=%d async=%d kayip |fark| %.2e, gradyan goreli %.2e (%s), kopya farki %.1e\n",
                  cs.name, v.tp, v.sp, v.async, l, g, worst.c_str(), r.replica_diff);
      nexus_test::expect(l < 1e-5 && g < 1e-4 && r.replica_diff < 1e-6 && r.weight_replica_diff == 0.0,
                         std::string(cs.name) + ": tp=" + std::to_string(v.tp) + (v.sp ? " +SP" : "") + (v.async ? " +async" : "") +
                             " ileri/geri == tp=1");
    }
  }
}

void test_vocab() {
  // Dolgulu sozluk (61 -> 64) ve buyuk sozluk; kafa/gomme gradyani ve kayip tp'den bagimsiz.
  for (int vocab : {61, 250}) {
    lm::LmConfig c = cfg_gpt(2);
    c.vocab_size = vocab;
    c.tie_embeddings = vocab == 61;
    const lm::LanguageModel model(c);
    const auto mbs = micro_batches(1, 2, 16, vocab, 9);
    const RefResult ref = reference(model, mbs, 2, 16);
    for (int tp : {1, 2, 4}) {
      nd::ParallelConfig pc;
      pc.tp = tp;
      const RunOut r = run_parallel(model, pc, mbs, 2, 16, 1);
      const double l = std::fabs(r.losses[0][0] - ref.losses[0]);
      const double g = rel_diff(r.grads.at("wte"), ref.grads.at("wte"));
      double gh = 0.0;
      if (!c.tie_embeddings) gh = rel_diff(r.grads.at("lm_head.weight"), ref.grads.at("lm_head.weight"));
      std::printf("  info  sozluk %d (dolgulu %d) tp=%d: CE |fark| %.2e, wte goreli %.2e, head goreli %.2e\n", vocab,
                  c.padded_vocab(), tp, l, g, gh);
      nexus_test::expect(l < 1e-5 && g < 1e-4 && gh < 1e-4,
                         "sozluk-parcali CE (V=" + std::to_string(vocab) + ", tp=" + std::to_string(tp) + ") == tam CE");
    }
  }
}

// ==========================================================================================
// 12.E PP
// ==========================================================================================

void test_pp() {
  const lm::LmConfig c = cfg_gpt(8);
  const lm::LanguageModel model(c);
  const int B = 2, T = 16, M = 8;
  const auto mbs = micro_batches(M, B, T, c.vocab_size, 21);
  const RefResult ref = reference(model, mbs, B, T);
  struct S {
    nd::PipeSchedule s;
    int v;
  };
  RunOut base;
  bool have = false;
  for (const S sc : {S{nd::PipeSchedule::GPipe, 1}, S{nd::PipeSchedule::OneF1B, 1}, S{nd::PipeSchedule::Interleaved1F1B, 2},
                     S{nd::PipeSchedule::ZeroBubbleH1, 1}}) {
    nd::ParallelConfig pc;
    pc.pp = 4;
    pc.pp_schedule = sc.s;
    pc.virtual_stages = sc.v;
    pc.trace = true;
    const RunOut r = run_parallel(model, pc, mbs, B, T, 1);
    std::string worst;
    const double g = compare_grads(r, ref.grads, &worst);
    double bubble = 0.0, peak = 0.0;
    int wpeak = 0;
    for (const auto& st : r.stats) {
      bubble += st.bubble() / static_cast<double>(r.stats.size());
      peak = std::max(peak, static_cast<double>(st.peak_activation_bytes));
      wpeak = std::max(wpeak, st.deferred_w_peak);
    }
    std::printf("  info  %s v=%d: gradyan goreli %.2e (%s); olculen kabarcik %.3f (teorik %.3f); tepe aktivasyon %.0f KB; ertelenmis W tepe %d\n",
                nd::to_string(sc.s), sc.v, g, worst.c_str(), bubble, nd::theoretical_bubble(sc.s, 4, M, sc.v), peak / 1024.0, wpeak);
    std::printf("  info    rank 0 izi: ");
    for (std::size_t i = 0; i < r.stats[0].trace.size() && i < 40; ++i) std::printf("%s ", r.stats[0].trace[i].c_str());
    std::printf("\n  info    yerlesim: %s | %s\n", r.layout[0].c_str(), r.layout[3].c_str());
    nexus_test::expect(g < 1e-4, std::string(nd::to_string(sc.s)) + ": gradyan == pp=1 otograd");
    if (!have) {
      base = r;
      have = true;
    } else {
      const double d = compare_maps_abs(r.grads, base.grads);
      std::printf("  info    gpipe ile en buyuk |gradyan farki| %.3e\n", d);
      nexus_test::expect(d == 0.0, std::string(nd::to_string(sc.s)) + ": gradyan GPipe ile BIT BIT ayni");
    }
  }
  // dengeli bolumleme
  const auto part = nd::balance_partition({1, 1, 1, 1, 1, 1, 1, 4}, 4);
  nexus_test::expect(part.back().first == 7 && part.front().first == 0, "maliyet dengeli bolumleme: agir son katman tek basina");
  bool threw = false;
  try {
    (void)nd::make_schedule(nd::PipeSchedule::Interleaved1F1B, 4, 0, 6, 2);
  } catch (const nd::DistError&) {
    threw = true;
  }
  nexus_test::expect(threw, "interleaved_1f1b m % pp != 0 acik hata");
}

// ==========================================================================================
// 12.F CP
// ==========================================================================================

void test_ring() {
  const int B = 2, T = 64, D = 16;
  struct Case {
    int cp, H, Hk;
    nd::CpAlgorithm alg;
    nd::CpLayout layout;
  };
  for (const Case cs : {Case{4, 4, 2, nd::CpAlgorithm::Ring, nd::CpLayout::Contiguous}, Case{4, 4, 2, nd::CpAlgorithm::Ring, nd::CpLayout::Zigzag},
                        Case{2, 4, 2, nd::CpAlgorithm::Ulysses, nd::CpLayout::Zigzag}, Case{4, 4, 4, nd::CpAlgorithm::Ulysses, nd::CpLayout::Contiguous},
                        Case{8, 2, 1, nd::CpAlgorithm::Ring, nd::CpLayout::Zigzag}}) {
    const int qw = cs.H * D, kw = cs.Hk * D;
    std::vector<float> q(static_cast<std::size_t>(B) * T * qw), k(static_cast<std::size_t>(B) * T * kw), v(k.size()), dout(q.size());
    std::uint32_t s = 99;
    auto rnd = [&] {
      s = s * 1664525u + 1013904223u;
      return static_cast<float>((s >> 8) % 2001) / 1000.0f - 1.0f;
    };
    for (auto& x : q) x = rnd();
    for (auto& x : k) x = rnd();
    for (auto& x : v) x = rnd();
    for (auto& x : dout) x = rnd();
    nexus::AttentionParams p;
    p.batch = B;
    p.heads = cs.H;
    p.kv_heads = cs.Hk;
    p.tq = T;
    p.tk = T;
    p.head_dim = D;
    p.dtype = DType::F32;
    p.causal = true;
    p.set_contiguous_bthd();
    std::vector<float> o(q.size()), lse(static_cast<std::size_t>(B) * cs.H * T), dq(q.size()), dk(k.size()), dv(v.size());
    nexus::attention::flash_attention_forward_cpu(o.data(), lse.data(), q.data(), k.data(), v.data(), p, 1);
    nexus::attention::flash_attention_backward_cpu(dq.data(), dk.data(), dv.data(), dout.data(), q.data(), k.data(), v.data(),
                                                   o.data(), lse.data(), p, 1);
    std::vector<float> go(q.size()), gq(q.size()), gk(k.size()), gv(v.size());
    std::vector<int> blocks(static_cast<std::size_t>(cs.cp));
    run_world(cs.cp, [&](int rank, nd::GroupPtr world) {
      nd::CpAttention a{B, T, cs.H, cs.Hk, D, 0.0f, cs.layout, 1};
      const std::size_t rows = static_cast<std::size_t>(B) * T / cs.cp;
      std::vector<float> lq(rows * qw), lk(rows * kw), lv(rows * kw), ld(rows * qw), lo(rows * qw), lq2(rows * qw), lk2(rows * kw), lv2(rows * kw);
      nd::cp_scatter_sequence(q.data(), lq.data(), B, T, qw, cs.cp, rank, cs.layout);
      nd::cp_scatter_sequence(k.data(), lk.data(), B, T, kw, cs.cp, rank, cs.layout);
      nd::cp_scatter_sequence(v.data(), lv.data(), B, T, kw, cs.cp, rank, cs.layout);
      nd::cp_scatter_sequence(dout.data(), ld.data(), B, T, qw, cs.cp, rank, cs.layout);
      if (cs.alg == nd::CpAlgorithm::Ring) {
        std::vector<float> l(static_cast<std::size_t>(nd::cp_chunks(cs.cp, rank, cs.layout).size()) * B * cs.H * a.chunk_len(cs.cp));
        nd::ring_attention_forward(*world, a, lq.data(), lk.data(), lv.data(), lo.data(), l.data());
        nd::ring_attention_backward(*world, a, lq.data(), lk.data(), lv.data(), lo.data(), l.data(), ld.data(), lq2.data(), lk2.data(), lv2.data());
      } else {
        std::vector<float> l(nd::ulysses_lse_size(a, cs.cp));
        nd::ulysses_attention_forward(*world, a, lq.data(), lk.data(), lv.data(), lo.data(), l.data());
        nd::ulysses_attention_backward(*world, a, lq.data(), lk.data(), lv.data(), lo.data(), l.data(), ld.data(), lq2.data(), lk2.data(), lv2.data());
      }
      std::lock_guard<std::mutex> lock(g_mu);
      nd::cp_gather_sequence(lo.data(), go.data(), B, T, qw, cs.cp, rank, cs.layout);
      nd::cp_gather_sequence(lq2.data(), gq.data(), B, T, qw, cs.cp, rank, cs.layout);
      nd::cp_gather_sequence(lk2.data(), gk.data(), B, T, kw, cs.cp, rank, cs.layout);
      nd::cp_gather_sequence(lv2.data(), gv.data(), B, T, kw, cs.cp, rank, cs.layout);
      blocks[static_cast<std::size_t>(rank)] = nd::ring_attention_blocks(cs.cp, rank, cs.layout);
    });
    auto md = [](const std::vector<float>& a, const std::vector<float>& b) {
      double m = 0.0;
      for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
      return m;
    };
    const double eo = md(go, o), eq = md(gq, dq), ek = md(gk, dk), ev = md(gv, dv);
    const auto mm = std::minmax_element(blocks.begin(), blocks.end());
    std::printf("  info  %s %s cp=%d H=%d Hkv=%d: |dO| %.1e |dQ| %.1e |dK| %.1e |dV| %.1e; rank blok sayisi min %d max %d\n",
                nd::to_string(cs.alg), nd::to_string(cs.layout), cs.cp, cs.H, cs.Hk, eo, eq, ek, ev, *mm.first, *mm.second);
    nexus_test::expect(eo < 1e-5 && eq < 1e-5 && ek < 1e-5 && ev < 1e-5,
                       std::string(nd::to_string(cs.alg)) + " " + nd::to_string(cs.layout) + " cp=" + std::to_string(cs.cp) + " == tam Flash (ileri+geri)");
    if (cs.alg == nd::CpAlgorithm::Ring && cs.layout == nd::CpLayout::Zigzag) {
      nexus_test::expect(*mm.first == *mm.second, "zigzag: her rank ayni sayida nedensel blok (dengeli)");
    }
  }
}

// ==========================================================================================
// ZeRO-1, 3B, 4B
// ==========================================================================================

void test_zero() {
  const lm::LmConfig c = cfg_llama(2);
  const lm::LanguageModel model(c);
  const auto mbs = micro_batches(4, 2, 16, c.vocab_size, 31);
  for (int tp : {1, 2}) {
    nd::ParallelConfig z0;
    z0.dp = 4 / tp;
    z0.tp = tp;
    nd::ParallelConfig z1 = z0;
    z1.zero = 1;
    const RunOut a = run_parallel(model, z0, mbs, 2, 16, 5);
    const RunOut b = run_parallel(model, z1, mbs, 2, 16, 5);
    const double w = compare_maps_abs(a.weights, b.weights);
    const double l = max_loss_diff(a, b);
    std::printf("  info  dp=%d tp=%d: ZeRO-1 vs ZeRO-0 5 adim agirlik |fark| %.3e, kayip |fark| %.3e\n", z0.dp, tp, w, l);
    nexus_test::expect(w == 0.0 && l == 0.0, "ZeRO-1 (dp=" + std::to_string(z0.dp) + ", tp=" + std::to_string(tp) + ") == ZeRO-0 BIT BIT");
  }
}

/// Agirlik farki ||w_a - w_b|| / ||w_b - w_0|| (parametre basina; guncelleme buyuklugune gore). Max-abs olcutu
/// gradyani gurultu tabanindaki ogelerde Adam isaret cevirmesine (|fark| ~ 2 lr) duyarlidir; o da raporlanir.
double update_rel(const RunOut& r, const RunOut& s, const lm::LanguageModel& model, std::string* worst, double* max_abs) {
  double mx = 0.0;
  *max_abs = 0.0;
  for (const auto& p : model.parameters()) {
    const auto& a = r.weights.at(p.name);
    const auto& b = s.weights.at(p.name);
    const auto& w0 = p.value.values();
    double dd = 0.0, uu = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      dd += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
      uu += (static_cast<double>(b[i]) - w0[i]) * (static_cast<double>(b[i]) - w0[i]);
      *max_abs = std::max(*max_abs, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    const double rel = uu > 0.0 ? std::sqrt(dd / uu) : std::sqrt(dd);
    if (rel > mx) {
      mx = rel;
      if (worst) *worst = p.name;
    }
  }
  return mx;
}

void multi_d(const char* name, nd::ParallelConfig pc, const lm::LmConfig& c, int seq) {
  const lm::LanguageModel model(c);
  const int B = 2, M = 2 * pc.dp;
  const auto mbs = micro_batches(M, B, seq, c.vocab_size, 41);
  const RefResult ref = reference(model, mbs, B, seq);
  nd::ParallelConfig single;
  const RunOut s = run_parallel(model, single, mbs, B, seq, 3);
  const RunOut r = run_parallel(model, pc, mbs, B, seq, 3);
  std::string worst, wworst;
  // ZeRO-1: Param::grad dp x cp indirgemesi oncesi yereldir (indirgenmis gradyan yalniz parcada) -> karsilastirilmaz.
  const double g = pc.zero == 1 ? 0.0 : compare_grads(r, ref.grads, &worst);
  const double l = max_loss_diff(r, s);
  double wabs = 0.0;
  const double w = update_rel(r, s, model, &wworst, &wabs);
  char gs[32];
  std::snprintf(gs, sizeof(gs), "%.2e", g);
  std::printf("  info  %s: gradyan goreli %s (%s), 3 adim kayip |fark| %.2e, agirlik farki/guncelleme %.2e (%s; max |fark| %.1e), "
              "kopya farki grad %.1e agirlik %.1e\n",
              name, pc.zero == 1 ? "n/a (ZeRO-1)" : gs, worst.c_str(), l, w, wworst.c_str(), wabs, r.replica_diff,
              r.weight_replica_diff);
  for (std::size_t i = 0; i < r.layout.size() && i < 4; ++i) std::printf("  info    %s\n", r.layout[i].c_str());
  nexus_test::expect(g < 1e-4 && l < 1e-5 && w < 1e-3 && r.weight_replica_diff == 0.0 && (pc.zero == 1 || r.replica_diff < 1e-5),
                     std::string(name) + " == tek rank (3 adim)");
}

void test_3d() {
  nd::ParallelConfig pc;
  pc.dp = 2;
  pc.tp = 2;
  pc.pp = 2;
  pc.sp = true;
  pc.pp_schedule = nd::PipeSchedule::OneF1B;
  multi_d("3B (dp=2, tp=2+SP, pp=2 1F1B) 8 rank", pc, cfg_gpt(4), 16);
  pc.pp_schedule = nd::PipeSchedule::Interleaved1F1B;
  pc.virtual_stages = 2;
  pc.async_tp = true;
  multi_d("3B (dp=2, tp=2+SP+async, pp=2 interleaved v=2) 8 rank", pc, cfg_llama(4), 16);
  pc.zero = 1;
  multi_d("3B (dp=2 ZeRO-1, tp=2+SP+async, pp=2 interleaved v=2) 8 rank", pc, cfg_llama(4), 16);
}

void test_4d() {
  nd::ParallelConfig pc;
  pc.dp = 2;
  pc.tp = 2;
  pc.pp = 2;
  pc.cp = 2;
  pc.sp = true;
  pc.cp_layout = nd::CpLayout::Zigzag;
  multi_d("4B (dp=2, pp=2, cp=2 ring zigzag, tp=2+SP) 16 rank", pc, cfg_llama(4), 32);
  pc.cp_algorithm = nd::CpAlgorithm::Ulysses;
  pc.pp_schedule = nd::PipeSchedule::ZeroBubbleH1;
  lm::LmConfig c = cfg_llama(4, 4);
  multi_d("4B (dp=2, pp=2 ZB-H1, cp=2 Ulysses, tp=2+SP) 16 rank", pc, c, 32);
}

// ==========================================================================================
// 12.8 API: LanguageTrainerBuilder(lm model).parallel(world, {...})
// ==========================================================================================

class GlobalRows final : public nt::TokenBatchSource {
 public:
  /// Cagri basina rows satir; kopya r kuresel [calls * rows * replicas + r * rows, + rows) satirlarini alir.
  GlobalRows(int replica, int replicas, int vocab) : replica_(replica), replicas_(replicas), vocab_(vocab) {}
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
    for (int j = 0; j < rows; ++j) {
      const std::uint64_t id = calls_ * static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(replicas_) +
                               static_cast<std::uint64_t>(replica_) * static_cast<std::uint64_t>(rows) + static_cast<std::uint64_t>(j);
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
  int replica_, replicas_, vocab_;
  std::uint64_t calls_ = 0;
};

std::vector<double> train_losses(const lm::LanguageModel& model, nd::GroupPtr world, nt::LanguageTrainerBuilder::Parallel p,
                                 int replica, int rows, int steps) {
  const int T = 16;
  GlobalRows src(replica, world ? p.dp : 1, model.config().vocab_size);
  nt::SpikeOptions no_spike;
  no_spike.enabled = false;
  nt::NanOptions nan;
  nan.diagnose = false;
  nt::LanguageTrainerBuilder b(model);
  b.data(src)
      .optimizer(nt::AdamW{1.0e-2, 0.9, 0.95, 1.0e-8, 0.1})
      .micro_batch(rows)
      .seq_len(T)
      .global_batch_tokens(static_cast<std::int64_t>(4) * T * 2)  // 4 satir x 2 mikro adim
      .grad_clip(1.0)
      .spike_detector(no_spike)
      .nan_policy(nan)
      .log_console(false);
  if (world) b.parallel(world, p);
  auto trainer = b.build();
  trainer->train(steps);
  std::vector<double> out;
  for (const auto& r : trainer->history()) out.push_back(r.loss);
  return out;
}

void test_trainer() {
  const lm::LanguageModel model(cfg_llama(4));
  const std::vector<double> single = train_losses(model, nullptr, {}, 0, 4, 4);
  std::vector<double> par;
  run_world(8, [&](int rank, nd::GroupPtr world) {
    nt::LanguageTrainerBuilder::Parallel p;
    p.dp = 2;
    p.tp = 2;
    p.pp = 2;
    p.sp = true;
    p.pp_schedule = "1f1b";
    const int replica = rank / 4;  // mesh {dp, pp, cp, tp}: dp en yavas
    const std::vector<double> l = train_losses(model, world, p, replica, 2, 4);
    if (rank == 0) par = l;
  });
  double mx = 0.0;
  for (std::size_t i = 0; i < single.size() && i < par.size(); ++i) mx = std::max(mx, std::fabs(single[i] - par[i]));
  std::printf("  info  egitmen 4 adim: tek rank %.5f .. %.5f, dp2 tp2+SP pp2 %.5f .. %.5f, en buyuk |fark| %.2e\n",
              single.front(), single.back(), par.empty() ? 0.0 : par.front(), par.empty() ? 0.0 : par.back(), mx);
  nexus_test::expect(par.size() == single.size() && mx < 1e-5, "trainer.parallel({dp=2, tp=2, sp, pp=2}) kayiplari == tek rank egitmen");
  bool threw = false;
  try {
    nt::LanguageTrainerBuilder::Parallel p;
    p.tp = 2;
    nt::LanguageTrainerBuilder b(nt::BackendFactory([](const nt::LmBackendRequest&) -> std::unique_ptr<nt::LmBackend> { return nullptr; }));
    nd::ThreadWorld w(1);
    b.parallel(w.create_group(0), p);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  nexus_test::expect(threw, "tp > 1 lm::LanguageModel olmayan builder'da acik hata");
}

// ==========================================================================================
// TCP (surecler) ve olcum modlari
// ==========================================================================================

/// nexus-run -n 4 -- test_phase12_parallel tcp : dp=2, tp=2 + SP + async TP (halka send_recv) surecler arasi
/// 3 adim == tek rank (rank 0 kendi icinde hesaplar).
int tcp_mode() {
  nd::GroupPtr g = nd::init_process_group("tcp");
  const lm::LanguageModel model(cfg_llama(2));
  const int B = 2, T = 16;
  const auto mbs = micro_batches(4, B, T, model.config().vocab_size, 51);
  nd::ParallelConfig pc;
  pc.dp = 2;
  pc.tp = g->size() / 2;
  pc.sp = true;
  pc.async_tp = true;
  auto run = [&](nd::ParallelLm& pl, int replicas) {
    const int dpi = pl.mesh().coordinate("dp");
    const int M = static_cast<int>(mbs.size()) / replicas;
    std::vector<const std::int32_t*> micro;
    for (int i = 0; i < M; ++i) micro.push_back(mbs[static_cast<std::size_t>(dpi * M + i)].data());
    const std::vector<float> w(static_cast<std::size_t>(M), 1.0f / static_cast<float>(mbs.size()));
    std::vector<float> all;
    for (int s = 0; s < 3; ++s) {
      const std::vector<float> l = pl.forward_backward(micro, w, B, T);
      std::vector<float> step(static_cast<std::size_t>(M) * replicas);
      pl.data_parallel_group()->all_gather(l.data(), step.data(), l.size(), DType::F32)->wait();
      all.insert(all.end(), step.begin(), step.end());
      (void)pl.step(nd::FsdpAdamW{1e-2, 0.9, 0.95, 1e-8, 0.1}, 1e-2, 1.0);
    }
    return all;
  };
  nd::ParallelLm pl(model, g, pc);
  const std::vector<float> got = run(pl, pc.dp);
  if (g->rank() != 0) return 0;
  nd::ThreadWorld one(1);
  nd::ParallelLm ref(model, one.create_group(0), nd::ParallelConfig{});
  const std::vector<float> want = run(ref, 1);
  double mx = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) mx = std::max(mx, static_cast<double>(std::fabs(got[i] - want[i])));
  const bool ok = got.size() == want.size() && mx < 1e-5;
  std::printf("  %s  tcp %d surec (dp=2, tp=%d+SP+async): 3 adim kayip == tek rank (|fark| %.2e)\n", ok ? "PASS" : "FAIL",
              g->size(), pc.tp, mx);
  return ok ? 0 : 1;
}

/// Olcum modeli (TP/SP): llama benzeri, sozluk 2048, d 256, 8 bas, 4 katman, T 256, B 2.
lm::LmConfig cfg_bench() {
  lm::LmConfig c = cfg_llama(4, 8);
  c.vocab_size = 2048;
  c.d_model = 256;
  c.n_head = 8;
  c.d_ff = 704;
  c.max_position = 512;
  return c;
}

double model_flops_per_token(const lm::LmConfig& c, int seq) {
  const lm::LanguageModel m(c);
  const double n = static_cast<double>(m.num_parameters());
  return 6.0 * n + 12.0 * c.n_layer * c.d_model * static_cast<double>(seq);
}

/// bench_tp TP SP ASYNC STEPS: tek surec thread rank'leri (ya da nexus-run altinda tcp). JSON satiri.
int bench_tp(int tp, bool sp, bool async_tp, int steps) {
  const lm::LmConfig c = cfg_bench();
  const lm::LanguageModel model(c);
  const int B = 2, T = 256;
  const auto mbs = micro_batches(1, B, T, c.vocab_size, 61);
  const bool tcp = nexus::dist::env_int("WORLD_SIZE", 0) > 0;
  std::vector<double> times;
  auto body = [&](int rank, nd::GroupPtr world) {
    nd::ParallelConfig pc;
    pc.tp = tp;
    pc.sp = sp;
    pc.async_tp = async_tp;
    nd::ParallelLm pl(model, world, pc);
    std::vector<const std::int32_t*> micro{mbs[0].data()};
    for (int s = 0; s < steps; ++s) {
      world->barrier()->wait();
      const auto t0 = std::chrono::steady_clock::now();
      (void)pl.forward_backward(micro, {1.0f}, B, T);
      (void)pl.step(nd::FsdpAdamW{1e-3, 0.9, 0.95, 1e-8, 0.1}, 1e-3, 1.0);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      world->all_reduce(&ms, 1, DType::F64, nd::ReduceOp::Max)->wait();
      if (rank == 0) times.push_back(ms);
    }
  };
  if (tcp) {
    nd::GroupPtr g = nd::init_process_group("tcp");
    body(g->rank(), g);
    if (g->rank() != 0) return 0;
  } else {
    run_world(tp, body);
  }
  std::sort(times.begin() + 1, times.end());
  const double med = times[1 + (times.size() - 1) / 2];
  const double tok = static_cast<double>(B) * T;
  const double gflops = model_flops_per_token(c, T) * tok / (med * 1e-3) / 1e9;
  std::printf("{\"mode\": \"bench_tp\", \"backend\": \"%s\", \"tp\": %d, \"sp\": %d, \"async\": %d, \"median_step_ms\": %.3f, "
              "\"tokens_per_s\": %.1f, \"model_gflops\": %.2f}\n",
              tcp ? "tcp" : "thread", tp, sp ? 1 : 0, async_tp ? 1 : 0, med, tok / (med * 1e-3), gflops);
  return 0;
}

/// bench_pp SCHEDULE V M: pp=4 (thread), hesap agir model; olculen kabarcik vs teorik (tF/tB/tW olculen).
int bench_pp(const std::string& schedule, int v, int m) {
  lm::LmConfig c = cfg_gpt(8);
  c.d_model = 256;
  c.n_head = 8;
  c.n_kv_head = 8;
  c.d_ff = 1024;
  c.max_position = 256;
  const lm::LanguageModel model(c);
  const int B = 2, T = 128;
  const auto mbs = micro_batches(m, B, T, c.vocab_size, 71);
  nd::ParallelConfig pc;
  pc.pp = 4;
  pc.pp_schedule = nd::parse_pipe_schedule(schedule);
  pc.virtual_stages = v;
  std::vector<nd::ParallelStepStats> st(4);
  run_world(4, [&](int rank, nd::GroupPtr world) {
    nd::ParallelLm pl(model, world, pc);
    std::vector<const std::int32_t*> micro;
    for (const auto& x : mbs) micro.push_back(x.data());
    const std::vector<float> w(static_cast<std::size_t>(m), 1.0f / static_cast<float>(m));
    nd::ParallelStepStats best;
    for (int s = 0; s < 4; ++s) {
      world->barrier()->wait();
      (void)pl.forward_backward(micro, w, B, T);
      (void)pl.step(nd::FsdpAdamW{1e-3, 0.9, 0.95, 1e-8, 0.1}, 1e-3, 1.0);
      if (s >= 1 && (best.step_ms == 0.0 || pl.last_stats().step_ms < best.step_ms)) best = pl.last_stats();
    }
    st[static_cast<std::size_t>(rank)] = best;
  });
  double bubble = 0.0, step = 0.0, tf = 0.0, tb = 0.0, tw = 0.0;
  std::uint64_t peak = 0;
  int wpeak = 0;
  for (const auto& x : st) {
    bubble += x.bubble() / 4.0;
    step = std::max(step, x.step_ms);
    tf += x.forward_ms / std::max(1, x.forwards) / 4.0;
    tb += x.backward_ms / std::max(1, x.backwards) / 4.0;
    tw += x.weights > 0 ? x.weight_ms / x.weights / 4.0 : 0.0;
    peak = std::max(peak, x.peak_activation_bytes);
    wpeak = std::max(wpeak, x.deferred_w_peak);
  }
  const nd::PipeSchedule ps = nd::parse_pipe_schedule(schedule);
  const double theo = ps == nd::PipeSchedule::ZeroBubbleH1 ? nd::theoretical_bubble(ps, 4, m, v, tf, tb, tw)
                                                            : nd::theoretical_bubble(ps, 4, m, v);
  std::printf("{\"mode\": \"bench_pp\", \"schedule\": \"%s\", \"pp\": 4, \"v\": %d, \"m\": %d, \"step_ms\": %.2f, "
              "\"bubble_measured\": %.4f, \"bubble_theory\": %.4f, \"tF_ms\": %.3f, \"tB_ms\": %.3f, \"tW_ms\": %.3f, "
              "\"peak_activation_mb\": %.2f, \"deferred_w_peak\": %d}\n",
              schedule.c_str(), v, m, step, bubble, theo, tf, tb, tw, static_cast<double>(peak) / 1048576.0, wpeak);
  return 0;
}

/// bench_cp T CP ALG LAYOUT: ring/Ulysses dikkat ileri + geri (1 bas, D 64, B 1). Rank basina tampon bellegi
/// (q,k,v,o,lse,dout,dq,dk,dv + halka tamponlari) ve sure; cp=1 tam Flash ile karsilastirma.
int bench_cp(int T, int cp, const std::string& alg, const std::string& layout) {
  const int B = 1, H = 1, D = 64;
  const nd::CpLayout lay = layout == "zigzag" ? nd::CpLayout::Zigzag : nd::CpLayout::Contiguous;
  const nd::CpAlgorithm al = alg == "ulysses" ? nd::CpAlgorithm::Ulysses : nd::CpAlgorithm::Ring;
  std::vector<double> fwd(static_cast<std::size_t>(cp)), bwd(static_cast<std::size_t>(cp));
  std::uint64_t bytes = 0;
  std::vector<int> blocks(static_cast<std::size_t>(cp));
  auto body = [&](int rank, nd::GroupPtr world) {
    nd::CpAttention a{B, T, H, H, D, 0.0f, lay, 1};
    const std::size_t rows = static_cast<std::size_t>(B) * T / cp;
    const std::size_t w = static_cast<std::size_t>(H) * D;
    std::vector<float> q(rows * w), k(rows * w), v(rows * w), o(rows * w), dout(rows * w), dq(rows * w), dk(rows * w), dv(rows * w);
    std::uint32_t sd = 1234u + static_cast<std::uint32_t>(rank);
    for (auto* vec : {&q, &k, &v, &dout}) {
      for (auto& x : *vec) {
        sd = sd * 1664525u + 1013904223u;
        x = static_cast<float>((sd >> 8) % 2001) / 1000.0f - 1.0f;
      }
    }
    const std::size_t lse_n = cp == 1 ? static_cast<std::size_t>(B) * H * T
                              : al == nd::CpAlgorithm::Ring ? nd::cp_chunks(cp, rank, lay).size() * B * H * a.chunk_len(cp)
                                                            : nd::ulysses_lse_size(a, cp);
    std::vector<float> lse(lse_n);
    world->barrier()->wait();
    auto t0 = std::chrono::steady_clock::now();
    if (cp == 1) {
      nexus::AttentionParams p;
      p.batch = B;
      p.heads = H;
      p.kv_heads = H;
      p.tq = T;
      p.tk = T;
      p.head_dim = D;
      p.dtype = DType::F32;
      p.causal = true;
      p.set_contiguous_bthd();
      nexus::attention::flash_attention_forward_cpu(o.data(), lse.data(), q.data(), k.data(), v.data(), p, 1);
      fwd[0] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      t0 = std::chrono::steady_clock::now();
      nexus::attention::flash_attention_backward_cpu(dq.data(), dk.data(), dv.data(), dout.data(), q.data(), k.data(), v.data(),
                                                     o.data(), lse.data(), p, 1);
      bwd[0] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    } else if (al == nd::CpAlgorithm::Ring) {
      nd::ring_attention_forward(*world, a, q.data(), k.data(), v.data(), o.data(), lse.data());
      fwd[static_cast<std::size_t>(rank)] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      world->barrier()->wait();
      t0 = std::chrono::steady_clock::now();
      nd::ring_attention_backward(*world, a, q.data(), k.data(), v.data(), o.data(), lse.data(), dout.data(), dq.data(), dk.data(), dv.data());
      bwd[static_cast<std::size_t>(rank)] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    } else {
      nd::ulysses_attention_forward(*world, a, q.data(), k.data(), v.data(), o.data(), lse.data());
      fwd[static_cast<std::size_t>(rank)] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      world->barrier()->wait();
      t0 = std::chrono::steady_clock::now();
      nd::ulysses_attention_backward(*world, a, q.data(), k.data(), v.data(), o.data(), lse.data(), dout.data(), dq.data(), dk.data(), dv.data());
      bwd[static_cast<std::size_t>(rank)] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    // rank basina kalici + gecici tampon (analitik): 9 [rows, w] + lse + halka (ring: kv cur/next 4, dkv 6 blok;
    // ulysses: tam dizi bas parcasi 8 [T, w/cp] + gonder/al 2 [rows, w])
    std::uint64_t b = (9 * rows * w + lse.size()) * sizeof(float);
    if (cp > 1 && al == nd::CpAlgorithm::Ring) b += 10 * rows * w * sizeof(float);
    if (cp > 1 && al == nd::CpAlgorithm::Ulysses) b += (8 * static_cast<std::size_t>(B) * T * w / cp + 2 * rows * w) * sizeof(float);
    if (rank == 0) bytes = b;
    blocks[static_cast<std::size_t>(rank)] = cp > 1 ? nd::ring_attention_blocks(cp, rank, lay) : 1;
  };
  run_world(cp, body);
  const double f = *std::max_element(fwd.begin(), fwd.end());
  const double bk = *std::max_element(bwd.begin(), bwd.end());
  const auto mm = std::minmax_element(blocks.begin(), blocks.end());
  std::printf("{\"mode\": \"bench_cp\", \"T\": %d, \"cp\": %d, \"alg\": \"%s\", \"layout\": \"%s\", \"fwd_ms\": %.1f, \"bwd_ms\": %.1f, "
              "\"per_rank_buffer_mb\": %.2f, \"blocks_min\": %d, \"blocks_max\": %d}\n",
              T, cp, alg.c_str(), layout.c_str(), f, bk, static_cast<double>(bytes) / 1048576.0, *mm.first, *mm.second);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "dtensor";
  try {
    if (mode == "dtensor") test_dtensor();
    else if (mode == "tp") test_tp();
    else if (mode == "vocab") test_vocab();
    else if (mode == "pp") test_pp();
    else if (mode == "ring") test_ring();
    else if (mode == "zero") test_zero();
    else if (mode == "3d") test_3d();
    else if (mode == "4d") test_4d();
    else if (mode == "trainer") test_trainer();
    else if (mode == "tcp") return tcp_mode();
    else if (mode == "bench_tp" && argc > 5) return bench_tp(std::stoi(argv[2]), std::stoi(argv[3]) != 0, std::stoi(argv[4]) != 0, std::stoi(argv[5]));
    else if (mode == "bench_pp" && argc > 4) return bench_pp(argv[2], std::stoi(argv[3]), std::stoi(argv[4]));
    else if (mode == "bench_cp" && argc > 5) return bench_cp(std::stoi(argv[2]), std::stoi(argv[3]), argv[4], argv[5]);
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  nexus_test::expect(g_fail.load() == 0, "rank ici denetimler");
  return nexus_test::finish("test_phase12_parallel");
}
