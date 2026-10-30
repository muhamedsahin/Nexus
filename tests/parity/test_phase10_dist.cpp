/**
 * @file test_phase10_dist.cpp
 * @brief FAZ 10: ProcessGroup kollektifleri (thread + tcp), dogrulama modu, rank olumu, busbw.
 *
 *   test_phase10_dist collectives_thread        : 1..8 rank (is parcacigi), tum islemler ve dtype'lar
 *   nexus-run -n N -- test_phase10_dist collectives_tcp : ayni govde, rank = surec (TCP)
 *   test_phase10_dist mismatch_thread           : dogrulama modu, rank'ler farkli kollektif -> acik hata
 *   nexus-run -n 2 -- test_phase10_dist mismatch_tcp
 *   test_phase10_dist failure NEXUS_RUN         : rank 1 olur -> rank 0 hata ile temiz cikar, baslatici kodu raporlar
 *   test_phase10_dist nccl                      : NCCL arka ucu Windows'ta acik hata verir
 *   nexus-run -n N -- test_phase10_dist busbw_tcp [MAX_BAYT] / test_phase10_dist busbw_thread N [MAX_BAYT]
 *   test_phase10_dist ddp_thread                : dp=4 x micro 2 == dp=1 x micro 8 (FP32, CPU), 100 adim
 *   nexus-run -n 4 -- test_phase10_dist ddp_tcp : ayni, rank = surec
 *   test_phase10_dist overlap_thread            : kova iletisimi geri gecis bitmeden baslar (zaman damgasi)
 *   test_phase10_dist ddp_bench N STEPS         : CPU DDP adim suresi (thread arka ucu), olcekleme
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/half.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus/dist/store.hpp"
#include "nexus/dist/topology.hpp"
#include "nexus/dist/ddp.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/train/lm_trainer.hpp"
#include "nexus_test.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace nd = nexus::dist;
using nexus::DType;

namespace {

std::mutex g_print;
std::atomic<int> g_fail{0};

void check(bool ok, int rank, const std::string& what) {
  if (!ok) {
    g_fail.fetch_add(1);
    std::lock_guard<std::mutex> lock(g_print);
    std::printf("  FAIL  rank %d: %s\n", rank, what.c_str());
  }
}

template <typename T>
T value_of(double v);
template <>
float value_of<float>(double v) { return static_cast<float>(v); }
template <>
double value_of<double>(double v) { return v; }
template <>
std::int32_t value_of<std::int32_t>(double v) { return static_cast<std::int32_t>(v); }
template <>
std::int64_t value_of<std::int64_t>(double v) { return static_cast<std::int64_t>(v); }

struct Bf {
  std::uint16_t b;
};

double as_double(float v) { return v; }
double as_double(double v) { return v; }
double as_double(std::int32_t v) { return static_cast<double>(v); }
double as_double(std::int64_t v) { return static_cast<double>(v); }

// Kucuk tamsayi degerleri: tum dtype'larda (BF16 dahil) toplamlar tam temsil edilir.
double contrib(int rank, std::size_t i) { return static_cast<double>((rank + 1) * ((i % 5) + 1)); }

template <typename T>
void typed_all_reduce(nd::ProcessGroup& g, DType dt, std::size_t count) {
  const int n = g.size();
  const int r = g.rank();
  for (const nd::ReduceOp op : {nd::ReduceOp::Sum, nd::ReduceOp::Max, nd::ReduceOp::Min}) {
    std::vector<T> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = value_of<T>(contrib(r, i));
    g.all_reduce(v.data(), count, dt, op)->wait();
    bool ok = true;
    for (std::size_t i = 0; i < count; ++i) {
      double want = 0.0;
      if (op == nd::ReduceOp::Sum) {
        for (int k = 0; k < n; ++k) want += contrib(k, i);
      } else if (op == nd::ReduceOp::Max) {
        want = contrib(n - 1, i);
      } else {
        want = contrib(0, i);
      }
      ok = ok && as_double(v[i]) == want;
    }
    check(ok, r, std::string("all_reduce ") + nd::to_string(op) + " dt=" + std::to_string(int(dt)) + " n=" + std::to_string(count));
  }
}

void half_all_reduce(nd::ProcessGroup& g, DType dt, std::size_t count) {
  const int n = g.size();
  std::vector<std::uint16_t> v(count);
  auto to = [dt](float f) { return dt == DType::BF16 ? nexus::float_to_bfloat16_bits(f) : nexus::float_to_half_bits(f); };
  auto from = [dt](std::uint16_t b) { return dt == DType::BF16 ? nexus::bfloat16_bits_to_float(b) : nexus::half_bits_to_float(b); };
  for (std::size_t i = 0; i < count; ++i) v[i] = to(static_cast<float>(contrib(g.rank(), i)));
  g.all_reduce(v.data(), count, dt, nd::ReduceOp::Sum)->wait();
  bool ok = true;
  for (std::size_t i = 0; i < count; ++i) {
    double want = 0.0;
    for (int k = 0; k < n; ++k) want += contrib(k, i);
    ok = ok && from(v[i]) == static_cast<float>(want);
  }
  check(ok, g.rank(), std::string("all_reduce ") + (dt == DType::BF16 ? "bf16" : "f16"));
}

void suite(nd::ProcessGroup& g) {
  const int n = g.size();
  const int r = g.rank();
  for (const std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{1000}, std::size_t{65537}}) {
    typed_all_reduce<float>(g, DType::F32, count);
    typed_all_reduce<double>(g, DType::F64, count);
    typed_all_reduce<std::int32_t>(g, DType::I32, count);
    typed_all_reduce<std::int64_t>(g, DType::I64, count);
    half_all_reduce(g, DType::BF16, count);
    half_all_reduce(g, DType::F16, count);
  }
  {  // Avg
    std::vector<float> v(33, static_cast<float>(r));
    g.all_reduce(v.data(), v.size(), DType::F32, nd::ReduceOp::Avg)->wait();
    check(std::fabs(v[0] - (n - 1) / 2.0f) < 1e-6f && std::fabs(v[32] - (n - 1) / 2.0f) < 1e-6f, r, "all_reduce avg");
    bool threw = false;
    std::int32_t x = 1;
    try {
      (void)g.all_reduce(&x, 1, DType::I32, nd::ReduceOp::Avg);
    } catch (const nd::DistError&) {
      threw = true;
    }
    check(threw, r, "tamsayi Avg acik hata");
  }
  {  // reduce_scatter + all_gather
    const std::size_t c = 129;
    std::vector<double> in(static_cast<std::size_t>(n) * c), out(c);
    for (std::size_t i = 0; i < in.size(); ++i) in[i] = contrib(r, i);
    g.reduce_scatter(in.data(), out.data(), c, DType::F64, nd::ReduceOp::Sum)->wait();
    bool ok = true;
    for (std::size_t i = 0; i < c; ++i) {
      double want = 0.0;
      for (int k = 0; k < n; ++k) want += contrib(k, static_cast<std::size_t>(r) * c + i);
      ok = ok && out[i] == want;
    }
    check(ok, r, "reduce_scatter");
    std::vector<std::int64_t> mine(c), all(static_cast<std::size_t>(n) * c);
    for (std::size_t i = 0; i < c; ++i) mine[i] = r * 1000 + static_cast<std::int64_t>(i);
    g.all_gather(mine.data(), all.data(), c, DType::I64)->wait();
    ok = true;
    for (int k = 0; k < n; ++k) {
      for (std::size_t i = 0; i < c; ++i) ok = ok && all[static_cast<std::size_t>(k) * c + i] == k * 1000 + static_cast<std::int64_t>(i);
    }
    check(ok, r, "all_gather");
  }
  for (int root = 0; root < n; ++root) {  // broadcast her kokten
    std::vector<std::int32_t> v(1001, r == root ? root * 7 + 3 : -1);
    g.broadcast(v.data(), v.size(), DType::I32, root)->wait();
    check(v[0] == root * 7 + 3 && v[1000] == root * 7 + 3, r, "broadcast kok " + std::to_string(root));
  }
  {  // all_to_all_v, esit olmayan boyutlar: rank r, rank d'ye (r + d) % 3 + d eleman gonderir
    std::vector<std::size_t> sc(static_cast<std::size_t>(n)), rc(static_cast<std::size_t>(n));
    std::vector<float> in;
    for (int d = 0; d < n; ++d) {
      sc[static_cast<std::size_t>(d)] = static_cast<std::size_t>((r + d) % 3 + d);
      for (std::size_t i = 0; i < sc[static_cast<std::size_t>(d)]; ++i) in.push_back(static_cast<float>(r * 100 + d * 10 + static_cast<int>(i)));
    }
    std::size_t total = 0;
    for (int s = 0; s < n; ++s) {
      rc[static_cast<std::size_t>(s)] = static_cast<std::size_t>((s + r) % 3 + r);
      total += rc[static_cast<std::size_t>(s)];
    }
    std::vector<float> out(total, -1.0f);
    g.all_to_all_v(in.data(), sc, out.data(), rc, DType::F32)->wait();
    bool ok = true;
    std::size_t pos = 0;
    for (int s = 0; s < n; ++s) {
      for (std::size_t i = 0; i < rc[static_cast<std::size_t>(s)]; ++i) ok = ok && out[pos++] == static_cast<float>(s * 100 + r * 10 + static_cast<int>(i));
    }
    check(ok, r, "all_to_all_v (esit olmayan)");
  }
  if (n > 1) {  // send/recv halkasi, iki etiket
    std::vector<std::int64_t> a(17, r), b(17, -1), c2(5, r * 2), d(5, -1);
    const int right = (r + 1) % n;
    const int left = (r - 1 + n) % n;
    auto s1 = g.send(a.data(), a.size(), DType::I64, right, 1);
    auto s2 = g.send(c2.data(), c2.size(), DType::I64, right, 2);
    g.recv(b.data(), b.size(), DType::I64, left, 1)->wait();
    g.recv(d.data(), d.size(), DType::I64, left, 2)->wait();
    s1->wait();
    s2->wait();
    check(b[16] == left && d[4] == left * 2, r, "send/recv");
  }
  g.barrier()->wait();
  {  // alt grup: cift rank'ler
    std::vector<int> evens;
    for (int k = 0; k < n; k += 2) evens.push_back(k);
    nd::GroupPtr sub = g.new_group(evens);
    check((r % 2 == 0) == (sub != nullptr), r, "new_group uyelik");
    if (sub) {
      std::int64_t v = r;
      sub->all_reduce(&v, 1, DType::I64, nd::ReduceOp::Sum)->wait();
      std::int64_t want = 0;
      for (int k : evens) want += k;
      check(v == want && sub->size() == static_cast<int>(evens.size()), r, "alt grup all_reduce");
    }
  }
  for (const int k : {2, 4}) {  // hiyerarsik
    if (n % k != 0 || n == k) continue;
    std::vector<float> v(64 * static_cast<std::size_t>(k));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(contrib(r, i));
    g.all_reduce_hierarchical(v.data(), v.size(), DType::F32, nd::ReduceOp::Sum, k)->wait();
    bool ok = true;
    for (std::size_t i = 0; i < v.size(); ++i) {
      double want = 0.0;
      for (int q = 0; q < n; ++q) want += contrib(q, i);
      ok = ok && v[i] == static_cast<float>(want);
    }
    check(ok && std::strncmp(g.last_algorithm(), "hierarchical", 12) == 0, r,
          "hiyerarsik all_reduce k=" + std::to_string(k) + " (" + g.last_algorithm() + ")");
  }
  g.synchronize();
}

void mismatch(nd::ProcessGroup& g) {
  std::vector<float> v(10, 1.0f);
  bool caught = false;
  std::string msg;
  try {
    if (g.rank() == 0) {
      g.all_reduce(v.data(), v.size(), DType::F32, nd::ReduceOp::Sum)->wait();
    } else {
      g.broadcast(v.data(), v.size(), DType::F32, 0)->wait();
    }
  } catch (const nd::DistError& e) {
    caught = true;
    msg = e.what();
  }
  check(caught && msg.find("uyusmazligi") != std::string::npos, g.rank(), "uyusmazlik acik hata: " + msg);
  if (g.rank() == 0) {
    std::lock_guard<std::mutex> lock(g_print);
    std::printf("  info  %s\n", msg.c_str());
  }
}

int run_threads(int n, const std::function<void(nd::ProcessGroup&)>& body, nd::GroupOptions opt = {}) {
  nd::ThreadWorld world(n);
  std::vector<std::thread> ts;
  for (int r = 0; r < n; ++r) {
    ts.emplace_back([&, r] {
      try {
        nd::GroupPtr g = world.create_group(r, opt);
        body(*g);
      } catch (const std::exception& e) {
        check(false, r, std::string("istisna: ") + e.what());
      }
    });
  }
  for (auto& t : ts) t.join();
  return 0;
}

void busbw(nd::ProcessGroup& g, std::size_t max_bytes) {
  const int n = g.size();
  if (g.rank() == 0) {
    std::printf("  %-12s %10s %10s %10s   (%s, %d rank, float32 sum; busbw = algbw * 2(n-1)/n)\n", "bayt", "us", "algbw GB/s",
                "busbw GB/s", g.backend().c_str(), n);
  }
  for (std::size_t bytes = 8; bytes <= max_bytes; bytes *= 8) {  // 8 B, 64 B, ..., 128 MB, 1 GB
    const std::size_t count = bytes / 4;
    std::vector<float> v(count, 1.0f);
    const int iters = bytes <= (1u << 20) ? 50 : (bytes <= (64u << 20) ? 8 : (bytes <= (256u << 20) ? 3 : 2));
    g.all_reduce(v.data(), count, DType::F32, nd::ReduceOp::Sum)->wait();  // isinma
    g.barrier()->wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) g.all_reduce(v.data(), count, DType::F32, nd::ReduceOp::Sum)->wait();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
    double worst = s;
    g.all_reduce(&worst, 1, DType::F64, nd::ReduceOp::Max)->wait();  // en yavas rank
    if (g.rank() == 0) {
      const double alg = static_cast<double>(bytes) / worst / 1e9;
      std::printf("  %-12zu %10.1f %10.3f %10.3f\n", bytes, worst * 1e6, alg, alg * 2.0 * (n - 1) / n);
    }
  }
}

// ------------------------------------------------------------------------------------------ DDP
namespace nt = nexus::train;
constexpr int kVocab = 11;

nexus::nn::Sequential ddp_model(int blocks, int embed) {
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

std::vector<std::vector<float>> weights(const nexus::nn::Module& m) {
  std::vector<std::vector<float>> out;
  for (auto* p : m.parameters()) out.emplace_back(p->data.data(), p->data.data() + p->data.numel());
  return out;
}

/// Kuresel satir kimligi: adim k'de rank r, satir j -> k * global + r * per_rank + j (dp'den bagimsiz veri).
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

nt::SpikeOptions no_spike() {
  nt::SpikeOptions o;
  o.enabled = false;
  return o;
}

std::unique_ptr<nt::LanguageModelTrainer> ddp_trainer(nexus::nn::Sequential& model, nt::TokenBatchSource& src, int micro,
                                                      int global_rows, int T, nd::GroupPtr g, nd::DdpOptions opt,
                                                      std::vector<double>* grad_rel = nullptr) {
  nt::LanguageTrainerBuilder b(model);
  if (grad_rel != nullptr) {
    // Referans kosu: her adimda parametre gradyan normunun toplam norma orani (en buyugu tutulur).
    b.after_backward([&model, grad_rel](nt::LmBackend&, std::int64_t) {
      const auto ps = model.parameters();
      grad_rel->resize(ps.size(), 0.0);
      std::vector<double> sq(ps.size(), 0.0);
      double total = 0.0;
      for (std::size_t i = 0; i < ps.size(); ++i) {
        for (std::size_t k = 0; k < ps[i]->grad.numel(); ++k) sq[i] += static_cast<double>(ps[i]->grad[k]) * ps[i]->grad[k];
        total += sq[i];
      }
      for (std::size_t i = 0; i < ps.size(); ++i) (*grad_rel)[i] = std::max((*grad_rel)[i], std::sqrt(sq[i] / total));
    });
  }
  b.data(src)
      .optimizer(nt::AdamW{1.0e-3, 0.9, 0.95, 1.0e-8, 0.1})
      .schedule(nt::WarmupCosine{10, 100, 0.1})
      .micro_batch(micro)
      .seq_len(T)
      .global_batch_tokens(static_cast<std::int64_t>(global_rows) * T)
      .grad_clip(1.0)
      .spike_detector(no_spike())
      .log_console(false);
  if (g) b.process_group(g, opt);
  return b.build();
}

/// dp rank'li egitim (rank basina model kopyasi); rank 0'in agirliklari ve kayiplari doner.
struct DdpResult {
  std::vector<std::vector<float>> w;
  std::vector<double> losses;
  nd::DdpStepStats stats;
  double mean_step_ms = 0.0;
  std::vector<double> grad_rel;  // parametre basina max_adim(|g_p| / |g|) (referans kosu)
};

DdpResult reference_run(const nexus::nn::Sequential& init, int global_rows, int T, int steps, int blocks, int embed) {
  DdpResult r;
  nexus::nn::Sequential m = ddp_model(blocks, embed);
  copy_weights(init, m);
  GlobalRows src(0, global_rows, global_rows);
  auto t = ddp_trainer(m, src, global_rows, global_rows, T, nullptr, {}, &r.grad_rel);
  t->train(steps);
  r.w = weights(m);
  for (const auto& h : t->history()) r.losses.push_back(h.loss);
  return r;
}

double max_abs(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t k = 0; k < a[i].size(); ++k) worst = std::max(worst, std::fabs(static_cast<double>(a[i][k]) - b[i][k]));
  return worst;
}

void ddp_rank(nd::ProcessGroup& g, const nexus::nn::Sequential& init, int per_rank, int T, int steps, int blocks, int embed,
              nd::DdpOptions opt, DdpResult* out) {
  nexus::nn::Sequential m = ddp_model(blocks, embed);
  if (g.rank() == 0) copy_weights(init, m);  // digerleri yayinla almali (baslangicta farkli)
  GlobalRows src(g.rank(), per_rank, per_rank * g.size());
  std::shared_ptr<nd::ProcessGroup> sp(&g, [](nd::ProcessGroup*) {});
  auto t = ddp_trainer(m, src, per_rank, per_rank * g.size(), T, sp, opt);
  t->train(steps);
  if (out != nullptr) {
    out->w = weights(m);
    double ms = 0.0;
    for (const auto& h : t->history()) {
      out->losses.push_back(h.loss);
      ms += h.step_ms;
    }
    out->mean_step_ms = ms / static_cast<double>(t->history().size());
    out->stats = t->history().back().ddp;
  }
  // tum rank'ler ayni agirliklarda mi?
  std::vector<float> flat;
  for (const auto& v : weights(m)) flat.insert(flat.end(), v.begin(), v.end());
  std::vector<float> mx = flat, mn = flat;
  g.all_reduce(mx.data(), mx.size(), DType::F32, nd::ReduceOp::Max)->wait();
  g.all_reduce(mn.data(), mn.size(), DType::F32, nd::ReduceOp::Min)->wait();
  bool same = true;
  for (std::size_t i = 0; i < flat.size(); ++i) same = same && mx[i] == mn[i];
  check(same, g.rank(), "tum rank'lerin agirliklari bit bit ayni");
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "collectives_thread";
  try {
    if (mode == "collectives_thread") {
      nexus_test::section("test_collectives_thread: 1..8 rank, tum islemler, F32/F64/I32/I64/BF16/F16");
      for (int n = 1; n <= 8; ++n) {
        run_threads(n, suite);
        std::printf("  info  %d rank: %s\n", n, g_fail.load() == 0 ? "tamam" : "HATA");
      }
      nexus_test::expect(g_fail.load() == 0, "tum kollektifler 1..8 rank'te dogru (thread)");
      nd::GroupOptions v;
      v.validate = true;
      run_threads(4, suite, v);
      nexus_test::expect(g_fail.load() == 0, "dogrulama modu acikken ayni sonuc (4 rank)");
    } else if (mode == "collectives_tcp" || mode == "mismatch_tcp" || mode == "busbw_tcp" || mode == "die") {
      nd::GroupOptions opt;
      opt.validate = mode == "mismatch_tcp";
      opt.timeout = std::chrono::milliseconds(mode == "die" ? 5000 : 60000);
      nd::GroupPtr g = nd::init_process_group("tcp", opt);
      if (mode == "collectives_tcp") {
        suite(*g);
        if (g->rank() == 0) std::printf("  info  tcp %d rank\n", g->size());
      } else if (mode == "mismatch_tcp") {
        mismatch(*g);
      } else if (mode == "busbw_tcp") {
        busbw(*g, argc > 2 ? std::stoull(argv[2]) : (256u << 20));
      } else {  // die: rank 1 kod 3 ile olur; digerleri iletisim hatasini gorup kod 4 ile cikar
        g->barrier()->wait();
        if (g->rank() == 1) {
          std::fflush(stdout);
          std::_Exit(3);
        }
        std::vector<float> v(1 << 16, 1.0f);
        try {
          for (int i = 0; i < 100; ++i) g->all_reduce(v.data(), v.size(), DType::F32, nd::ReduceOp::Sum)->wait();
          std::printf("  FAIL  rank %d: olu rank'e ragmen kollektif bitti\n", g->rank());
          std::_Exit(5);
        } catch (const nd::DistError& e) {
          std::printf("  info  rank %d hatayi gordu: %s\n", g->rank(), e.what());
          std::_Exit(4);
        }
      }
      const int fails = g_fail.load();
      if (fails == 0 && g->rank() == 0) std::printf("  PASS  %s (%d rank)\n", mode.c_str(), g->size());
      return fails == 0 ? 0 : 1;
    } else if (mode == "mismatch_thread") {
      nexus_test::section("test_mismatched_collective_detected (thread)");
      nd::GroupOptions v;
      v.validate = true;
      v.timeout = std::chrono::milliseconds(5000);
      run_threads(2, mismatch, v);
      nexus_test::expect(g_fail.load() == 0, "farkli kollektif kilitlenme yerine acik hata");
    } else if (mode == "failure" && argc > 3) {
      nexus_test::section("test_rank_failure_clean_exit: rank 1 olur (nexus-run, 3 surec, tcp)");
      const auto log = std::filesystem::temp_directory_path() /
          ("nexus_rank_failure_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".log");
      std::string cmd = std::string("\"") + argv[2] + "\" --nproc-per-node 3 --master-port 29611 --grace-s 20 -- \"" +
                        argv[3] + "\" die 2>\"" + log.string() + "\"";
#if defined(_WIN32)
      cmd = "\"" + cmd + "\"";
#endif
      const auto t0 = std::chrono::steady_clock::now();
      const int status = std::system(cmd.c_str());
#if defined(_WIN32)
      const int rc = status;
#else
      const int rc = status == -1 ? -1 : WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::ifstream input(log);
      const std::string output((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
      std::printf("%s",output.c_str());
      const auto first = output.find("nexus-run: rank ");
      int observed_rank = -1, observed_code = -1;
      bool observed = false;
      if (first != std::string::npos) {
        std::istringstream parser(output.substr(first + std::strlen("nexus-run: rank ")));
        std::string separator;
        observed = static_cast<bool>(parser >> observed_rank >> separator >> observed_code) && separator=="kod";
      }
      nexus_test::info("baslatici_donus", rc);
      nexus_test::info("sure_s", s);
      nexus_test::expect(observed && (observed_code==3 || observed_code==4), "ilk gozlenen hata kaydi mevcut");
      nexus_test::expect(observed && rc==observed_code, "baslatici ilk gozlenen hata kodunu korur");
      nexus_test::expect(s < 15.0, "digerleri zaman asimi/baglanti hatasiyla grace suresinden once cikar");
    } else if (mode == "topology") {
      nexus_test::section("topoloji (10.5) + LR olcek yardimcisi (10.4)");
      const nd::ClusterTopology one = nd::discover_cluster(4, 1, 4);
      const nd::ClusterTopology two = nd::discover_cluster(8, 5, 4);
      std::printf("  info  %s\n  info  %s\n", one.describe().c_str(), two.describe().c_str());
      nexus_test::expect(one.algorithm == "ring" && two.algorithm == "hierarchical" && two.node_rank == 1 && two.local_rank == 1,
                         "dugum ici/arasi ayrimi ve algoritma onerisi");
      nexus_test::expect(one.logical_cpus >= 1, "donanim kesfi");
      nexus_test::expect(nd::scale_learning_rate(6e-4, 256, 1024) == 2.4e-3 &&
                             std::fabs(nd::scale_learning_rate(6e-4, 256, 1024, true) - 1.2e-3) < 1e-15,
                         "lineer ve karekok LR olcekleme");
    } else if (mode == "nccl") {
      nexus_test::section("nccl arka ucu: acik hata");
      nd::TcpStore store("127.0.0.1", 0, 1, true);
      bool threw = false;
      try {
        (void)nd::create_nccl_group(store, 0, 1);
      } catch (const nd::DistError& e) {
        threw = true;
        std::printf("  info  %s\n", e.what());
      }
      nexus_test::expect(threw, "NCCL sessiz yedek yerine acik hata");
    } else if (mode == "ddp_thread" || mode == "ddp_tcp") {
      const int T = 12;
      const int steps = 100;
      nexus::nn::Sequential init = ddp_model(1, 16);
      DdpResult ref = reference_run(init, 8, T, steps, 1, 16);
      DdpResult got;
      if (mode == "ddp_thread") {
        nexus_test::section("test_ddp_equivalence: dp=4 x micro 2 == dp=1 x micro 8 (FP32, CPU, thread), 100 adim");
        nd::GroupOptions go;
        run_threads(4, [&](nd::ProcessGroup& g) { ddp_rank(g, init, 2, T, steps, 1, 16, {}, g.rank() == 0 ? &got : nullptr); }, go);
      } else {
        nd::GroupPtr g = nd::init_process_group("tcp");
        if (g->size() != 4) throw std::runtime_error("ddp_tcp 4 rank ister");
        ddp_rank(*g, init, 2, T, steps, 1, 16, {}, g->rank() == 0 ? &got : nullptr);
        if (g->rank() != 0) return g_fail.load() == 0 ? 0 : 1;
        nexus_test::section("test_ddp_equivalence_tcp: dp=4 surec x micro 2 == dp=1 x micro 8, 100 adim");
      }
      // Gercek gradyani matematiksel olarak sifir olan parametreler (dikkatte anahtar bias'i: softmax tum
      // anahtarlara eklenen sabite duyarsiz) yalniz yuvarlama gurultusu tasir; Adam bu gurultuyu ~lr adima
      // buyutur, toplama sirasi farkli iki kosuda (dp=1 / dp=4) rastgele yurur. Ayri raporlanir.
      double w = 0.0;
      double noise_only = 0.0;
      for (std::size_t i = 0; i < ref.w.size(); ++i) {
        double d = 0.0;
        for (std::size_t k = 0; k < ref.w[i].size(); ++k) d = std::max(d, std::fabs(static_cast<double>(got.w[i][k]) - ref.w[i][k]));
        const bool noise = ref.grad_rel[i] < 1.0e-6;
        if (noise) {
          noise_only = std::max(noise_only, d);
          std::printf("  info  param %zu (%zu oge): gradyani yalniz gurultu (max |g_p|/|g| = %.1e), fark %.2e\n", i,
                      ref.w[i].size(), ref.grad_rel[i], d);
        } else {
          w = std::max(w, d);
        }
      }
      nexus_test::info("gurultu_parametresi_max_fark", noise_only);
      double lmax = 0.0;
      for (std::size_t i = 0; i < ref.losses.size(); ++i) lmax = std::max(lmax, std::fabs(got.losses[i] - ref.losses[i]));
      nexus_test::info("agirlik_max_abs_fark_100_adim", w);
      nexus_test::info("kayip_max_abs_fark", lmax);
      nexus_test::info("kayip_0", ref.losses.front());
      nexus_test::info("kayip_99", ref.losses.back());
      nexus_test::expect(g_fail.load() == 0, "rank'ler arasi agirliklar ayni");
      nexus_test::expect(w <= 1.0e-6, "100 adim sonra agirlik farki <= 1e-6");
      nexus_test::expect(lmax <= 1.0e-5, "adim kayiplari ayni (<= 1e-5)");
    } else if (mode == "overlap_thread") {
      nexus_test::section("test_overlap_occurs (CPU): 2 rank, 6 blok, kova 64 KB");
      nexus::nn::Sequential init = ddp_model(6, 64);
      DdpResult got;
      nd::DdpOptions opt;
      opt.bucket_bytes = 64 << 10;
      run_threads(2, [&](nd::ProcessGroup& g) { ddp_rank(g, init, 4, 32, 4, 6, 64, opt, g.rank() == 0 ? &got : nullptr); });
      const nd::DdpStepStats& st = got.stats;
      nexus_test::info("kova", st.buckets);
      nexus_test::info("geri_gecis_ms", st.backward_ms);
      nexus_test::info("ilk_iletisim_baslangici_ms", st.comm_first_start_ms);
      nexus_test::info("son_iletisim_bitisi_ms", st.comm_last_end_ms);
      nexus_test::info("iletisim_mesgul_ms", st.comm_busy_ms);
      nexus_test::info("ortusen_ms", st.comm_overlapped_ms);
      nexus_test::expect(st.buckets > 1 && st.comm_first_start_ms >= 0.0 && st.comm_first_start_ms < st.backward_ms,
                         "ilk kova iletisimi geri gecis bitmeden basladi");
      nexus_test::expect(st.comm_overlapped_ms > 0.0, "iletisimin bir kismi geri gecisle ortusuyor");
    } else if (mode == "ddp_bench" && argc > 3) {
      const int n = std::stoi(argv[2]);
      const int steps = std::stoi(argv[3]);
      nexus::nn::Sequential init = ddp_model(4, 128);
      DdpResult got;
      if (n == 1) {
        nexus::nn::Sequential m = ddp_model(4, 128);
        copy_weights(init, m);
        GlobalRows src(0, 4, 4);
        auto t = ddp_trainer(m, src, 4, 4, 64, nullptr, {});
        t->train(steps);
        double ms = 0.0;
        for (std::size_t i = 2; i < t->history().size(); ++i) ms += t->history()[i].step_ms;
        got.mean_step_ms = ms / static_cast<double>(t->history().size() - 2);
      } else {
        run_threads(n, [&](nd::ProcessGroup& g) { ddp_rank(g, init, 4, 64, steps, 4, 128, {}, g.rank() == 0 ? &got : nullptr); });
      }
      std::printf("{\"ranks\":%d,\"mean_step_ms\":%.2f,\"rows_per_rank\":4,\"T\":64,\"backward_ms\":%.2f,\"comm_busy_ms\":%.2f,"
                  "\"overlapped_ms\":%.2f}\n", n, got.mean_step_ms, got.stats.backward_ms, got.stats.comm_busy_ms,
                  got.stats.comm_overlapped_ms);
      return 0;
    } else if (mode == "busbw_thread" && argc > 2) {
      const int n = std::stoi(argv[2]);
      const std::size_t max_bytes = argc > 3 ? std::stoull(argv[3]) : (256u << 20);
      run_threads(n, [&](nd::ProcessGroup& g) { busbw(g, max_bytes); });
      return 0;
    } else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase10_dist");
}
