/**
 * @file test_phase7_memory.cpp
 * @brief FAZ 7 CPU testleri: non-reentrant checkpoint (bit-bit gradyan, dropout dahil, ic ice,
 *        retain_graph), kaydedilen bayt olcumu ve bellek planlayicisi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/model.hpp"
#include "nexus/memory/planner.hpp"
#include "nexus_test.hpp"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace ag = nexus::autograd;
namespace lm = nexus::lm;
using ag::Shape;
using ag::Variable;

namespace {

std::vector<std::int32_t> tokens(int n, int vocab, std::uint64_t seed) {
  std::vector<std::int32_t> t(static_cast<std::size_t>(n));
  std::uint64_t s = seed;
  for (auto& x : t) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    x = static_cast<std::int32_t>((s >> 33) % static_cast<std::uint64_t>(vocab));
  }
  return t;
}

struct Run {
  float loss = 0.0f;
  std::vector<std::vector<float>> grads;
  std::size_t saved = 0;
};

Run run_lm(lm::RecomputeKind kind, int every) {
  lm::LmConfig c;
  c.vocab_size = 101;
  c.n_layer = 4;
  c.d_model = 32;
  c.n_head = 4;
  c.n_kv_head = 2;
  c.d_ff = 64;
  c.max_position = 64;
  c.resid_dropout = 0.1f;
  c.attn_dropout = 0.1f;
  c.qk_norm = true;
  c.recompute = kind;
  c.recompute_every = every;
  const lm::LanguageModel model(c);
  lm::ForwardOptions o;
  o.training = true;
  o.dropout_seed = 77;
  Run r;
  Variable loss;
  {
    ag::SavedBytesCounter counter;
    loss = model.loss(tokens(2 * 32, c.vocab_size, 3), 2, 32, o);
    r.saved = counter.bytes();
  }
  loss.backward();
  r.loss = loss.values()[0];
  for (const lm::NamedParam& p : model.parameters()) {
    r.grads.push_back(p.value.grad().values());
  }
  return r;
}

void test_checkpoint_lm() {
  nexus_test::section("test_checkpoint_grad_equal: lm (dropout 0.1 artik + dikkat), none/full/every_k");
  const Run base = run_lm(lm::RecomputeKind::None, 1);
  for (const auto& [kind, every, name] : {std::tuple{lm::RecomputeKind::Full, 1, "full"},
                                          std::tuple{lm::RecomputeKind::EveryK, 2, "every_k=2"}}) {
    const Run r = run_lm(kind, every);
    nexus_test::expect(r.loss == base.loss, std::string(name) + ": kayip bit-bit ayni");
    nexus_test::expect(r.grads == base.grads, std::string(name) + ": tum gradyanlar bit-bit ayni");
    std::printf("  info  %-10s kaydedilen aktivasyon %.1f KiB (none: %.1f KiB, %.1f%%)\n", name, r.saved / 1024.0,
                base.saved / 1024.0, 100.0 * static_cast<double>(r.saved) / static_cast<double>(base.saved));
    nexus_test::expect(r.saved < base.saved, std::string(name) + ": daha az kayit");
  }
}

void test_checkpoint_semantics() {
  nexus_test::section("checkpoint: ic ice, kismi backward, retain_graph");
  Variable w1 = Variable::randn(Shape{{6, 6}}, 1, true);
  Variable w2 = Variable::randn(Shape{{6, 6}}, 2, true);
  const Variable x0 = Variable::randn(Shape{{4, 6}}, 3, false);
  auto inner = [&](const std::vector<Variable>& in) { return ag::tanh(ag::matmul(in[0], w2)); };
  auto outer = [&](const std::vector<Variable>& in) {
    const Variable h = ag::gelu(ag::matmul(in[0], w1));
    return ag::checkpoint(inner, {h});
  };
  auto grads = [&](bool use_ckpt) {
    w1.zero_grad();
    w2.zero_grad();
    const Variable x = Variable::make(x0.values(), x0.shape(), true);
    const Variable y = use_ckpt ? ag::checkpoint(outer, {x}) : inner({ag::gelu(ag::matmul(x, w1))});
    const Variable loss = ag::sum(ag::mul(y, y));
    loss.backward(Variable::make({1.0f}, loss.shape(), false), true);  // retain_graph
    loss.backward(Variable::make({1.0f}, loss.shape(), false));        // ikinci gecis ayni grafi okur
    return std::vector<std::vector<float>>{w1.grad().values(), w2.grad().values(), x.grad().values()};
  };
  nexus_test::expect(grads(true) == grads(false), "ic ice checkpoint + retain_graph: gradyanlar bit-bit ayni");
  // Kismi backward: yalniz bir dal.
  w1.zero_grad();
  const Variable x = Variable::make(x0.values(), x0.shape(), true);
  const Variable a = ag::checkpoint([&](const std::vector<Variable>& in) { return ag::matmul(in[0], w1); }, {x});
  ag::sum(a).backward();
  const auto g1 = w1.grad().values();
  w1.zero_grad();
  const Variable x2 = Variable::make(x0.values(), x0.shape(), true);
  ag::sum(ag::matmul(x2, w1)).backward();
  nexus_test::expect(g1 == w1.grad().values(), "kismi backward: gradyan ayni");
}

// ---- Planlayici ---------------------------------------------------------------------------
void test_planner() {
  nexus_test::section("test_planner_no_overlap: yasam suresi cakisan tensörler ortusmez");
  namespace nm = nexus::memory;
  std::mt19937_64 rng(5);
  for (int trial = 0; trial < 50; ++trial) {
    std::vector<nm::Lifetime> items;
    const int n = 20 + static_cast<int>(rng() % 200);
    for (int i = 0; i < n; ++i) {
      nm::Lifetime t;
      t.begin = static_cast<int>(rng() % 400);
      t.end = t.begin + 1 + static_cast<int>(rng() % 120);
      t.bytes = 1 + rng() % (1u << 20);
      items.push_back(t);
    }
    const nm::Plan plan = nm::plan_offsets(items, 256);
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
      ok = ok && plan.offsets[static_cast<std::size_t>(i)] % 256 == 0 &&
           plan.offsets[static_cast<std::size_t>(i)] + items[static_cast<std::size_t>(i)].bytes <= plan.arena_bytes;
      for (int j = i + 1; j < n && ok; ++j) {
        const auto& a = items[static_cast<std::size_t>(i)];
        const auto& b = items[static_cast<std::size_t>(j)];
        const bool live = a.begin < b.end && b.begin < a.end;
        const std::uint64_t oa = plan.offsets[static_cast<std::size_t>(i)];
        const std::uint64_t ob = plan.offsets[static_cast<std::size_t>(j)];
        const bool space = oa < ob + b.bytes && ob < oa + a.bytes;
        ok = !(live && space);
      }
    }
    nexus_test::expect(ok, "deneme " + std::to_string(trial) + ": ortusme yok, hizali, arena icinde");
    nexus_test::expect(plan.arena_bytes >= plan.peak_live_bytes, "arena >= tepe canli bayt (alt sinir)");
    if (trial == 0) {
      std::printf("  info  n=%d arena=%.2f MiB tepe canli=%.2f MiB (asim %.2f%%)\n", n, plan.arena_bytes / 1048576.0,
                  plan.peak_live_bytes / 1048576.0,
                  100.0 * (static_cast<double>(plan.arena_bytes) / plan.peak_live_bytes - 1.0));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    if (mode == "all" || mode == "checkpoint") {
      test_checkpoint_lm();
      test_checkpoint_semantics();
    }
    if (mode == "all" || mode == "planner") test_planner();
  } catch (const std::exception& e) {
    std::printf("  FAIL  %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("faz7-memory");
}
