/**
 * @file test_phase15_durability.cpp
 * @brief FAZ 15: dayaniklilik — tam durum kaydi, RNG, parcali kayit, yeniden parcalama, cokme, bekci, elastik.
 *
 *   test_phase15_durability exact    : test_save_load_exact — 6 adim + kayit (senkron ve eszamansiz), YENI SURECTE
 *                                      devam 6 adim == kesintisiz 12 adim: kayip, agirlik, gradyan BIT BIT
 *   test_phase15_durability rng      : test_rng_restore — dropout maskesi (seed, offset)'ten birebir; FAZ 9 kaydi
 *                                      (RNG'siz) devamda farkli maske verir (negatif kontrol), FAZ 15 kaydi ayni
 *   test_phase15_durability sharded  : test_sharded_save_load — 4 ve 8 rank ZeRO-3: rank basina kendi parcasi
 *                                      (toplama yok), yukleme + devam == kesintisiz (bit bit)
 *   test_phase15_durability reshard  : test_reshard_2_to_4 — 2 rank kaydi 4 rankta (ve 4 -> 2) devam eder
 *   test_phase15_durability crash    : test_crash_injection — kaydin 3 noktasinda surec oldurulur: son gecerli kayit
 *                                      saglam, devam kesintisizle ayni; bozuk dosya / surum acik hata
 *   test_phase15_durability watchdog : test_watchdog_timeout — takilan rank tespit + rapor, diger rank'ler zaman
 *                                      asimiyla cikar (asili kalmaz); straggler tespiti
 *   test_phase15_durability elastic <nexus-run> : nexus-run --max-restarts 1 --restart-nproc 2: 4 tcp surecinden biri
 *                                      adim 5'te olur -> 2 surecle adim 4 kaydindan (4 -> 2 yeniden parcalama) devam
 *   test_phase15_durability bench    : kayit GB/s, adim suresine etkisi (senkron / eszamansiz), boyut
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/ckpt/checkpoint.hpp"
#include "nexus/dist/fsdp.hpp"
#include "nexus/dist/process_group.hpp"
#include "nexus/dist/watchdog.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/train/lm_trainer.hpp"
#include "nexus_test.hpp"


#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace nt = nexus::train;
namespace nd = nexus::dist;
namespace ck = nexus::ckpt;

namespace {

constexpr int kVocab = 13;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

void set_env(const char* k, const std::string& v) {
#if defined(_WIN32)
  _putenv_s(k, v.c_str());
#else
  if (v.empty()) unsetenv(k);
  else setenv(k, v.c_str(), 1);
#endif
}

int run_status(const std::string& command) {
#if defined(_WIN32)
  const int rc = std::system(("\"" + command + "\"").c_str());
  return rc;
#else
  const int rc = std::system(command.c_str());
  return WIFEXITED(rc) ? WEXITSTATUS(rc) : 128;
#endif
}

fs::path scratch(const std::string& name) {
  const fs::path d = fs::temp_directory_path() / ("nexus_faz15_" + name);
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d);
  return d;
}

nexus::nn::Sequential make_model(float dropout, int blocks = 1, int embed = 16) {
  nexus::nn::Sequential m;
  m.add(std::make_shared<nexus::nn::Embedding>(kVocab, embed, -1));
  m.add(std::make_shared<nexus::nn::PositionalEncoding>(embed, 128, false));
  if (dropout > 0.0f) m.add(std::make_shared<nexus::nn::Dropout>(dropout));
  for (int i = 0; i < blocks; ++i) m.add(std::make_shared<nexus::nn::CausalTransformerBlock>(embed, 2, 2 * embed));
  m.add(std::make_shared<nexus::nn::Linear>(static_cast<std::size_t>(embed), static_cast<std::size_t>(kVocab), true));
  return m;
}

void write_weights(const fs::path& p, const nexus::nn::Module& m) {
  std::ofstream o(p, std::ios::binary);
  for (auto* q : m.parameters()) o.write(reinterpret_cast<const char*>(q->data.data()), static_cast<std::streamsize>(q->data.numel() * 4));
}
void read_weights(const fs::path& p, nexus::nn::Module& m) {
  std::ifstream i(p, std::ios::binary);
  for (auto* q : m.parameters()) i.read(reinterpret_cast<char*>(q->data.data()), static_cast<std::streamsize>(q->data.numel() * 4));
  if (!i) throw std::runtime_error("baslangic agirliklari okunamadi: " + p.string());
}
std::vector<float> flat(const nexus::nn::Module& m, bool grad) {
  std::vector<float> v;
  for (auto* q : m.parameters()) {
    const auto& t = grad ? q->grad : q->data;
    v.insert(v.end(), t.data(), t.data() + t.numel());
  }
  return v;
}

/// Kuresel imlecli kaynak: adim c'nin kuresel satirlari [c*G, (c+1)*G); rank r satir [r*per, (r+1)*per).
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
  bool global_cursor() const override { return true; }

 private:
  int rank_, per_rank_, global_;
  std::uint64_t calls_ = 0;
};

struct Cfg {
  int global_rows = 8;
  int T = 12;
  int steps = 12;
  int zero = -1;  // -1: DDP / tek rank, 0..3 ZeRO
  float dropout = 0.0f;
  std::string dir;  // bos: kayit yok
  std::int64_t interval = 0;
  bool async = false;
};

std::unique_ptr<nt::LanguageModelTrainer> make_trainer(nexus::nn::Sequential& m, GlobalRows& src, const Cfg& c,
                                                       nd::GroupPtr g, std::shared_ptr<nd::Watchdog> wd = nullptr,
                                                       std::function<void(nt::LmBackend&, std::int64_t)> hook = {}) {
  const int dp = g ? g->size() : 1;
  nt::SpikeOptions no_spike;
  no_spike.enabled = false;
  nt::LanguageTrainerBuilder b(m);
  b.data(src)
      .optimizer(nt::AdamW{3.0e-3, 0.9, 0.95, 1.0e-8, 0.1})
      .schedule(nt::WarmupCosine{3, 40, 0.1})
      .micro_batch(c.global_rows / dp)
      .seq_len(c.T)
      .global_batch_tokens(static_cast<std::int64_t>(c.global_rows) * c.T)
      .grad_clip(1.0)
      .seed(4242)
      .spike_detector(no_spike)
      .log_console(false);
  if (!c.dir.empty()) {
    ck::CheckpointOptions o;
    o.dir = c.dir;
    o.interval = c.interval;
    o.keep = 2;
    o.async = c.async;
    o.timeout = std::chrono::seconds(60);
    b.checkpointing(o);
  }
  if (g && dp > 1) {
    nt::LanguageTrainerBuilder::Parallel p;
    p.zero = c.zero;
    b.parallel(g, p);
  }
  if (wd) b.watchdog(wd);
  if (hook) b.after_backward(hook);
  return b.build();
}

struct Out {
  std::vector<double> losses;  // bu calismanin adimlari
  std::vector<float> w, g;
  std::int64_t first_step = 0;
};

Out collect(nt::LanguageModelTrainer& t, const nexus::nn::Module& m, std::int64_t first) {
  Out o;
  o.first_step = first;
  for (const auto& h : t.history()) o.losses.push_back(h.loss);
  if (nd::FsdpEngine* e = t.backend().fsdp()) {
    o.w = e->full_parameters();
  } else {
    o.w = flat(m, false);
    o.g = flat(m, true);
  }
  return o;
}

/// Tek rank ya da ThreadWorld(n) kosusu; rank 0 sonucu.
Out run(int n, const fs::path& init, const Cfg& c, bool resume, std::int64_t until, std::vector<std::string>* errors = nullptr,
        std::function<void(nt::LmBackend&, std::int64_t, int)> hook = {}, std::vector<std::shared_ptr<nd::Watchdog>>* wds = nullptr,
        nd::GroupOptions gopt = {}) {
  Out res;
  std::mutex mu;
  auto body = [&](int r, nd::GroupPtr g) {
    nexus::nn::Sequential m = make_model(c.dropout);
    if (!resume) read_weights(init, m);
    else
      for (auto* q : m.parameters()) q->data.zero();  // kaydin gercekten yuklendigini gostermek icin
    GlobalRows src(r, c.global_rows / n, c.global_rows);
    std::function<void(nt::LmBackend&, std::int64_t)> h;
    if (hook) h = [&, r](nt::LmBackend& b, std::int64_t s) { hook(b, s, r); };
    auto t = make_trainer(m, src, c, g, wds ? (*wds)[static_cast<std::size_t>(r)] : nullptr, h);
    std::int64_t first = 0;
    if (resume) {
      if (!t->resume_latest()) throw std::runtime_error("kayit bulunamadi");
      first = t->step();
    }
    t->train(until);
    Out o = collect(*t, m, first);
    if (r == 0) {
      std::lock_guard<std::mutex> lock(mu);
      res = std::move(o);
    }
  };
  if (n == 1) {
    body(0, nullptr);
    return res;
  }
  nd::ThreadWorld world(n);
  std::vector<std::thread> ts;
  for (int r = 0; r < n; ++r) {
    ts.emplace_back([&, r] {
      try {
        nd::GroupPtr g = world.create_group(r, gopt);
        body(r, g);
        g->synchronize();
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mu);
        if (errors) errors->push_back("rank " + std::to_string(r) + ": " + e.what());
        else std::printf("  FAIL  rank %d istisna: %s\n", r, e.what());
      }
    });
  }
  for (auto& t : ts) t.join();
  return res;
}

bool same_tail(const Out& ref, const Out& got, double* worst = nullptr) {
  const std::size_t off = static_cast<std::size_t>(got.first_step);
  if (ref.losses.size() != off + got.losses.size()) return false;
  double w = 0.0;
  for (std::size_t i = 0; i < got.losses.size(); ++i)
    w = std::max(w, std::fabs(ref.losses[off + i] - got.losses[i]) / std::max(1e-12, std::fabs(ref.losses[off + i])));
  if (worst) *worst = w;
  return w == 0.0 && ref.w == got.w && ref.g == got.g;
}

void save_out(const fs::path& p, const Out& o) {
  std::ofstream f(p, std::ios::binary);
  auto vec = [&](const auto& v) {
    const std::uint64_t n = v.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * sizeof(v[0])));
  };
  f.write(reinterpret_cast<const char*>(&o.first_step), 8);
  vec(o.losses);
  vec(o.w);
  vec(o.g);
}
Out load_out(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  Out o;
  auto vec = [&](auto& v) {
    std::uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(v[0])));
  };
  f.read(reinterpret_cast<char*>(&o.first_step), 8);
  vec(o.losses);
  vec(o.w);
  vec(o.g);
  if (!f) throw std::runtime_error("alt surec ciktisi okunamadi: " + p.string());
  return o;
}

std::string q(const fs::path& p) { return "\"" + p.string() + "\""; }

// ==========================================================================================

// alt surec: exact_child <dir> <init> <A|B> <async 0|1> [out]
int exact_child(int argc, char** argv) {
  if (argc < 6) return 2;
  Cfg c;
  c.dropout = 0.1f;
  c.dir = argv[2];
  c.interval = 3;
  c.async = std::string(argv[5]) == "1";
  if (std::string(argv[4]) == "A") {
    (void)run(1, argv[3], c, false, 6);
    return 0;
  }
  const Out o = run(1, argv[3], c, true, 12);
  save_out(argv[6], o);
  return 0;
}

void test_exact(const char* exe) {
  nexus_test::section("test_save_load_exact: 6 adim + kayit, yeni surecte 6 adim == kesintisiz 12");
  const fs::path dir = scratch("exact");
  const fs::path init = dir / "init.bin";
  write_weights(init, make_model(0.1f));
  Cfg c;
  c.dropout = 0.1f;
  const Out ref = run(1, init, c, false, 12);
  for (int async = 0; async < 2; ++async) {
    const fs::path cd = dir / (async ? "async" : "sync");
    const fs::path out = dir / (async ? "async.bin" : "sync.bin");
    const std::string base = std::string("\"") + exe + "\" exact_child " + q(cd) + " " + q(init);
    const int a = run_status(base + " A " + std::to_string(async));
    const int b = run_status(base + " B " + std::to_string(async) + " " + q(out));
    if (a != 0 || b != 0) {
      nexus_test::expect(false, "alt surecler calisti");
      continue;
    }
    const Out got = load_out(out);
    ck::CheckpointManager probe(ck::CheckpointOptions{cd.string(), 0, 2, false, std::chrono::seconds(5)}, 1, 1);
    const auto steps = probe.committed_steps();  // keep=2: 9 ve 12
    const ck::Manifest m = ck::read_manifest(ck::CheckpointManager::step_dir(cd, steps.back()));
    std::printf("  info  %s: kayitlar 3, 6 (A) + 9, 12 (B), keep=2 -> %zu kaldi; devam adim %lld; manifest %s v%u, %zu dosya, %llu bayt, sha %s...\n",
                async ? "eszamansiz" : "senkron", steps.size(), static_cast<long long>(got.first_step), m.format.c_str(), m.version,
                m.files.size(), static_cast<unsigned long long>(m.files[0].bytes), m.files[0].sha256.substr(0, 12).c_str());
    double worst = 0.0;
    const bool ok = same_tail(ref, got, &worst);
    std::printf("  info  kayip farki %.1e, agirlik %s, gradyan %s\n", worst, ref.w == got.w ? "bit bit" : "FARKLI",
                ref.g == got.g ? "bit bit" : "FARKLI");
    nexus_test::expect(got.first_step == 6 && ok,
                       std::string("test_save_load_exact (") + (async ? "eszamansiz" : "senkron") +
                           "): yeni surecte devam == kesintisiz (kayip, agirlik, gradyan bit bit; dropout 0.1)");
  }
}

void test_rng() {
  nexus_test::section("test_rng_restore: dropout maskesi (seed, offset)'ten birebir");
  nexus::nn::Dropout d(0.3f);
  nexus_model::Tensor x = nexus_model::Tensor::zeros({4, 64});
  for (std::size_t i = 0; i < x.numel(); ++i) x.data()[i] = 1.0f;
  auto mask = [&](std::uint64_t seed, std::uint64_t off, int call) {
    nexus::nn::set_dropout_rng(seed, off);
    nexus_model::Tensor y;
    for (int k = 0; k <= call; ++k) y = d.forward(x);
    return std::vector<float>(y.data(), y.data() + y.numel());
  };
  const auto a = mask(7, 100, 1), b = mask(7, 100, 1), c2 = mask(7, 101, 1), e = mask(7, 100, 0);
  nexus::nn::clear_dropout_rng();
  nexus_test::expect(a == b && a != c2 && a != e, "ayni (seed, offset, cagri) -> ayni maske; offset ya da cagri degisince farkli");

  const fs::path dir = scratch("rng");
  const fs::path init = dir / "init.bin";
  write_weights(init, make_model(0.2f));
  Cfg c;
  c.dropout = 0.2f;
  const Out ref = run(1, init, c, false, 10);
  // FAZ 9 kaydi (RNG sayaci yok): devamda maske dizisi bastan baslar
  Out old;
  {
    nexus::nn::Sequential m = make_model(c.dropout);
    read_weights(init, m);
    GlobalRows src(0, 8, 8);
    auto t = make_trainer(m, src, c, nullptr);
    t->train(5);
    t->save((dir / "faz9.nxlm").string());
  }
  {
    nexus::nn::Sequential m = make_model(c.dropout);
    GlobalRows src(0, 8, 8);
    auto t = make_trainer(m, src, c, nullptr);
    t->resume((dir / "faz9.nxlm").string());
    t->train(10);
    old = collect(*t, m, 5);
  }
  c.dir = (dir / "ck").string();
  c.interval = 5;
  (void)run(1, init, c, false, 5);
  const Out neu = run(1, init, c, true, 10);
  double wo = 0.0, wn = 0.0;
  const bool old_same = same_tail(ref, old, &wo);
  const bool new_same = same_tail(ref, neu, &wn);
  std::printf("  info  FAZ 9 kaydi (RNG yok): kayip farki %.2e; FAZ 15 kaydi: %.1e\n", wo, wn);
  nexus_test::expect(!old_same && wo > 0.0, "negatif kontrol: RNG sayaci kaydedilmezse devamda maske farkli (kayip farkli)");
  nexus_test::expect(new_same, "test_rng_restore: FAZ 15 kaydiyla devamda ayni maske dizisi (kayip ve agirlik bit bit)");
}

void test_sharded() {
  nexus_test::section("test_sharded_save_load: ZeRO-3 rank basina parca");
  for (int n : {4, 8}) {
    const fs::path dir = scratch("sharded" + std::to_string(n));
    const fs::path init = dir / "init.bin";
    write_weights(init, make_model(0.0f));
    Cfg c;
    c.global_rows = 16;
    c.zero = 3;
    const Out ref = run(n, init, c, false, 8);
    c.dir = (dir / "ck").string();
    c.interval = 4;
    (void)run(n, init, c, false, 4);
    const ck::Manifest m = ck::read_manifest(ck::CheckpointManager::step_dir(c.dir, 4));
    std::uint64_t total = 0, mx = 0, mn = ~0ull;
    for (const auto& f : m.files) {
      total += f.bytes;
      mx = std::max(mx, f.bytes);
      mn = std::min(mn, f.bytes);
    }
    std::uint64_t params = 0;
    {
      const nexus::nn::Sequential probe_model = make_model(0.0f);  // gecici model dongu boyunca yasamali
      for (auto* p : probe_model.parameters()) params += p->data.numel();
    }
    const Out got = run(n, init, c, true, 8);
    std::printf("  info  %d rank: duzen %s, dosya %llu..%llu bayt (toplam %llu; tam durum w+m+v %llu bayt), devam adim %lld\n", n,
                m.layout.c_str(), static_cast<unsigned long long>(mn), static_cast<unsigned long long>(mx),
                static_cast<unsigned long long>(total), static_cast<unsigned long long>(params * 12),
                static_cast<long long>(got.first_step));
    nexus_test::expect(m.world == n && mx * n <= total * 3 / 2 && mx < params * 12 / 2,
                       std::to_string(n) + " rank: her rank yalniz kendi parcasini yazdi (en buyuk dosya < tam durumun yarisi)");
    nexus_test::expect(got.first_step == 4 && same_tail(ref, got),
                       "test_sharded_save_load (" + std::to_string(n) + " rank): devam == kesintisiz (bit bit)");
  }
}

void test_reshard() {
  nexus_test::section("test_reshard_2_to_4: rank sayisi degisince devam");
  for (auto [from, to] : {std::pair<int, int>{2, 4}, std::pair<int, int>{4, 2}}) {
    const fs::path dir = scratch("reshard" + std::to_string(from) + std::to_string(to));
    const fs::path init = dir / "init.bin";
    write_weights(init, make_model(0.0f));
    Cfg c;
    c.zero = 3;
    const Out ref = run(from, init, c, false, 8);
    const Out ref_to = run(to, init, c, false, 8);
    c.dir = (dir / "ck").string();
    c.interval = 4;
    (void)run(from, init, c, false, 4);
    const Out got = run(to, init, c, true, 8);
    double w_from = 0.0, w_to = 0.0, w_par = 0.0;
    (void)same_tail(ref, got, &w_from);
    (void)same_tail(ref_to, got, &w_to);
    double w_base = 0.0;  // kesintisiz from-rank vs kesintisiz to-rank: yalniz indirgeme sirasinin etkisi
    for (std::size_t i = 0; i < got.w.size() && i < ref.w.size(); ++i) {
      w_par = std::max(w_par, static_cast<double>(std::fabs(got.w[i] - ref.w[i])));
      w_base = std::max(w_base, static_cast<double>(std::fabs(ref_to.w[i] - ref.w[i])));
    }
    std::printf("  info  %d -> %d: devam adim %lld; kayip goreli farki kesintisiz %d rank'e %.1e, %d rank'e %.1e; agirlik max %.1e "
                "(kesintisiz %d rank vs %d rank: %.1e)\n",
                from, to, static_cast<long long>(got.first_step), from, w_from, to, w_to, w_par, from, to, w_base);
    nexus_test::expect(got.first_step == 4 && got.losses.size() == 4 && w_from <= 1e-5 && w_par <= 2.0 * w_base + 1e-6,
                       "test_reshard_" + std::to_string(from) + "_to_" + std::to_string(to) +
                           ": kayip devam eder (kesintisiz egriyle <= 1e-5; agirlik farki rank sayisi farkinin etkisi kadar)");
  }
}

// alt surec: crash_child <dir> <init>
int crash_child(int argc, char** argv) {
  if (argc < 4) return 2;
  Cfg c;
  c.dropout = 0.1f;
  c.dir = argv[2];
  c.interval = 2;
  (void)run(1, argv[3], c, false, 6);
  return 0;
}

void test_crash(const char* exe) {
  nexus_test::section("test_crash_injection: kayit ortasinda surec oldurulur");
  const fs::path dir = scratch("crash");
  const fs::path init = dir / "init.bin";
  write_weights(init, make_model(0.1f));
  Cfg c;
  c.dropout = 0.1f;
  const Out ref = run(1, init, c, false, 6);
  for (const char* stage : {"rank_file", "before_manifest", "before_rename"}) {
    const fs::path cd = dir / stage;
    set_env("NEXUS_CKPT_CRASH", stage);
    set_env("NEXUS_CKPT_CRASH_STEP", "4");
    const int rc = run_status(std::string("\"") + exe + "\" crash_child " + q(cd) + " " + q(init));
    set_env("NEXUS_CKPT_CRASH", "");
    set_env("NEXUS_CKPT_CRASH_STEP", "");
    bool tmp_left = false;
    for (const auto& e : fs::directory_iterator(cd)) tmp_left = tmp_left || e.path().filename().string().rfind(".tmp-", 0) == 0;
    ck::CheckpointManager probe(ck::CheckpointOptions{cd.string(), 0, 2, false, std::chrono::seconds(5)}, 1, 1);  // rank 1: temizlemez
    const auto steps = probe.committed_steps();
    Cfg r = c;
    r.dir = cd.string();
    r.interval = 2;
    const Out got = run(1, init, r, true, 6);
    std::printf("  info  %s: cikis kodu %d, yarim kayit dizini %s, tamamlanmis kayitlar %zu (son %lld), devam adim %lld\n", stage, rc,
                tmp_left ? "kaldi (yok sayildi)" : "yok", steps.size(), steps.empty() ? -1LL : static_cast<long long>(steps.back()),
                static_cast<long long>(got.first_step));
    nexus_test::expect(rc == 86 && steps.size() == 1 && steps.back() == 2 && got.first_step == 2 && same_tail(ref, got),
                       std::string("test_crash_injection (") + stage + "): son gecerli kayit (adim 2) saglam, devam kesintisizle ayni");
  }
  // bozulma: tamamlanmis dosyada tek bayt
  const fs::path cd = dir / "before_rename";
  const fs::path f = ck::CheckpointManager::step_dir(cd, 6) / "rank0.nxs3";
  std::string err;
  {
    std::vector<unsigned char> b = ck::read_file(f);
    b[b.size() / 2] ^= 0x40;
    ck::write_file_atomic(f, b.data(), b.size());
    try {
      ck::CheckpointManager m(ck::CheckpointOptions{cd.string(), 0, 2, false, std::chrono::seconds(5)}, 0, 1);
      (void)m.load_latest();
    } catch (const std::exception& e) {
      err = e.what();
    }
  }
  std::printf("  info  bozuk dosya: %s\n", err.c_str());
  nexus_test::expect(err.find("SHA-256") != std::string::npos, "bozuk kayit acik hatayla reddedilir (sessiz yedek yok)");
  {
    std::vector<unsigned char> e = ck::encode_archive({ck::Entry{"x", {1, 2, 3}}});
    e[8] = 2;  // surum 2
    std::string verr;
    try {
      (void)ck::decode_archive(e, "test");
    } catch (const std::exception& ex) {
      verr = ex.what();
    }
    nexus_test::expect(verr.find("SHA-256") != std::string::npos || verr.find("surum") != std::string::npos,
                       "surum/sema uyusmazligi acik hata");
    std::vector<unsigned char> g = ck::encode_archive({ck::Entry{"x", {1, 2, 3}}});
    nexus_test::expect(ck::decode_archive(g, "test").at(0).data == std::vector<unsigned char>{1, 2, 3}, "NXS3 gidis-donus");
    // SHA-256 yollari: donanim (SHA-NI) == skaler, tum dolgu sinirlari + buyuk tampon; bilinen vektor
    bool same = true;
    std::vector<unsigned char> buf(1 << 20);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<unsigned char>((i * 2654435761u) >> 13);
    for (std::size_t n = 0; n <= 300 && same; ++n) same = ck::sha256(buf.data(), n, true) == ck::sha256(buf.data(), n, false);
    same = same && ck::sha256(buf.data(), buf.size(), true) == ck::sha256(buf.data(), buf.size(), false);
    const std::string abc = ck::sha256_hex("abc", 3);
    std::printf("  info  SHA-256 yolu: %s; sha256(\"abc\") = %s...\n", ck::sha256_backend(), abc.substr(0, 16).c_str());
    nexus_test::expect(same && abc == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                       "SHA-256: donanim yolu == skaler (0..300 bayt + 1 MiB), FIPS 180-2 'abc' vektoru");
  }
}

void test_watchdog() {
  nexus_test::section("test_watchdog_timeout: takilan rank tespit edilir");
  const fs::path dir = scratch("watchdog");
  const fs::path init = dir / "init.bin";
  write_weights(init, make_model(0.0f));
  {
    const int n = 4;
    auto board = std::make_shared<nd::LocalBoard>(n);
    std::vector<std::shared_ptr<nd::Watchdog>> wds;
    nd::WatchdogOptions wo;
    wo.stall_timeout = std::chrono::milliseconds(500);
    wo.poll = std::chrono::milliseconds(10);
    for (int r = 0; r < n; ++r) wds.push_back(std::make_shared<nd::Watchdog>(board, r, n, wo));
    std::atomic<std::int64_t> stall_start{0};
    auto hook = [&](nt::LmBackend&, std::int64_t step, int rank) {
      if (rank == 2 && step == 3) {
        stall_start = nd::Watchdog::now_ms();
        std::this_thread::sleep_for(std::chrono::milliseconds(3500));
      }
    };
    nd::GroupOptions go;
    go.timeout = std::chrono::milliseconds(2000);
    std::vector<std::string> errors;
    Cfg c;
    const auto t0 = Clock::now();
    (void)run(n, init, c, false, 8, &errors, hook, &wds, go);
    const double total = ms_since(t0);
    const auto ev = wds[0]->events();
    bool found = false, only2 = true;
    double latency = 0.0;
    for (const auto& e : ev) {
      if (e.kind != nd::WatchdogEvent::Kind::Stalled) continue;
      std::printf("  info  %s\n", e.message.c_str());
      if (e.rank == 2 && !found) {
        found = true;
        latency = static_cast<double>(e.detected_ms - stall_start.load());
      }
      only2 = only2 && e.rank == 2;
    }
    int timeouts = 0;
    for (const auto& e : errors) {
      std::printf("  info  %s\n", e.substr(0, 160).c_str());
      timeouts += e.find("zaman") != std::string::npos || e.find("timeout") != std::string::npos ? 1 : 0;
    }
    std::printf("  info  tespit gecikmesi (rank 2 takildiktan sonra) %.0f ms; toplam %.0f ms; %zu rank hata ile cikti\n", latency, total,
                errors.size());
    nexus_test::expect(found && only2, "test_watchdog_timeout: takilan rank (2) tespit edilip raporlandi, baska rank suclanmadi");
    nexus_test::expect(timeouts >= 3 && total < 15000, "diger rank'ler kollektif zaman asimiyla cikti (asili kalmadi)");
  }
  {
    const int n = 4;
    auto board = std::make_shared<nd::LocalBoard>(n);
    std::vector<std::shared_ptr<nd::Watchdog>> wds;
    nd::WatchdogOptions wo;
    wo.poll = std::chrono::milliseconds(2);
    wo.straggler_window = 4;
    for (int r = 0; r < n; ++r) wds.push_back(std::make_shared<nd::Watchdog>(board, r, n, wo));
    auto hook = [&](nt::LmBackend&, std::int64_t, int rank) {
      if (rank == 1) std::this_thread::sleep_for(std::chrono::milliseconds(60));
    };
    Cfg c;
    (void)run(n, init, c, false, 14, nullptr, hook, &wds);
    {
      const auto beats = board->read(n);
      std::printf("  info  son adim sureleri (ms):");
      for (const auto& b : beats) std::printf(" %.1f", b.step_ms);
      std::printf("\n");
    }
    bool found = false, only1 = true;
    for (const auto& e : wds[0]->events()) {
      if (e.kind != nd::WatchdogEvent::Kind::Straggler) continue;
      std::printf("  info  %s\n", e.message.c_str());
      found = found || e.rank == 1;
      only1 = only1 && e.rank == 1;
    }
    nexus_test::expect(found && only1, "straggler tespiti: her adimda 60 ms geciken rank 1 (varis gecikmesiyle)");
  }
}

// alt surec (nexus-run): elastic_child <dir> <init>
int elastic_child(int argc, char** argv) {
  if (argc < 4) return 2;
  const fs::path dir = argv[2];
  nd::GroupOptions go;
  go.timeout = std::chrono::seconds(30);
  nd::GroupPtr g = nd::init_process_group("tcp", go);
  const int attempt = nd::env_int("NEXUS_RESTART_COUNT", 0);
  Cfg c;
  c.zero = 3;
  c.dir = (dir / "ck").string();
  c.interval = 2;
  nexus::nn::Sequential m = make_model(0.0f);
  read_weights(argv[3], m);
  GlobalRows src(g->rank(), c.global_rows / g->size(), c.global_rows);
  bool first = true;
  auto hook = [&](nt::LmBackend&, std::int64_t step) {
    if (attempt == 0 && g->rank() == 3 && step == 5) {
      const std::string t = std::to_string(nd::Watchdog::now_ms());
      ck::write_file_atomic(dir / "crash_ms", t.data(), t.size());
      std::fprintf(stderr, "elastic_child: rank 3 adim 5'te olduruluyor\n");
      std::_Exit(9);
    }
    if (attempt > 0 && first && g->rank() == 0) {
      first = false;
      const std::string t = std::to_string(nd::Watchdog::now_ms());
      ck::write_file_atomic(dir / "resume_ms", t.data(), t.size());
    }
  };
  auto t = make_trainer(m, src, c, g, nullptr, hook);
  std::int64_t from = 0;
  if (t->resume_latest()) from = t->step();
  std::fprintf(stderr, "elastic_child: deneme %d rank %d/%d, devam adim %lld\n", attempt, g->rank(), g->size(),
               static_cast<long long>(from));
  t->train(10);
  const Out o = collect(*t, m, from);  // full_parameters kollektif: tum rank'ler cagirir
  if (g->rank() == 0) save_out(dir / ("attempt" + std::to_string(attempt) + ".bin"), o);
  g->synchronize();
  return 0;
}

void test_elastic(const char* exe, const char* runner) {
  nexus_test::section("elastik yeniden baslatma: 4 surec, rank 3 olur -> 2 surecle devam");
  const fs::path dir = scratch("elastic");
  const fs::path init = dir / "init.bin";
  write_weights(init, make_model(0.0f));
  Cfg c;
  c.zero = 3;
  const Out ref = run(4, init, c, false, 10);
  const auto t0 = Clock::now();
  const int rc = run_status(std::string("\"") + runner + "\" --nproc-per-node 4 --master-port 29731 --max-restarts 1 --restart-nproc 2 -- \"" +
                            exe + "\" elastic_child " + q(dir) + " " + q(init));
  const double wall = ms_since(t0);
  if (rc != 0 || !fs::exists(dir / "attempt1.bin")) {
    nexus_test::expect(false, "nexus-run elastik kosu basarili (kod " + std::to_string(rc) + ")");
    return;
  }
  const Out got = load_out(dir / "attempt1.bin");
  auto read_ms = [&](const char* n) {
    const auto b = ck::read_file(dir / n);
    return std::stoll(std::string(b.begin(), b.end()));
  };
  const double mttr = static_cast<double>(read_ms("resume_ms") - read_ms("crash_ms"));
  double w = 0.0;
  (void)same_tail(ref, got, &w);
  std::printf("  info  cokme adim 5'te, devam adim %lld (kayit araligi 2 -> kaybedilen %lld adim); MTTR %.0f ms (olumden ilk devam "
              "adimina: surec baslatma + tcp grup + yukleme + 4->2 yeniden parcalama); toplam %.1f s\n",
              static_cast<long long>(got.first_step), 5 - static_cast<long long>(got.first_step), mttr, wall / 1000.0);
  std::printf("  info  devam kaybi kesintisiz 4 rank egrisine goreli %.1e\n", w);
  std::printf("{\"mode\": \"bench_elastic\", \"crash_step\": 5, \"resume_step\": %lld, \"interval\": 2, \"mttr_ms\": %.0f, \"loss_rel_vs_uninterrupted\": %.2e}\n",
              static_cast<long long>(got.first_step), mttr, w);
  nexus_test::expect(got.first_step == 4 && 5 - got.first_step <= 2 && w <= 1e-5,
                     "kurtarma kayit araligindan fazla adim kaybettirmez; devam kaybi kesintisiz egriyi izler (<= 1e-5)");
}

int bench() {
  const fs::path dir = scratch("bench");
  const int embed = 256, blocks = 2;
  nexus::nn::Sequential init_model = make_model(0.0f, blocks, embed);
  std::uint64_t params = 0;
  for (auto* p : init_model.parameters()) params += p->data.numel();
  std::printf("== model: %.1fM parametre (durum w+m+v %.0f MB)\n", params / 1e6, params * 12 / 1e6);
  auto one = [&](const std::string& name, bool ck, bool async, std::int64_t interval) {
    nexus::nn::Sequential m = make_model(0.0f, blocks, embed);
    const auto src_p = init_model.parameters();
    const auto dst_p = m.parameters();
    for (std::size_t i = 0; i < src_p.size(); ++i) std::memcpy(dst_p[i]->data.data(), src_p[i]->data.data(), src_p[i]->data.numel() * 4);
    GlobalRows src(0, 4, 4);
    Cfg c;
    c.global_rows = 4;
    c.T = 32;
    if (ck) {
      c.dir = (dir / name).string();
      c.interval = interval;
      c.async = async;
    }
    auto t = make_trainer(m, src, c, nullptr);
    const auto tw = Clock::now();
    t->train(2);  // isinma
    std::fprintf(stderr, "  bench %s: isinma 2 adim %.0f ms\n", name.c_str(), ms_since(tw));
    const int steps = 21;  // son adim kayit adimi degil: train() sonundaki bekleme, onceki kaydin ortusmesini olcer
    const auto t0 = Clock::now();
    t->train(2 + steps);  // kayitlar adim icinde; eszamansiz yazimlar train() sonunda beklenir (sona dahil)
    double block = 0.0, write = 0.0, gbs = 0.0;
    std::uint64_t bytes = 0;
    int saves = 0;
    if (ck) {
      const ck::SaveStats st = t->checkpoints()->last_stats();  // son kayit (her kayit ayni boyutta)
      block = st.blocking_ms;
      write = st.write_ms;
      bytes = st.bytes;
      gbs = st.gb_per_s();
      saves = static_cast<int>(steps / interval);
    }
    const double per_step = ms_since(t0) / steps;
    std::printf("{\"mode\": \"bench_ckpt\", \"case\": \"%s\", \"params\": %llu, \"interval\": %lld, \"ms_per_step\": %.1f, "
                "\"saves\": %d, \"file_bytes\": %llu, \"save_blocking_ms\": %.1f, \"save_write_ms\": %.1f, \"write_gb_s\": %.3f}\n",
                name.c_str(), static_cast<unsigned long long>(params), static_cast<long long>(interval), per_step, saves,
                static_cast<unsigned long long>(bytes), block, write, gbs);
    return per_step;
  };
  // Ham kayit hizi: sentetik yuk (SHA + yazma + tamamlama), tek rank ve 4 rank (is parcacigi) parcali
  for (int world : {1, 4}) {
    for (std::uint64_t mb : {64ull, 512ull}) {
      const fs::path d = dir / ("raw" + std::to_string(world) + "_" + std::to_string(mb));
      const std::size_t per = static_cast<std::size_t>(mb << 20) / static_cast<std::size_t>(world);
      std::vector<double> wms(static_cast<std::size_t>(world));
      const auto t0 = Clock::now();
      std::vector<std::thread> ts;
      for (int r = 0; r < world; ++r) {
        ts.emplace_back([&, r] {
          ck::CheckpointManager m(ck::CheckpointOptions{d.string(), 0, 1, false, std::chrono::seconds(120)}, r, world);
          std::vector<unsigned char> payload(per, static_cast<unsigned char>(r + 1));
          ck::TrainerState st;
          m.save(1, st, std::move(payload), "raw");
          wms[static_cast<std::size_t>(r)] = m.last_stats().write_ms;
        });
      }
      for (auto& t : ts) t.join();
      const double wall = ms_since(t0);
      std::printf("{\"mode\": \"bench_ckpt_raw\", \"world\": %d, \"total_mb\": %llu, \"wall_ms\": %.1f, \"gb_s\": %.2f, \"rank_write_ms_max\": %.1f}\n",
                  world, static_cast<unsigned long long>(mb), wall, static_cast<double>(mb << 20) / (wall * 1e6),
                  *std::max_element(wms.begin(), wms.end()));
    }
  }
  const double base = one("none", false, false, 0);
  for (std::int64_t interval : {2, 10}) {
    const double sync = one("sync_i" + std::to_string(interval), true, false, interval);
    const double asy = one("async_i" + std::to_string(interval), true, true, interval);
    std::printf("  info  adim %.1f ms; her %lld adimda kayit: senkron +%.1f%%, eszamansiz +%.1f%%\n", base,
                static_cast<long long>(interval), 100.0 * (sync / base - 1.0), 100.0 * (asy / base - 1.0));
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::string mode = argc > 1 ? argv[1] : "exact";
  try {
    if (mode == "exact_child") return exact_child(argc, argv);
    if (mode == "crash_child") return crash_child(argc, argv);
    if (mode == "elastic_child") return elastic_child(argc, argv);
    if (mode == "exact") test_exact(argv[0]);
    else if (mode == "rng") test_rng();
    else if (mode == "sharded") test_sharded();
    else if (mode == "reshard") test_reshard();
    else if (mode == "crash") test_crash(argv[0]);
    else if (mode == "watchdog") test_watchdog();
    else if (mode == "elastic" && argc > 2) test_elastic(argv[0], argv[2]);
    else if (mode == "bench") return bench();
    else {
      std::printf("bilinmeyen mod: %s\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("test_phase15_durability");
}
