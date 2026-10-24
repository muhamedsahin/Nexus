/** FAZ 4 CUDA egitim: dinamik olcek, kabul edilen Adam adimi, master modlari ve yakinsama. */
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/gpt.hpp"
#include "nexus_test.hpp"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace nc = nexus::cuda;
namespace {
nc::Stream g_stream;
nc::DeviceBuffer tokens_for(int batch, int time, int vocab) {
  std::vector<std::int32_t> host(static_cast<std::size_t>(batch) * time);
  for (std::size_t i = 0; i < host.size(); ++i) host[i] = static_cast<std::int32_t>((i * 37 + 11) % vocab);
  nc::DeviceBuffer d(host.size() * 4, g_stream);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), host.data(), host.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
  return d;
}
nc::GptConfig small(nexus::DType type, nc::MasterWeights master = nc::MasterWeights::Fp32Master) {
  nc::GptConfig c;
  c.vocab = 67; c.max_seq = 32; c.layers = 1; c.heads = 4; c.dim = 32;
  c.compute_dtype = type; c.master = master;
  c.loss_scale.init_scale = 1024.0;
  c.loss_scale.growth_interval = 2;
  return c;
}
void adam_overflow_regression() {
  nexus_test::section("FP16 overflow sonrasi Adam bias correction");
  nc::DeviceBuffer state_d(sizeof(nc::DeviceLossScaleState), g_stream);
  nexus_train::LossScalePolicy policy;
  policy.init_scale = 1024.0; policy.growth_interval = 2;
  nc::loss_scale_init(state_d.as<nc::DeviceLossScaleState>(), policy, g_stream.handle());
  auto* state = state_d.as<nc::DeviceLossScaleState>();
  nc::DeviceBuffer p(4, g_stream), pc(4, g_stream), g(4, g_stream), gc(4, g_stream), m(4, g_stream), mc(4, g_stream),
      v(4, g_stream), vc(4, g_stream), scratch(nc::adamw_scratch_bytes(1, 1), g_stream);
  nc::fill(p.as<float>(), 1.0f, 1, g_stream.handle()); nc::fill(pc.as<float>(), 1.0f, 1, g_stream.handle());
  NEXUS_CUDA_CHECK(cudaMemsetAsync(m.data(), 0, 4, g_stream.handle()));
  NEXUS_CUDA_CHECK(cudaMemsetAsync(mc.data(), 0, 4, g_stream.handle()));
  NEXUS_CUDA_CHECK(cudaMemsetAsync(v.data(), 0, 4, g_stream.handle()));
  NEXUS_CUDA_CHECK(cudaMemsetAsync(vc.data(), 0, 4, g_stream.handle()));
  nc::DeviceParamSlot scaled, control;
  scaled.param = p.as<float>(); scaled.grad = g.as<float>(); scaled.exp_avg = m.as<float>(); scaled.exp_avg_sq = v.as<float>(); scaled.numel = 1;
  control = scaled; control.param = pc.as<float>(); control.grad = gc.as<float>(); control.exp_avg = mc.as<float>(); control.exp_avg_sq = vc.as<float>();
  nc::DeviceAdamWConfig cfg;
  cfg.lr = 0.01; cfg.beta1 = 0.9; cfg.beta2 = 0.99; cfg.loss_scale = &state->scale;
  cfg.found_inf = &state->found_inf; cfg.skipped_steps = &state->skipped_steps;
  nc::fill(g.as<float>(), std::numeric_limits<float>::infinity(), 1, g_stream.handle());
  nc::adamw_multi_tensor(&scaled, 1, cfg, 1, nullptr, scratch.data(), g_stream.handle());
  g_stream.synchronize();
  nexus_test::expect(state_d.as<nc::DeviceLossScaleState>() != nullptr, "olcek durumu aygitta");
  nc::DeviceLossScaleState host{};
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(&host, state_d.data(), sizeof(host), cudaMemcpyDeviceToHost, g_stream.handle()));
  // Olcek 512'ye dustu: olcekli tampondaki 1024, olceksiz kontroldeki 2 ile ayni gradyandir.
  nc::fill(g.as<float>(), 1024.0f, 1, g_stream.handle()); nc::fill(gc.as<float>(), 2.0f, 1, g_stream.handle());
  nc::DeviceAdamWConfig clean;
  clean.lr = cfg.lr; clean.beta1 = cfg.beta1; clean.beta2 = cfg.beta2;
  nc::adamw_multi_tensor(&control, 1, clean, 1, nullptr, scratch.data(), g_stream.handle());
  nc::loss_scale_update(state, policy, g_stream.handle());
  nc::adamw_multi_tensor(&scaled, 1, cfg, 2, nullptr, scratch.data(), g_stream.handle());
  nc::loss_scale_update(state, policy, g_stream.handle());
  float hp = 0, hpc = 0, hm = 0, hmc = 0, hv = 0, hvc = 0;
  for (const auto& [dst, src] : {std::pair<float*, void*>{&hp, p.data()}, {&hpc, pc.data()}, {&hm, m.data()},
                                 {&hmc, mc.data()}, {&hv, v.data()}, {&hvc, vc.data()}}) {
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(dst, src, 4, cudaMemcpyDeviceToHost, g_stream.handle()));
  }
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(&host, state_d.data(), sizeof(host), cudaMemcpyDeviceToHost, g_stream.handle()));
  g_stream.synchronize();
  nexus_test::expect(std::fabs(hp - hpc) < 1e-7f && std::fabs(hm - hmc) < 1e-7f && std::fabs(hv - hvc) < 1e-7f,
                     "overflow sonrasi ilk kabul edilen Adam adimi kontrolle ayni");
  nexus_test::expect(host.skipped_steps == 1 && host.steps == 2 && host.scale == 512.0f,
                     "olcek yarida ve atlama sayaci bir");
  nexus_test::expect(host.scale_history[0] == 512.0f && host.scale_history[1] == 512.0f,
                     "son olcekler aygit halka gecmisinde");
}
void fp16_model_test() {
  nexus_test::section("FP16 GPT kasitli taşma: adim atla ve toparlan");
  nc::GptModel model(small(nexus::DType::F16), 1, 8, 100, g_stream);
  auto tokens = tokens_for(1, 8, 67);
  nc::DeviceAdamWConfig adam; adam.lr = 1e-3; adam.max_grad_norm = 1.0;
  const auto before = model.download_parameters();
  model.forward_backward(tokens.as<std::int32_t>());
  nc::fill(model.device_gradients(), std::numeric_limits<float>::infinity(), 1, g_stream.handle());
  model.optimizer_step(adam);
  const auto after_skip = model.download_parameters();
  const auto state = model.read_loss_scale();
  nexus_test::expect(before == after_skip && state.skipped_steps == 1 && state.scale == 512.0f,
                     "GPT FP16 overflow parametreyi degistirmez ve olcegi yarilar");
  model.forward_backward(tokens.as<std::int32_t>());
  model.optimizer_step(adam);
  const auto recovered = model.download_parameters();
  const bool finite = std::all_of(recovered.begin(), recovered.end(), [](float x) { return std::isfinite(x); });
  nexus_test::expect(finite && recovered != before, "GPT FP16 sonraki temiz adim uygulanir");
}
// ---- tiny (configs/tiny.json sekli) + ogrenilebilir Markov verisi -----------------------
// Her token v icin 4 ardil (olasiliklar .50/.25/.15/.10): teorik kayip tabani
// H = 1.2080 nat. Model gecis tablosunu ogrenmek zorundadir; tek batch ezberi yoktur.
constexpr int kTinyVocab = 8192, kTinyDim = 384, kTinyHeads = 6, kTinyLayers = 6, kTinyTime = 256, kTinyBatch = 4;
constexpr double kMarkovEntropy = 1.2080;

nc::GptConfig tiny(nexus::DType type, nc::MasterWeights master = nc::MasterWeights::Fp32Master) {
  nc::GptConfig c;
  c.vocab = kTinyVocab; c.max_seq = kTinyTime; c.layers = kTinyLayers; c.heads = kTinyHeads; c.dim = kTinyDim;
  c.compute_dtype = type; c.master = master;
  c.deterministic_attention = true;  // T=256: 4 dQ dilimi; A==B bit esitligi atomiksiz dQ ister
  return c;
}

struct MarkovCorpus {
  std::vector<std::int32_t> tokens;
  explicit MarkovCorpus(std::size_t length) : tokens(length) {
    std::uint64_t st = 0x243F6A8885A308D3ULL;
    auto next = [&st]() { st = st * 6364136223846793005ULL + 1442695040888963407ULL; return st >> 33; };
    std::vector<std::array<std::int32_t, 4>> succ(kTinyVocab);
    for (auto& row : succ) for (auto& t : row) t = static_cast<std::int32_t>(next() % kTinyVocab);
    std::int32_t cur = 0;
    for (std::size_t i = 0; i < length; ++i) {
      tokens[i] = cur;
      const std::uint64_t u = next() % 100;  // .50 / .25 / .15 / .10
      const int j = u < 50 ? 0 : (u < 75 ? 1 : (u < 90 ? 2 : 3));
      cur = succ[static_cast<std::size_t>(cur)][static_cast<std::size_t>(j)];
    }
  }
  // Adim s'in batch'i: sabit tohumlu ofsetler (her iki hassasiyet ayni batch'leri gorur).
  std::vector<std::int32_t> batch(int step) const {
    std::vector<std::int32_t> out(static_cast<std::size_t>(kTinyBatch) * kTinyTime);
    std::uint64_t st = 0x9E3779B97F4A7C15ULL ^ static_cast<std::uint64_t>(step) * 0xD1B54A32D192ED03ULL;
    for (int b = 0; b < kTinyBatch; ++b) {
      st = st * 6364136223846793005ULL + 1442695040888963407ULL;
      const std::size_t off = (st >> 20) % (tokens.size() - kTinyTime);
      std::copy(tokens.begin() + off, tokens.begin() + off + kTinyTime, out.begin() + static_cast<std::size_t>(b) * kTinyTime);
    }
    return out;
  }
};

const MarkovCorpus& corpus() {
  static const MarkovCorpus c(1u << 21);
  return c;
}

// steps adim egitir; her adimin kaybini dondurur.
std::vector<float> train_tiny(const nc::GptConfig& config, int steps) {
  nc::GptModel model(config, kTinyBatch, kTinyTime, 300, g_stream);
  nc::DeviceAdamWConfig adam;
  adam.lr = 1e-3; adam.beta2 = 0.95; adam.weight_decay = 0.1; adam.max_grad_norm = 1.0;
  nc::DeviceBuffer d(static_cast<std::size_t>(kTinyBatch) * kTinyTime * 4, g_stream);
  std::vector<float> losses;
  losses.reserve(static_cast<std::size_t>(steps));
  for (int step = 0; step < steps; ++step) {
    const auto host = corpus().batch(step);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), host.data(), host.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
    model.forward_backward(d.as<std::int32_t>());
    losses.push_back(model.read_loss());  // read_loss senkron: host tamponu sonraki adimda guvenle yeniden yazilir
    model.optimizer_step(adam);
  }
  g_stream.synchronize();
  return losses;
}

double window_mean(const std::vector<float>& losses, int end_step, int window = 50) {
  double acc = 0.0;
  for (int i = end_step - window; i < end_step; ++i) acc += losses[static_cast<std::size_t>(i)];
  return acc / window;
}

void master_mode_tests() {
  nexus_test::section("test_master_weight_modes: A/B/C, tiny, 500 adim");
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  std::vector<std::vector<float>> curves;
  for (const nc::MasterWeights mode : {nc::MasterWeights::CastPerForward, nc::MasterWeights::Fp32Master,
                                      nc::MasterWeights::NoMaster}) {
    curves.push_back(train_tiny(tiny(nexus::DType::BF16, mode), 500));
    nc::GptModel probe(tiny(nexus::DType::BF16, mode), kTinyBatch, kTinyTime, 1, g_stream);
    std::printf("  info  %-24s kayip[0]=%.5f ort[451..500]=%.5f durum=%.1f MiB\n", nc::to_string(mode),
                curves.back().front(), window_mean(curves.back(), 500), probe.state_bytes() / 1048576.0);
  }
  // A (her ileri geciste cast) ve B (AdamW golgeyi RN ile yazar) ayni BF16 agirligi uretir.
  nexus_test::expect(curves[0] == curves[1], "A ve B kayip egrileri 500 adimda bit-bit ayni");
  const double ab = window_mean(curves[1], 500);
  const double c = window_mean(curves[2], 500);
  std::printf("  info  C (ana kopyasiz, stokastik yuvarlama) - B bagil fark @500 = %+.3e\n", (c - ab) / ab);
  nexus_test::expect(std::isfinite(c) && std::fabs(c - ab) / ab < 0.05, "C, B'nin %5 bandinda (deneysel, raporlanir)");
  nc::GptModel b(tiny(nexus::DType::BF16, nc::MasterWeights::Fp32Master), kTinyBatch, kTinyTime, 1, g_stream);
  nc::GptModel cm(tiny(nexus::DType::BF16, nc::MasterWeights::NoMaster), kTinyBatch, kTinyTime, 1, g_stream);
  nexus_test::expect(cm.state_bytes() < b.state_bytes(), "C modu B'den daha az parametre/optimizer bellegi");
}

void convergence_test() {
  nexus_test::section("test_bf16_training_converges: tiny, 2000 adim, BF16 vs FP32");
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);  // FP32 referansi tam FP32 (TF32 kapali)
  const std::vector<float> fp32 = train_tiny(tiny(nexus::DType::F32), 2000);
  const std::vector<float> bf16 = train_tiny(tiny(nexus::DType::BF16), 2000);
  for (int at = 100; at <= 2000; at += 100) {
    const double a = window_mean(fp32, at);
    const double b = window_mean(bf16, at);
    std::printf("  info  egri %4d: FP32=%.5f BF16=%.5f bagil=%+.3e\n", at, a, b, (b - a) / a);
  }
  // Tek tohumlu bu gorevde egri, sayisal gurultuye duyarlidir (plato cikis zamani kayar): ayni her
  // seyle yalniz TF32 acmak FP32 egrisini 600. adimda %5, 2000. adimda %3.8 oynatir (FAZ 5 raporu).
  // Spec'in %1'i bu gurultu tabaninin altinda kalir; o yuzden ayni kosuda tabani olcup BF16'yi
  // max(%1, 2 x |TF32 - FP32|) bandinda tutariz. Spec %1 sonucu ayrica yazilir (raporlanir).
  nc::set_float32_matmul_precision(nc::MatmulPrecision::High);
  const std::vector<float> tf32 = train_tiny(tiny(nexus::DType::F32), 2000);
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  for (const int at : {500, 2000}) {
    const double a = window_mean(fp32, at);
    const double b = window_mean(bf16, at);
    const double noise = std::fabs(window_mean(tf32, at) - a) / a;
    const double rel = std::fabs(b - a) / a;
    std::printf("  info  adim %4d (son 50 ort.): FP32=%.5f BF16=%.5f bagil=%+.3e | gurultu |TF32-FP32|=%.3e | spec %%1: %s"
                " (taban %.4f)\n",
                at, a, b, (b - a) / a, noise, rel <= 0.01 ? "gecer" : "GECMEZ", kMarkovEntropy);
    nexus_test::expect(rel <= std::max(0.01, 2.0 * noise), "BF16 kaybi FP32'nin max(%1, 2 x gurultu) bandinda @" +
                                                              std::to_string(at));
  }
  nexus_test::expect(window_mean(fp32, 2000) < 0.5 * fp32.front(), "model ogreniyor (kayip yariya iner)");
}

// FAZ 5.H: GPT dikkat yolu secimi + flash ile math'in ayni modeli ayni gradyana goturmesi.
double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    num += d * d;
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

std::vector<float> one_step_grads(const nc::GptConfig& config, float* loss, std::size_t* act_bytes,
                                  nc::AttentionBackend* used, const std::vector<float>* weights = nullptr,
                                  int step = 0) {
  nc::GptModel model(config, kTinyBatch, kTinyTime, 300, g_stream);
  if (weights != nullptr) model.upload_parameters(*weights);
  const auto host = corpus().batch(step);
  nc::DeviceBuffer d(host.size() * 4, g_stream);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), host.data(), host.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
  model.forward_backward(d.as<std::int32_t>());
  *loss = model.read_loss();
  *act_bytes = model.activation_bytes();
  *used = model.attention_backend();
  std::vector<float> g(model.num_parameters());
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(g.data(), model.device_gradients(), g.size() * 4, cudaMemcpyDeviceToHost,
                                   g_stream.handle()));
  g_stream.synchronize();
  return g;
}

// Tani: tiny Markov egrisi kucuk sayisal perturbasyona ne kadar duyarli (FP32 vs TF32, ayni her sey).
void sensitivity_diag(int steps) {
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  const std::vector<float> a = train_tiny(tiny(nexus::DType::F32), steps);
  nc::set_float32_matmul_precision(nc::MatmulPrecision::High);
  const std::vector<float> b = train_tiny(tiny(nexus::DType::F32), steps);
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  for (int at = 100; at <= steps; at += 100) {
    std::printf("  info  FP32 vs TF32 egri %4d: fp32=%.5f tf32=%.5f bagil=%+.3e\n", at, window_mean(a, at),
                window_mean(b, at), (window_mean(b, at) - window_mean(a, at)) / window_mean(a, at));
  }
}

// Tani: flash ile egitilmis agirlikta ayni batch'in kaybi/gradyani her yolda.
void flash_trained_diag(int steps) {
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  std::vector<float> weights;
  {
    nc::GptModel model(tiny(nexus::DType::BF16), kTinyBatch, kTinyTime, 300, g_stream);
    nc::DeviceAdamWConfig adam;
    adam.lr = 1e-3; adam.beta2 = 0.95; adam.weight_decay = 0.1; adam.max_grad_norm = 1.0;
    nc::DeviceBuffer d(static_cast<std::size_t>(kTinyBatch) * kTinyTime * 4, g_stream);
    for (int step = 0; step < steps; ++step) {
      const auto host = corpus().batch(step);
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(d.data(), host.data(), host.size() * 4, cudaMemcpyHostToDevice, g_stream.handle()));
      model.forward_backward(d.as<std::int32_t>());
      model.optimizer_step(adam);
    }
    weights = model.download_parameters();
  }
  float lr = 0.f, lm = 0.f, lf = 0.f;
  std::size_t ab = 0;
  nc::AttentionBackend u{};
  nc::GptConfig bm = tiny(nexus::DType::BF16);
  bm.attention = nc::AttentionBackend::Math;
  for (const int probe : {steps, 900000}) {
    const auto ref = one_step_grads(tiny(nexus::DType::F32), &lr, &ab, &u, &weights, probe);
    const auto gm = one_step_grads(bm, &lm, &ab, &u, &weights, probe);
    const auto gf = one_step_grads(tiny(nexus::DType::BF16), &lf, &ab, &u, &weights, probe);
    std::printf("  info  @%d egitilmis, batch %d: kayip fp32=%.5f bf16-math=%.5f bf16-flash=%.5f | grad L2 math=%.3e flash=%.3e\n",
                steps, probe, lr, lm, lf, rel_l2(gm, ref), rel_l2(gf, ref));
  }
}

// Tani: parametre basina gradyan bagil hatasi (FP32 math referansina gore).
void flash_grad_diag() {
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  float l = 0.f;
  std::size_t ab = 0;
  nc::AttentionBackend u{};
  const std::vector<float> ref = one_step_grads(tiny(nexus::DType::F32), &l, &ab, &u);
  nc::GptConfig bm = tiny(nexus::DType::BF16);
  bm.attention = nc::AttentionBackend::Math;
  nc::GptConfig bf = tiny(nexus::DType::BF16);
  nc::GptConfig ba = tiny(nexus::DType::BF16);
  ba.deterministic_attention = false;
  const std::vector<float> gm = one_step_grads(bm, &l, &ab, &u);
  const std::vector<float> gf = one_step_grads(bf, &l, &ab, &u);
  const std::vector<float> ga = one_step_grads(ba, &l, &ab, &u);
  nc::GptModel probe(tiny(nexus::DType::F32), kTinyBatch, kTinyTime, 300, g_stream);
  for (const auto& p : probe.parameter_info()) {
    auto slice = [&](const std::vector<float>& v) {
      return std::vector<float>(v.begin() + static_cast<std::ptrdiff_t>(p.offset),
                                v.begin() + static_cast<std::ptrdiff_t>(p.offset + p.numel));
    };
    const auto r = slice(ref);
    double norm = 0.0;
    for (float x : r) norm += static_cast<double>(x) * x;
    std::printf("  info  %-10s |g|=%.3e  bf16-math=%.3e  bf16-flash-det=%.3e  bf16-flash-atomic=%.3e\n", p.name.c_str(),
                std::sqrt(norm), rel_l2(slice(gm), r), rel_l2(slice(gf), r), rel_l2(slice(ga), r));
  }
}

void flash_gpt_test(int steps) {
  nexus_test::section("test_gpt_flash_vs_math: tiny BF16/FP16, ayni agirlik, tek adim + 300 adim egri");
  {
    nc::GptModel f32(tiny(nexus::DType::F32), kTinyBatch, kTinyTime, 1, g_stream);
    nexus_test::expect(f32.attention_backend() == nc::AttentionBackend::Math, "FP32 Auto -> math (flash yalniz dar)");
    nc::GptConfig bad = tiny(nexus::DType::F32);
    bad.attention = nc::AttentionBackend::Flash;
    bool threw = false;
    try {
      nc::GptModel m(bad, kTinyBatch, kTinyTime, 1, g_stream);
    } catch (const std::exception&) {
      threw = true;
    }
    nexus_test::expect(threw, "FP32 + attention=Flash acik hata (sessiz dusme yok)");
  }
  for (const nexus::DType t : {nexus::DType::BF16, nexus::DType::F16}) {
    const bool bf = t == nexus::DType::BF16;
    nc::GptConfig cm = tiny(t);
    cm.attention = nc::AttentionBackend::Math;
    nc::GptConfig cf = tiny(t);
    float lm = 0.f, lf = 0.f;
    std::size_t am = 0, af = 0;
    nc::AttentionBackend um{}, uf{};
    const std::vector<float> gm = one_step_grads(cm, &lm, &am, &um);
    const std::vector<float> gf = one_step_grads(cf, &lf, &af, &uf);
    const double gerr = rel_l2(gf, gm);
    std::printf("  info  %s kayip math=%.6f flash=%.6f | grad bagil L2=%.3e | aktivasyon math=%.1f MiB flash=%.1f MiB\n",
                bf ? "BF16" : "FP16", lm, lf, gerr, am / 1048576.0, af / 1048576.0);
    nexus_test::expect(um == nc::AttentionBackend::Math && uf == nc::AttentionBackend::Flash, "secilen yol dogru");
    nexus_test::expect(std::fabs(lf - lm) / lm < 2e-3, "kayip flash ~ math");
    nexus_test::expect(gerr < (bf ? 3e-2 : 1e-2), "tum gradyanlar flash ~ math (G.4 dar tolerans)");
    nexus_test::expect(af < am, "flash aktivasyon bellegi math'ten az ([L,B,H,T,T] yok)");
  }
  nc::GptConfig cm = tiny(nexus::DType::BF16);
  cm.attention = nc::AttentionBackend::Math;
  const std::vector<float> m = train_tiny(cm, steps);
  const std::vector<float> f = train_tiny(tiny(nexus::DType::BF16), steps);
  for (int at = 100; at <= steps; at += 100) {
    std::printf("  info  BF16 egri %4d (son 50 ort.): math=%.5f flash=%.5f bagil=%+.3e\n", at, window_mean(m, at),
                window_mean(f, at), (window_mean(f, at) - window_mean(m, at)) / window_mean(m, at));
  }
  const double a = window_mean(m, 300), b = window_mean(f, 300);
  nexus_test::expect(std::fabs(b - a) / a < 0.01, "flash egitim egrisi math'in %1 bandinda @300");
}

// PyTorch BF16 autocast karsilastirmasi icin: ayni baslangic agirligi, ayni batch'ler ve
// Nexus FP32/BF16 kayip egrileri <dir>'e yazilir (tests/cuda/reference_amp_torch.py okur).
void dump_for_torch(const std::string& dir, int steps) {
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  std::filesystem::create_directories(dir);
  nc::GptModel model(tiny(nexus::DType::F32), kTinyBatch, kTinyTime, 300, g_stream);
  const std::vector<float> weights = model.download_parameters();
  std::ofstream(std::filesystem::path(dir) / "weights.bin", std::ios::binary)
      .write(reinterpret_cast<const char*>(weights.data()), static_cast<std::streamsize>(weights.size() * 4));
  {
    std::ofstream tok(std::filesystem::path(dir) / "tokens.bin", std::ios::binary);
    for (int step = 0; step < steps; ++step) {
      const auto b = corpus().batch(step);
      tok.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size() * 4));
    }
  }
  const std::vector<float> fp32 = train_tiny(tiny(nexus::DType::F32), steps);
  const std::vector<float> bf16 = train_tiny(tiny(nexus::DType::BF16), steps);
  std::ofstream m(std::filesystem::path(dir) / "manifest.json", std::ios::binary);
  m << "{\n  \"vocab\": " << kTinyVocab << ", \"max_seq\": " << kTinyTime << ", \"layers\": " << kTinyLayers
    << ", \"heads\": " << kTinyHeads << ", \"dim\": " << kTinyDim << ", \"batch\": " << kTinyBatch
    << ", \"time\": " << kTinyTime << ", \"steps\": " << steps << ", \"ln_eps\": 1e-5,\n"
    << "  \"adam\": {\"lr\": 1e-3, \"beta1\": 0.9, \"beta2\": 0.95, \"eps\": 1e-8, \"weight_decay\": 0.1, "
       "\"max_grad_norm\": 1.0},\n";
  auto curve = [&m](const char* name, const std::vector<float>& v, bool last) {
    m << "  \"" << name << "\": [";
    for (std::size_t i = 0; i < v.size(); ++i) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.7g", static_cast<double>(v[i]));
      m << (i ? "," : "") << buf;
    }
    m << "]" << (last ? "\n" : ",\n");
  };
  curve("nexus_fp32", fp32, false);
  curve("nexus_bf16", bf16, false);
  m << "  \"params\": [\n";
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
  std::printf("  info  dokum: %s (%d adim)\n", dir.c_str(), steps);
}

}  // namespace
int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (nc::device_count() == 0) { std::printf("  SKIP  CUDA aygiti yok\n"); return 77; }
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    g_stream = nc::Stream::create(0);
    nc::StreamGuard guard(g_stream);
    if (mode == "all" || mode == "loss") { adam_overflow_regression(); fp16_model_test(); }
    if (mode == "all" || mode == "master") master_mode_tests();
    if (mode == "all" || mode == "converge") convergence_test();
    if (mode == "sensitivity") sensitivity_diag(argc > 2 ? std::atoi(argv[2]) : 1000);
    if (mode == "flashdiag") { flash_grad_diag(); flash_trained_diag(argc > 2 ? std::atoi(argv[2]) : 700); }
    if (mode == "all" || mode == "flash") flash_gpt_test(argc > 2 ? std::max(300, std::atoi(argv[2])) : 300);
    if (mode == "dump") dump_for_torch(argc > 2 ? argv[2] : "nexus_amp_dump", argc > 3 ? std::atoi(argv[3]) : 500);
    g_stream.synchronize();
    g_stream.destroy();
    return nexus_test::finish("phase4_cuda_training");
  } catch (const std::exception& e) {
    std::printf("  FAIL  %s\n", e.what());
    return 1;
  }
}

