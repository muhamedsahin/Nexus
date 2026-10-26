#pragma once

/**
 * @file moe.hpp
 * @brief FAZ 14: uzman karisimi (MoE) cekirdegi — yonlendirici, kapasite, siralama permutasyonu, gruplu GEMM,
 *        birlestirme, yuk dengeleme kayiplari ve el yazimi geri gecis (ham FP32 tamponlar; CPU).
 *
 * MoEFFN(x) = sum_{j < k} g_j Expert_{e_j}(x), Expert = SwiGLU MLP (up [d, 2F] = [W1 | W3], down [F, d]; FAZ 6 ile ayni).
 * Yonlendirici (her zaman FP32): logits = x W_gate [N, E]. Gurultulu top-k (Shazeer ve ark. 2017, yalniz egitimde):
 *   H = logits + eps * softplus(x W_noise), eps ~ N(0, 1) (tohumlu, belirlenimci); secim H uzerinde top-k.
 * Kapi: g = softmax(H'nin secilen k degeri) (Mixtral: secilenlerin uzerinde normalize).
 * Kapasite (GShard): tokenlar `groups` ardisik gruba bolunur; grup basina uzman kapasitesi
 *   C = max(min_capacity, round(k * N_g / E * capacity_factor)) (k = 1'de spec formulu N/E * cf).
 *   Atama onceligi grupta (secim sirasi j, sonra token) — once herkesin 1. secimi.
 *   Drop: dolu uzmana dusen atama dusurulur (katkisi 0). Reroute: tokenin secmedigi, logit sirasina gore bos
 *   ilk uzmana yonlendirilir (kapi agirligi korunur).
 * Permutasyon: kabul edilen atamalar (uzman, grup, kabul sirasi) duzeninde bitisik dilimler — sayim sirali (histogram
 * + cumsum), yerinde ve belirlenimci; geri gecis ayni sirayi tersine kullanir.
 * Gruplu GEMM (CPU): uzman basina bitisik dilim -> uzmanlar is parcaciklarina dagitilir (her uzman ayri cikti ve
 * ayri gradyan dilimi: yaris yok, sonuc is parcacigi sayisindan bagimsiz, bit bit ayni).
 * Kayiplar (yonlendiricinin temiz logitleri uzerinden):
 *   aux = alpha * E * sum_i f_i P_i, f_i = (gruplar oncesi) i'ye yonlenen atama / (N k), P_i = ortalama softmax olasiligi.
 *   z = z_alpha * mean_t (logsumexp_t)^2 (ST-MoE).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace moe {

enum class Overflow : std::uint8_t { Drop, Reroute };

struct Spec {
  int experts = 8;
  int top_k = 2;
  int d_model = 0;
  int d_ff = 0;  // uzman basina F (up [d, 2F])
  float capacity_factor = 1.25f;
  int min_capacity = 4;
  Overflow overflow = Overflow::Drop;
  int groups = 1;
  bool noisy = false;
  float aux_alpha = 0.01f;
  float z_alpha = 1.0e-3f;
  int threads = 0;  // 0: donanim
  void validate() const;
};

/// Uzman agirlik deposu: FP32 (egitim) ya da BF16 / FP8-E4M3 (uzman basina olcek; yalniz ileri, bellek / 2 ya da / 4).
enum class WeightType : std::uint8_t { F32, BF16, FP8 };

struct Routing {
  std::int64_t N = 0;
  std::vector<float> logits;     // [N, E] temiz
  std::vector<float> probs;      // [N, E] softmax(temiz)
  std::vector<float> select;     // [N, E] secim logitleri (gurultulu ya da temiz)
  std::vector<float> noise_eps;  // [N, E] (gurultulu)
  std::vector<float> noise_pre;  // [N, E] x W_noise (gurultulu)
  std::vector<int> expert;       // [N, k]
  std::vector<float> gate;       // [N, k]
  std::vector<std::int64_t> row; // [N, k] permute satiri, -1: dusuruldu
  std::vector<std::int64_t> offsets;  // [E + 1]
  std::vector<std::int64_t> routed;   // [E] kapasite oncesi atama
  std::vector<std::int32_t> perm_token;  // [M] satir -> token
  std::int64_t capacity = 0, dropped = 0, rerouted = 0;
  float aux = 0.0f, z = 0.0f;
  // Yuk dengeleme istatistikleri (EP: rank'ler arasi toplanip aux/z kuresel yeniden hesaplanir).
  std::vector<double> prob_sum;   // [E] sum_t p_te
  double lse_sq_sum = 0.0;        // sum_t lse_t^2
  std::int64_t loss_tokens = 0;   // aux/z normalizasyonu ve gradyanlarinda N (0: yerel N)
  /// prob_sum / routed / lse_sq_sum'dan aux ve z (loss_tokens ile).
  void recompute_losses(const struct Spec& s);
  [[nodiscard]] std::int64_t rows() const noexcept { return offsets.empty() ? 0 : offsets.back(); }
  [[nodiscard]] int idle_experts() const;
};

/// Yonlendirme (ileri). noise_w nullptr ya da !training: gurultu yok.
[[nodiscard]] Routing route(const float* x, const float* router_w, const float* noise_w, std::int64_t N, const Spec& s,
                            bool training, std::uint64_t seed);

/// Gruplu uzman ileri: xp [M, d] (permutasyonlu), offsets [E + 1]; up [E][d][2F], down [E][F][d].
/// Onbellek (h = xp up, a = swiglu(h)) geri gecis icin doldurulur (nullptr: tutulmaz).
struct ExpertCache {
  std::vector<float> xp, h, a, yp;
};
void experts_forward(const float* xp, const std::int64_t* offsets, int experts, const float* up, const float* down, int d,
                     int F, float* yp, ExpertCache* cache, int threads);
/// dyp [M, d] -> dxp [M, d] (yazilir); dup/ddown += (uzman dilimleri).
void experts_backward(const float* dyp, const std::int64_t* offsets, int experts, const float* up, const float* down, int d,
                      int F, const ExpertCache& cache, float* dxp, float* dup, float* ddown, int threads);

/// Dar uzman agirliklari (14.F): ileri gecis kullanilirken uzman basina FP32'ye acilir.
struct QuantizedExperts {
  WeightType type = WeightType::F32;
  int experts = 0, d = 0, F = 0;
  std::vector<std::uint16_t> bf16_up, bf16_down;
  std::vector<std::uint8_t> fp8_up, fp8_down;
  std::vector<float> scale_up, scale_down;  // FP8: uzman basina olcek (amax / 448)
  [[nodiscard]] std::uint64_t bytes() const;
};
[[nodiscard]] QuantizedExperts quantize_experts(const float* up, const float* down, int experts, int d, int F, WeightType t);
void experts_forward_quantized(const float* xp, const std::int64_t* offsets, const QuantizedExperts& q, float* yp, int threads);

/// Tek rank MoE katmani (ileri + geri; onbellek ileri gecisten).
struct LayerCache {
  Routing routing;
  ExpertCache experts;
  std::vector<float> x;  // [N, d]
};
/// y [N, d] yazilir.
void layer_forward(const float* x, std::int64_t N, const float* router_w, const float* noise_w, const float* up,
                   const float* down, const Spec& s, bool training, std::uint64_t seed, float* y, LayerCache* cache);
/// dy -> dx (yazilir), drouter/dnoise/dup/ddown += . router_loss_grad: aux + z teriminin ust gradyani (0: kayip yok).
void layer_backward(const float* dy, const LayerCache& cache, const float* router_w, const float* noise_w, const float* up,
                    const float* down, const Spec& s, float router_loss_grad, float* dx, float* drouter, float* dnoise,
                    float* dup, float* ddown);

/// Yonlendirici geri (kapi gradyanlari dgate [N, k] + aux/z terimi): dx += , drouter += , dnoise += .
void router_backward(const Routing& r, const float* x, const float* router_w, const float* noise_w, const float* dgate,
                     const Spec& s, float router_loss_grad, float* dx, float* drouter, float* dnoise);

/// Referans (14 testleri): tamamen ayri, token basina double hassasiyetli ileri + geri (permutasyon / gruplama yok).
struct Reference {
  std::vector<double> y, dx, drouter, dup, ddown;
  double aux = 0.0, z = 0.0;
};
[[nodiscard]] Reference reference(const float* x, std::int64_t N, const float* router_w, const float* up, const float* down,
                                  const Routing& routing, const Spec& s, const float* dy, float router_loss_grad);

/// Aktif / toplam FLOP muhasebesi (token basina ileri, MLP kismi): 2 d 2F + 2 F d her aktif uzman icin + yonlendirici.
[[nodiscard]] double active_mlp_flops_per_token(const Spec& s);

}  // namespace moe
}  // namespace nexus
