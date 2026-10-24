#pragma once

/**
 * @file estimate.hpp
 * @brief Kapali bicim parametre/bellek/FLOP tahmin formulleri (FAZ 0).
 *
 * Kaynaklar ve kapsamlari:
 *  - Parametre sayimi: dogrudan mimari tanimindan (GPT-2 / LLaMA ailesi).
 *    Kalibrasyon: GPT-2 124M (n_layer=12, d_model=768, n_head=12, d_ff=3072,
 *    vocab=50257, ogrenilen konum=1024, bias=var, LN, bagli head) icin
 *    124.439.808 parametre; nominal 124M'e gore ~%0,35 fark. LLaMA-7B
 *    (32/4096/32/11008, vocab=32000, bias yok, SwiGLU, RMSNorm, ayrik head)
 *    icin 6.738.415.616; yayimlanan ~6,74B ile ~%0,02 fark.
 *  - Egitim FLOP: Kaplan ve ark. "Scaling Laws for Neural Language Models"
 *    (arXiv:2001.08361) ve Hoffmann ve ark. Chinchilla (arXiv:2203.15556)
 *    yaklasimi C ≈ 6·N·T; dikkat puan matrislerinin ek terimi
 *    PaLM (arXiv:2204.02311) Ek B'den: token basina 12·n_layer·d_model·s.
 *  - Aktivasyon bellegi: Korthikanti ve ark. "Reducing Activation
 *    Recomputation in Large Transformer Models" (arXiv:2205.05198), katman
 *    basina s·b·h·(34 + 5·a·s/h) bayt (2 bayt/eleman varsayimi); tam
 *    yeniden hesaplama secilirse katman basina yalnizca 2·s·b·h saklanir.
 *    FAZ 0 taban cizgisi FP32 oldugundan eleman boyutuyla olceklenir.
 *  - ZeRO bolusumu: Rajbhandari ve ark. (arXiv:1910.02054): optimizer
 *    durumu evre 1'de, gradyan evre 2'de, parametre evre 3'te dp'ye bolunur.
 *  - Cati/parcalanma orani: FAZ 0'da k64 havuzu (CachingAllocator) henuz
 *    yok; defter yalnız gercek ayirmalari sayar. Parcalanma orani FAZ 1'in
 *    metriğidir (reserved/current); burada uydurulmaz, 'n/a' raporlanir.
 *
 * MFU ust siniri cati modeliyle uretilmez: flops_per_step ve adim basina
 * hareket eden bayt verilir; kullanicinin verdigi aygit zirve FLOP/s ve
 * bant genisligiyle roofline siniri (min(zirve, yogunluk·bant)) hesaplanir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>

namespace nexus {
namespace estimate {

/// Mimari tanimi (GPT-2 / LLaMA ailesinin ortak parametreleri).
struct ModelShape {
  std::int64_t n_layer = 12;
  std::int64_t d_model = 768;
  std::int64_t n_head = 12;
  std::int64_t n_kv_head = 12;  // GQA: < n_head; MHA: = n_head
  std::int64_t d_ff = 3072;
  std::int64_t vocab = 50257;
  std::int64_t seq_len = 1024;
  std::int64_t micro_batch = 1;
  std::int64_t learned_pos = 0;  // GPT-2: seq_len; LLaMA/RoPE: 0
  bool tie_head = true;          // GPT-2: true; LLaMA: false
  bool ffn_swiglu = false;       // GPT-2: false (GELU, 2 matris); LLaMA: true (3 matris)
  bool bias = true;              // GPT-2: true; LLaMA: false
  bool rmsnorm = false;          // GPT-2: false (LayerNorm); LLaMA: true (RMSNorm)
  std::int64_t gradient_accum = 1;
};

/// Eleman boyutlari (bayt). FAZ 0 taban cizgisi FP32.
struct DTypePlan {
  std::int64_t param_bytes = 4;
  std::int64_t grad_bytes = 4;
  std::int64_t optim_bytes = 8;   // Adam m+v; FP32'de 4+4
  std::int64_t master_bytes = 0;  // karisik kesinlikte FP32 ana kopya
  std::int64_t activation_elem_bytes = 4;  // aktivasyon eleman boyutu (FP32 taban)
};

/// Paralellik ve yeniden hesaplama plani.
struct Plan {
  std::int64_t dp = 1;
  std::int64_t tp = 1;
  std::int64_t pp = 1;
  std::int64_t zero_stage = 0;  // 0..3 (Rajbhandari ve ark.)
  bool full_recompute = false;  // tam aktivasyon yeniden hesaplama
};

/// Parametre dokumu.
struct ParamBreakdown {
  std::int64_t token_embed = 0;
  std::int64_t pos_embed = 0;
  std::int64_t attention = 0;
  std::int64_t ffn = 0;
  std::int64_t norms = 0;
  std::int64_t head = 0;  // yalnizca ayrik (untied) head
  std::int64_t total = 0;
};

/// Bellek tahmini (bayt). "static" = param+grad+optim+master, planla bolunmus.
struct MemoryEstimate {
  std::int64_t param_bytes = 0;       // bolunmemis toplam
  std::int64_t grad_bytes = 0;
  std::int64_t optim_bytes = 0;
  std::int64_t master_bytes = 0;
  std::int64_t static_sharded = 0;    // aygit basina (tp·pp·ZeRO bolunumlu)
  std::int64_t activation_bytes = 0;  // aygit basina, mikro-yigin basina
  std::int64_t logits_bytes = 0;      // b·s·V·elem + grad kopyasi
  std::int64_t per_gpu_bytes = 0;     // static_sharded + activation + logits
  bool fits_budget = false;
  std::int64_t budget_bytes = 8ll << 30;  // 8 GiB varsayilan
};

/// FLOP tahmini.
struct FlopEstimate {
  std::int64_t matmul_params = 0;   // 6N'deki N (embedding lookup haric)
  std::int64_t tokens_per_step = 0; // micro_batch·seq_len·dp·grad_accum
  double flops_per_step = 0.0;      // tokens·(6N + 12·L·d·s)
  double flops_per_token = 0.0;
};

/// Parametre sayimi (dogrudan mimariden).
[[nodiscard]] ParamBreakdown count_parameters(const ModelShape& shape);

/// Bellek tahmini; d_model % n_head != 0 veya n_kv_head bolmuyorsa tanimsizdir
/// ve std::invalid_argument firlatir.
[[nodiscard]] MemoryEstimate estimate_memory(const ModelShape& shape, const DTypePlan& dtype, const Plan& plan);

/// FLOP tahmini (Kaplan/Chinchilla 6N + PaLM dikkat terimi).
[[nodiscard]] FlopEstimate estimate_flops(const ModelShape& shape, const Plan& plan);

/**
 * @brief Roofline ust siniri: verilen aygit zirvesi ve bant genisligiyle
 *        adim basina erisilebilir en yuksek FLOP hizi.
 * @param flops_per_step adim basina FLOP (estimate_flops'tan)
 * @param bytes_per_step adim basina hareket eden en az bayt (statik+aktiv)
 * @param peak_flops_per_s aygit zirve FLOP/s (kullanicidan; uydurulmaz)
 * @param mem_bytes_per_s aygit bant genisligi (kullanicidan)
 */
struct Roofline {
  double intensity = 0.0;         // flops/bayt
  double ceiling_flops_per_s = 0.0;  // min(zirve, yogunluk·bant)
};
[[nodiscard]] Roofline roofline_bound(double flops_per_step, double bytes_per_step, double peak_flops_per_s,
                                      double mem_bytes_per_s);

}  // namespace estimate
}  // namespace nexus
