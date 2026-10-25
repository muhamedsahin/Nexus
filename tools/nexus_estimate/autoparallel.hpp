#pragma once

/**
 * @file autoparallel.hpp
 * @brief FAZ 12.G: otomatik paralellik onericisi — verilen kume ve model icin (dp, tp, pp, cp, mikro-yigin,
 *        yeniden hesaplama, zero) kombinasyonlarini tahmini adim suresi ve bellekle siralar; en iyi N'i verir.
 *
 * Kapali bicim model (varsayimlar acik; olcum degil — KURAL 3: raporda "tahmin"):
 *  - Bellek (aygit basina): statik = param/grad/optim/ana kopya (tp * pp) ile, ZeRO evresine gore ayrica
 *    dp * cp ile bolunur (FAZ 0 estimate_memory ile ayni ZeRO kurali). Aktivasyon Korthikanti ve ark.
 *    (arXiv:2205.05198) Tablo 2: katman basina s b h (10 + 24/t + 5 a s/(h t)) bayt (TP), SP ile
 *    s b h (34/t + 5 a s/(h t)); tam yeniden hesaplamada 2 s b h / t (SP). CP dizi s'yi cp'ye boler. 1F1B'de
 *    ilk asama pp mikro yigin x L/pp katman = L katman aktivasyonu tutar. Logit: 2 b s V / (tp cp) eleman.
 *  - Hesap: 6 N + 12 L h s FLOP/token (FAZ 0), tam yeniden hesaplama 4/3 kat; aygit verimi `mfu`
 *    (buyuk GEMM'lerde erisilen zirve kesri; kullanicidan) x parca boyu carpani r/(r+128) * c/(c+128)
 *    (r = mikro yigin basina yerel token b s / cp, c = h / tp; kalibre edilmemis varsayim).
 *  - TP iletisimi: katman ve mikro yigin basina 4 all-reduce (SP: 4 AG + 4 RS, ayni hacim) b s h elem;
 *    halka 2 (t-1)/t bayt / bant. tp <= dugum basina GPU ise dugum ici bant, aksi halde dugumler arasi.
 *  - DP iletisimi: gradyan all-reduce (ZeRO-3: 1.5x hacim) param/(tp pp) * grad_bytes; geri gecisle
 *    `dp_overlap` kesri ortusur. Grup dugum disina tasarsa dugumler arasi bant.
 *  - PP: 1F1B kabarcik orani (p - 1)/(m + p - 1); toplam = hesap / (1 - kabarcik).
 *  - CP: ring K/V dolasimi (cp - 1) 2 b (s/cp) h_kv elem katman basina; yarisi ortusmez varsayilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "estimate.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace nexus {
namespace estimate {

struct Cluster {
  int gpus = 8;
  int gpus_per_node = 8;
  double mem_gib = 80.0;
  double peak_tflops = 312.0;    // kullanilan sayi bicimi icin tepe dense TFLOPS
  double intra_gbps = 300.0;     // dugum ici kollektif bant (GB/s, busbw)
  double inter_gbps = 25.0;      // dugumler arasi (GB/s, busbw)
  double mfu = 0.5;              // GEMM verimi (zirve kesri)
  double dp_overlap = 0.7;       // DP gradyan iletisiminin geri gecisle ortusen kesri
  double reserve_gib = 2.0;      // ayirici/CUDA baglami payi
};

struct AutoOptions {
  std::int64_t global_batch_tokens = 1 << 22;  // 4M token
  std::vector<int> micro_batches{1, 2, 4, 8};
  std::vector<int> zero_stages{0, 1, 2, 3};
  bool try_recompute = true;
  bool sequence_parallel = true;
  int max_cp = 8;
  int top = 3;
};

struct Candidate {
  int dp = 1, tp = 1, pp = 1, cp = 1, zero = 0, micro_batch = 1;
  bool recompute = false;
  std::int64_t grad_accum = 1;
  double memory_gib = 0.0;
  double static_gib = 0.0, activation_gib = 0.0, logits_gib = 0.0;
  double compute_s = 0.0, tp_comm_s = 0.0, cp_comm_s = 0.0, dp_comm_s = 0.0, bubble = 0.0;
  double step_s = 0.0;
  double mfu = 0.0;  // model FLOP / (adim * gpus * zirve)
  bool fits = false;
};

struct AutoResult {
  std::vector<Candidate> best;      // siralanmis, sigan en iyi `top`
  std::int64_t evaluated = 0;       // degerlendirilen gecerli kombinasyon
  std::int64_t feasible = 0;        // bellege sigan
  std::string note;                 // hic sigan yoksa neden
};

[[nodiscard]] AutoResult auto_parallel(const ModelShape& shape, const DTypePlan& dtype, const Cluster& cluster,
                                       const AutoOptions& options);

/// Tek bir plani degerlendirir (test ve rapor icin).
[[nodiscard]] Candidate evaluate_plan(const ModelShape& shape, const DTypePlan& dtype, const Cluster& cluster,
                                      const AutoOptions& options, int dp, int tp, int pp, int cp, int zero, int micro_batch,
                                      bool recompute);

}  // namespace estimate
}  // namespace nexus
