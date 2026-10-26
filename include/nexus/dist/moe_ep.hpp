#pragma once

/**
 * @file moe_ep.hpp
 * @brief FAZ 14.D: uzman paralelligi (EP) — uzmanlar EP grubunun rank'lerine bitisik bloklar halinde bolunur
 *        (rank r: [r E/ep, (r + 1) E/ep)); her rank kendi tokenlarini yonlendirir, satirlar hedef uzmanin rank'ina
 *        all-to-all(v) ile gider -> yerel gruplu GEMM -> all-to-all geri -> birlestirme. Geri gecis ayni yolun tersi.
 *
 * Tek rank esdegerligi: ep rank'li EP (rank basina `groups` grup) == tek rank, ayni tokenlar arka arkaya ve
 * groups = ep * groups — kapasite ve satir sirasi (uzman, grup, kabul) birebir ayni oldugundan y, dx ve uzman
 * gradyanlari BIT BIT aynidir. Yonlendirici gradyani rank'te kismidir (cagiran all-reduce eder); aux/z kaybi
 * rank'ler arasi toplanan sayim ve olasiliklarla kuresel hesaplanir.
 * Iletisim: once uzman basina satir sayilari (I64), sonra satirlar (FP32). Hesapla ortusme yok (olculur, raporlanir).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/process_group.hpp"
#include "nexus/moe/moe.hpp"

#include <cstdint>
#include <vector>

namespace nexus {
namespace dist {

struct MoeEpStats {
  double a2a_ms = 0.0;      // tum all-to-all (sayim + satir, ileri + geri)
  double compute_ms = 0.0;  // yerel uzman hesabi
  std::uint64_t bytes_sent = 0;
  std::vector<std::int64_t> rows_per_rank;  // bu rank'in her hedefe gonderdigi satir
};

class MoeExpertParallel {
 public:
  MoeExpertParallel(GroupPtr ep, moe::Spec spec);
  [[nodiscard]] int local_experts() const noexcept { return local_; }
  [[nodiscard]] int first_expert() const noexcept { return first_; }

  /// x [N, d] (bu rank'in tokenlari), router_w tam [d, E], up/down yalniz yerel uzmanlar. y [N, d] yazilir.
  void forward(const float* x, std::int64_t N, const float* router_w, const float* up, const float* down, bool training,
               std::uint64_t seed, float* y);
  /// dx yazilir; drouter (kismi), dup/ddown (yerel uzmanlar) += .
  void backward(const float* dy, const float* router_w, const float* up, const float* down, float router_loss_grad, float* dx,
                float* drouter, float* dup, float* ddown);

  [[nodiscard]] const moe::Routing& routing() const noexcept { return cache_.routing; }
  [[nodiscard]] const MoeEpStats& stats() const noexcept { return stats_; }

 private:
  void exchange(const float* send, const std::vector<std::int64_t>& send_rows, float* recv,
                const std::vector<std::int64_t>& recv_rows);
  GroupPtr ep_;
  moe::Spec spec_;
  int ranks_ = 1, local_ = 0, first_ = 0;
  moe::LayerCache cache_;          // yerel yonlendirme + x + xp/yp (gonderen tarafi)
  moe::ExpertCache remote_;        // alinan satirlar uzerinde (uzman sahibi tarafi)
  std::vector<std::int64_t> send_rows_, recv_rows_;     // rank basina satir
  std::vector<std::int64_t> recv_counts_;               // [ep][local] alinan uzman basina satir
  std::vector<std::int64_t> local_offsets_;             // [local + 1] yeniden duzenlenmis
  std::vector<std::int64_t> order_;                     // yeniden duzenlenmis satir -> alinan tampondaki satir
  MoeEpStats stats_;
};

}  // namespace dist
}  // namespace nexus
