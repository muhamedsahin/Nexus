#pragma once

/**
 * @file pipeline.hpp
 * @brief FAZ 12.E: ardisik (pipeline) paralellik zamanlamalari — GPipe, 1F1B, interleaved 1F1B (sanal
 *        asamalar), ZB-H1 (zero-bubble, B'yi giris gradyani B ve agirlik gradyani W'ye boler).
 *
 * Model pp * v parcaya (chunk) bolunur; parca g, pp rank'i g % pp'de (yerel indeks g / pp) — Megatron
 * interleaved duzeni. v = 1: klasik asamalar. Eylemler: F(mb, parca), B(mb, parca) (ZB-H1'de yalniz giris
 * gradyani), W(mb, parca) (ZB-H1 agirlik gradyani).
 *
 * Siralar (her parca icin mikro yiginlarin B/W sirasi artan: gradyan birikimi mikro yigin sirasiyla,
 * zamanlamalar arasi BIT BIT ayni sonuc):
 *   GPipe: tum F'ler (mb 0..m-1), sonra tum B'ler (mb 0..m-1).
 *   1F1B: isinma min(pp - r - 1, m) F, sonra F/B donusumlu, sogutma B'leri.
 *   Interleaved 1F1B (Narayanan ve ark. 2021): m % pp == 0; k. ileri eylem parca (k / pp) % v,
 *     mb (k / (pp v)) pp + k % pp; geri eylemde parca v - 1 - (k / pp) % v. Isinma
 *     (pp - r - 1) * 2 + (v - 1) * pp (en cok m v).
 *   ZB-H1 (Qi ve ark. 2023): 1F1B sirasi; W'ler FIFO ertelenir, bir sonraki F/B'nin girisi henuz
 *     gelmemisse (kabarcik) ya da bekleyen W sayisi pp - r'yi gecerse yurutulur; kalanlar sonda.
 *     Yurutucu karari calisma aninda verir (statik listede W yok; make_schedule ZB-H1 icin W'siz 1F1B dondurur).
 *
 * Teorik kabarcik orani (bos / toplam, tF + tB + tW ayni her mikro yigin):
 *   GPipe, 1F1B: (p - 1) / (m + p - 1)
 *   Interleaved: (p - 1) / (v m + p - 1)
 *   ZB-H1: (p - 1)(tF + tB - tW) / (m (tF + tB + tW) + (p - 1)(tF + tB - tW))   (tB: giris gradyani)
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace nexus {
namespace dist {

enum class PipeSchedule : std::uint8_t { GPipe, OneF1B, Interleaved1F1B, ZeroBubbleH1 };
[[nodiscard]] const char* to_string(PipeSchedule s) noexcept;
/// "gpipe" | "1f1b" | "interleaved_1f1b" | "zb_h1" (bilinmeyen: DistError).
[[nodiscard]] PipeSchedule parse_pipe_schedule(const std::string& name);

struct PipeAction {
  enum class Kind : std::uint8_t { F, B, W };
  Kind kind = Kind::F;
  int mb = 0;
  int chunk = 0;  // global parca indeksi
  [[nodiscard]] std::string str() const;
};

/// Rank r'nin eylem sirasi. Gecersiz kombinasyonlar (interleaved: m % pp != 0) acik hata.
[[nodiscard]] std::vector<PipeAction> make_schedule(PipeSchedule s, int pp, int rank, int microbatches,
                                                    int virtual_stages);

[[nodiscard]] double theoretical_bubble(PipeSchedule s, int pp, int microbatches, int virtual_stages, double tf = 1.0,
                                        double tb = 1.0, double tw = 1.0);

/// Katman maliyetlerini `parts` ardisik parcaya boler (en buyuk parca maliyetini en aza indirir; DP).
/// Donus: parca basina [baslangic, bitis) katman araligi. Her parca en az bir katman (parts <= n).
[[nodiscard]] std::vector<std::pair<int, int>> balance_partition(const std::vector<double>& costs, int parts);

}  // namespace dist
}  // namespace nexus
