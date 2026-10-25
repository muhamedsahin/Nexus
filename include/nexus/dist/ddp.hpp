#pragma once

/**
 * @file ddp.hpp
 * @brief FAZ 10.3: DistributedDataParallel ortak parcalari — secenekler, kova planlayici,
 *        host kova indirgeyici, ortusme istatistigi. Arka uca ozel DDP (CPU nn::Module:
 *        katman katman geri gecis; CUDA GptModel: katman kancasi + iletisim is parcacigi)
 *        dil modeli egitmeni arka uclarinda.
 *
 * Kova: geri gecis sirasiyla hazir olan "asama"lar (katman / kafa / gomme dilimleri) ardisik
 * birlestirilir, toplam bayt bucket_bytes'i gecince kova kapanir (PyTorch DDP gibi ~25 MB).
 * Ilk senkron adimda asamalarin gercek hazir olma sirasi kaydedilir; ilk tahminden farkliysa
 * kovalar bu sirayla yeniden kurulur (rebuilt_buckets()). Indirgeme SUM: egitmen mikro adim
 * agirligini (gecerli_token_i / KURESEL toplam) dlogits'e uyguladigi icin toplam = kuresel token
 * agirlikli ortalama. no_sync: biriktirme mikro adimlarinda iletisim yok (yalniz son mikro adim).
 * bf16_compress: kova BF16'ya cevrilip BF16 all-reduce (her adimda FP32'de toplanip yuvarlanir),
 * sonra FP32'ye acilir (PyTorch bf16_compress_hook). PowerSGD yok.
 * find_unused_parameters: kovalar erken baslatilmaz; geri gecis bitince (kullanilmayan
 * parametrelerin sifir gradyani dahil) hepsi indirgenir. ranks_per_node > 0: hiyerarsik all-reduce.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/process_group.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nexus {
namespace dist {

struct DdpOptions {
  std::size_t bucket_bytes = std::size_t{25} << 20;
  bool overlap = true;
  bool bf16_compress = false;
  bool find_unused_parameters = false;
  int ranks_per_node = 0;
};

/// FAZ 10.4 olcek kurallari: global yigin = micro * accum * dp. Taban LR'yi taban global yigindan yeni
/// global yigina tasir: "linear" (Goyal ve ark. 2017) ya da "sqrt" (Adam icin sik kullanilan; Hoffer ve ark.).
[[nodiscard]] inline double scale_learning_rate(double base_lr, std::int64_t base_global_batch,
                                                std::int64_t new_global_batch, bool sqrt_rule = false) {
  if (base_global_batch < 1 || new_global_batch < 1) {
    throw DistError("Nexus DDP: yigin boyutlari >= 1");
  }
  const double r = static_cast<double>(new_global_batch) / static_cast<double>(base_global_batch);
  return base_lr * (sqrt_rule ? std::sqrt(r) : r);
}

/// Duz bir gradyan uzayinda [offset, offset + numel) dilimi (eleman cinsinden).
struct GradSlice {
  std::size_t offset = 0;
  std::size_t numel = 0;
};

/// Ayni anda hazir olan dilimler (bir katmanin tum gradyanlari gibi).
struct Stage {
  int id = 0;
  std::vector<GradSlice> slices;
};

struct Bucket {
  std::vector<int> stages;          // icerdigi asama kimlikleri (hazir olma sirasiyla)
  std::vector<GradSlice> slices;    // paketleme sirasi
  std::size_t numel = 0;
};

/// stages hazir olma sirasinda verilir.
[[nodiscard]] std::vector<Bucket> plan_buckets(const std::vector<Stage>& stages, std::size_t bucket_bytes,
                                               std::size_t element_bytes = 4);

/// Host'ta tek kova: paketlenmis FP32 tampon -> asenkron all-reduce (SUM) -> acma.
class HostBucket {
 public:
  /// Kendi (sayfalanabilir) tamponu; pad_to > 1: hiyerarsik icin uzunluk pad_to'nun katina yuvarlanir.
  void resize(std::size_t numel, int pad_to);
  /// Dis tampon (or. pinned host): capacity >= numel, dolgu alani sifir olmali.
  void bind(float* external, std::size_t numel, std::size_t capacity);
  [[nodiscard]] float* data() noexcept { return external_ != nullptr ? external_ : data_.data(); }
  [[nodiscard]] std::size_t numel() const noexcept { return numel_; }
  void launch(ProcessGroup& pg, const DdpOptions& options);
  void finish();
  [[nodiscard]] const WorkPtr& work() const noexcept { return work_; }

 private:
  [[nodiscard]] std::size_t capacity() const noexcept { return external_ != nullptr ? capacity_ : data_.size(); }
  std::vector<float> data_;
  float* external_ = nullptr;
  std::size_t capacity_ = 0;
  std::vector<std::uint16_t> half_;
  std::size_t numel_ = 0;
  WorkPtr work_;
  bool compressed_ = false;
};

/// Bir senkron adimin olcumu (ms, geri gecis baslangicina gore).
struct DdpStepStats {
  double backward_ms = 0.0;           // geri gecis (son mikro adim) suresi
  double comm_first_start_ms = -1.0;  // ilk kovanin iletisim baslangici
  double comm_last_end_ms = 0.0;      // son kovanin iletisim bitisi
  double comm_busy_ms = 0.0;          // kovalarin iletisim surelerinin toplami
  double comm_overlapped_ms = 0.0;    // iletisimin geri gecisle kesisen kismi
  int buckets = 0;
  bool rebuilt = false;               // bu adimda kovalar gercek siraya gore yeniden kuruldu
};

}  // namespace dist
}  // namespace nexus
