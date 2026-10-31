#pragma once

/**
 * @file tensor.hpp
 * @brief Dalga 2: dtype'li, adimli (strided), paylasilan depolamali birlesik tensör.
 *
 * Model (PyTorch Storage / TensorImpl ayrimina benzer):
 *   - `Storage`: bayt tamponu + bellek alani + cihaz + surum sayaci. Sahip olan (ayirici) ya da sahip olmayan
 *     (`alias`, `adopt_shared` + yasam tutucu). Birden cok tensör ayni Storage'i paylasabilir.
 *   - `Tensor`: Storage + dtype + sekil + eleman cinsinden adimlar (int64) + eleman ofseti.
 * Gorunum islemleri (`view`, `reshape` bitisikse, `narrow`, `slice`, `transpose`, `permute`, `expand`, `squeeze`,
 * `unsqueeze`) KOPYALAMAZ: ayni Storage'i yeni metadata ile paylasir; yazilar tum gorunumlerde gorunur.
 *
 * Geriye uyumluluk (FAZ 1 float32 tensörü): kurucular, `zeros/full/pinned/alias/adopt_shared`, `data()`,
 * `view()`, `operator[]`, `clone()`, `swap()` ayni anlamda kalir. Kopya kurucusu bilerek DERIN kopyadir
 * (bitisik sonuc); paylasim icin gorunum islemleri kullanilir. `data()` / `operator[]` / `view()` yalniz F32
 * icin gecerlidir; diger dtype'larda acik `TensorError` (sessiz yeniden yorumlama yok) -> `data_as<T>()`.
 *
 * Hesap islemleri `nexus/core/tensor_ops.hpp` (dagitici: islem x dtype x cihaz -> cekirdek; eksik -> acik hata).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"
#include "nexus/core/memory.hpp"
#include "nexus/core/tensor_view.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace nexus {

/// Paylasilan bayt tamponu. Surum sayaci yerinde yazimlarda artar (autograd / onbellek gecersizlestirme icin).
class Storage {
 public:
  /// `bytes` kadar yeni bellek (alanina gore aligned / pinned / cihaz). 0 bayt: bos depolama.
  static std::shared_ptr<Storage> allocate(std::size_t bytes, MemorySpace space, DeviceRef device);
  /// Dis tampon. `keep` bos ise saf alias (yasam cagiranda), degilse `keep` yasadikca gecerli.
  static std::shared_ptr<Storage> wrap(void* data, std::size_t bytes, MemorySpace space, DeviceRef device,
                                       std::shared_ptr<void> keep = {});

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  ~Storage();

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] MemorySpace space() const noexcept { return space_; }
  [[nodiscard]] DeviceRef device() const noexcept { return device_; }
  [[nodiscard]] bool owned() const noexcept { return owned_; }
  [[nodiscard]] std::uint64_t version() const noexcept { return version_.load(std::memory_order_relaxed); }
  void bump_version() noexcept { version_.fetch_add(1, std::memory_order_relaxed); }

 private:
  Storage() = default;
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
  MemorySpace space_ = MemorySpace::Host;
  DeviceRef device_{};
  bool owned_ = false;
  std::shared_ptr<void> keep_{};
  std::atomic<std::uint64_t> version_{0};
};

using Strides = std::array<std::int64_t, kMaxRank>;

class Tensor {
 public:
  Tensor() = default;

  /// Ayrilmis ama doldurulmamis F32 bellek. Sifir icin `zeros` kullanin.
  explicit Tensor(Shape shape, MemorySpace space = MemorySpace::Host, DeviceRef device = DeviceRef::cpu());
  Tensor(Shape shape, std::span<const float> values, MemorySpace space = MemorySpace::Host,
         DeviceRef device = DeviceRef::cpu());

  // ---- olusturma ------------------------------------------------------------------------
  static Tensor empty(Shape shape, DType dtype, MemorySpace space = MemorySpace::Host, DeviceRef device = DeviceRef::cpu());
  static Tensor zeros(Shape shape, MemorySpace space = MemorySpace::Host, DeviceRef device = DeviceRef::cpu());
  static Tensor zeros(Shape shape, DType dtype, MemorySpace space = MemorySpace::Host, DeviceRef device = DeviceRef::cpu());
  static Tensor full(Shape shape, float value, MemorySpace space = MemorySpace::Host,
                     DeviceRef device = DeviceRef::cpu());
  /// Host degerlerinden (double ara deger; tamsayi tipleri icin tam sayi olmayan / aralik disi deger acik hata).
  static Tensor from_values(Shape shape, std::span<const double> values, DType dtype);
  static Tensor pinned(Shape shape);

  /// Dis F32 isaretciye bakan gorunum. Yikici bu adresi serbest birakmaz.
  static Tensor alias(float* data, Shape shape, MemorySpace space = MemorySpace::Host,
                      DeviceRef device = DeviceRef::cpu());
  /// Dtype'li alias (bitisik).
  static Tensor alias_raw(void* data, Shape shape, DType dtype, MemorySpace space = MemorySpace::Host,
                          DeviceRef device = DeviceRef::cpu());
  /// `keep` yasadikca `data` gecerlidir. Veri kopyalanmaz.
  static Tensor adopt_shared(float* data, Shape shape, MemorySpace space, DeviceRef device, std::shared_ptr<void> keep);
  static Tensor adopt_shared_raw(void* data, Shape shape, DType dtype, MemorySpace space, DeviceRef device,
                                 std::shared_ptr<void> keep);
  /// Var olan depolama uzerinde keyfi gorunum (sinirlar denetlenir).
  static Tensor from_storage(std::shared_ptr<Storage> storage, DType dtype, Shape shape, const Strides& strides,
                             std::int64_t offset);

  Tensor(const Tensor& other);             // derin kopya (bitisik)
  Tensor& operator=(const Tensor& other);  // derin kopya
  /// Tasima: kaynak bos tensör olur (FAZ 1 anlami: sekil {}, F32, host).
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;
  ~Tensor() = default;

  // ---- metadata --------------------------------------------------------------------------
  [[nodiscard]] DType dtype() const noexcept { return dtype_; }
  [[nodiscard]] std::size_t element_size() const noexcept { return size_of(dtype_); }
  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] std::uint8_t rank() const noexcept { return shape_.rank(); }
  [[nodiscard]] std::size_t numel() const noexcept { return shape_.numel(); }
  [[nodiscard]] bool empty() const noexcept { return shape_.numel() == 0; }
  [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
  [[nodiscard]] std::int64_t stride(std::size_t axis) const;
  [[nodiscard]] std::int64_t offset() const noexcept { return offset_; }
  [[nodiscard]] bool is_contiguous() const noexcept;
  [[nodiscard]] MemorySpace space() const noexcept { return storage_ ? storage_->space() : space_; }
  [[nodiscard]] DeviceRef device() const noexcept { return storage_ ? storage_->device() : device_; }
  [[nodiscard]] bool owns_storage() const noexcept { return storage_ && storage_->owned(); }
  [[nodiscard]] const std::shared_ptr<Storage>& storage() const noexcept { return storage_; }
  [[nodiscard]] bool shares_storage_with(const Tensor& other) const noexcept {
    return storage_ && storage_ == other.storage_;
  }
  /// Depolama surumu (yerinde yazimlarda artar). Bos tensörde 0.
  [[nodiscard]] std::uint64_t version() const noexcept { return storage_ ? storage_->version() : 0; }
  void bump_version() noexcept {
    if (storage_) storage_->bump_version();
  }

  // ---- veri erisimi ----------------------------------------------------------------------
  /// Ilk elemanin adresi (ofset dahil), her dtype.
  [[nodiscard]] void* raw_data() const noexcept;
  /// F32 isaretci (geriye uyum). F32 degilse TensorError.
  [[nodiscard]] float* data();
  [[nodiscard]] const float* data() const;
  /// T, dtype ile ayni boyutta olmali (orn. F32 -> float, BF16/F16 -> std::uint16_t, I64 -> std::int64_t).
  template <typename T>
  [[nodiscard]] T* data_as() const {
    if (sizeof(T) != element_size()) throw TensorError("Nexus: data_as<T> boyutu dtype ile uyusmuyor");
    return static_cast<T*>(raw_data());
  }

  /// Bitisik F32 gorunum (geriye uyum; adimli tensörde TensorView adimlariyla).
  [[nodiscard]] TensorView<float> view();
  [[nodiscard]] TensorView<const float> view() const;

  void zero();
  void fill(float value);
  [[nodiscard]] Tensor clone() const { return Tensor(*this); }

  float& operator[](std::size_t index);
  float operator[](std::size_t index) const;
  /// Host'ta cok boyutlu indeksle tek eleman (dtype -> double).
  [[nodiscard]] double item_at(std::span<const std::int64_t> index) const;
  /// Host'ta dtype -> double, mantiksal (satir-ana) sirada tum elemanlar.
  [[nodiscard]] std::vector<double> to_double_vector() const;

  // ---- gorunum islemleri (kopyasiz) -------------------------------------------------------
  /// Ayni elemanlar yeni sekil; bitisik olmayan tensörde acik hata (bkz. reshape).
  [[nodiscard]] Tensor view_as(Shape shape) const;
  /// Bitisikse gorunum, degilse bitisik kopya + yeni sekil.
  [[nodiscard]] Tensor reshape(Shape shape) const;
  [[nodiscard]] Tensor narrow(std::size_t axis, std::size_t start, std::size_t length) const;
  /// [start, end) adim step (> 0).
  [[nodiscard]] Tensor slice(std::size_t axis, std::size_t start, std::size_t end, std::size_t step = 1) const;
  [[nodiscard]] Tensor transpose(std::size_t a, std::size_t b) const;
  [[nodiscard]] Tensor permute(std::span<const std::size_t> order) const;
  /// Boyu 1 olan eksenler (ya da yeni bastaki eksenler) adim 0 ile genisler.
  [[nodiscard]] Tensor expand(Shape shape) const;
  [[nodiscard]] Tensor unsqueeze(std::size_t axis) const;
  [[nodiscard]] Tensor squeeze(std::size_t axis) const;
  [[nodiscard]] Tensor select(std::size_t axis, std::size_t index) const;

  /// Ayni depolama + ayni metadata (sig kopya; kopya kurucusu derindir).
  [[nodiscard]] Tensor shallow() const;

  // ---- kopyali donusumler ----------------------------------------------------------------
  /// Bitisikse kendisi (paylasimli), degilse bitisik kopya.
  [[nodiscard]] Tensor contiguous() const;
  /// Dtype donusumu (her zaman yeni depolama; ayni dtype ise bitisik kopya). Kayan -> tamsayi: sifira dogru
  /// kesme; NaN / aralik disi acik hata. Kayan -> BF16 / F16: RNE.
  [[nodiscard]] Tensor to(DType dtype) const;
  /// Bellek alani / cihaz degisimi (bitisik kopya).
  [[nodiscard]] Tensor to(MemorySpace space, DeviceRef device) const;
  /// Ayni sekilli `src`'den yerinde kopya (dtype donusumu + adimlar). Surumu artirir.
  void copy_from(const Tensor& src);

  void swap(Tensor& other) noexcept;

 private:
  void require_host_f32_index(std::size_t index) const;
  void require_f32() const;

  std::shared_ptr<Storage> storage_{};
  DType dtype_ = DType::F32;
  Shape shape_{};
  Strides strides_{};
  std::int64_t offset_ = 0;
  // storage_ yokken (bos tensör) bildirilen alan / cihaz
  MemorySpace space_ = MemorySpace::Host;
  DeviceRef device_{};
};

[[nodiscard]] Strides contiguous_strides(const Shape& shape) noexcept;

inline void swap(Tensor& lhs, Tensor& rhs) noexcept { lhs.swap(rhs); }

}  // namespace nexus
