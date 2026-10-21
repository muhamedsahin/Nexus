#pragma once

/**
 * @file tensor.hpp
 * @brief Sahip olan ya da baska bir tamponu goruntuleyen float32 tensör.
 *
 * `zeros` / `pinned` / cihaz kurucusu bellek ayirir. `alias` ve
 * `adopt_shared` ayirmaz: `data()` kaynagin isaretcisiyle ayni adrestir.
 * Kopya kurucusu bilerek derin kopyadir; sifir kopya icin `view()` kullanilir.
 *
 * Host doldurma ve okuma bu basliktadir. Cihaz kopyasi `memory.cpp` icindeki
 * CUDA yoluna gider. ROCm / TPU / NPU icin ayirma istegi acik hata verir;
 * o bellege zaten sahip olan cagiran `alias` ile gorunum kurar.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/memory.hpp"
#include "nexus/core/tensor_view.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <utility>

namespace nexus {

class Tensor {
 public:
  Tensor() = default;

  /// Ayrilmis ama doldurulmamis bellek. Sifir icin `zeros` kullanin.
  explicit Tensor(Shape shape, MemorySpace space = MemorySpace::Host, DeviceRef device = DeviceRef::cpu());

  Tensor(Shape shape, std::span<const float> values, MemorySpace space = MemorySpace::Host,
         DeviceRef device = DeviceRef::cpu());

  static Tensor zeros(Shape shape, MemorySpace space = MemorySpace::Host, DeviceRef device = DeviceRef::cpu());
  static Tensor full(Shape shape, float value, MemorySpace space = MemorySpace::Host,
                     DeviceRef device = DeviceRef::cpu());
  static Tensor pinned(Shape shape);

  /// Dis isaretciye bakan gorunum. Yikici bu adresi serbest birakmaz.
  static Tensor alias(float* data, Shape shape, MemorySpace space = MemorySpace::Host,
                      DeviceRef device = DeviceRef::cpu());

  /// `keep` yasadikca `data` gecerlidir. Veri kopyalanmaz; shared_ptr sayaci artar.
  static Tensor adopt_shared(float* data, Shape shape, MemorySpace space, DeviceRef device,
                             std::shared_ptr<void> keep);

  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;
  ~Tensor();

  [[nodiscard]] float* data() noexcept { return data_; }
  [[nodiscard]] const float* data() const noexcept { return data_; }
  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] std::size_t numel() const noexcept { return shape_.numel(); }
  [[nodiscard]] bool empty() const noexcept { return shape_.numel() == 0; }
  [[nodiscard]] MemorySpace space() const noexcept { return space_; }
  [[nodiscard]] DeviceRef device() const noexcept { return device_; }
  [[nodiscard]] bool owns_storage() const noexcept { return ownership_ == Ownership::Owned; }

  [[nodiscard]] TensorView<float> view() noexcept;
  [[nodiscard]] TensorView<const float> view() const noexcept;

  void zero();
  void fill(float value);
  [[nodiscard]] Tensor clone() const { return Tensor(*this); }

  float& operator[](std::size_t index);
  float operator[](std::size_t index) const;

  void swap(Tensor& other) noexcept;

 private:
  enum class Ownership : std::uint8_t { Alias, Owned };

  void allocate();
  void reset() noexcept;
  void require_host_index(std::size_t index) const;

  float* data_ = nullptr;
  Shape shape_{};
  MemorySpace space_ = MemorySpace::Host;
  DeviceRef device_{};
  Ownership ownership_ = Ownership::Alias;
  std::shared_ptr<void> lifetime_{};
};

inline Tensor::Tensor(Shape shape, MemorySpace space, DeviceRef device)
    : shape_(std::move(shape)), space_(space), device_(device) {
  if (shape_.numel() == 0) {
    return;
  }
  allocate();
}

inline Tensor::Tensor(Shape shape, std::span<const float> values, MemorySpace space, DeviceRef device)
    : Tensor(std::move(shape), space, device) {
  if (values.size() != numel()) {
    throw TensorError("Nexus: deger sayisi sekille uyusmuyor");
  }
  if (numel() == 0) {
    return;
  }
  copy_bytes(data_, space_, device_, values.data(), MemorySpace::Host, DeviceRef::cpu(), numel() * sizeof(float));
}

inline Tensor Tensor::zeros(Shape shape, MemorySpace space, DeviceRef device) {
  Tensor tensor(std::move(shape), space, device);
  tensor.zero();
  return tensor;
}

inline Tensor Tensor::full(Shape shape, float value, MemorySpace space, DeviceRef device) {
  Tensor tensor(std::move(shape), space, device);
  tensor.fill(value);
  return tensor;
}

inline Tensor Tensor::pinned(Shape shape) { return zeros(std::move(shape), MemorySpace::PinnedHost, DeviceRef::cpu()); }

inline Tensor Tensor::alias(float* data, Shape shape, MemorySpace space, DeviceRef device) {
  if (shape.numel() > 0 && data == nullptr) {
    throw TensorError("Nexus: bos olmayan alias icin isaretci gerekli");
  }
  Tensor tensor;
  tensor.data_ = data;
  tensor.shape_ = std::move(shape);
  tensor.space_ = space;
  tensor.device_ = device;
  tensor.ownership_ = Ownership::Alias;
  return tensor;
}

inline Tensor Tensor::adopt_shared(float* data, Shape shape, MemorySpace space, DeviceRef device,
                                   std::shared_ptr<void> keep) {
  if (shape.numel() > 0 && data == nullptr) {
    throw TensorError("Nexus: bos olmayan paylasilan tensör icin isaretci gerekli");
  }
  Tensor tensor = alias(data, std::move(shape), space, device);
  tensor.lifetime_ = std::move(keep);
  return tensor;
}

inline Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), space_(other.space_), device_(other.device_), ownership_(Ownership::Alias) {
  if (other.numel() == 0) {
    return;
  }
  allocate();
  copy_bytes(data_, space_, device_, other.data_, other.space_, other.device_, other.numel() * sizeof(float));
}

inline Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    Tensor copy(other);
    swap(copy);
  }
  return *this;
}

inline Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      shape_(other.shape_),
      space_(other.space_),
      device_(other.device_),
      ownership_(other.ownership_),
      lifetime_(std::move(other.lifetime_)) {
  other.data_ = nullptr;
  other.shape_ = Shape{};
  other.ownership_ = Ownership::Alias;
  other.space_ = MemorySpace::Host;
  other.device_ = DeviceRef::cpu();
}

inline Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = other.data_;
    shape_ = other.shape_;
    space_ = other.space_;
    device_ = other.device_;
    ownership_ = other.ownership_;
    lifetime_ = std::move(other.lifetime_);
    other.data_ = nullptr;
    other.shape_ = Shape{};
    other.ownership_ = Ownership::Alias;
    other.space_ = MemorySpace::Host;
    other.device_ = DeviceRef::cpu();
  }
  return *this;
}

inline Tensor::~Tensor() { reset(); }

inline TensorView<float> Tensor::view() noexcept {
  return TensorView<float>(data_, shape_, shape_.contiguous_strides(), true, space_, device_);
}

inline TensorView<const float> Tensor::view() const noexcept {
  return TensorView<const float>(data_, shape_, shape_.contiguous_strides(), true, space_, device_);
}

inline void Tensor::zero() {
  if (numel() == 0 || data_ == nullptr) {
    return;
  }
  zero_bytes(data_, space_, device_, numel() * sizeof(float));
}

inline void Tensor::fill(float value) {
  if (numel() == 0 || data_ == nullptr) {
    return;
  }
  fill_float(data_, space_, device_, numel(), value);
}

inline float& Tensor::operator[](std::size_t index) {
  require_host_index(index);
  return data_[index];
}

inline float Tensor::operator[](std::size_t index) const {
  require_host_index(index);
  return data_[index];
}

inline void Tensor::swap(Tensor& other) noexcept {
  using std::swap;
  swap(data_, other.data_);
  swap(shape_, other.shape_);
  swap(space_, other.space_);
  swap(device_, other.device_);
  swap(ownership_, other.ownership_);
  swap(lifetime_, other.lifetime_);
}

inline void Tensor::allocate() {
  const std::size_t bytes = shape_.numel() * sizeof(float);
  if (space_ == MemorySpace::Host) {
    if (!device_.is_host()) {
      throw TensorError("Nexus: sayfalanabilir host bellegi yalnizca cpu cihaziyla kullanilir");
    }
    data_ = static_cast<float*>(allocate_aligned(bytes));
  } else if (space_ == MemorySpace::PinnedHost) {
    if (!device_.is_host()) {
      throw TensorError("Nexus: sayfa kilitli host bellegi yalnizca cpu cihaziyla kullanilir");
    }
    data_ = static_cast<float*>(allocate_pinned(bytes));
  } else if (space_ == MemorySpace::Device) {
    if (!device_.is_accelerator()) {
      throw TensorError("Nexus: cihaz bellegi bir hizlandirici DeviceRef ister");
    }
    data_ = static_cast<float*>(allocate_device(device_, bytes));
  } else {
    throw TensorError("Nexus: bilinmeyen bellek alani");
  }
  ownership_ = Ownership::Owned;
}

inline void Tensor::reset() noexcept {
  if (ownership_ == Ownership::Owned && data_ != nullptr) {
    release_owned(data_);
  }
  data_ = nullptr;
  lifetime_.reset();
  ownership_ = Ownership::Alias;
}

inline void Tensor::require_host_index(std::size_t index) const {
  if (space_ == MemorySpace::Device) {
    throw TensorError("Nexus: cihaz isaretcisi host tarafinda operator[] ile okunamaz");
  }
  if (data_ == nullptr || index >= numel()) {
    throw TensorError("Nexus: tensör indeksi aralik disinda");
  }
}

inline void swap(Tensor& lhs, Tensor& rhs) noexcept { lhs.swap(rhs); }

}  // namespace nexus
