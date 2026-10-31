/**
 * @file tensor.cpp
 * @brief Dalga 2: Storage + dtype'li adimli Tensor (gorunumler, adimli kopya, dtype donusumu).
 *
 * Host'ta adimli kopya / donusum burada (genel eleman dongusu). Cihazda yalniz bitisik + ayni dtype kopya
 * (`copy_bytes`) yerlesiktir; adimli ya da donusumlu cihaz kopyasi dagitici uzerinden ("copy" cekirdegi, cihaz
 * kutuphanesi kaydeder) yapilir, yoksa acik hata (sessiz host'a kacis yok).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/tensor.hpp"

#include "nexus/core/fp8.hpp"
#include "nexus/core/half.hpp"
#include "nexus/core/tensor_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

namespace nexus {

// ---- Storage ---------------------------------------------------------------------------------------
std::shared_ptr<Storage> Storage::allocate(std::size_t bytes, MemorySpace space, DeviceRef device) {
  std::shared_ptr<Storage> s(new Storage());
  s->bytes_ = bytes;
  s->space_ = space;
  s->device_ = device;
  if (bytes == 0) return s;
  if (space == MemorySpace::Host) {
    if (!device.is_host()) throw TensorError("Nexus: sayfalanabilir host bellegi yalnizca cpu cihaziyla kullanilir");
    s->data_ = allocate_aligned(bytes);
  } else if (space == MemorySpace::PinnedHost) {
    if (!device.is_host()) throw TensorError("Nexus: sayfa kilitli host bellegi yalnizca cpu cihaziyla kullanilir");
    s->data_ = allocate_pinned(bytes);
  } else if (space == MemorySpace::Device) {
    if (!device.is_accelerator()) throw TensorError("Nexus: cihaz bellegi bir hizlandirici DeviceRef ister");
    s->data_ = allocate_device(device, bytes);
  } else {
    throw TensorError("Nexus: bilinmeyen bellek alani");
  }
  s->owned_ = true;
  return s;
}

std::shared_ptr<Storage> Storage::wrap(void* data, std::size_t bytes, MemorySpace space, DeviceRef device,
                                       std::shared_ptr<void> keep) {
  if (bytes > 0 && data == nullptr) throw TensorError("Nexus: bos olmayan alias icin isaretci gerekli");
  std::shared_ptr<Storage> s(new Storage());
  s->data_ = data;
  s->bytes_ = bytes;
  s->space_ = space;
  s->device_ = device;
  s->owned_ = false;
  s->keep_ = std::move(keep);
  return s;
}

Storage::~Storage() {
  if (owned_ && data_ != nullptr) release_owned(data_);
}

// ---- yardimcilar -------------------------------------------------------------------------------------
Strides contiguous_strides(const Shape& shape) noexcept {
  Strides s{};
  std::int64_t acc = 1;
  for (int axis = static_cast<int>(shape.rank()) - 1; axis >= 0; --axis) {
    s[static_cast<std::size_t>(axis)] = acc;
    acc *= static_cast<std::int64_t>(shape[static_cast<std::size_t>(axis)]);
  }
  return s;
}

namespace {

Shape make_shape(const std::size_t* dims, std::size_t rank) { return Shape(dims, static_cast<std::uint8_t>(rank)); }

void require_axis(const Tensor& t, std::size_t axis, const char* op) {
  if (axis >= t.rank()) throw TensorError(std::string("Nexus: ") + op + ": eksen aralik disinda");
}

// Tek eleman: tamsayi tipleri tam (int64), kayanlar double.
struct Scalar {
  bool is_int;
  std::int64_t i;
  double d;
};

bool is_integer_like(DType t) noexcept {
  return t == DType::I32 || t == DType::I64 || t == DType::I8 || t == DType::U8 || t == DType::Bool;
}

Scalar load(DType t, const void* p) {
  switch (t) {
    case DType::F32: return {false, 0, *static_cast<const float*>(p)};
    case DType::F64: return {false, 0, *static_cast<const double*>(p)};
    case DType::F16: return {false, 0, half_bits_to_float(*static_cast<const std::uint16_t*>(p))};
    case DType::BF16: return {false, 0, bfloat16_bits_to_float(*static_cast<const std::uint16_t*>(p))};
    case DType::F8E4M3: return {false, 0, fp8_to_float(*static_cast<const std::uint8_t*>(p), Fp8Format::E4M3)};
    case DType::F8E5M2: return {false, 0, fp8_to_float(*static_cast<const std::uint8_t*>(p), Fp8Format::E5M2)};
    case DType::I32: return {true, *static_cast<const std::int32_t*>(p), 0};
    case DType::I64: return {true, *static_cast<const std::int64_t*>(p), 0};
    case DType::I8: return {true, *static_cast<const std::int8_t*>(p), 0};
    case DType::U8: return {true, *static_cast<const std::uint8_t*>(p), 0};
    case DType::Bool: return {true, *static_cast<const std::uint8_t*>(p) != 0 ? 1 : 0, 0};
  }
  throw TensorError("Nexus: bilinmeyen dtype");
}

template <typename I>
I checked_int(const Scalar& s, const char* name) {
  if (s.is_int) {
    if (s.i < static_cast<std::int64_t>(std::numeric_limits<I>::min()) ||
        s.i > static_cast<std::int64_t>(std::numeric_limits<I>::max()))
      throw TensorError(std::string("Nexus: tamsayi donusumu aralik disi (") + name + ")");
    return static_cast<I>(s.i);
  }
  if (!std::isfinite(s.d)) throw TensorError(std::string("Nexus: NaN/inf tamsayiya donusturulemez (") + name + ")");
  const double t = std::trunc(s.d);
  // [min, max] sinirlari double'da tam temsil edilmeyebilir (I64): max icin acik ust sinir 2^63
  if (t < static_cast<double>(std::numeric_limits<I>::min()) ||
      t >= static_cast<double>(std::numeric_limits<I>::max()) + 1.0)
    throw TensorError(std::string("Nexus: kayan -> tamsayi aralik disi (") + name + ")");
  return static_cast<I>(t);
}

double as_double(const Scalar& s) noexcept { return s.is_int ? static_cast<double>(s.i) : s.d; }

void store(DType t, void* p, const Scalar& s) {
  switch (t) {
    case DType::F32: *static_cast<float*>(p) = s.is_int ? static_cast<float>(s.i) : static_cast<float>(s.d); return;
    case DType::F64: *static_cast<double*>(p) = as_double(s); return;
    case DType::F16: *static_cast<std::uint16_t*>(p) = float_to_half_bits(static_cast<float>(as_double(s))); return;
    case DType::BF16: *static_cast<std::uint16_t*>(p) = float_to_bfloat16_bits(static_cast<float>(as_double(s))); return;
    case DType::F8E4M3: *static_cast<std::uint8_t*>(p) = float_to_fp8(static_cast<float>(as_double(s)), Fp8Format::E4M3); return;
    case DType::F8E5M2: *static_cast<std::uint8_t*>(p) = float_to_fp8(static_cast<float>(as_double(s)), Fp8Format::E5M2); return;
    case DType::I32: *static_cast<std::int32_t*>(p) = checked_int<std::int32_t>(s, "i32"); return;
    case DType::I64: *static_cast<std::int64_t*>(p) = checked_int<std::int64_t>(s, "i64"); return;
    case DType::I8: *static_cast<std::int8_t*>(p) = checked_int<std::int8_t>(s, "i8"); return;
    case DType::U8: *static_cast<std::uint8_t*>(p) = checked_int<std::uint8_t>(s, "u8"); return;
    case DType::Bool: *static_cast<std::uint8_t*>(p) = (s.is_int ? s.i != 0 : s.d != 0.0) ? 1 : 0; return;
  }
  throw TensorError("Nexus: bilinmeyen dtype");
}

// Satir-ana sirada coklu indeks uzerinde gez; f(src_offset, dst_offset) eleman cinsinden.
template <typename F>
void for_each_index(const Shape& shape, const Strides& a, std::int64_t a0, const Strides& b, std::int64_t b0, F&& f) {
  const std::size_t n = shape.numel();
  if (n == 0) return;
  const std::size_t r = shape.rank();
  if (r == 0) {
    f(a0, b0);
    return;
  }
  std::array<std::size_t, kMaxRank> idx{};
  std::int64_t oa = a0, ob = b0;
  const std::size_t inner = shape[r - 1];
  const std::int64_t sa = a[r - 1], sb = b[r - 1];
  for (std::size_t done = 0; done < n; done += inner) {
    for (std::size_t k = 0; k < inner; ++k) f(oa + static_cast<std::int64_t>(k) * sa, ob + static_cast<std::int64_t>(k) * sb);
    // bir ust eksenleri ilerlet
    for (int axis = static_cast<int>(r) - 2; axis >= 0; --axis) {
      const auto ax = static_cast<std::size_t>(axis);
      ++idx[ax];
      oa += a[ax];
      ob += b[ax];
      if (idx[ax] < shape[ax]) break;
      oa -= a[ax] * static_cast<std::int64_t>(shape[ax]);
      ob -= b[ax] * static_cast<std::int64_t>(shape[ax]);
      idx[ax] = 0;
    }
  }
}

bool host_space(MemorySpace s) noexcept { return s == MemorySpace::Host || s == MemorySpace::PinnedHost; }

}  // namespace

// ---- kurucular -----------------------------------------------------------------------------------------
Tensor::Tensor(Shape shape, MemorySpace space, DeviceRef device) {
  *this = empty(std::move(shape), DType::F32, space, device);
}

Tensor::Tensor(Shape shape, std::span<const float> values, MemorySpace space, DeviceRef device)
    : Tensor(std::move(shape), space, device) {
  if (values.size() != numel()) throw TensorError("Nexus: deger sayisi sekille uyusmuyor");
  if (numel() == 0) return;
  copy_bytes(raw_data(), this->space(), this->device(), values.data(), MemorySpace::Host, DeviceRef::cpu(),
             numel() * sizeof(float));
}

Tensor Tensor::empty(Shape shape, DType dtype, MemorySpace space, DeviceRef device) {
  Tensor t;
  t.dtype_ = dtype;
  t.shape_ = std::move(shape);
  t.strides_ = contiguous_strides(t.shape_);
  t.space_ = space;
  t.device_ = device;
  if (t.shape_.numel() > 0) t.storage_ = Storage::allocate(t.shape_.numel() * size_of(dtype), space, device);
  return t;
}

Tensor Tensor::zeros(Shape shape, MemorySpace space, DeviceRef device) { return zeros(std::move(shape), DType::F32, space, device); }

Tensor Tensor::zeros(Shape shape, DType dtype, MemorySpace space, DeviceRef device) {
  Tensor t = empty(std::move(shape), dtype, space, device);
  t.zero();
  return t;
}

Tensor Tensor::full(Shape shape, float value, MemorySpace space, DeviceRef device) {
  Tensor t(std::move(shape), space, device);
  t.fill(value);
  return t;
}

Tensor Tensor::from_values(Shape shape, std::span<const double> values, DType dtype) {
  Tensor t = empty(std::move(shape), dtype);
  if (values.size() != t.numel()) throw TensorError("Nexus: deger sayisi sekille uyusmuyor");
  auto* base = static_cast<char*>(t.raw_data());
  const std::size_t es = t.element_size();
  for (std::size_t i = 0; i < values.size(); ++i) {
    Scalar s{false, 0, values[i]};
    if (is_integer_like(dtype)) {
      if (!std::isfinite(values[i]) || std::trunc(values[i]) != values[i])
        throw TensorError("Nexus: tamsayi dtype'a tam sayi olmayan deger");
      s = Scalar{true, static_cast<std::int64_t>(values[i]), 0};
      if (dtype == DType::Bool && s.i != 0 && s.i != 1) throw TensorError("Nexus: Bool yalniz 0 / 1");
    }
    store(dtype, base + i * es, s);
  }
  return t;
}

Tensor Tensor::pinned(Shape shape) { return zeros(std::move(shape), MemorySpace::PinnedHost, DeviceRef::cpu()); }

Tensor Tensor::alias(float* data, Shape shape, MemorySpace space, DeviceRef device) {
  return alias_raw(data, std::move(shape), DType::F32, space, device);
}

Tensor Tensor::alias_raw(void* data, Shape shape, DType dtype, MemorySpace space, DeviceRef device) {
  return adopt_shared_raw(data, std::move(shape), dtype, space, device, {});
}

Tensor Tensor::adopt_shared(float* data, Shape shape, MemorySpace space, DeviceRef device, std::shared_ptr<void> keep) {
  return adopt_shared_raw(data, std::move(shape), DType::F32, space, device, std::move(keep));
}

Tensor Tensor::adopt_shared_raw(void* data, Shape shape, DType dtype, MemorySpace space, DeviceRef device,
                                std::shared_ptr<void> keep) {
  if (shape.numel() > 0 && data == nullptr) throw TensorError("Nexus: bos olmayan paylasilan tensör icin isaretci gerekli");
  Tensor t;
  t.dtype_ = dtype;
  t.shape_ = std::move(shape);
  t.strides_ = contiguous_strides(t.shape_);
  t.space_ = space;
  t.device_ = device;
  t.storage_ = Storage::wrap(data, t.shape_.numel() * size_of(dtype), space, device, std::move(keep));
  return t;
}

Tensor Tensor::from_storage(std::shared_ptr<Storage> storage, DType dtype, Shape shape, const Strides& strides,
                            std::int64_t offset) {
  if (!storage) throw TensorError("Nexus: from_storage bos depolama");
  if (offset < 0) throw TensorError("Nexus: from_storage negatif ofset");
  // en kucuk / en buyuk erisilen eleman sinirlar icinde mi
  std::int64_t lo = offset, hi = offset;
  if (shape.numel() > 0) {
    for (std::size_t a = 0; a < shape.rank(); ++a) {
      const std::int64_t span = strides[a] * (static_cast<std::int64_t>(shape[a]) - 1);
      (span < 0 ? lo : hi) += span;
    }
    if (lo < 0 || static_cast<std::size_t>(hi + 1) * size_of(dtype) > storage->bytes())
      throw TensorError("Nexus: from_storage gorunumu depolama sinirlarini asiyor");
  }
  Tensor t;
  t.storage_ = std::move(storage);
  t.dtype_ = dtype;
  t.shape_ = std::move(shape);
  t.strides_ = strides;
  for (std::size_t a = t.shape_.rank(); a < kMaxRank; ++a) t.strides_[a] = 0;
  t.offset_ = offset;
  return t;
}

Tensor::Tensor(const Tensor& other) {
  if (!other.storage_) {
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    strides_ = contiguous_strides(shape_);
    space_ = other.space();
    device_ = other.device();
    return;
  }
  *this = empty(other.shape_, other.dtype_, other.space(), other.device());
  copy_from(other);
}

Tensor::Tensor(Tensor&& other) noexcept { swap(other); }

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    Tensor empty_tensor;
    swap(other);
    other.swap(empty_tensor);  // eski icerik empty_tensor ile birlikte yikilir; kaynak bos kalir
  }
  return *this;
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    Tensor copy(other);
    swap(copy);
  }
  return *this;
}

// ---- metadata / erisim ------------------------------------------------------------------------------------
std::int64_t Tensor::stride(std::size_t axis) const {
  require_axis(*this, axis, "stride");
  return strides_[axis];
}

bool Tensor::is_contiguous() const noexcept {
  std::int64_t expect = 1;
  for (int a = static_cast<int>(shape_.rank()) - 1; a >= 0; --a) {
    const auto ax = static_cast<std::size_t>(a);
    if (shape_[ax] != 1 && strides_[ax] != expect) return false;
    expect *= static_cast<std::int64_t>(shape_[ax]);
  }
  return true;
}

void* Tensor::raw_data() const noexcept {
  if (!storage_ || storage_->data() == nullptr) return nullptr;
  return static_cast<char*>(storage_->data()) + offset_ * static_cast<std::int64_t>(size_of(dtype_));
}

void Tensor::require_f32() const {
  if (dtype_ != DType::F32)
    throw TensorError(std::string("Nexus: bu erisim yalniz F32 icin; tensör ") + std::string(to_string(dtype_)) +
                      " (data_as<T>() kullanin)");
}

float* Tensor::data() {
  require_f32();
  return static_cast<float*>(raw_data());
}

const float* Tensor::data() const {
  require_f32();
  return static_cast<const float*>(raw_data());
}

TensorView<float> Tensor::view() {
  require_f32();
  std::array<std::size_t, kMaxRank> s{};
  for (std::size_t a = 0; a < kMaxRank; ++a) s[a] = static_cast<std::size_t>(std::max<std::int64_t>(strides_[a], 0));
  return TensorView<float>(static_cast<float*>(raw_data()), shape_, s, is_contiguous(), space(), device());
}

TensorView<const float> Tensor::view() const {
  require_f32();
  std::array<std::size_t, kMaxRank> s{};
  for (std::size_t a = 0; a < kMaxRank; ++a) s[a] = static_cast<std::size_t>(std::max<std::int64_t>(strides_[a], 0));
  return TensorView<const float>(static_cast<const float*>(raw_data()), shape_, s, is_contiguous(), space(), device());
}

void Tensor::zero() {
  if (numel() == 0 || raw_data() == nullptr) return;
  if (is_contiguous()) {
    zero_bytes(raw_data(), space(), device(), numel() * element_size());
    bump_version();
    return;
  }
  ops::fill_(*this, 0.0);
}

void Tensor::fill(float value) {
  if (numel() == 0 || raw_data() == nullptr) return;
  if (dtype_ == DType::F32 && is_contiguous()) {
    fill_float(static_cast<float*>(raw_data()), space(), device(), numel(), value);
    bump_version();
    return;
  }
  ops::fill_(*this, value);
}

void Tensor::require_host_f32_index(std::size_t index) const {
  require_f32();
  if (space() == MemorySpace::Device) throw TensorError("Nexus: cihaz isaretcisi host tarafinda operator[] ile okunamaz");
  if (raw_data() == nullptr || index >= numel()) throw TensorError("Nexus: tensör indeksi aralik disinda");
  if (!is_contiguous()) throw TensorError("Nexus: operator[] duz indeksi yalniz bitisik tensörde (item_at kullanin)");
}

float& Tensor::operator[](std::size_t index) {
  require_host_f32_index(index);
  return static_cast<float*>(raw_data())[index];
}

float Tensor::operator[](std::size_t index) const {
  require_host_f32_index(index);
  return static_cast<const float*>(raw_data())[index];
}

double Tensor::item_at(std::span<const std::int64_t> index) const {
  if (!host_space(space())) throw TensorError("Nexus: item_at yalniz host tensöründe");
  if (index.size() != rank()) throw TensorError("Nexus: item_at indeks sayisi rank ile uyusmuyor");
  std::int64_t off = offset_;
  for (std::size_t a = 0; a < rank(); ++a) {
    if (index[a] < 0 || static_cast<std::size_t>(index[a]) >= shape_[a]) throw TensorError("Nexus: item_at indeks aralik disi");
    off += index[a] * strides_[a];
  }
  return as_double(load(dtype_, static_cast<const char*>(storage_->data()) + off * static_cast<std::int64_t>(element_size())));
}

std::vector<double> Tensor::to_double_vector() const {
  if (!host_space(space())) throw TensorError("Nexus: to_double_vector yalniz host tensöründe");
  std::vector<double> out;
  out.reserve(numel());
  if (numel() == 0) return out;
  const auto* base = static_cast<const char*>(storage_->data());
  const auto es = static_cast<std::int64_t>(element_size());
  const Strides zero{};
  for_each_index(shape_, strides_, offset_, zero, 0, [&](std::int64_t o, std::int64_t) {
    out.push_back(as_double(load(dtype_, base + o * es)));
  });
  return out;
}

// ---- gorunum islemleri ----------------------------------------------------------------------------------
Tensor Tensor::view_as(Shape shape) const {
  if (shape.numel() != numel()) throw TensorError("Nexus: view eleman sayisi uyusmuyor");
  if (!is_contiguous()) throw TensorError("Nexus: view bitisik olmayan tensörde yapilamaz (reshape / contiguous kullanin)");
  Tensor t = shallow();
  t.shape_ = std::move(shape);
  t.strides_ = contiguous_strides(t.shape_);
  return t;
}

Tensor Tensor::reshape(Shape shape) const {
  if (shape.numel() != numel()) throw TensorError("Nexus: reshape eleman sayisi uyusmuyor");
  if (is_contiguous()) return view_as(std::move(shape));
  return contiguous().view_as(std::move(shape));
}

Tensor Tensor::narrow(std::size_t axis, std::size_t start, std::size_t length) const {
  return slice(axis, start, start + length, 1);
}

Tensor Tensor::slice(std::size_t axis, std::size_t start, std::size_t end, std::size_t step) const {
  require_axis(*this, axis, "slice");
  if (step == 0) throw TensorError("Nexus: slice adimi 0 olamaz");
  if (start > end || end > shape_[axis]) throw TensorError("Nexus: slice araligi disinda");
  std::vector<std::size_t> dims(shape_.dims(), shape_.dims() + rank());
  dims[axis] = (end - start + step - 1) / step;
  Strides s = strides_;
  s[axis] = strides_[axis] * static_cast<std::int64_t>(step);
  const std::int64_t off = offset_ + strides_[axis] * static_cast<std::int64_t>(start);
  if (!storage_) throw TensorError("Nexus: bos tensör dilimlenemez");
  return from_storage(storage_, dtype_, Shape(dims), s, dims[axis] == 0 ? offset_ : off);
}

Tensor Tensor::transpose(std::size_t a, std::size_t b) const {
  require_axis(*this, a, "transpose");
  require_axis(*this, b, "transpose");
  std::vector<std::size_t> order(rank());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::swap(order[a], order[b]);
  return permute(order);
}

Tensor Tensor::permute(std::span<const std::size_t> order) const {
  if (order.size() != rank()) throw TensorError("Nexus: permute sira uzunlugu rank ile uyusmuyor");
  std::array<bool, kMaxRank> seen{};
  std::vector<std::size_t> dims(rank());
  Strides s{};
  for (std::size_t i = 0; i < order.size(); ++i) {
    if (order[i] >= rank() || seen[order[i]]) throw TensorError("Nexus: permute gecersiz sira");
    seen[order[i]] = true;
    dims[i] = shape_[order[i]];
    s[i] = strides_[order[i]];
  }
  if (!storage_) throw TensorError("Nexus: bos tensör permute edilemez");
  return from_storage(storage_, dtype_, Shape(dims), s, offset_);
}

Tensor Tensor::expand(Shape target) const {
  if (target.rank() < rank()) throw TensorError("Nexus: expand hedef rank'i kucuk");
  const std::size_t lead = target.rank() - rank();
  std::vector<std::size_t> dims(target.rank());
  Strides s{};
  for (std::size_t i = 0; i < target.rank(); ++i) {
    dims[i] = target[i];
    if (i < lead) {
      s[i] = 0;
      continue;
    }
    const std::size_t src = shape_[i - lead];
    if (src == target[i]) {
      s[i] = strides_[i - lead];
    } else if (src == 1) {
      s[i] = 0;
    } else {
      throw TensorError("Nexus: expand yalniz boyu 1 olan eksenleri genisletir");
    }
  }
  if (!storage_) throw TensorError("Nexus: bos tensör genisletilemez");
  return from_storage(storage_, dtype_, Shape(dims), s, offset_);
}

Tensor Tensor::unsqueeze(std::size_t axis) const {
  if (axis > rank()) throw TensorError("Nexus: unsqueeze ekseni aralik disinda");
  if (rank() + 1 > kMaxRank) throw TensorError("Nexus: unsqueeze azami rank'i asiyor");
  std::vector<std::size_t> dims(shape_.dims(), shape_.dims() + rank());
  dims.insert(dims.begin() + static_cast<std::ptrdiff_t>(axis), 1);
  Strides s{};
  for (std::size_t i = 0, j = 0; i < dims.size(); ++i) {
    if (i == axis) {
      s[i] = (axis < rank()) ? strides_[axis] * static_cast<std::int64_t>(shape_[axis]) : 1;
    } else {
      s[i] = strides_[j++];
    }
  }
  if (!storage_) throw TensorError("Nexus: bos tensör unsqueeze edilemez");
  return from_storage(storage_, dtype_, Shape(dims), s, offset_);
}

Tensor Tensor::squeeze(std::size_t axis) const {
  require_axis(*this, axis, "squeeze");
  if (shape_[axis] != 1) throw TensorError("Nexus: squeeze yalniz boyu 1 olan eksende");
  std::vector<std::size_t> dims;
  Strides s{};
  for (std::size_t i = 0, j = 0; i < rank(); ++i) {
    if (i == axis) continue;
    dims.push_back(shape_[i]);
    s[j++] = strides_[i];
  }
  if (!storage_) throw TensorError("Nexus: bos tensör squeeze edilemez");
  return from_storage(storage_, dtype_, make_shape(dims.data(), dims.size()), s, offset_);
}

Tensor Tensor::select(std::size_t axis, std::size_t index) const {
  require_axis(*this, axis, "select");
  if (index >= shape_[axis]) throw TensorError("Nexus: select indeksi aralik disinda");
  return narrow(axis, index, 1).squeeze(axis);
}

// ---- kopyali donusumler -----------------------------------------------------------------------------------
Tensor Tensor::shallow() const {
  Tensor t;
  t.storage_ = storage_;
  t.dtype_ = dtype_;
  t.shape_ = shape_;
  t.strides_ = strides_;
  t.offset_ = offset_;
  t.space_ = space();
  t.device_ = device();
  return t;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) return shallow();
  Tensor out = empty(shape_, dtype_, space(), device());
  out.copy_from(*this);
  return out;
}

Tensor Tensor::to(DType dtype) const {
  Tensor out = empty(shape_, dtype, space(), device());
  out.copy_from(*this);
  return out;
}

Tensor Tensor::to(MemorySpace space, DeviceRef device) const {
  const Tensor src = contiguous();
  Tensor out = empty(shape_, dtype_, space, device);
  if (numel() > 0) copy_bytes(out.raw_data(), space, device, src.raw_data(), src.space(), src.device(), numel() * element_size());
  return out;
}

void Tensor::copy_from(const Tensor& src) {
  if (src.shape_ != shape_) throw TensorError("Nexus: copy_from sekiller uyusmuyor (once expand)");
  if (numel() == 0) return;
  const bool same = src.dtype_ == dtype_;
  // hizli yol: ikisi de bitisik, ayni dtype -> bayt kopyasi (alanlar arasi, cihaz dahil)
  if (same && is_contiguous() && src.is_contiguous()) {
    copy_bytes(raw_data(), space(), device(), src.raw_data(), src.space(), src.device(), numel() * element_size());
    bump_version();
    return;
  }
  if (host_space(space()) && host_space(src.space())) {
    auto* d = static_cast<char*>(storage_->data());
    const auto* s = static_cast<const char*>(src.storage_->data());
    const auto des = static_cast<std::int64_t>(element_size()), ses = static_cast<std::int64_t>(src.element_size());
    if (same) {
      const std::size_t es = element_size();
      for_each_index(shape_, src.strides_, src.offset_, strides_, offset_, [&](std::int64_t so, std::int64_t dso) {
        std::memcpy(d + dso * des, s + so * ses, es);
      });
    } else {
      const DType sd = src.dtype_, dd = dtype_;
      for_each_index(shape_, src.strides_, src.offset_, strides_, offset_, [&](std::int64_t so, std::int64_t dso) {
        store(dd, d + dso * des, load(sd, s + so * ses));
      });
    }
    bump_version();
    return;
  }
  // cihaz: adimli / donusumlu kopya dagitici uzerinden (yoksa acik hata)
  ops::copy_(*this, src);
}

void Tensor::swap(Tensor& other) noexcept {
  using std::swap;
  swap(storage_, other.storage_);
  swap(dtype_, other.dtype_);
  swap(shape_, other.shape_);
  swap(strides_, other.strides_);
  swap(offset_, other.offset_);
  swap(space_, other.space_);
  swap(device_, other.device_);
}

}  // namespace nexus
