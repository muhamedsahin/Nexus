/**
 * @file test_phase1_mirrored_tensor.cpp
 * @brief Faz 1: MirroredTensor host/cihaz tutarlilik sozlesmesi.
 *
 * Denenenler:
 *   - CPU (unified) modda tek tampon, senkronlarin no-op olmasi,
 *   - durum makinesi: HostAhead -> Synced -> DeviceAhead -> Synced,
 *   - iki tarafi senkronsuz degistirmenin TensorError atmasi,
 *   - CUDA'da H2D/D2H gidis-donus, bayt sayaclari, asenkron kopya,
 *   - kopya akisinin cihaz akisiyla olaylar uzerinden siralanmasi,
 *   - kararli durumda senkron dongusunun heap ayirmamasi,
 *   - nexus_model::Tensor ve matrix_pro::Tensor aynalarinin sifir kopya olmasi,
 *   - tasima semantigi ve yikicinin kayitli bellegi geri vermesi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/nexus.hpp"
#include "nexus_test.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#if defined(NEXUS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

using nexus::DeviceRef;
using nexus::MemorySpace;
using nexus::MirroredTensor;
using nexus::MirrorState;
using nexus::Shape;
using nexus_test::AllocationScope;
using nexus_test::expect;

template <typename Action>
bool throws_tensor_error(Action&& action) {
  try {
    action();
  } catch (const nexus::TensorError&) {
    return true;
  }
  return false;
}

void test_unified_cpu() {
  nexus_test::section("unified (CPU) ayna");
  MirroredTensor mirror(Shape{4, 8}, DeviceRef::cpu());
  expect(mirror.unified(), "CPU cihazi unified moddur");
  expect(mirror.state() == MirrorState::Synced, "unified ayna Synced baslar");
  expect(mirror.host_data() != nullptr && mirror.device_data() == mirror.host_data(),
         "unified modda host ve cihaz ayni adres");
  expect(mirror.host_view().aligned_to(64), "unified host 64 bayt hizali");
  expect(nexus::owns_allocation(mirror.host_data()), "unified host Nexus kaydinda");

  bool zeroed = true;
  for (std::size_t i = 0; i < mirror.numel(); ++i) {
    zeroed = zeroed && mirror.host_data()[i] == 0.0f;
  }
  expect(zeroed, "sahip olunan host sifirla baslar");

  {
    AllocationScope scope;
    mirror.modify_host();
    mirror.sync_device();
    mirror.modify_device();
    mirror.sync_host();
    mirror.synchronize();
    expect(scope.count() == 0, "unified senkronlar heap ayirmaz");
  }
  expect(mirror.bytes_to_device() == 0 && mirror.bytes_to_host() == 0, "unified modda hicbir bayt kopyalanmaz");
  expect(!mirror.pin_host(), "unified modda pin_host anlamsizdir (false)");
  expect(mirror.device_view().data() == mirror.host_view().data(), "unified cihaz gorunumu host adresidir");

  nexus_model::Tensor model = nexus_model::Tensor::from_values({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  const std::uint64_t before = nexus::allocation_count();
  MirroredTensor model_mirror = nexus::mirror(model, DeviceRef::cpu());
  expect(model_mirror.host_data() == model.data(), "mirror(nexus_model::Tensor) host adresini korur");
  expect(model_mirror.device_data() == model.data(), "CPU aynasi cihaz tarafinda da ayni adres");
  expect(nexus::allocation_count() == before, "CPU aynasi bellek ayirmaz");
  expect(model_mirror.shape() == Shape({2, 3}), "ayna sekli model sekliyle ayni");

  float values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float other[4] = {};
  expect(throws_tensor_error([&] { (void)MirroredTensor::wrap(values, other, Shape{4}, DeviceRef::cpu(),
                                                               MirrorState::Synced); }),
         "CPU wrap'inda iki farkli adres reddedilir");
  expect(throws_tensor_error([] { MirroredTensor hybrid(Shape{4}, DeviceRef::hybrid()); }),
         "hybrid politika somut ayna cihazi degildir");
  expect(throws_tensor_error([] { (void)MirroredTensor::wrap_host(nullptr, Shape{4}, DeviceRef::cpu()); }),
         "bos olmayan wrap_host icin nullptr reddedilir");

  MirroredTensor empty(Shape{0}, DeviceRef::cpu());
  expect(empty.bytes() == 0 && empty.host_data() == nullptr, "bos ayna bellek ayirmaz");
}

void test_move_and_release() {
  nexus_test::section("tasima ve serbest birakma");
  MirroredTensor source(Shape{16}, DeviceRef::cpu());
  float* const host = source.host_data();
  source.host_data()[3] = 7.0f;

  MirroredTensor moved(std::move(source));
  expect(moved.host_data() == host && moved.host_data()[3] == 7.0f, "tasima isaretciyi devreder");
  expect(source.host_data() == nullptr && source.numel() == 0, "tasinan kaynak bos kalir");

  MirroredTensor assigned(Shape{2}, DeviceRef::cpu());
  float* const replaced = assigned.host_data();
  assigned = std::move(moved);
  expect(assigned.host_data() == host, "tasima atamasi yeni tamponu alir");
  expect(!nexus::owns_allocation(replaced), "tasima atamasi eski tamponu serbest birakir");
  {
    MirroredTensor scoped(Shape{8}, DeviceRef::cpu());
    float* const scoped_host = scoped.host_data();
    expect(nexus::owns_allocation(scoped_host), "kapsam icinde host kayitli");
    scoped.swap(assigned);
    expect(assigned.host_data() == scoped_host, "swap tamponlari degistirir");
  }
  expect(!nexus::owns_allocation(host), "yikici sahip oldugu host'u geri verir");
}

#if defined(NEXUS_ENABLE_CUDA)

bool cuda_available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::vector<float> read_device(MirroredTensor& mirror) {
  std::vector<float> values(mirror.numel());
  nexus::copy_bytes(values.data(), MemorySpace::Host, DeviceRef::cpu(), mirror.device_data(), MemorySpace::Device,
                    mirror.device(), mirror.bytes());
  return values;
}

void write_device(MirroredTensor& mirror, const std::vector<float>& values) {
  nexus::copy_bytes(mirror.device_data(), MemorySpace::Device, mirror.device(), values.data(), MemorySpace::Host,
                    DeviceRef::cpu(), mirror.bytes());
}

void test_cuda_round_trip() {
  nexus_test::section("CUDA ayna gidis-donus");
  constexpr std::size_t count = 4096;
  MirroredTensor mirror(Shape{count}, DeviceRef::cuda(0));
  expect(!mirror.unified() && mirror.host_pinned(), "CUDA aynasi sayfa kilitli host kullanir");
  expect(mirror.state() == MirrorState::HostAhead, "yeni ayna HostAhead baslar");
  expect(!mirror.device_allocated(), "cihaz tamponu tembel ayrilir");

  mirror.modify_host();
  for (std::size_t i = 0; i < count; ++i) {
    mirror.host_data()[i] = static_cast<float>(i) * 0.5f;
  }
  mirror.sync_device();
  expect(mirror.device_allocated() && mirror.state() == MirrorState::Synced, "sync_device sonrasi Synced");
  expect(mirror.bytes_to_device() == mirror.bytes(), "H2D sayaci bir tam kopya");

  std::vector<float> on_device = read_device(mirror);
  bool copied = true;
  for (std::size_t i = 0; i < count; ++i) {
    copied = copied && on_device[i] == static_cast<float>(i) * 0.5f;
  }
  expect(copied, "cihaz host verisini tasir");

  mirror.sync_device();
  expect(mirror.bytes_to_device() == mirror.bytes(), "Synced iken sync_device kopya yapmaz");

  mirror.modify_device();
  std::vector<float> doubled(count);
  for (std::size_t i = 0; i < count; ++i) {
    doubled[i] = static_cast<float>(i) * 2.0f;
  }
  write_device(mirror, doubled);
  expect(mirror.state() == MirrorState::DeviceAhead, "modify_device sonrasi DeviceAhead");
  expect(mirror.needs_sync_host() && !mirror.needs_sync_device(), "needs_sync_* durumla tutarli");
  expect(throws_tensor_error([&] { mirror.modify_host(); }), "DeviceAhead iken modify_host reddedilir");

  mirror.sync_host();
  bool returned = true;
  for (std::size_t i = 0; i < count; ++i) {
    returned = returned && mirror.host_data()[i] == doubled[i];
  }
  expect(returned && mirror.state() == MirrorState::Synced, "sync_host cihaz verisini host'a getirir");
  expect(mirror.bytes_to_host() == mirror.bytes(), "D2H sayaci bir tam kopya");

  mirror.modify_host();
  expect(throws_tensor_error([&] { mirror.modify_device(); }), "HostAhead iken modify_device reddedilir");

  mirror.host_data()[0] = -1.0f;
  mirror.sync_device(false);
  expect(mirror.copy_pending(), "engellemeyen sync_device kopyayi bekleyen olarak isaretler");
  mirror.synchronize();
  expect(!mirror.copy_pending() && read_device(mirror)[0] == -1.0f, "asenkron sync_device + synchronize tamamlanir");

  mirror.modify_host();
  mirror.host_data()[1] = -2.0f;
  mirror.sync_device(false);
  mirror.modify_host();
  expect(!mirror.copy_pending(), "modify_host bekleyen H2D kopyasini once tamamlar (DMA yarisi yok)");
  mirror.host_data()[1] = -3.0f;
  mirror.sync_device();
  expect(read_device(mirror)[1] == -3.0f, "bekleme sonrasi yazilan deger cihaza ulasir");

  {
    AllocationScope scope;
    for (int step = 0; step < 16; ++step) {
      mirror.modify_host();
      mirror.sync_device(false);
      mirror.modify_device();
      mirror.sync_host();
    }
    expect(scope.count() == 0, "kararli durumda 16 gidis-donus 0 heap ayirma");
  }
  expect(mirror.bytes_to_device() == mirror.bytes() * 20 && mirror.bytes_to_host() == mirror.bytes() * 17,
         "bayt sayaclari her gercek kopyayi sayar");
  expect(nexus::owns_allocation(mirror.device_data()), "cihaz tamponu Nexus kaydinda");
}

void test_cuda_stream_ordering() {
  nexus_test::section("CUDA akis siralamasi");
  const DeviceRef gpu = DeviceRef::cuda(0);
  nexus::StreamHandle user = nexus::create_stream(gpu);
  {
    constexpr std::size_t count = 64;
    MirroredTensor mirror(Shape{count}, gpu);
    mirror.set_device_stream(user);
    expect(mirror.device_stream() == user, "cihaz akisi kullanicinin akisi");
    mirror.sync_device();

    std::vector<float> produced(count);
    for (std::size_t i = 0; i < count; ++i) {
      produced[i] = 10.0f + static_cast<float>(i);
    }
    mirror.modify_device();
    nexus::copy_bytes_async(mirror.device_data(), MemorySpace::Device, gpu, produced.data(), MemorySpace::Host,
                            DeviceRef::cpu(), mirror.bytes(), user);
    mirror.sync_host();
    bool waited = true;
    for (std::size_t i = 0; i < count; ++i) {
      waited = waited && mirror.host_data()[i] == produced[i];
    }
    expect(waited, "sync_host kullanici akisindaki yazmayi olayla bekler");

    mirror.modify_host();
    for (std::size_t i = 0; i < count; ++i) {
      mirror.host_data()[i] = -static_cast<float>(i);
    }
    mirror.sync_device(false);
    std::vector<float> consumed(count, 1.0f);
    nexus::copy_bytes_async(consumed.data(), MemorySpace::Host, DeviceRef::cpu(), mirror.device_data(),
                            MemorySpace::Device, gpu, mirror.bytes(), user);
    nexus::synchronize_stream(gpu, user);
    bool ordered = true;
    for (std::size_t i = 0; i < count; ++i) {
      ordered = ordered && consumed[i] == -static_cast<float>(i);
    }
    expect(ordered, "kullanici akisi engellemeyen H2D kopyasini host beklemeden izler");
  }
  nexus::destroy_stream(gpu, user);
}

void test_cuda_bridges() {
  nexus_test::section("CUDA kopruleri");
  nexus_model::Tensor model = nexus_model::Tensor::from_values({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  {
    MirroredTensor mirror = nexus::mirror(model, DeviceRef::cuda(0));
    expect(mirror.host_data() == model.data(), "model aynasi host'u kopyalamaz");
    expect(mirror.state() == MirrorState::HostAhead && !mirror.host_pinned(), "model aynasi sayfalanabilir baslar");
    expect(mirror.pin_host() && mirror.host_pinned(), "pin_host model bellegini CUDA'ya kaydeder");
    mirror.sync_device();
    mirror.modify_device();
    std::vector<float> negated(mirror.numel());
    for (std::size_t i = 0; i < negated.size(); ++i) {
      negated[i] = -static_cast<float>(i + 1);
    }
    write_device(mirror, negated);
    mirror.sync_host();
    expect(model[0] == -1.0f && model[7] == -8.0f, "sync_host dogrudan model tensörunu gunceller");
  }
  model[0] = 3.0f;
  expect(model[0] == 3.0f, "ayna yikildiktan sonra model bellegi gecerli (kayit kaldirildi)");

  matrix_pro::Tensor tensor(std::vector<std::size_t>{2, 4});
  const std::uint64_t before = nexus::allocation_count();
  MirroredTensor flash = nexus::mirror(tensor, 0);
  expect(flash.host_data() == tensor.host_ptr() && flash.device_data() == tensor.device_data(),
         "MatrixFlash aynasi iki tamponu da kopyalamadan sarar");
  expect(nexus::allocation_count() == before, "MatrixFlash aynasi bellek ayirmaz");
  expect(flash.state() == (tensor.host_current() ? MirrorState::Synced : MirrorState::DeviceAhead),
         "MatrixFlash aynasi host_current bayragindan durum alir");
  matrix_pro::Tensor slice = tensor.narrow(0, 1, 1);
  expect(throws_tensor_error([&] { (void)nexus::mirror(slice, 0); }), "ofsetli MatrixFlash dilimi reddedilir");
}

#endif

void test_accelerator_without_cuda() {
#if !defined(NEXUS_ENABLE_CUDA)
  nexus_test::section("CUDA kapali hizlandirici aynasi");
  MirroredTensor mirror(Shape{8}, DeviceRef::cuda(0));
  expect(mirror.state() == MirrorState::HostAhead, "host tarafi CUDA olmadan da kurulur");
  expect(throws_tensor_error([&] { mirror.sync_device(); }), "CUDA olmadan cihaz senkronu acik hata verir");
#endif
}

}  // namespace

int main() {
  std::cout << "Nexus Faz 1 MirroredTensor testi\n";
  try {
    test_unified_cpu();
    test_move_and_release();
    test_accelerator_without_cuda();
#if defined(NEXUS_ENABLE_CUDA)
    if (cuda_available()) {
      test_cuda_round_trip();
      test_cuda_stream_ordering();
      test_cuda_bridges();
    } else {
      nexus_test::section("CUDA aygiti yok, cihaz testleri atlandi");
    }
#endif
  } catch (const std::exception& error) {
    std::cout << "  FAIL  istisna: " << error.what() << "\n";
    return 1;
  }
  return nexus_test::finish("test_phase1_mirrored_tensor");
}
