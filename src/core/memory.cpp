/**
 * @file memory.cpp
 * @brief Host, sayfa kilitli host ve CUDA bellek kapisi.
 *
 * Kayit tablosu ayirma ile serbest birakmayi esler. Tablo kilitlidir; ayni
 * isaretci iki kez serbest birakilirsa ikinci cagri isletim sistemi free'sine
 * gitmez. Tensör elemanlarina eszamanli yazmak bu kapiyla ilgili degildir ve
 * cagiranin sorumlulugundadir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/memory.hpp"

#include "nexus/core/error.hpp"
#include "nexus/core/shape.hpp"

#include <atomic>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <malloc.h>
#include <windows.h>
#else
#include <stdlib.h>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#endif

#if defined(NEXUS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace nexus {
namespace {

enum class Backend : std::uint8_t { CrtAligned = 1, CudaHost = 2, OsLocked = 3, CudaDevice = 4 };

struct Record {
  Backend backend = Backend::CrtAligned;
  AllocKind kind = AllocKind::AlignedHost;
  DeviceRef device{};
  std::size_t bytes = 0;
  std::size_t reserved = 0;
};

struct RegistryNode {
  void* key = nullptr;
  Record value;
  RegistryNode* next = nullptr;
};

std::atomic_flag registry_flag = ATOMIC_FLAG_INIT;
RegistryNode* registry_head = nullptr;
std::atomic<std::uint64_t> g_allocations{0};

struct RegistrySpin {
  RegistrySpin() {
    while (registry_flag.test_and_set(std::memory_order_acquire)) {
    }
  }
  ~RegistrySpin() { registry_flag.clear(std::memory_order_release); }
};

RegistryNode* find_node(void* pointer) {
  for (RegistryNode* node = registry_head; node != nullptr; node = node->next) {
    if (node->key == pointer) {
      return node;
    }
  }
  return nullptr;
}

void track(void* pointer, Record record) {
  if (pointer == nullptr) {
    throw TensorError("Nexus: bos isaretci bellek kaydina yazilamaz");
  }
  RegistrySpin guard;
  if (find_node(pointer) != nullptr) {
    throw TensorError("Nexus: ayni adres ikinci kez kaydedildi");
  }
  registry_head = new RegistryNode{pointer, record, registry_head};
  g_allocations.fetch_add(1, std::memory_order_relaxed);
}

std::optional<Record> untrack(void* pointer) {
  RegistrySpin guard;
  RegistryNode** link = &registry_head;
  while (*link != nullptr) {
    if ((*link)->key == pointer) {
      RegistryNode* node = *link;
      Record record = node->value;
      *link = node->next;
      delete node;
      return record;
    }
    link = &(*link)->next;
  }
  return std::nullopt;
}

bool recorded(const void* pointer) {
  RegistrySpin guard;
  return find_node(const_cast<void*>(pointer)) != nullptr;
}

std::size_t round_up(std::size_t bytes, std::size_t alignment) {
  return (bytes + alignment - 1u) & ~(alignment - 1u);
}

void free_block(void* pointer, const Record& record) noexcept {
  switch (record.backend) {
    case Backend::CrtAligned:
#if defined(_WIN32)
      _aligned_free(pointer);
#else
      std::free(pointer);
#endif
      break;
    case Backend::CudaHost:
#if defined(NEXUS_ENABLE_CUDA)
      cudaFreeHost(pointer);
#endif
      break;
    case Backend::OsLocked:
#if defined(_WIN32)
      VirtualUnlock(pointer, record.reserved);
      VirtualFree(pointer, 0, MEM_RELEASE);
#elif defined(__linux__) || defined(__APPLE__)
      munlock(pointer, record.reserved);
      std::free(pointer);
#else
      std::free(pointer);
#endif
      break;
    case Backend::CudaDevice:
#if defined(NEXUS_ENABLE_CUDA)
    {
      int previous = 0;
      const cudaError_t remembered = cudaGetDevice(&previous);
      if (record.device.index >= 0) {
        cudaSetDevice(record.device.index);
      }
      cudaFree(pointer);
      if (remembered == cudaSuccess) {
        cudaSetDevice(previous);
      }
    }
#endif
      break;
  }
}

#if defined(NEXUS_ENABLE_CUDA)

struct CudaDeviceGuard {
  int previous = 0;
  bool restore = false;

  explicit CudaDeviceGuard(int index) {
    if (cudaGetDevice(&previous) != cudaSuccess) {
      previous = 0;
    }
    if (cudaSetDevice(index) != cudaSuccess) {
      throw TensorError("Nexus: cudaSetDevice basarisiz");
    }
    restore = true;
  }

  ~CudaDeviceGuard() {
    if (restore) {
      cudaSetDevice(previous);
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;
};

int cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 0) {
    return 0;
  }
  return count;
}

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw TensorError(std::string("Nexus: ") + what + " basarisiz: " + cudaGetErrorString(status));
  }
}

#endif

void* lock_pages(std::size_t bytes, std::size_t& reserved) {
#if defined(_WIN32)
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  const std::size_t page = info.dwPageSize == 0 ? 4096u : static_cast<std::size_t>(info.dwPageSize);
  reserved = round_up(bytes, page);
  void* pointer = VirtualAlloc(nullptr, reserved, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (pointer == nullptr) {
    throw TensorError("Nexus: VirtualAlloc sayfa kilitli bellek ayiramadi");
  }
  HANDLE process = GetCurrentProcess();
  SIZE_T minimum = 0;
  SIZE_T maximum = 0;
  if (GetProcessWorkingSetSize(process, &minimum, &maximum)) {
    const SIZE_T extra = static_cast<SIZE_T>(reserved) + (8u << 20);
    SetProcessWorkingSetSize(process, minimum + extra, maximum + extra);
  }
  if (!VirtualLock(pointer, reserved)) {
    VirtualFree(pointer, 0, MEM_RELEASE);
    throw TensorError("Nexus: VirtualLock sayfalari bellege sabitleyemedi");
  }
  return pointer;
#elif defined(__linux__) || defined(__APPLE__)
  const long page_size = sysconf(_SC_PAGESIZE);
  const std::size_t page = page_size > 0 ? static_cast<std::size_t>(page_size) : 4096u;
  reserved = round_up(bytes, page);
  void* pointer = nullptr;
  if (posix_memalign(&pointer, page, reserved) != 0 || pointer == nullptr) {
    throw TensorError("Nexus: sayfa hizali kilit bellegi ayrilamadi");
  }
  if (mlock(pointer, reserved) != 0) {
    std::free(pointer);
    throw TensorError("Nexus: mlock sayfalari sabitleyemedi");
  }
  return pointer;
#else
  (void)bytes;
  (void)reserved;
  throw TensorError("Nexus: bu platformda sayfa kilidi yok");
#endif
}

}  // namespace

void* allocate_aligned(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  const std::size_t reserved = round_up(bytes, kAlignment);
  void* pointer = nullptr;
#if defined(_WIN32)
  pointer = _aligned_malloc(reserved, kAlignment);
#else
  pointer = std::aligned_alloc(kAlignment, reserved);
#endif
  if (pointer == nullptr) {
    throw TensorError("Nexus: 64 bayt hizali bellek ayrilamadi");
  }
  Record record;
  record.backend = Backend::CrtAligned;
  record.kind = AllocKind::AlignedHost;
  record.device = DeviceRef::cpu();
  record.bytes = bytes;
  record.reserved = reserved;
  track(pointer, record);
  return pointer;
}

void* allocate_pinned(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
#if defined(NEXUS_ENABLE_CUDA)
  if (cuda_device_count() > 0) {
    void* pointer = nullptr;
    if (cudaMallocHost(&pointer, bytes) == cudaSuccess && pointer != nullptr) {
      Record record;
      record.backend = Backend::CudaHost;
      record.kind = AllocKind::PinnedHost;
      record.device = DeviceRef::cpu();
      record.bytes = bytes;
      record.reserved = bytes;
      track(pointer, record);
      return pointer;
    }
  }
#endif
  std::size_t reserved = 0;
  void* pointer = lock_pages(bytes, reserved);
  Record record;
  record.backend = Backend::OsLocked;
  record.kind = AllocKind::PinnedHost;
  record.device = DeviceRef::cpu();
  record.bytes = bytes;
  record.reserved = reserved;
  track(pointer, record);
  return pointer;
}

void* allocate_device(DeviceRef device, std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  if (device.kind != DeviceKind::Cuda) {
    throw TensorError(
        std::string("Nexus: ") + to_string(device.kind) +
        " bellegi Faz 1'de ayirilmaz. Mevcut bir aygit isaretcisini Tensor::alias ile goruntuleyin.");
  }
#if !defined(NEXUS_ENABLE_CUDA)
  (void)device;
  throw TensorError("Nexus: CUDA toolkit bagli degil, cihaz bellegi ayirilamaz");
#else
  if (device.index < 0) {
    throw TensorError("Nexus: CUDA cihaz indeksi negatif olamaz");
  }
  const int count = cuda_device_count();
  if (device.index >= count) {
    throw TensorError("Nexus: CUDA cihaz indeksi sistemdeki aygit sayisinin disinda");
  }
  CudaDeviceGuard guard(device.index);
  void* pointer = nullptr;
  check_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
  if (pointer == nullptr) {
    throw TensorError("Nexus: cudaMalloc bos isaretci dondurdu");
  }
  Record record;
  record.backend = Backend::CudaDevice;
  record.kind = AllocKind::Device;
  record.device = device;
  record.bytes = bytes;
  record.reserved = bytes;
  track(pointer, record);
  return pointer;
#endif
}

void release_owned(void* ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }
  const std::optional<Record> record = untrack(ptr);
  if (!record.has_value()) {
    return;
  }
  free_block(ptr, *record);
}

bool owns_allocation(const void* ptr) noexcept {
  if (ptr == nullptr) {
    return false;
  }
  try {
    return recorded(ptr);
  } catch (...) {
    return false;
  }
}

AllocInfo allocation_info(const void* ptr) {
  if (ptr == nullptr) {
    throw TensorError("Nexus: bos isaretcinin ayirma kaydi yok");
  }
  RegistrySpin guard;
  RegistryNode* found = find_node(const_cast<void*>(ptr));
  if (found == nullptr) {
    throw TensorError("Nexus: isaretci bu bellek kapisina ait degil");
  }
  AllocInfo info;
  info.kind = found->value.kind;
  info.device = found->value.device;
  info.bytes = found->value.bytes;
  return info;
}

std::uint64_t allocation_count() noexcept { return g_allocations.load(std::memory_order_relaxed); }

void copy_bytes(void* dst, MemorySpace dst_space, DeviceRef dst_device, const void* src, MemorySpace src_space,
                DeviceRef src_device, std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw TensorError("Nexus: kopya icin kaynak ve hedef isaretci gerekli");
  }
  const bool dst_on_device = dst_space == MemorySpace::Device;
  const bool src_on_device = src_space == MemorySpace::Device;
  if (!dst_on_device && !src_on_device) {
    std::memcpy(dst, src, bytes);
    return;
  }
#if !defined(NEXUS_ENABLE_CUDA)
  (void)dst_device;
  (void)src_device;
  throw TensorError("Nexus: cihaz kopyasi icin CUDA toolkit gerekli");
#else
  if (dst_on_device && src_on_device && dst_device.index != src_device.index) {
    check_cuda(cudaMemcpyPeer(dst, dst_device.index, src, src_device.index, bytes), "cudaMemcpyPeer");
    return;
  }
  const int index = dst_on_device ? dst_device.index : src_device.index;
  if (index < 0) {
    throw TensorError("Nexus: cihaz kopyasinda gecersiz cihaz indeksi");
  }
  CudaDeviceGuard guard(index);
  cudaMemcpyKind kind = cudaMemcpyHostToHost;
  if (!src_on_device && dst_on_device) {
    kind = cudaMemcpyHostToDevice;
  } else if (src_on_device && !dst_on_device) {
    kind = cudaMemcpyDeviceToHost;
  } else {
    kind = cudaMemcpyDeviceToDevice;
  }
  check_cuda(cudaMemcpy(dst, src, bytes, kind), "cudaMemcpy");
#endif
}

void zero_bytes(void* dst, MemorySpace space, DeviceRef device, std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr) {
    throw TensorError("Nexus: sifirlama icin isaretci gerekli");
  }
  if (space != MemorySpace::Device) {
    std::memset(dst, 0, bytes);
    return;
  }
#if !defined(NEXUS_ENABLE_CUDA)
  (void)device;
  throw TensorError("Nexus: cihaz bellegi sifirlanamadi, CUDA toolkit yok");
#else
  if (device.index < 0) {
    throw TensorError("Nexus: cihaz sifirlama icin gecersiz indeks");
  }
  CudaDeviceGuard guard(device.index);
  check_cuda(cudaMemset(dst, 0, bytes), "cudaMemset");
#endif
}

void fill_float(float* dst, MemorySpace space, DeviceRef device, std::size_t count, float value) {
  if (count == 0) {
    return;
  }
  if (dst == nullptr) {
    throw TensorError("Nexus: doldurma icin isaretci gerekli");
  }
  if (value == 0.0f) {
    zero_bytes(dst, space, device, count * sizeof(float));
    return;
  }
  if (space != MemorySpace::Device) {
    for (std::size_t i = 0; i < count; ++i) {
      dst[i] = value;
    }
    return;
  }
  std::vector<float> stage(count, value);
  copy_bytes(dst, space, device, stage.data(), MemorySpace::Host, DeviceRef::cpu(), count * sizeof(float));
}

StreamHandle create_stream(DeviceRef device) {
  if (device.kind != DeviceKind::Cuda) {
    return nullptr;
  }
#if !defined(NEXUS_ENABLE_CUDA)
  throw TensorError("Nexus: CUDA akisi icin CUDA toolkit gerekli");
#else
  CudaDeviceGuard guard(device.index);
  cudaStream_t stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  return static_cast<StreamHandle>(stream);
#endif
}

void destroy_stream(DeviceRef device, StreamHandle stream) noexcept {
  if (stream == nullptr || device.kind != DeviceKind::Cuda) {
    return;
  }
#if defined(NEXUS_ENABLE_CUDA)
  int previous = 0;
  const bool remembered = cudaGetDevice(&previous) == cudaSuccess;
  cudaSetDevice(device.index);
  cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  cudaStreamDestroy(static_cast<cudaStream_t>(stream));
  if (remembered) {
    cudaSetDevice(previous);
  }
#endif
}

void synchronize_stream(DeviceRef device, StreamHandle stream) {
  if (device.kind != DeviceKind::Cuda) {
    return;
  }
#if !defined(NEXUS_ENABLE_CUDA)
  (void)stream;
  throw TensorError("Nexus: CUDA akisi icin CUDA toolkit gerekli");
#else
  CudaDeviceGuard guard(device.index);
  check_cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
#endif
}

EventHandle create_event(DeviceRef device) {
  if (device.kind != DeviceKind::Cuda) {
    return nullptr;
  }
#if !defined(NEXUS_ENABLE_CUDA)
  throw TensorError("Nexus: CUDA olayi icin CUDA toolkit gerekli");
#else
  CudaDeviceGuard guard(device.index);
  cudaEvent_t event = nullptr;
  check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
  return static_cast<EventHandle>(event);
#endif
}

void destroy_event(DeviceRef device, EventHandle event) noexcept {
  if (event == nullptr || device.kind != DeviceKind::Cuda) {
    return;
  }
#if defined(NEXUS_ENABLE_CUDA)
  cudaEventDestroy(static_cast<cudaEvent_t>(event));
#endif
}

void stream_wait(DeviceRef device, StreamHandle consumer, StreamHandle producer, EventHandle event) {
  if (device.kind != DeviceKind::Cuda || consumer == producer) {
    return;
  }
#if !defined(NEXUS_ENABLE_CUDA)
  (void)event;
  throw TensorError("Nexus: CUDA akis siralamasi icin CUDA toolkit gerekli");
#else
  if (event == nullptr) {
    throw TensorError("Nexus: akis siralamasi icin olay gerekli");
  }
  CudaDeviceGuard guard(device.index);
  cudaStream_t from = producer == nullptr ? cudaStreamLegacy : static_cast<cudaStream_t>(producer);
  cudaStream_t to = consumer == nullptr ? cudaStreamLegacy : static_cast<cudaStream_t>(consumer);
  check_cuda(cudaEventRecord(static_cast<cudaEvent_t>(event), from), "cudaEventRecord");
  check_cuda(cudaStreamWaitEvent(to, static_cast<cudaEvent_t>(event), 0), "cudaStreamWaitEvent");
#endif
}

void copy_bytes_async(void* dst, MemorySpace dst_space, DeviceRef dst_device, const void* src, MemorySpace src_space,
                      DeviceRef src_device, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) {
    return;
  }
  if (dst == nullptr || src == nullptr) {
    throw TensorError("Nexus: kopya icin kaynak ve hedef isaretci gerekli");
  }
  const bool dst_on_device = dst_space == MemorySpace::Device;
  const bool src_on_device = src_space == MemorySpace::Device;
  if (!dst_on_device && !src_on_device) {
    std::memcpy(dst, src, bytes);
    return;
  }
#if !defined(NEXUS_ENABLE_CUDA)
  (void)dst_device;
  (void)src_device;
  (void)stream;
  throw TensorError("Nexus: cihaz kopyasi icin CUDA toolkit gerekli");
#else
  auto* cuda_stream = static_cast<cudaStream_t>(stream);
  if (dst_on_device && src_on_device && dst_device.index != src_device.index) {
    check_cuda(cudaMemcpyPeerAsync(dst, dst_device.index, src, src_device.index, bytes, cuda_stream),
               "cudaMemcpyPeerAsync");
    return;
  }
  const int index = dst_on_device ? dst_device.index : src_device.index;
  if (index < 0) {
    throw TensorError("Nexus: cihaz kopyasinda gecersiz cihaz indeksi");
  }
  CudaDeviceGuard guard(index);
  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  if (!src_on_device && dst_on_device) {
    kind = cudaMemcpyHostToDevice;
  } else if (src_on_device && !dst_on_device) {
    kind = cudaMemcpyDeviceToHost;
  }
  check_cuda(cudaMemcpyAsync(dst, src, bytes, kind, cuda_stream), "cudaMemcpyAsync");
#endif
}

bool register_host_memory(void* ptr, std::size_t bytes) noexcept {
  if (ptr == nullptr || bytes == 0) {
    return false;
  }
#if defined(NEXUS_ENABLE_CUDA)
  if (cuda_device_count() <= 0) {
    return false;
  }
  if (cudaHostRegister(ptr, bytes, cudaHostRegisterPortable) == cudaSuccess) {
    return true;
  }
  // Reddedilen kayit hatasi sonraki bir cekirdegin cudaGetLastError kontrolune sizmasin.
  (void)cudaGetLastError();
  return false;
#else
  return false;
#endif
}

void unregister_host_memory(void* ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }
#if defined(NEXUS_ENABLE_CUDA)
  cudaHostUnregister(ptr);
#endif
}

}  // namespace nexus
