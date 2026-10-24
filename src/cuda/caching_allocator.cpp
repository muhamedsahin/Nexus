/**
 * @file caching_allocator.cpp
 * @brief FAZ 1.4 onbellekli CUDA aygit ayiricisi ve pinned host havuzu.
 *
 * Veri yapisi: her segment bir cudaMalloc bolgesidir; icindeki bloklar cift yonlu
 * listeyle (prev/next) baglidir. Serbest bloklar havuz basina (kucuk/buyuk) bir
 * `std::set` icinde (akis, boyut, adres) sirasiyla durur; best-fit lower_bound ile
 * bulunur. Canli bloklar adres -> Block haritasindadir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/caching_allocator.hpp"

#include "nexus/core/error.hpp"
#include "nexus/prof/memory_ledger.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nexus {
namespace cuda {

// ================================ Aygit ayiricisi =================================

namespace {

struct Block;

struct BlockComparator {
  bool operator()(const Block* a, const Block* b) const noexcept;
};

using FreeSet = std::set<Block*, BlockComparator>;

struct Block {
  int device = 0;
  cudaStream_t stream = nullptr;
  std::size_t size = 0;       // blok boyutu (yuvarlanmis)
  std::size_t requested = 0;  // kullanicinin istedigi
  char* ptr = nullptr;
  FreeSet* pool = nullptr;
  Block* prev = nullptr;
  Block* next = nullptr;
  bool allocated = false;
  int pending_events = 0;
  std::unordered_set<cudaStream_t> stream_uses;
};

bool BlockComparator::operator()(const Block* a, const Block* b) const noexcept {
  if (a->stream != b->stream) {
    return reinterpret_cast<std::uintptr_t>(a->stream) < reinterpret_cast<std::uintptr_t>(b->stream);
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<std::uintptr_t>(a->ptr) < reinterpret_cast<std::uintptr_t>(b->ptr);
}

struct DeviceState {
  FreeSet small_pool;
  FreeSet large_pool;
  DeviceAllocatorStats stats;
};

constexpr int kMaxDevices = 16;

}  // namespace

struct CachingDeviceAllocator::Impl {
  mutable std::recursive_mutex mutex;
  std::array<DeviceState, kMaxDevices> devices;
  std::unordered_map<void*, Block*> active;
  // Erteleme: blok, akislarina kaydedilen olaylar tamamlanana kadar bekler.
  std::deque<std::pair<cudaEvent_t, Block*>> pending;
  std::vector<cudaEvent_t> event_pool;

  DeviceState& state(int device) {
    if (device < 0 || device >= kMaxDevices) {
      throw TensorError("Nexus CUDA ayirici: aygit indeksi gecersiz");
    }
    return devices[static_cast<std::size_t>(device)];
  }

  static bool is_split(const Block* block) noexcept { return block->prev != nullptr || block->next != nullptr; }

  cudaEvent_t take_event() {
    if (!event_pool.empty()) {
      cudaEvent_t event = event_pool.back();
      event_pool.pop_back();
      return event;
    }
    cudaEvent_t event = nullptr;
    NEXUS_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
  }

  void update_inactive(DeviceState& st, const Block* block, bool add) {
    if (!is_split(block)) {
      return;
    }
    if (add) {
      st.stats.inactive_split_bytes += block->size;
    } else {
      st.stats.inactive_split_bytes -= block->size;
    }
  }

  void insert_free(Block* block) {
    DeviceState& st = state(block->device);
    block->pool->insert(block);
    update_inactive(st, block, true);
  }

  void erase_free(Block* block) {
    DeviceState& st = state(block->device);
    block->pool->erase(block);
    update_inactive(st, block, false);
  }

  // Serbest blogu ayni segmentteki serbest komsularla birlestirip havuza koyar.
  void free_block(Block* block) {
    DeviceState& st = state(block->device);
    st.stats.active_bytes -= block->size;
    block->allocated = false;
    block->stream_uses.clear();
    for (Block* neighbour : {block->prev, block->next}) {
      if (neighbour == nullptr || neighbour->allocated || neighbour->pending_events > 0) {
        continue;
      }
      erase_free(neighbour);
      if (neighbour == block->prev) {
        block->ptr = neighbour->ptr;
        block->prev = neighbour->prev;
        if (block->prev != nullptr) {
          block->prev->next = block;
        }
      } else {
        block->next = neighbour->next;
        if (block->next != nullptr) {
          block->next->prev = block;
        }
      }
      block->size += neighbour->size;
      delete neighbour;
    }
    insert_free(block);
  }

  void process_events() {
    while (!pending.empty()) {
      auto [event, block] = pending.front();
      const cudaError_t status = cudaEventQuery(event);
      if (status == cudaErrorNotReady) {
        cudaGetLastError();
        break;
      }
      NEXUS_CUDA_CHECK(status);
      pending.pop_front();
      event_pool.push_back(event);
      block->pending_events -= 1;
      if (block->pending_events == 0) {
        free_block(block);
      }
    }
  }

  void synchronize_and_drain() {
    for (auto& [event, block] : pending) {
      NEXUS_CUDA_CHECK(cudaEventSynchronize(event));
      (void)block;
    }
    process_events();
  }

  // Bolunmemis (tum segment) serbest bloklari cudaFree ile birakir.
  void release_cached(int device) {
    DeviceState& st = state(device);
    for (FreeSet* pool : {&st.small_pool, &st.large_pool}) {
      for (auto it = pool->begin(); it != pool->end();) {
        Block* block = *it;
        if (is_split(block)) {
          ++it;
          continue;
        }
        it = pool->erase(it);
        DeviceGuard guard(device);
        NEXUS_CUDA_CHECK(cudaFree(block->ptr));
        st.stats.reserved_bytes -= block->size;
        st.stats.num_segments -= 1;
        delete block;
      }
    }
  }

  std::uint64_t largest_free(const DeviceState& st) const {
    std::uint64_t largest = 0;
    for (const FreeSet* pool : {&st.small_pool, &st.large_pool}) {
      for (const Block* block : *pool) {
        largest = std::max<std::uint64_t>(largest, block->size);
      }
    }
    return largest;
  }
};

std::size_t CachingDeviceAllocator::round_size(std::size_t bytes) noexcept {
  if (bytes < kMinBlock) {
    return kMinBlock;
  }
  return kMinBlock * ((bytes + kMinBlock - 1) / kMinBlock);
}

CachingDeviceAllocator::CachingDeviceAllocator() : impl_(std::make_unique<Impl>()) {}

CachingDeviceAllocator::~CachingDeviceAllocator() {
  // Surec sonunda CUDA baglami zaten yikilmis olabilir; hatalari yut.
  try {
    std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
    for (auto& [event, block] : impl_->pending) {
      cudaEventDestroy(event);
      (void)block;
    }
    for (cudaEvent_t event : impl_->event_pool) {
      cudaEventDestroy(event);
    }
  } catch (...) {
  }
}

void* CachingDeviceAllocator::allocate(std::size_t bytes, const Stream& stream) {
  if (bytes == 0) {
    return nullptr;
  }
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->process_events();
  const int device = stream.device();
  DeviceState& st = impl_->state(device);
  const std::size_t size = round_size(bytes);
  const bool small = size <= kSmallSize;
  FreeSet& pool = small ? st.small_pool : st.large_pool;

  Block key;
  key.stream = stream.handle();
  key.size = size;
  key.ptr = nullptr;
  auto found = pool.lower_bound(&key);
  Block* block = nullptr;
  if (found != pool.end() && (*found)->stream == stream.handle()) {
    block = *found;
    impl_->erase_free(block);
    st.stats.cache_hits += 1;
  } else {
    std::size_t segment = small ? kSmallBuffer
                                : (size < kMinLargeAlloc ? kLargeBuffer
                                                         : kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge));
    void* raw = nullptr;
    DeviceGuard guard(device);
    cudaError_t status = cudaMalloc(&raw, segment);
    if (status == cudaErrorMemoryAllocation) {
      cudaGetLastError();
      // OOM: ertelenmisleri bitir, onbellegi bosalt, tekrar dene.
      st.stats.num_alloc_retries += 1;
      impl_->synchronize_and_drain();
      impl_->release_cached(device);
      status = cudaMalloc(&raw, segment);
      if (status == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        st.stats.num_ooms += 1;
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        std::ostringstream message;
        message << "Nexus CUDA OOM: " << bytes << " bayt istendi (segment " << segment << "). Aygit " << device
                << ": toplam " << total_bytes << ", bos " << free_bytes << "; Nexus ayrilmis " << st.stats.allocated_bytes
                << ", tutulan " << st.stats.reserved_bytes << ", en buyuk serbest blok " << impl_->largest_free(st)
                << ", bolunmus serbest " << st.stats.inactive_split_bytes << ". Etiket kirilimi: ";
        const auto snapshot = prof::ledger().snapshot();
        for (std::size_t i = 0; i < snapshot.by_label.size(); ++i) {
          message << prof::label_name(static_cast<prof::MemoryLabel>(i)) << "=" << snapshot.by_label[i].current_bytes
                  << " ";
        }
        throw TensorError(message.str());
      }
    }
    NEXUS_CUDA_CHECK(status);
    st.stats.num_cuda_mallocs += 1;
    st.stats.num_segments += 1;
    st.stats.reserved_bytes += segment;
    st.stats.peak_reserved_bytes = std::max(st.stats.peak_reserved_bytes, st.stats.reserved_bytes);
    block = new Block();
    block->device = device;
    block->stream = stream.handle();
    block->size = segment;
    block->ptr = static_cast<char*>(raw);
    block->pool = &pool;
  }

  // Bolme: kalan kucuk havuzda >= 512, buyuk havuzda > 1 MiB ise.
  const std::size_t remaining = block->size - size;
  if ((small && remaining >= kMinBlock) || (!small && remaining > kSmallSize)) {
    Block* rest = new Block();
    rest->device = device;
    rest->stream = block->stream;
    rest->size = remaining;
    rest->ptr = block->ptr + size;
    rest->pool = block->pool;
    rest->prev = block;
    rest->next = block->next;
    if (rest->next != nullptr) {
      rest->next->prev = rest;
    }
    block->next = rest;
    block->size = size;
    impl_->insert_free(rest);
  }
  block->allocated = true;
  block->requested = bytes;
  impl_->active.emplace(block->ptr, block);
  st.stats.num_allocs += 1;
  st.stats.allocated_bytes += block->size;
  st.stats.requested_bytes += bytes;
  st.stats.active_bytes += block->size;
  st.stats.peak_allocated_bytes = std::max(st.stats.peak_allocated_bytes, st.stats.allocated_bytes);
  return block->ptr;
}

void CachingDeviceAllocator::deallocate(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto it = impl_->active.find(ptr);
  if (it == impl_->active.end()) {
    throw TensorError("Nexus CUDA ayirici: bu ayiricidan gelmeyen veya iki kez serbest birakilan isaretci");
  }
  Block* block = it->second;
  impl_->active.erase(it);
  DeviceState& st = impl_->state(block->device);
  st.stats.num_frees += 1;
  st.stats.allocated_bytes -= block->size;
  st.stats.requested_bytes -= block->requested;
  if (!block->stream_uses.empty()) {
    // Baska akislar hala okuyor olabilir: her birine olay kaydet, tamamlaninca birak.
    DeviceGuard guard(block->device);
    for (cudaStream_t used : block->stream_uses) {
      cudaEvent_t event = impl_->take_event();
      NEXUS_CUDA_CHECK(cudaEventRecord(event, used));
      block->pending_events += 1;
      impl_->pending.emplace_back(event, block);
    }
    block->stream_uses.clear();
    return;
  }
  impl_->free_block(block);
}

void CachingDeviceAllocator::record_stream(void* ptr, const Stream& stream) {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto it = impl_->active.find(ptr);
  if (it == impl_->active.end()) {
    throw TensorError("Nexus CUDA ayirici: record_stream icin canli blok bulunamadi");
  }
  if (stream.handle() != it->second->stream) {
    it->second->stream_uses.insert(stream.handle());
  }
}

void CachingDeviceAllocator::empty_cache() {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  impl_->synchronize_and_drain();
  for (int device = 0; device < kMaxDevices; ++device) {
    if (impl_->devices[static_cast<std::size_t>(device)].stats.num_segments > 0) {
      impl_->release_cached(device);
    }
  }
}

DeviceAllocatorStats CachingDeviceAllocator::stats(int device) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  if (device < 0) {
    device = current_device();
  }
  DeviceAllocatorStats out = impl_->devices.at(static_cast<std::size_t>(device)).stats;
  out.largest_free_block = impl_->largest_free(impl_->devices[static_cast<std::size_t>(device)]);
  return out;
}

void CachingDeviceAllocator::reset_peak_stats() {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  for (DeviceState& st : impl_->devices) {
    st.stats.peak_allocated_bytes = st.stats.allocated_bytes;
    st.stats.peak_reserved_bytes = st.stats.reserved_bytes;
  }
}

std::size_t CachingDeviceAllocator::allocation_size(const void* ptr) const {
  std::lock_guard<std::recursive_mutex> lock(impl_->mutex);
  const auto it = impl_->active.find(const_cast<void*>(ptr));
  return it == impl_->active.end() ? 0 : it->second->requested;
}

std::string CachingDeviceAllocator::summary(int device) const {
  const DeviceAllocatorStats s = stats(device);
  std::ostringstream out;
  out << "allocated=" << s.allocated_bytes << " reserved=" << s.reserved_bytes << " active=" << s.active_bytes
      << " inactive_split=" << s.inactive_split_bytes << " segments=" << s.num_segments
      << " cudaMalloc=" << s.num_cuda_mallocs << " hits=" << s.cache_hits << " retries=" << s.num_alloc_retries
      << " ooms=" << s.num_ooms;
  return out.str();
}

namespace {
DeviceAllocator* g_override = nullptr;
}  // namespace

CachingDeviceAllocator& caching_device_allocator() {
  // Bilerek sizdirilir: statik yikim sirasi CUDA baglamindan sonraya kalabilir.
  static auto* allocator = new CachingDeviceAllocator();
  return *allocator;
}

DeviceAllocator& device_allocator() { return g_override != nullptr ? *g_override : caching_device_allocator(); }

void set_device_allocator(DeviceAllocator* allocator) noexcept { g_override = allocator; }

// ================================ Pinned host havuzu ==============================

struct CachingHostAllocator::Impl {
  struct HostBlock {
    std::size_t size = 0;
    bool allocated = false;
    std::vector<cudaEvent_t> events;
    std::unordered_set<cudaStream_t> stream_uses;
  };
  mutable std::mutex mutex;
  std::unordered_map<void*, HostBlock> blocks;
  std::multimap<std::size_t, void*> free_by_size;
  std::deque<void*> deferred;
  HostAllocatorStats stats;

  static std::size_t round(std::size_t bytes) {
    std::size_t size = 4096;
    while (size < bytes) {
      size <<= 1;
    }
    return size;
  }

  void process() {
    for (auto it = deferred.begin(); it != deferred.end();) {
      HostBlock& block = blocks.at(*it);
      bool done = true;
      for (cudaEvent_t event : block.events) {
        const cudaError_t status = cudaEventQuery(event);
        if (status == cudaErrorNotReady) {
          cudaGetLastError();
          done = false;
          break;
        }
        NEXUS_CUDA_CHECK(status);
      }
      if (!done) {
        ++it;
        continue;
      }
      for (cudaEvent_t event : block.events) {
        cudaEventDestroy(event);
      }
      block.events.clear();
      free_by_size.emplace(block.size, *it);
      it = deferred.erase(it);
    }
  }
};

CachingHostAllocator::CachingHostAllocator() : impl_(std::make_unique<Impl>()) {}
CachingHostAllocator::~CachingHostAllocator() = default;

void* CachingHostAllocator::allocate(std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->process();
  const std::size_t size = Impl::round(bytes);
  auto it = impl_->free_by_size.find(size);
  void* ptr = nullptr;
  if (it != impl_->free_by_size.end()) {
    ptr = it->second;
    impl_->free_by_size.erase(it);
    impl_->stats.cache_hits += 1;
  } else {
    NEXUS_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    impl_->blocks[ptr].size = size;
    impl_->stats.reserved_bytes += size;
    impl_->stats.num_host_allocs += 1;
  }
  Impl::HostBlock& block = impl_->blocks.at(ptr);
  block.allocated = true;
  impl_->stats.allocated_bytes += size;
  return ptr;
}

void CachingHostAllocator::deallocate(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto it = impl_->blocks.find(ptr);
  if (it == impl_->blocks.end() || !it->second.allocated) {
    throw TensorError("Nexus CUDA pinned: bu havuzdan gelmeyen veya iki kez birakilan isaretci");
  }
  Impl::HostBlock& block = it->second;
  block.allocated = false;
  impl_->stats.allocated_bytes -= block.size;
  if (block.stream_uses.empty()) {
    impl_->free_by_size.emplace(block.size, ptr);
    return;
  }
  for (cudaStream_t stream : block.stream_uses) {
    cudaEvent_t event = nullptr;
    NEXUS_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    NEXUS_CUDA_CHECK(cudaEventRecord(event, stream));
    block.events.push_back(event);
  }
  block.stream_uses.clear();
  impl_->deferred.push_back(ptr);
  impl_->stats.deferred_frees += 1;
}

void CachingHostAllocator::record_stream(void* ptr, const Stream& stream) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto it = impl_->blocks.find(ptr);
  if (it == impl_->blocks.end() || !it->second.allocated) {
    throw TensorError("Nexus CUDA pinned: record_stream icin canli blok yok");
  }
  it->second.stream_uses.insert(stream.handle());
}

void CachingHostAllocator::empty_cache() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  for (void* ptr : impl_->deferred) {
    for (cudaEvent_t event : impl_->blocks.at(ptr).events) {
      NEXUS_CUDA_CHECK(cudaEventSynchronize(event));
    }
  }
  impl_->process();
  for (const auto& [size, ptr] : impl_->free_by_size) {
    NEXUS_CUDA_CHECK(cudaFreeHost(ptr));
    impl_->stats.reserved_bytes -= size;
    impl_->blocks.erase(ptr);
  }
  impl_->free_by_size.clear();
}

HostAllocatorStats CachingHostAllocator::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

CachingHostAllocator& pinned_allocator() {
  static auto* allocator = new CachingHostAllocator();
  return *allocator;
}

// ================================ DeviceBuffer ====================================

DeviceBuffer::DeviceBuffer(std::size_t bytes, const Stream& stream)
    : ptr_(device_allocator().allocate(bytes, stream)),
      bytes_(bytes),
      label_(static_cast<std::uint8_t>(prof::current_allocation_label())) {
  prof::ledger().record_alloc(static_cast<prof::MemoryLabel>(label_), prof::LedgerSpace::Device, bytes_);
}

DeviceBuffer::~DeviceBuffer() { reset(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)), label_(other.label_) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = std::exchange(other.ptr_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
    label_ = other.label_;
  }
  return *this;
}

void DeviceBuffer::reset() {
  if (ptr_ != nullptr) {
    prof::ledger().record_free(static_cast<prof::MemoryLabel>(label_), prof::LedgerSpace::Device, bytes_);
    device_allocator().deallocate(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
  }
}

}  // namespace cuda
}  // namespace nexus
