/**
 * @file directml.cpp
 * @brief FAZ 19.D: Windows DirectML (DirectX 12 uzerinden tum Windows GPU'lari — NVIDIA, AMD, Intel)
 *        arka ucu. Dinamik dxgi.dll / d3d12.dll / DirectML.dll yukleyici, D3D12 tampon havuzu ve
 *        DML_OPERATOR_GEMM + temel egitim/cikarim islemleri.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/backend/backend.hpp"
#include "nexus/cpu/gemm.hpp"
#include "nexus/cpu/threads.hpp"
#include "nexus/core/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32) && defined(NEXUS_ENABLE_DIRECTML)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#include <DirectML.h>
#include <wrl/client.h>

namespace nexus::backend {
namespace {

using Microsoft::WRL::ComPtr;

struct DmlApi {
  HMODULE dxgi = nullptr;
  HMODULE d3d12 = nullptr;
  HMODULE dml = nullptr;
  decltype(&CreateDXGIFactory1) create_factory = nullptr;
  decltype(&D3D12CreateDevice) create_d3d12 = nullptr;
  decltype(&DMLCreateDevice) create_dml = nullptr;
};

DmlApi& dml_api() {
  static DmlApi api = [] {
    DmlApi a;
    a.dxgi = LoadLibraryA("dxgi.dll");
    a.d3d12 = LoadLibraryA("d3d12.dll");
    a.dml = LoadLibraryA("DirectML.dll");
    if (a.dxgi) {
      a.create_factory = reinterpret_cast<decltype(&CreateDXGIFactory1)>(GetProcAddress(a.dxgi, "CreateDXGIFactory1"));
    }
    if (a.d3d12) {
      a.create_d3d12 = reinterpret_cast<decltype(&D3D12CreateDevice)>(GetProcAddress(a.d3d12, "D3D12CreateDevice"));
    }
    if (a.dml) {
      a.create_dml = reinterpret_cast<decltype(&DMLCreateDevice)>(GetProcAddress(a.dml, "DMLCreateDevice"));
    }
    return a;
  }();
  return api;
}

void check_hr(HRESULT hr, const char* what) {
  if (FAILED(hr)) {
    throw TensorError(std::string("Nexus DirectML: ") + what + " HRESULT=" + std::to_string(static_cast<long>(hr)));
  }
}

std::string wide_to_utf8(const wchar_t* w) {
  if (w == nullptr || *w == L'\0') return {};
  const int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  if (len <= 1) return {};
  std::string out(static_cast<std::size_t>(len - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), len, nullptr, nullptr);
  return out;
}

struct DmlBuffer {
  ComPtr<ID3D12Resource> resource;
  std::uint64_t size = 0;
};

struct DmlContext {
  ComPtr<IDXGIAdapter1> adapter;
  ComPtr<ID3D12Device> d3d_device;
  ComPtr<IDMLDevice> dml_device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> cmd_alloc;
  ComPtr<ID3D12GraphicsCommandList> cmd_list;
  ComPtr<IDMLCommandRecorder> recorder;
  ComPtr<IDMLOperatorInitializer> initializer;
  ComPtr<ID3D12Fence> fence;
  HANDLE fence_event = nullptr;
  std::uint64_t fence_value = 0;

  ComPtr<ID3D12Resource> upload_buf;
  std::uint64_t upload_size = 0;
  void* upload_mapped = nullptr;

  ComPtr<ID3D12Resource> readback_buf;
  std::uint64_t readback_size = 0;
  void* readback_mapped = nullptr;

  std::unordered_map<std::uint64_t, std::vector<DmlBuffer*>> free_pool;
  std::unordered_set<DmlBuffer*> all_buffers;
  mutable std::mutex mutex;
  AllocatorStats stats{};

  ~DmlContext() {
    if (upload_buf && upload_mapped) upload_buf->Unmap(0, nullptr);
    if (readback_buf && readback_mapped) readback_buf->Unmap(0, nullptr);
    for (DmlBuffer* b : all_buffers) delete b;
    all_buffers.clear();
    free_pool.clear();
    if (fence_event) CloseHandle(fence_event);
  }
};

ComPtr<ID3D12Resource> make_d3d_buffer(ID3D12Device* dev, std::uint64_t bytes, D3D12_HEAP_TYPE heap_type,
                                       D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES init_state) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = heap_type;
  heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap.CreationNodeMask = 1;
  heap.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Alignment = 0;
  desc.Width = (bytes + 255ull) & ~255ull;
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.Format = DXGI_FORMAT_UNKNOWN;
  desc.SampleDesc.Count = 1;
  desc.SampleDesc.Quality = 0;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = flags;

  ComPtr<ID3D12Resource> res;
  check_hr(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, init_state, nullptr,
                                        IID_PPV_ARGS(&res)),
           "CreateCommittedResource");
  return res;
}

void wait_gpu(DmlContext& c) {
  const std::uint64_t val = ++c.fence_value;
  check_hr(c.queue->Signal(c.fence.Get(), val), "Queue::Signal");
  if (c.fence->GetCompletedValue() < val) {
    check_hr(c.fence->SetEventOnCompletion(val, c.fence_event), "SetEventOnCompletion");
    WaitForSingleObject(c.fence_event, INFINITE);
  }
}

void begin_cmd(DmlContext& c) {
  check_hr(c.cmd_alloc->Reset(), "CommandAllocator::Reset");
  check_hr(c.cmd_list->Reset(c.cmd_alloc.Get(), nullptr), "CommandList::Reset");
}

void submit_cmd(DmlContext& c) {
  check_hr(c.cmd_list->Close(), "CommandList::Close");
  ID3D12CommandList* lists[] = {c.cmd_list.Get()};
  c.queue->ExecuteCommandLists(1, lists);
  wait_gpu(c);
}

void ensure_staging(DmlContext& c, std::uint64_t bytes) {
  const std::uint64_t aligned = (bytes + 255ull) & ~255ull;
  if (c.upload_size < aligned) {
    if (c.upload_buf && c.upload_mapped) c.upload_buf->Unmap(0, nullptr);
    c.upload_mapped = nullptr;
    c.upload_buf = make_d3d_buffer(c.d3d_device.Get(), aligned, D3D12_HEAP_TYPE_UPLOAD,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    c.upload_size = aligned;
    check_hr(c.upload_buf->Map(0, nullptr, &c.upload_mapped), "Map upload");
  }
  if (c.readback_size < aligned) {
    if (c.readback_buf && c.readback_mapped) c.readback_buf->Unmap(0, nullptr);
    c.readback_mapped = nullptr;
    c.readback_buf = make_d3d_buffer(c.d3d_device.Get(), aligned, D3D12_HEAP_TYPE_READBACK,
                                     D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    c.readback_size = aligned;
    check_hr(c.readback_buf->Map(0, nullptr, &c.readback_mapped), "Map readback");
  }
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* res,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
  if (before == after) return;
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = res;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = before;
  barrier.Transition.StateAfter = after;
  list->ResourceBarrier(1, &barrier);
}

void uav_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* res) {
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  barrier.UAV.pResource = res;
  list->ResourceBarrier(1, &barrier);
}

void run_dml_operator(DmlContext& c, IDMLCompiledOperator* compiled,
                      const DML_BINDING_DESC* inputs, std::uint32_t input_count,
                      const DML_BINDING_DESC* outputs, std::uint32_t output_count) {
  IDMLCompiledOperator* ops[] = {compiled};
  check_hr(c.initializer->Reset(1, ops), "Initializer::Reset");

  DML_BINDING_PROPERTIES init_props = c.initializer->GetBindingProperties();
  DML_BINDING_PROPERTIES exec_props = compiled->GetBindingProperties();
  const std::uint32_t desc_count = std::max(1u, init_props.RequiredDescriptorCount + exec_props.RequiredDescriptorCount);

  D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
  heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  heap_desc.NumDescriptors = desc_count;
  heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ComPtr<ID3D12DescriptorHeap> desc_heap;
  check_hr(c.d3d_device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&desc_heap)), "CreateDescriptorHeap");

  const std::uint32_t step_size = c.d3d_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

  ComPtr<ID3D12Resource> persistent_res;
  ComPtr<ID3D12Resource> temp_res;
  if (exec_props.PersistentResourceSize > 0) {
    persistent_res = make_d3d_buffer(c.d3d_device.Get(), exec_props.PersistentResourceSize,
                                     D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }
  const std::uint64_t max_temp = std::max(init_props.TemporaryResourceSize, exec_props.TemporaryResourceSize);
  if (max_temp > 0) {
    temp_res = make_d3d_buffer(c.d3d_device.Get(), max_temp, D3D12_HEAP_TYPE_DEFAULT,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }

  begin_cmd(c);
  ID3D12DescriptorHeap* heaps[] = {desc_heap.Get()};
  c.cmd_list->SetDescriptorHeaps(1, heaps);

  // 1) Initialize operator
  D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle = desc_heap->GetCPUDescriptorHandleForHeapStart();
  D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle = desc_heap->GetGPUDescriptorHandleForHeapStart();

  DML_BINDING_TABLE_DESC init_table_desc{};
  init_table_desc.Dispatchable = c.initializer.Get();
  init_table_desc.CPUDescriptorHandle = cpu_handle;
  init_table_desc.GPUDescriptorHandle = gpu_handle;
  init_table_desc.SizeInDescriptors = init_props.RequiredDescriptorCount;

  ComPtr<IDMLBindingTable> binding_table;
  check_hr(c.dml_device->CreateBindingTable(&init_table_desc, IID_PPV_ARGS(&binding_table)), "CreateBindingTable(init)");
  if (init_props.TemporaryResourceSize > 0) {
    DML_BUFFER_BINDING b{temp_res.Get(), 0, init_props.TemporaryResourceSize};
    DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &b};
    binding_table->BindTemporaryResource(&d);
  }
  if (exec_props.PersistentResourceSize > 0) {
    DML_BUFFER_BINDING b{persistent_res.Get(), 0, exec_props.PersistentResourceSize};
    DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &b};
    binding_table->BindOutputs(1, &d);
  }
  c.recorder->RecordDispatch(c.cmd_list.Get(), c.initializer.Get(), binding_table.Get());
  if (persistent_res) uav_barrier(c.cmd_list.Get(), persistent_res.Get());

  // 2) Execute operator
  cpu_handle.ptr += static_cast<SIZE_T>(init_props.RequiredDescriptorCount) * step_size;
  gpu_handle.ptr += static_cast<UINT64>(init_props.RequiredDescriptorCount) * step_size;

  DML_BINDING_TABLE_DESC exec_table_desc{};
  exec_table_desc.Dispatchable = compiled;
  exec_table_desc.CPUDescriptorHandle = cpu_handle;
  exec_table_desc.GPUDescriptorHandle = gpu_handle;
  exec_table_desc.SizeInDescriptors = exec_props.RequiredDescriptorCount;
  check_hr(binding_table->Reset(&exec_table_desc), "BindingTable::Reset(exec)");

  binding_table->BindInputs(input_count, inputs);
  binding_table->BindOutputs(output_count, outputs);
  if (exec_props.PersistentResourceSize > 0) {
    DML_BUFFER_BINDING b{persistent_res.Get(), 0, exec_props.PersistentResourceSize};
    DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &b};
    binding_table->BindPersistentResource(&d);
  }
  if (exec_props.TemporaryResourceSize > 0) {
    DML_BUFFER_BINDING b{temp_res.Get(), 0, exec_props.TemporaryResourceSize};
    DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &b};
    binding_table->BindTemporaryResource(&d);
  }
  c.recorder->RecordDispatch(c.cmd_list.Get(), compiled, binding_table.Get());
  submit_cmd(c);
}

class DirectMlDriver final : public Driver {
 public:
  DirectMlDriver() {
    DmlApi& a = dml_api();
    if (!a.create_factory || !a.create_d3d12 || !a.create_dml) {
      reason_ = "dxgi.dll / d3d12.dll / DirectML.dll giris noktalari yuklenemedi";
      return;
    }
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(a.create_factory(IID_PPV_ARGS(&factory)))) {
      reason_ = "DXGI factory olusturulamadi";
      return;
    }
    for (UINT idx = 0;; ++idx) {
      ComPtr<IDXGIAdapter1> adapter;
      if (factory->EnumAdapters1(idx, &adapter) == DXGI_ERROR_NOT_FOUND) break;
      DXGI_ADAPTER_DESC1 desc{};
      if (FAILED(adapter->GetDesc1(&desc))) continue;
      if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) continue;

      try {
        auto ctx = std::make_unique<DmlContext>();
        ctx->adapter = adapter;
        check_hr(a.create_d3d12(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ctx->d3d_device)), "D3D12CreateDevice");
        check_hr(a.create_dml(ctx->d3d_device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&ctx->dml_device)), "DMLCreateDevice");

        D3D12_COMMAND_QUEUE_DESC qdesc{};
        qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        qdesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        check_hr(ctx->d3d_device->CreateCommandQueue(&qdesc, IID_PPV_ARGS(&ctx->queue)), "CreateCommandQueue");
        check_hr(ctx->d3d_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ctx->cmd_alloc)),
                 "CreateCommandAllocator");
        check_hr(ctx->d3d_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ctx->cmd_alloc.Get(), nullptr,
                                                    IID_PPV_ARGS(&ctx->cmd_list)),
                 "CreateCommandList");
        check_hr(ctx->cmd_list->Close(), "Close initial CommandList");
        check_hr(ctx->dml_device->CreateCommandRecorder(IID_PPV_ARGS(&ctx->recorder)), "CreateCommandRecorder");
        check_hr(ctx->dml_device->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&ctx->initializer)),
                 "CreateOperatorInitializer");
        check_hr(ctx->d3d_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ctx->fence)), "CreateFence");
        ctx->fence_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);

        DeviceInfo info;
        info.device = DeviceRef::directml(static_cast<int>(contexts_.size()));
        info.name = wide_to_utf8(desc.Description);
        info.vendor = std::to_string(desc.VendorId) + ":" + std::to_string(desc.DeviceId);
        info.api_version = "D3D12+DML1";
        info.driver_version = std::to_string(desc.Revision);
        info.memory_bytes = desc.DedicatedVideoMemory > 0 ? desc.DedicatedVideoMemory : desc.SharedSystemMemory;
        info.subgroup_size = (desc.VendorId == 0x1002) ? 64 : 32;
        info.unified_memory = (desc.DedicatedVideoMemory == 0);
        devices_.push_back(std::move(info));
        contexts_.push_back(std::move(ctx));
      } catch (const std::exception& e) {
        if (reason_.empty()) reason_ = e.what();
      }
    }
    if (contexts_.empty() && reason_.empty()) {
      reason_ = "DirectML uyumlu D3D12 donanim bagdastiricisi bulunamadi";
    }
  }

  Kind kind() const noexcept override { return Kind::DirectMl; }
  std::string_view name() const noexcept override { return "directml"; }
  std::string_view version() const noexcept override { return "1.0"; }
  const std::vector<DeviceInfo>& devices() const noexcept override { return devices_; }
  std::string unavailable_reason() const override { return reason_; }
  bool supports(Operation op) const noexcept override {
    return !devices_.empty() && op == Operation::MatMul;
  }

  void* allocate(int device, std::size_t bytes) override {
    if (bytes == 0) return nullptr;
    DmlContext& c = context(device);
    std::lock_guard lock(c.mutex);
    const std::uint64_t size = (bytes + 255ull) & ~255ull;
    auto& free = c.free_pool[size];
    ++c.stats.allocations;
    DmlBuffer* buf = nullptr;
    if (!free.empty()) {
      buf = free.back();
      free.pop_back();
      ++c.stats.cache_hits;
    } else {
      buf = new DmlBuffer();
      buf->size = size;
      buf->resource = make_d3d_buffer(c.d3d_device.Get(), size, D3D12_HEAP_TYPE_DEFAULT,
                                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      c.all_buffers.insert(buf);
      c.stats.reserved_bytes += size;
    }
    ++c.stats.live_blocks;
    c.stats.current_bytes += size;
    return buf;
  }

  void deallocate(int device, void* pointer) noexcept override {
    if (pointer == nullptr) return;
    try {
      DmlContext& c = context(device);
      std::lock_guard lock(c.mutex);
      auto* buf = static_cast<DmlBuffer*>(pointer);
      if (c.all_buffers.find(buf) == c.all_buffers.end()) return;
      c.free_pool[buf->size].push_back(buf);
      if (c.stats.live_blocks > 0) --c.stats.live_blocks;
      c.stats.current_bytes = c.stats.current_bytes >= buf->size ? c.stats.current_bytes - buf->size : 0;
    } catch (...) {}
  }

  void copy(DeviceRef dst_device, void* dst, DeviceRef src_device, const void* src, std::size_t bytes) override {
    if (bytes == 0) return;
    const bool dst_dml = dst_device.kind == DeviceKind::DirectMl;
    const bool src_dml = src_device.kind == DeviceKind::DirectMl;
    if (!dst_dml && !src_dml) {
      std::memcpy(dst, src, bytes);
      return;
    }
    if (dst_dml && src_dml && dst_device.index != src_device.index) {
      std::vector<std::byte> host(bytes);
      copy(DeviceRef::cpu(), host.data(), src_device, src, bytes);
      copy(dst_device, dst, DeviceRef::cpu(), host.data(), bytes);
      return;
    }
    DmlContext& c = context(dst_dml ? dst_device.index : src_device.index);
    std::lock_guard lock(c.mutex);
    ensure_staging(c, bytes);
    begin_cmd(c);
    if (dst_dml && src_dml) {
      auto* s = static_cast<const DmlBuffer*>(src);
      auto* d = static_cast<DmlBuffer*>(dst);
      transition(c.cmd_list.Get(), s->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
      transition(c.cmd_list.Get(), d->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
      c.cmd_list->CopyBufferRegion(d->resource.Get(), 0, s->resource.Get(), 0, bytes);
      transition(c.cmd_list.Get(), s->resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      transition(c.cmd_list.Get(), d->resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      submit_cmd(c);
    } else if (dst_dml) {
      auto* d = static_cast<DmlBuffer*>(dst);
      std::memcpy(c.upload_mapped, src, bytes);
      transition(c.cmd_list.Get(), d->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
      c.cmd_list->CopyBufferRegion(d->resource.Get(), 0, c.upload_buf.Get(), 0, bytes);
      transition(c.cmd_list.Get(), d->resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      submit_cmd(c);
    } else {
      auto* s = static_cast<const DmlBuffer*>(src);
      transition(c.cmd_list.Get(), s->resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
      c.cmd_list->CopyBufferRegion(c.readback_buf.Get(), 0, s->resource.Get(), 0, bytes);
      transition(c.cmd_list.Get(), s->resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      submit_cmd(c);
      std::memcpy(dst, c.readback_mapped, bytes);
    }
  }

  void synchronize(int device) override {
    DmlContext& c = context(device);
    std::lock_guard lock(c.mutex);
    wait_gpu(c);
  }

  void matmul(DeviceRef device, const MatMulArgs& args) override {
    if (args.m <= 0 || args.n <= 0 || args.k < 0 || args.lda <= 0 || args.ldb <= 0 || args.ldc <= 0) {
      throw TensorError("Nexus DirectML: gecersiz GEMM boyutu");
    }
    DmlContext& c = context(device.index);
    std::lock_guard lock(c.mutex);
    const auto* a_buf = static_cast<const DmlBuffer*>(static_cast<const void*>(args.a));
    const auto* b_buf = static_cast<const DmlBuffer*>(static_cast<const void*>(args.b));
    auto* c_buf = static_cast<DmlBuffer*>(static_cast<void*>(args.c));

    const UINT a_rows = static_cast<UINT>(args.a_trans ? args.k : args.m);
    const UINT a_cols = static_cast<UINT>(args.a_trans ? args.m : args.k);
    const UINT b_rows = static_cast<UINT>(args.b_trans ? args.n : args.k);
    const UINT b_cols = static_cast<UINT>(args.b_trans ? args.k : args.n);

    const UINT a_sizes[4] = {1, 1, a_rows, a_cols};
    const UINT a_strides[4] = {a_rows * static_cast<UINT>(args.lda), a_rows * static_cast<UINT>(args.lda),
                               static_cast<UINT>(args.lda), 1};
    const UINT b_sizes[4] = {1, 1, b_rows, b_cols};
    const UINT b_strides[4] = {b_rows * static_cast<UINT>(args.ldb), b_rows * static_cast<UINT>(args.ldb),
                               static_cast<UINT>(args.ldb), 1};
    const UINT c_sizes[4] = {1, 1, static_cast<UINT>(args.m), static_cast<UINT>(args.n)};
    const UINT c_strides[4] = {static_cast<UINT>(args.m * args.ldc), static_cast<UINT>(args.m * args.ldc),
                               static_cast<UINT>(args.ldc), 1};

    auto calc_bytes = [](UINT rows, UINT cols, UINT ld) -> UINT64 {
      const UINT64 elems = (rows > 0 ? static_cast<UINT64>(rows - 1) * ld + cols : 0);
      return ((elems * sizeof(float)) + 3ull) & ~3ull;
    };

    DML_BUFFER_TENSOR_DESC a_bdesc{};
    a_bdesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    a_bdesc.DimensionCount = 4;
    a_bdesc.Sizes = a_sizes;
    a_bdesc.Strides = (static_cast<int>(a_cols) == args.lda) ? nullptr : a_strides;
    a_bdesc.TotalTensorSizeInBytes = calc_bytes(a_rows, a_cols, static_cast<UINT>(args.lda));

    DML_BUFFER_TENSOR_DESC b_bdesc{};
    b_bdesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    b_bdesc.DimensionCount = 4;
    b_bdesc.Sizes = b_sizes;
    b_bdesc.Strides = (static_cast<int>(b_cols) == args.ldb) ? nullptr : b_strides;
    b_bdesc.TotalTensorSizeInBytes = calc_bytes(b_rows, b_cols, static_cast<UINT>(args.ldb));

    DML_BUFFER_TENSOR_DESC c_bdesc{};
    c_bdesc.DataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    c_bdesc.DimensionCount = 4;
    c_bdesc.Sizes = c_sizes;
    c_bdesc.Strides = (args.n == args.ldc) ? nullptr : c_strides;
    c_bdesc.TotalTensorSizeInBytes = calc_bytes(static_cast<UINT>(args.m), static_cast<UINT>(args.n), static_cast<UINT>(args.ldc));

    DML_TENSOR_DESC a_tdesc{DML_TENSOR_TYPE_BUFFER, &a_bdesc};
    DML_TENSOR_DESC b_tdesc{DML_TENSOR_TYPE_BUFFER, &b_bdesc};
    DML_TENSOR_DESC c_tdesc{DML_TENSOR_TYPE_BUFFER, &c_bdesc};

    DML_GEMM_OPERATOR_DESC gemm{};
    gemm.ATensor = &a_tdesc;
    gemm.BTensor = &b_tdesc;
    gemm.CTensor = (args.beta != 0.0f) ? &c_tdesc : nullptr;
    gemm.OutputTensor = &c_tdesc;
    gemm.TransA = args.a_trans ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    gemm.TransB = args.b_trans ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    gemm.Alpha = args.alpha;
    gemm.Beta = args.beta;
    gemm.FusedActivation = nullptr;

    DML_OPERATOR_DESC op_desc{DML_OPERATOR_GEMM, &gemm};
    ComPtr<IDMLOperator> dml_op;
    check_hr(c.dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&dml_op)), "CreateOperator(GEMM)");
    ComPtr<IDMLCompiledOperator> compiled;
    check_hr(c.dml_device->CompileOperator(dml_op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&compiled)),
             "CompileOperator(GEMM)");

    DML_BUFFER_BINDING a_bind{a_buf->resource.Get(), 0, a_buf->size};
    DML_BUFFER_BINDING b_bind{b_buf->resource.Get(), 0, b_buf->size};
    DML_BUFFER_BINDING c_bind{c_buf->resource.Get(), 0, c_buf->size};

    DML_BINDING_DESC in_binds[3] = {
        {DML_BINDING_TYPE_BUFFER, &a_bind},
        {DML_BINDING_TYPE_BUFFER, &b_bind},
        {args.beta != 0.0f ? DML_BINDING_TYPE_BUFFER : DML_BINDING_TYPE_NONE,
         args.beta != 0.0f ? &c_bind : nullptr},
    };
    DML_BINDING_DESC out_bind{DML_BINDING_TYPE_BUFFER, &c_bind};
    run_dml_operator(c, compiled.Get(), in_binds, 3, &out_bind, 1);
  }

  void elementwise(Operation op, DeviceRef device, const float* a, const float* b, float* out,
                   std::size_t count, float scalar) override {
    if (count == 0) return;
    // Route direct Driver calls through the same visible fallback policy.
    require_cpu_fallback(*this, op, count);
    std::vector<float> ha(count), hb, hout(count);
    copy(DeviceRef::cpu(), ha.data(), device, a, count * sizeof(float));
    if (b != nullptr && (op <= Operation::Div || op == Operation::Axpy)) {
      hb.resize(count);
      copy(DeviceRef::cpu(), hb.data(), device, b, count * sizeof(float));
    }
    Registry::instance().require(Kind::Cpu).elementwise(op, DeviceRef::cpu(), ha.data(),
                                                        hb.empty() ? nullptr : hb.data(),
                                                        hout.data(), count, scalar);
    copy(device, out, DeviceRef::cpu(), hout.data(), count * sizeof(float));
  }

  void reduce_sum(DeviceRef device, const float* input, float* output, std::size_t count) override {
    require_cpu_fallback(*this, Operation::ReduceSum, count);
    if (count == 0) {
      fill(device, output, 1, 0.0f);
      return;
    }
    std::vector<float> hin(count);
    copy(DeviceRef::cpu(), hin.data(), device, input, count * sizeof(float));
    float sum = 0.0f;
    Registry::instance().require(Kind::Cpu).reduce_sum(DeviceRef::cpu(), hin.data(), &sum, count);
    copy(device, output, DeviceRef::cpu(), &sum, sizeof(float));
  }

  AllocatorStats allocator_stats(int device) const override {
    DmlContext& c = context(device);
    std::lock_guard lock(c.mutex);
    return c.stats;
  }

 private:
  DmlContext& context(int index) const {
    if (index < 0 || static_cast<std::size_t>(index) >= contexts_.size()) {
      throw TensorError("Nexus DirectML: gecersiz cihaz indeksi");
    }
    return *contexts_[static_cast<std::size_t>(index)];
  }

  std::vector<std::unique_ptr<DmlContext>> contexts_;
  std::vector<DeviceInfo> devices_;
  std::string reason_;
};

}  // namespace

std::unique_ptr<Driver> make_directml_driver() { return std::make_unique<DirectMlDriver>(); }

}  // namespace nexus::backend

#else
namespace nexus::backend {
namespace {
class DisabledDirectMl final : public Driver {
 public:
  Kind kind() const noexcept override { return Kind::DirectMl; }
  std::string_view name() const noexcept override { return "directml"; }
  std::string_view version() const noexcept override { return "0"; }
  const std::vector<DeviceInfo>& devices() const noexcept override { return devices_; }
  std::string unavailable_reason() const override { return "DirectML yalniz Windows (DirectX 12) derlemesinde etkindir"; }
  bool supports(Operation) const noexcept override { return false; }
  void* allocate(int, std::size_t) override { fail(); }
  void deallocate(int, void*) noexcept override {}
  void copy(DeviceRef, void*, DeviceRef, const void*, std::size_t) override { fail(); }
  void synchronize(int) override { fail(); }
  void elementwise(Operation, DeviceRef, const float*, const float*, float*, std::size_t, float) override { fail(); }
  void matmul(DeviceRef, const MatMulArgs&) override { fail(); }
  void reduce_sum(DeviceRef, const float*, float*, std::size_t) override { fail(); }
 private:
  [[noreturn]] void fail() const { throw TensorError("Nexus: DirectML bu platformda etkin degil"); }
  std::vector<DeviceInfo> devices_;
};
}  // namespace
std::unique_ptr<Driver> make_directml_driver() { return std::make_unique<DisabledDirectMl>(); }
}  // namespace nexus::backend
#endif
