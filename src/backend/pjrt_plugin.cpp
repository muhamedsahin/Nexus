/**
 * Official PJRT C ABI client. No assumptions about plugin-owned object layouts.
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif
 * ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.
 */
#include "pjrt_plugin.hpp"
#include "nexus/backend/detail/pjrt_c_api.h"
#include "nexus/core/error.hpp"
#include <algorithm>
#include <cwctype>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <utility>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace nexus::backend::pjrt::detail {
namespace {
template<class T> T args(std::size_t size) { T a{}; a.struct_size = size; return a; }

struct Library {
  void* handle = nullptr;
  const ::PJRT_Api* api = nullptr;
  void discard(::PJRT_Error* error) const noexcept {
    if (!error) return;
    auto a = args<PJRT_Error_Destroy_Args>(PJRT_Error_Destroy_Args_STRUCT_SIZE);
    a.error = error; api->PJRT_Error_Destroy(&a);
  }
  void check(::PJRT_Error* error) const {
    if (!error) return;
    auto free_error = [this](::PJRT_Error* e) { discard(e); };
    std::unique_ptr<::PJRT_Error, decltype(free_error)> owned(error, free_error);
    auto a = args<PJRT_Error_Message_Args>(PJRT_Error_Message_Args_STRUCT_SIZE);
    a.error = error; api->PJRT_Error_Message(&a);
    throw TensorError("Nexus PJRT plugin: " + (a.message ? std::string(a.message, a.message_size) : "unspecified error"));
  }
  void await(::PJRT_Event* event) const {
    if (!event) return;
    auto a = args<PJRT_Event_Await_Args>(PJRT_Event_Await_Args_STRUCT_SIZE);
    a.event = event;
    auto* error = api->PJRT_Event_Await(&a);
    auto d = args<PJRT_Event_Destroy_Args>(PJRT_Event_Destroy_Args_STRUCT_SIZE);
    d.event = event;
    auto* destroy_error = api->PJRT_Event_Destroy(&d);
    if (error) { discard(destroy_error); check(error); }
    check(destroy_error);
  }
  // Adopt BOTH results before checking either: some failing calls still return
  // an owned transfer event, which must finish before host storage is released.
  void settle(::PJRT_Error* error, ::PJRT_Event* event) const {
    auto free_error = [this](::PJRT_Error* e) { discard(e); };
    std::unique_ptr<::PJRT_Error, decltype(free_error)> owned(error, free_error);
    await(event);
    check(owned.release());
  }
  void close() noexcept {
    if (!handle) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
    handle = nullptr;
  }
  explicit Library(const std::filesystem::path& path) {
    try {
      using GetApi = const ::PJRT_Api* (*)();
#if defined(_WIN32)
      handle = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      auto get = handle ? reinterpret_cast<GetApi>(GetProcAddress(static_cast<HMODULE>(handle), "GetPjrtApi")) : nullptr;
#else
      handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
      auto get = handle ? reinterpret_cast<GetApi>(dlsym(handle, "GetPjrtApi")) : nullptr;
#endif
      if (!get) throw TensorError("Nexus PJRT: cannot load plugin/GetPjrtApi: " + path.string());
      api = get();
      if (!api || api->struct_size < PJRT_STRUCT_SIZE(::PJRT_Api, PJRT_Buffer_ReadyEvent) ||
          api->pjrt_api_version.struct_size < PJRT_Api_Version_STRUCT_SIZE ||
          api->pjrt_api_version.major_version != PJRT_API_MAJOR)
        throw TensorError("Nexus PJRT: incompatible official C ABI/table size");
#define NEXUS_PJRT_REQUIRE(fn) if (!api->fn) throw TensorError("Nexus PJRT: missing API " #fn)
      NEXUS_PJRT_REQUIRE(PJRT_Error_Message); NEXUS_PJRT_REQUIRE(PJRT_Error_Destroy);
      NEXUS_PJRT_REQUIRE(PJRT_Plugin_Initialize);
      NEXUS_PJRT_REQUIRE(PJRT_Client_Create); NEXUS_PJRT_REQUIRE(PJRT_Client_Destroy);
      NEXUS_PJRT_REQUIRE(PJRT_Client_PlatformName); NEXUS_PJRT_REQUIRE(PJRT_Client_PlatformVersion);
      NEXUS_PJRT_REQUIRE(PJRT_Client_AddressableDevices); NEXUS_PJRT_REQUIRE(PJRT_Client_Compile);
      NEXUS_PJRT_REQUIRE(PJRT_Client_BufferFromHostBuffer);
      NEXUS_PJRT_REQUIRE(PJRT_LoadedExecutable_Destroy); NEXUS_PJRT_REQUIRE(PJRT_LoadedExecutable_GetExecutable);
      NEXUS_PJRT_REQUIRE(PJRT_LoadedExecutable_Execute);
      NEXUS_PJRT_REQUIRE(PJRT_Executable_Destroy); NEXUS_PJRT_REQUIRE(PJRT_Executable_NumOutputs);
      NEXUS_PJRT_REQUIRE(PJRT_Buffer_Destroy); NEXUS_PJRT_REQUIRE(PJRT_Buffer_ElementType);
      NEXUS_PJRT_REQUIRE(PJRT_Buffer_Dimensions); NEXUS_PJRT_REQUIRE(PJRT_Buffer_ToHostBuffer);
      NEXUS_PJRT_REQUIRE(PJRT_Event_Await); NEXUS_PJRT_REQUIRE(PJRT_Event_Destroy);
#undef NEXUS_PJRT_REQUIRE
      auto init = args<PJRT_Plugin_Initialize_Args>(PJRT_Plugin_Initialize_Args_STRUCT_SIZE);
      check(api->PJRT_Plugin_Initialize(&init));
    } catch (...) { close(); throw; }
  }
  ~Library() { close(); }
};

std::shared_ptr<Library> load(const std::string& path) {
  std::error_code ec;
  auto absolute = std::filesystem::canonical(std::filesystem::path(std::u8string(path.begin(), path.end())), ec);
  if (ec) throw TensorError("Nexus PJRT: plugin path does not exist: " + path);
  auto key = absolute.native();
#if defined(_WIN32)
  std::transform(key.begin(), key.end(), key.begin(), [](wchar_t c) { return std::towlower(c); });
#endif
  static std::mutex mutex;
  static std::map<decltype(key), std::weak_ptr<Library>> libraries;
  std::lock_guard lock(mutex);
  if (auto existing = libraries[key].lock()) return existing;
  for (auto it = libraries.begin(); it != libraries.end();) {
    if (it->second.expired()) it = libraries.erase(it); else ++it;
  }
  auto library = std::make_shared<Library>(absolute);
  libraries[key] = library;
  return library;
}
}  // namespace

struct PluginClient::Impl {
  std::shared_ptr<Library> lib;
  ::PJRT_Client* client = nullptr;
  std::vector<::PJRT_Device*> devices;
  std::string platform, version;
  ~Impl() {
    if (client) {
      auto a = args<PJRT_Client_Destroy_Args>(PJRT_Client_Destroy_Args_STRUCT_SIZE);
      a.client = client; lib->discard(lib->api->PJRT_Client_Destroy(&a));
    }
  }
};

PluginClient::PluginClient(const std::string& path) : impl_(std::make_unique<Impl>()) {
  auto& p = *impl_;
  p.lib = load(path);
  const auto* api = p.lib->api;
  auto create = args<PJRT_Client_Create_Args>(PJRT_Client_Create_Args_STRUCT_SIZE);
  auto* create_error = api->PJRT_Client_Create(&create); p.client = create.client; p.lib->check(create_error);
  if (!p.client) throw TensorError("Nexus PJRT: plugin returned a null client");
  auto pn = args<PJRT_Client_PlatformName_Args>(PJRT_Client_PlatformName_Args_STRUCT_SIZE);
  pn.client = p.client; p.lib->check(api->PJRT_Client_PlatformName(&pn));
  if (!pn.platform_name) throw TensorError("Nexus PJRT: missing platform name");
  p.platform.assign(pn.platform_name, pn.platform_name_size);
  auto pv = args<PJRT_Client_PlatformVersion_Args>(PJRT_Client_PlatformVersion_Args_STRUCT_SIZE);
  pv.client = p.client; p.lib->check(api->PJRT_Client_PlatformVersion(&pv));
  if (pv.platform_version) p.version.assign(pv.platform_version, pv.platform_version_size);
  auto d = args<PJRT_Client_AddressableDevices_Args>(PJRT_Client_AddressableDevices_Args_STRUCT_SIZE);
  d.client = p.client; p.lib->check(api->PJRT_Client_AddressableDevices(&d));
  if (!d.addressable_devices || !d.num_addressable_devices || d.num_addressable_devices > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw TensorError("Nexus PJRT: no valid addressable devices");
  p.devices.assign(d.addressable_devices, d.addressable_devices + d.num_addressable_devices);
}
PluginClient::~PluginClient() = default;
std::string_view PluginClient::platform() const noexcept { return impl_->platform; }
std::string_view PluginClient::version() const noexcept { return impl_->version; }
int PluginClient::device_count() const noexcept { return static_cast<int>(impl_->devices.size()); }
std::string PluginClient::device_name(int index) const {
  if (index < 0 || index >= device_count()) throw TensorError("Nexus PJRT: invalid device index");
  return impl_->platform + ":" + std::to_string(index) + " (official PJRT plugin)";
}

std::vector<std::vector<float>> PluginClient::run(std::string_view mlir,
    const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<std::int64_t>>& shapes) {
  auto& lib = *impl_->lib; const auto* api = lib.api;
  ::PJRT_LoadedExecutable* loaded = nullptr;
  ::PJRT_Executable* executable = nullptr;
  ::PJRT_Event* completion = nullptr;
  std::vector<::PJRT_Buffer*> in(inputs.size(), nullptr), out;
  auto cleanup = [&]() noexcept {
    if (completion) { auto a = args<PJRT_Event_Destroy_Args>(PJRT_Event_Destroy_Args_STRUCT_SIZE); a.event = completion; lib.discard(api->PJRT_Event_Destroy(&a)); }
    for (auto* b : out) if (b) { auto a = args<PJRT_Buffer_Destroy_Args>(PJRT_Buffer_Destroy_Args_STRUCT_SIZE); a.buffer = b; lib.discard(api->PJRT_Buffer_Destroy(&a)); }
    for (auto* b : in) if (b) { auto a = args<PJRT_Buffer_Destroy_Args>(PJRT_Buffer_Destroy_Args_STRUCT_SIZE); a.buffer = b; lib.discard(api->PJRT_Buffer_Destroy(&a)); }
    if (executable) { auto a = args<PJRT_Executable_Destroy_Args>(PJRT_Executable_Destroy_Args_STRUCT_SIZE); a.executable = executable; lib.discard(api->PJRT_Executable_Destroy(&a)); }
    if (loaded) { auto a = args<PJRT_LoadedExecutable_Destroy_Args>(PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE); a.executable = loaded; lib.discard(api->PJRT_LoadedExecutable_Destroy(&a)); }
  };
  try {
    auto program = args<PJRT_Program>(PJRT_Program_STRUCT_SIZE);
    program.code = const_cast<char*>(mlir.data()); program.code_size = mlir.size();
    program.format = "mlir"; program.format_size = 4;
    auto c = args<PJRT_Client_Compile_Args>(PJRT_Client_Compile_Args_STRUCT_SIZE);
    c.client = impl_->client; c.program = &program;
    // CompileOptionsProto: unset device ordinal (-1), one replica/partition.
    // Empty protobuf would request zero replicas/partitions. Validated with real XLA CPU.
    constexpr char options[] = "\x1a\x0f\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x20\x01\x28\x01";
    c.compile_options = options; c.compile_options_size = sizeof(options) - 1;
    auto* error = api->PJRT_Client_Compile(&c); loaded = c.executable; lib.check(error);
    if (!loaded) throw TensorError("Nexus PJRT: null loaded executable");
    auto e = args<PJRT_LoadedExecutable_GetExecutable_Args>(PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE);
    e.loaded_executable = loaded;
    error = api->PJRT_LoadedExecutable_GetExecutable(&e); executable = e.executable; lib.check(error);
    if (!executable) throw TensorError("Nexus PJRT: null executable metadata");
    auto n = args<PJRT_Executable_NumOutputs_Args>(PJRT_Executable_NumOutputs_Args_STRUCT_SIZE);
    n.executable = executable; lib.check(api->PJRT_Executable_NumOutputs(&n)); out.resize(n.num_outputs, nullptr);
    for (std::size_t i = 0; i < in.size(); ++i) {
      auto a = args<PJRT_Client_BufferFromHostBuffer_Args>(PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE);
      a.client = impl_->client; a.data = inputs[i].data(); a.type = PJRT_Buffer_Type_F32;
      a.dims = shapes[i].data(); a.num_dims = shapes[i].size(); a.device = impl_->devices[0];
      a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
      error = api->PJRT_Client_BufferFromHostBuffer(&a); in[i] = a.buffer;
      lib.settle(error, a.done_with_host_buffer);
      if (!in[i]) throw TensorError("Nexus PJRT: null input buffer");
    }
    auto opts = args<PJRT_ExecuteOptions>(PJRT_ExecuteOptions_STRUCT_SIZE);
    std::vector<std::int64_t> no_donate(in.size()); std::iota(no_donate.begin(), no_donate.end(), 0);
    opts.non_donatable_input_indices = no_donate.data(); opts.num_non_donatable_input_indices = no_donate.size();
    auto launch = args<PJRT_LoadedExecutable_Execute_Args>(PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE);
    ::PJRT_Buffer* const* in_list = in.data(); ::PJRT_Buffer** out_list = out.data();
    launch.executable = loaded; launch.options = &opts; launch.argument_lists = &in_list;
    launch.num_devices = 1; launch.num_args = in.size(); launch.output_lists = &out_list;
    launch.device_complete_events = &completion;
    error = api->PJRT_LoadedExecutable_Execute(&launch);
    lib.settle(error, std::exchange(completion, nullptr));
    std::vector<std::vector<float>> result(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
      if (!out[i]) throw TensorError("Nexus PJRT: null output buffer");
      auto ty = args<PJRT_Buffer_ElementType_Args>(PJRT_Buffer_ElementType_Args_STRUCT_SIZE);
      ty.buffer = out[i]; lib.check(api->PJRT_Buffer_ElementType(&ty));
      if (ty.type != PJRT_Buffer_Type_F32) throw TensorError("Nexus PJRT: only FP32 output is supported");
      auto dims = args<PJRT_Buffer_Dimensions_Args>(PJRT_Buffer_Dimensions_Args_STRUCT_SIZE);
      dims.buffer = out[i]; lib.check(api->PJRT_Buffer_Dimensions(&dims));
      if (dims.num_dims && !dims.dims) throw TensorError("Nexus PJRT: missing output dimensions");
      std::size_t elements = 1;
      for (std::size_t d = 0; d < dims.num_dims; ++d) {
        if (dims.dims[d] < 0 || (dims.dims[d] && elements > std::numeric_limits<std::size_t>::max() / sizeof(float) / static_cast<std::size_t>(dims.dims[d])))
          throw TensorError("Nexus PJRT: invalid/overflowing output dimensions");
        elements *= static_cast<std::size_t>(dims.dims[d]);
      }
      std::vector<std::int64_t> order(dims.num_dims);
      std::iota(order.rbegin(), order.rend(), 0);
      auto layout = args<PJRT_Buffer_MemoryLayout>(PJRT_Buffer_MemoryLayout_STRUCT_SIZE);
      layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
      layout.tiled = args<PJRT_Buffer_MemoryLayout_Tiled>(PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE);
      layout.tiled.minor_to_major = order.data(); layout.tiled.minor_to_major_size = order.size();
      auto h = args<PJRT_Buffer_ToHostBuffer_Args>(PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE);
      h.src = out[i]; h.host_layout = &layout;
      error = api->PJRT_Buffer_ToHostBuffer(&h); lib.settle(error, h.event);
      if (h.dst_size != elements * sizeof(float)) throw TensorError("Nexus PJRT: output shape/byte size mismatch");
      result[i].resize(h.dst_size / sizeof(float));
      // A non-null address also represents a valid zero-element result.
      float empty = 0.0f; h.dst = result[i].empty() ? &empty : result[i].data(); h.event = nullptr;
      error = api->PJRT_Buffer_ToHostBuffer(&h); lib.settle(error, h.event);
    }
    cleanup(); return result;
  } catch (...) { cleanup(); throw; }
}
}  // namespace nexus::backend::pjrt::detail
