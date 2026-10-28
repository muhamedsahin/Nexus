/** Contract-test double for the OFFICIAL PJRT ABI. This is NOT XLA or a TPU backend.
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif
 * ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.
 */
#include "nexus/backend/detail/pjrt_c_api.h"
#include <atomic>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

struct TestError : PJRT_Error {
  explicit TestError(const char* text) : PJRT_Error{}, message(text) {}
  std::string message;
};
struct PJRT_Client { int copies = 0; };
struct PJRT_Device { int ordinal = 0; };
struct PJRT_Executable { int outputs = 2; };
struct PJRT_LoadedExecutable { bool compiled = true; };
struct PJRT_Buffer { std::vector<std::int64_t> dims; std::vector<float> data; };
struct PJRT_Event { bool ready = true; };
namespace {
std::atomic<int> live{0}, init_count{0};
int mode = 0, copies_out = 0, awaits = 0;
PJRT_Device device;
PJRT_Device* devices[] = {&device};
PJRT_Error* fail(const char* message) { ++live; return new TestError(message); }
template<class T> T* make() { ++live; return new T; }
template<class T> void release(T* p) { if (p) { --live; delete p; } }
void error_destroy(PJRT_Error_Destroy_Args* a) { release(static_cast<TestError*>(a->error)); }
void error_message(PJRT_Error_Message_Args* a) {
  const auto& message = static_cast<const TestError*>(a->error)->message;
  a->message = message.data(); a->message_size = message.size();
}
PJRT_Error* initialize(PJRT_Plugin_Initialize_Args*) { ++init_count; return nullptr; }
PJRT_Error* create(PJRT_Client_Create_Args* a) {
  if (mode == 5) return fail("injected client creation failure");
  if (a->struct_size < PJRT_STRUCT_SIZE(PJRT_Client_Create_Args, client)) return fail("truncated create arguments");
  a->client = make<PJRT_Client>(); return nullptr;
}
PJRT_Error* destroy(PJRT_Client_Destroy_Args* a) { release(a->client); return nullptr; }
PJRT_Error* platform(PJRT_Client_PlatformName_Args* a) { a->platform_name = "nexus-pjrt-contract-test"; a->platform_name_size = std::strlen(a->platform_name); return nullptr; }
PJRT_Error* version(PJRT_Client_PlatformVersion_Args* a) { a->platform_version = "fixture-v1"; a->platform_version_size = 10; return nullptr; }
PJRT_Error* addressable(PJRT_Client_AddressableDevices_Args* a) { a->addressable_devices = devices; a->num_addressable_devices = 1; return nullptr; }
PJRT_Error* compile(PJRT_Client_Compile_Args* a) {
  if (mode == 1) return fail("injected compile failure");
  const char expected[] = "\x1a\x0f\x08\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\x20\x01\x28\x01";
  if (!a->program || std::string(a->program->format, a->program->format_size) != "mlir" ||
      a->compile_options_size != sizeof(expected) - 1 || std::memcmp(a->compile_options, expected, sizeof(expected) - 1))
    return fail("incorrect program/serialized single-device compile options");
  a->executable = make<PJRT_LoadedExecutable>(); return nullptr;
}
PJRT_Error* loaded_destroy(PJRT_LoadedExecutable_Destroy_Args* a) { release(a->executable); return nullptr; }
PJRT_Error* get_executable(PJRT_LoadedExecutable_GetExecutable_Args* a) { a->executable = make<PJRT_Executable>(); return nullptr; }
PJRT_Error* executable_destroy(PJRT_Executable_Destroy_Args* a) { release(a->executable); return nullptr; }
PJRT_Error* num_outputs(PJRT_Executable_NumOutputs_Args* a) { a->num_outputs = static_cast<std::size_t>(a->executable->outputs); return nullptr; }
PJRT_Error* from_host(PJRT_Client_BufferFromHostBuffer_Args* a) {
  const bool inject = mode == 2 && (++a->client->copies % 2 == 0);
  if (a->type != PJRT_Buffer_Type_F32 || a->host_buffer_semantics != PJRT_HostBufferSemantics_kImmutableOnlyDuringCall || a->device != &device)
    return fail("invalid host semantics/device/type");
  a->buffer = make<PJRT_Buffer>();
  a->buffer->dims.assign(a->dims, a->dims + a->num_dims);
  std::size_t n = 1; for (const auto d : a->buffer->dims) n *= static_cast<std::size_t>(d);
  const auto* data = static_cast<const float*>(a->data);
  if (n) a->buffer->data.assign(data, data + n);
  a->done_with_host_buffer = make<PJRT_Event>();
  return inject ? fail("injected second input failure with an owned event/buffer") : nullptr;
}
PJRT_Error* buffer_destroy(PJRT_Buffer_Destroy_Args* a) { release(a->buffer); return nullptr; }
PJRT_Error* element_type(PJRT_Buffer_ElementType_Args* a) { a->type = mode == 9 ? PJRT_Buffer_Type_S32 : PJRT_Buffer_Type_F32; return nullptr; }
PJRT_Error* dimensions(PJRT_Buffer_Dimensions_Args* a) {
  static const std::int64_t invalid = -1;
  a->dims = mode == 4 ? &invalid : a->buffer->dims.data();
  a->num_dims = mode == 4 ? 1 : a->buffer->dims.size(); return nullptr;
}
PJRT_Error* event_await(PJRT_Event_Await_Args* a) {
  if (!a->event->ready) return fail("event not ready");
  if (mode == 8 && ++awaits == 3) return fail("injected completion event failure");
  return nullptr;
}
PJRT_Error* event_destroy(PJRT_Event_Destroy_Args* a) { release(a->event); return nullptr; }
PJRT_Error* execute(PJRT_LoadedExecutable_Execute_Args* a) {
  if (a->num_devices != 1 || a->num_args != 2 || !a->options ||
      a->options->num_non_donatable_input_indices != 2 ||
      a->options->non_donatable_input_indices[0] != 0 || a->options->non_donatable_input_indices[1] != 1)
    return fail("invalid argument nesting/donation options");
  const auto* x = a->argument_lists[0][0]; const auto* y = a->argument_lists[0][1];
  if (x->dims != y->dims) return fail("shape mismatch");
  for (int o = 0; o < 2; ++o) {
    if (mode == 14 && o == 1) break;
    auto* b = make<PJRT_Buffer>(); b->dims = x->dims; b->data.resize(x->data.size());
    for (std::size_t i = 0; i < b->data.size(); ++i) b->data[i] = o ? x->data[i] - y->data[i] : x->data[i] + y->data[i];
    a->output_lists[0][o] = b;
    if (mode == 3) return fail("injected partial-output execute failure");
  }
  if (a->device_complete_events) a->device_complete_events[0] = make<PJRT_Event>();
  return nullptr;
}
PJRT_Error* to_host(PJRT_Buffer_ToHostBuffer_Args* a) {
  if (!a->host_layout || a->host_layout->type != PJRT_Buffer_MemoryLayout_Type_Tiled)
    return fail("explicit row-major host layout required");
  const auto& layout = a->host_layout->tiled;
  if (layout.minor_to_major_size != a->src->dims.size()) return fail("rank/layout mismatch");
  for (std::size_t i = 0; i < layout.minor_to_major_size; ++i)
    if (layout.minor_to_major[i] != static_cast<std::int64_t>(layout.minor_to_major_size - 1 - i)) return fail("not row major");
  const auto bytes = a->src->data.size() * sizeof(float);
  if (!a->dst) {
    if (mode == 7) { a->event = make<PJRT_Event>(); return fail("injected host size query failure with an owned event"); }
    a->dst_size = mode == 13 ? bytes + 1 : bytes; return nullptr;
  }
  if (mode == 6 && ++copies_out == 2) { a->event = make<PJRT_Event>(); return fail("injected second host copy failure with an owned event"); }
  if (a->dst_size != bytes) return fail("host size mismatch");
  if (bytes) std::memcpy(a->dst, a->src->data.data(), bytes);
  a->event = make<PJRT_Event>(); return nullptr;
}
}  // namespace

extern "C" {
void NexusPjrtTestMode(int value) { mode = value; copies_out = awaits = 0; }
int NexusPjrtTestLive() { return live.load(); }
int NexusPjrtTestInitializations() { return init_count.load(); }
const PJRT_Api* GetPjrtApi() {
  static PJRT_Api api;
  api = {};
  api.struct_size = PJRT_Api_STRUCT_SIZE;
  api.pjrt_api_version = {PJRT_Api_Version_STRUCT_SIZE, nullptr, PJRT_API_MAJOR, PJRT_API_MINOR};
  api.PJRT_Error_Destroy = error_destroy; api.PJRT_Error_Message = error_message;
  api.PJRT_Plugin_Initialize = initialize;
  api.PJRT_Client_Create = create; api.PJRT_Client_Destroy = destroy;
  api.PJRT_Client_PlatformName = platform; api.PJRT_Client_PlatformVersion = version;
  api.PJRT_Client_AddressableDevices = addressable; api.PJRT_Client_Compile = compile;
  api.PJRT_Client_BufferFromHostBuffer = from_host;
  api.PJRT_LoadedExecutable_Destroy = loaded_destroy; api.PJRT_LoadedExecutable_GetExecutable = get_executable;
  api.PJRT_Executable_Destroy = executable_destroy; api.PJRT_Executable_NumOutputs = num_outputs;
  api.PJRT_LoadedExecutable_Execute = execute;
  api.PJRT_Buffer_Destroy = buffer_destroy; api.PJRT_Buffer_ElementType = element_type;
  api.PJRT_Buffer_Dimensions = dimensions; api.PJRT_Buffer_ToHostBuffer = to_host;
  api.PJRT_Event_Destroy = event_destroy; api.PJRT_Event_Await = event_await;
  if (mode == 10) api.pjrt_api_version.major_version = PJRT_API_MAJOR + 1;
  if (mode == 11) api.PJRT_Buffer_ToHostBuffer = nullptr;
  if (mode == 12) api.struct_size = offsetof(PJRT_Api, PJRT_Client_Create);
  return &api;
}
}
