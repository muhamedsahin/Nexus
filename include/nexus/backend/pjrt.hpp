#pragma once

/**
 * @file pjrt.hpp
 * @brief FAZ 19.E: StableHLO CPU reference interpreter and
 *        DeviceMesh + DTensor -> XLA SPMD (GSPMD / mhlo.sharding) disa aktarimi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/ir.hpp"
#include "nexus/dist/dtensor.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace nexus::backend::pjrt {
namespace detail { class PluginClient; }

/// Legacy Nexus reference-interpreter table. NOT ABI-compatible with OpenXLA PJRT.
struct PJRT_Error;
struct PJRT_Client;
struct PJRT_Device;
struct PJRT_Buffer;
struct PJRT_LoadedExecutable;

struct PJRT_Api_Version {
  std::size_t struct_size;
  int major_version;
  int minor_version;
};

struct PJRT_Api {
  std::size_t struct_size;
  PJRT_Api_Version pjrt_api_version;
  const char* (*PlatformName)();
  const char* (*PlatformVersion)();
  PJRT_Error* (*Client_Create)(PJRT_Client** out_client);
  void (*Client_Destroy)(PJRT_Client* client);
  int (*Client_DeviceCount)(PJRT_Client* client);
  PJRT_Device* (*Client_AddressableDevice)(PJRT_Client* client, int index);
  const char* (*Device_DebugString)(PJRT_Device* device);
  PJRT_Error* (*Client_Compile)(PJRT_Client* client, const char* mlir_module, std::size_t mlir_size,
                                PJRT_LoadedExecutable** out_exec);
  void (*LoadedExecutable_Destroy)(PJRT_LoadedExecutable* exec);
  PJRT_Error* (*Buffer_FromHostBuffer)(PJRT_Client* client, const float* host_data,
                                       const std::int64_t* dims, std::size_t num_dims,
                                       PJRT_Device* device, PJRT_Buffer** out_buffer);
  PJRT_Error* (*Buffer_ToHostBuffer)(PJRT_Buffer* buffer, float* dst, std::size_t dst_bytes);
  void (*Buffer_Destroy)(PJRT_Buffer* buffer);
  PJRT_Error* (*LoadedExecutable_Execute)(PJRT_LoadedExecutable* exec,
                                          PJRT_Buffer* const* argument_lists, std::size_t num_args,
                                          PJRT_Buffer** output_lists, std::size_t num_outputs);
  const char* (*Error_Message)(const PJRT_Error* error);
  void (*Error_Destroy)(PJRT_Error* error);
};

using GetPjrtApiFn = const PJRT_Api* (*)();

/// Legacy CPU reference interpreter (not XLA compilation or a TPU device).
[[nodiscard]] const PJRT_Api* GetBuiltinCpuPjrtApi() noexcept;

/// GSPMD mhlo.sharding dizgesi uretir:
///   - Tumu Replicate -> "{replicated}"
///   - Partial -> explicit error; reduce to a concrete placement before export.
///   - Shard(d) -> "{devices=[d0,d1,...]0,1,2,...}"
[[nodiscard]] std::string to_mhlo_sharding(const std::vector<std::int64_t>& tensor_shape,
                                           const dist::DeviceMesh& mesh,
                                           const dist::Placements& placements);

/// Parametre/girdi adina gore mhlo.sharding oznitelikleriyle zenginlestirilmis StableHLO MLIR uretir.
[[nodiscard]] std::string export_stablehlo_spmd(
    const ir::Graph& g, const ir::Bindings& bindings,
    const std::map<std::string, std::string>& arg_shardings = {},
    const std::vector<std::string>& result_shardings = {});

class Client {
 public:
  /// Without a plugin, uses the explicitly labelled Nexus CPU reference interpreter.
  /// An explicit path (or NEXUS_PJRT_PLUGIN_PATH) loads the official OpenXLA C ABI.
  explicit Client(std::string_view plugin_path = "");
  ~Client();
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  [[nodiscard]] std::string_view platform_name() const noexcept;
  [[nodiscard]] std::string_view platform_version() const noexcept;
  [[nodiscard]] int device_count() const noexcept;
  [[nodiscard]] std::string device_name(int index) const;
  [[nodiscard]] bool uses_external_plugin() const noexcept { return external_ != nullptr; }

  /// StableHLO MLIR modulunu PJRT C API uzerinden derler ve verilen argumanlarla yurutur.
  [[nodiscard]] std::vector<std::vector<float>> compile_and_execute(
      std::string_view stablehlo_mlir,
      const std::vector<std::vector<float>>& inputs,
      const std::vector<std::vector<std::int64_t>>& input_shapes);

 private:
  std::unique_ptr<detail::PluginClient> external_;
  const PJRT_Api* api_ = nullptr;
  PJRT_Client* client_ = nullptr;
};

}  // namespace nexus::backend::pjrt
