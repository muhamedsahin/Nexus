#pragma once

/**
 * @file fsdp_options.hpp
 * @brief FAZ 11: FSDP / ZeRO secenekleri (agir bagimliligi olmayan baslik; motor: fsdp.hpp).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"

#include <cstdint>
#include <string>

namespace nexus {
namespace dist {

enum class ShardingStrategy : std::uint8_t { NoShard, ShardOptimizer, ShardGradOp, FullShard, Hybrid };
[[nodiscard]] const char* to_string(ShardingStrategy s) noexcept;
[[nodiscard]] ShardingStrategy zero_stage(int stage);  // 0..3

struct FsdpMixedPrecision {
  DType param = DType::F32;   // all-gather tipi (F32 | BF16)
  DType reduce = DType::F32;  // reduce-scatter tipi (F32 | BF16)
};

struct FsdpOffload {
  bool optimizer_nvme = false;
  bool params_nvme = false;  // yalniz FullShard / Hybrid
  std::string directory;     // bos: gecici dizin
};

struct FsdpOptions {
  ShardingStrategy sharding = ShardingStrategy::FullShard;
  enum class Wrap : std::uint8_t { PerBlock, Whole } wrap = Wrap::PerBlock;
  FsdpMixedPrecision mixed_precision{};
  FsdpOffload offload{};
  int limit_all_gathers = 2;  // 0: sinirsiz
  bool prefetch = true;
  int ranks_per_node = 0;     // Hybrid: dugum basina rank (0: LOCAL_WORLD_SIZE)
};

struct FsdpAdamW {
  double lr = 1.0e-3;
  double beta1 = 0.9;
  double beta2 = 0.95;
  double eps = 1.0e-8;
  double weight_decay = 0.0;
};

class FsdpEngine;

}  // namespace dist
}  // namespace nexus
