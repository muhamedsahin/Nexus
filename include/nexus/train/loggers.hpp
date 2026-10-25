#pragma once

/**
 * @file loggers.hpp
 * @brief FAZ 9.6: NexusTrain LoggerBase uzerine iki ek gunlukcu.
 *
 *  - TensorBoardLogger: protobuf kutuphanesi olmadan `events.out.tfevents.*` yazar. Dosya
 *    TFRecord dizisidir: [uint64 uzunluk][uint32 maskeli crc32c(uzunluk)][veri][uint32 maskeli
 *    crc32c(veri)]. Veri, elle kodlanan tensorflow.Event mesajidir: wall_time (1, double),
 *    step (2, int64), file_version (3, "brain.Event:2", ilk kayit) ve summary (5) ->
 *    Summary.value (1) -> {tag (1), simple_value (2, float)}. Yalniz skaler yazilir.
 *  - WandbJsonlLogger: W&B `wandb-history.jsonl` satir bicimi ({"_step", "_runtime",
 *    "_timestamp", metrikler...}); `wandb sync` ile ice aktarilabilir bicim, ag yok.
 * Ikisi de satir/kayit basina flush eder (cokme halinde yarim dosya kalmaz).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus_train/logging/logger_base.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace nexus {
namespace train {

/// CRC-32C (Castagnoli, yansimali 0x82F63B78). crc32c("123456789") = 0xE3069283.
[[nodiscard]] std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc = 0) noexcept;
/// TFRecord maskesi: ((crc >> 15) | (crc << 17)) + 0xA282EAD8.
[[nodiscard]] std::uint32_t masked_crc32c(const void* data, std::size_t size) noexcept;

class TensorBoardLogger final : public nexus_train::LoggerBase {
 public:
  /// directory icinde events.out.tfevents.<unix>.nexus dosyasini acar (dizin yoksa olusturur).
  explicit TensorBoardLogger(const std::string& directory);
  void log_step(const nexus_train::TrainingState& state) override;
  void log_epoch(const nexus_train::TrainingState& state) override;
  void scalar(const std::string& tag, double value, std::int64_t step);
  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  void write_record(const std::string& payload);
  void write_scalars(const std::vector<std::pair<std::string, double>>& values, std::int64_t step);
  std::string path_;
  std::ofstream out_;
};

/// TensorBoard dosyasini okur (testler icin): CRC'leri denetler, (adim, etiket, deger) dondurur.
struct TensorBoardScalar {
  std::int64_t step = 0;
  std::string tag;
  float value = 0.0f;
};
[[nodiscard]] std::vector<TensorBoardScalar> read_tensorboard_scalars(const std::string& path, std::string* file_version);

class WandbJsonlLogger final : public nexus_train::LoggerBase {
 public:
  explicit WandbJsonlLogger(const std::string& path);
  void log_step(const nexus_train::TrainingState& state) override;
  void log_epoch(const nexus_train::TrainingState& state) override;

 private:
  void write(const nexus_train::TrainingState& state, const char* prefix);
  std::ofstream out_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace train
}  // namespace nexus
