#pragma once
// Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif
// ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace nexus::backend::pjrt::detail {
class PluginClient {
 public:
  explicit PluginClient(const std::string& path);
  ~PluginClient();
  PluginClient(const PluginClient&) = delete;
  PluginClient& operator=(const PluginClient&) = delete;
  [[nodiscard]] std::string_view platform() const noexcept;
  [[nodiscard]] std::string_view version() const noexcept;
  [[nodiscard]] int device_count() const noexcept;
  [[nodiscard]] std::string device_name(int index) const;
  [[nodiscard]] std::vector<std::vector<float>> run(std::string_view mlir,
      const std::vector<std::vector<float>>& inputs,
      const std::vector<std::vector<std::int64_t>>& shapes);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace nexus::backend::pjrt::detail
