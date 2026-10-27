#pragma once

/**
 * @file json.hpp
 * @brief FAZ 17: kucuk JSON okuyucu/yazici (safetensors basligi, HF config.json, tokenizer.json). Dogrulamali:
 *        bozuk girdi acik hata (konum ile). Sayilar double + tamsayi olarak (int64 tasmasi kontrollu) tutulur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace nexus {
namespace interop {

class InteropError : public std::runtime_error {
 public:
  explicit InteropError(const std::string& what) : std::runtime_error("Nexus interop: " + what) {}
};

struct Json {
  enum class Kind : std::uint8_t { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool b = false;
  double num = 0.0;
  std::int64_t integer = 0;
  bool is_integer = false;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;  // sira korunur

  [[nodiscard]] bool has(const std::string& k) const;
  [[nodiscard]] const Json& at(const std::string& k) const;  // yoksa acik hata
  [[nodiscard]] const Json* get(const std::string& k) const;  // yoksa nullptr
  [[nodiscard]] std::int64_t as_int() const;
  [[nodiscard]] double as_double() const;
  [[nodiscard]] const std::string& as_string() const;
  [[nodiscard]] bool as_bool() const;
  [[nodiscard]] bool is_null() const noexcept { return kind == Kind::Null; }
};

[[nodiscard]] Json parse_json(std::string_view text, const std::string& what = "json");
[[nodiscard]] std::string json_quote(std::string_view s);

}  // namespace interop
}  // namespace nexus
