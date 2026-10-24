#pragma once

/**
 * @file json_mini.hpp
 * @brief FAZ 0 araclarinin paylastigi kucuk JSON okuyucu/erisim katmani.
 *
 * Yalnizca okuma: nesne, dizi, sayi, dize, bool, null. Yapilandirma
 * dosyalari kucuk ve el yazimidir; karmasik bir cozumleyici gerekmez.
 * Soz dizimi hatasinda JsonError firlatir (sessiz varsayilan degil).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexus {
namespace jsonmini {

class JsonError : public std::runtime_error {
 public:
  explicit JsonError(const std::string& message) : std::runtime_error(message) {}
};

enum class Kind { Null, Bool, Number, String, Array, Object };

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value {
 public:
  Value() = default;

  [[nodiscard]] Kind kind() const { return kind_; }
  [[nodiscard]] bool is_null() const { return kind_ == Kind::Null; }
  [[nodiscard]] bool as_bool() const { require(Kind::Bool); return bool_; }
  [[nodiscard]] double as_number() const { require(Kind::Number); return number_; }
  [[nodiscard]] const std::string& as_string() const { require(Kind::String); return string_; }
  [[nodiscard]] const Array& as_array() const { require(Kind::Array); return *array_; }
  [[nodiscard]] const Object& as_object() const { require(Kind::Object); return *object_; }

  /// Nesnede anahtar var mi? Nesne degilse false.
  [[nodiscard]] bool has(const std::string& key) const;
  /// Anahtar yoksa JsonError (sessiz varsayilan yok).
  [[nodiscard]] const Value& at(const std::string& key) const;

  static Value make_bool(bool value);
  static Value make_number(double value);
  static Value make_string(std::string value);
  static Value make_array(Array value);
  static Value make_object(Object value);

 private:
  void require(Kind expected) const;

  Kind kind_ = Kind::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::shared_ptr<Array> array_;
  std::shared_ptr<Object> object_;
};

/// JSON metnini cozumler; hata durumunda JsonError.
Value parse(const std::string& text);

/// Dosyadan okur ve cozumler; acilamazsa JsonError.
Value parse_file(const std::string& path);

// --- Erisim kisayollari: yapilandirma alanlarini dogrulamali okur -------------
[[nodiscard]] double get_number(const Value& object, const std::string& key, double fallback);
[[nodiscard]] std::int64_t get_int(const Value& object, const std::string& key, std::int64_t fallback);
[[nodiscard]] bool get_bool(const Value& object, const std::string& key, bool fallback);
[[nodiscard]] std::string get_string(const Value& object, const std::string& key, const std::string& fallback);

/// Zorunlu alan: yoksa JsonError (ayar sessizce uydurulmaz).
[[nodiscard]] double require_number(const Value& object, const std::string& key);
[[nodiscard]] std::int64_t require_int(const Value& object, const std::string& key);

}  // namespace jsonmini
}  // namespace nexus
