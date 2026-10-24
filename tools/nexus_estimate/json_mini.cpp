/**
 * @file json_mini.cpp
 * @brief Kucuk JSON cozumleyicinin gerceklemesi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "json_mini.hpp"

#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

namespace nexus {
namespace jsonmini {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}

  Value parse_document() {
    skip_ws();
    Value value = parse_value();
    skip_ws();
    if (position_ != text_.size()) {
      fail("belge sonunda beklenmeyen karakter");
    }
    return value;
  }

 private:
  [[noreturn]] void fail(const std::string& message) const {
    std::ostringstream out;
    out << "nexus json: konum " << position_ << ": " << message;
    throw JsonError(out.str());
  }

  char peek() const {
    if (position_ >= text_.size()) {
      fail("girdi beklenenden erken bitti");
    }
    return text_[position_];
  }

  char take() {
    const char result = peek();
    ++position_;
    return result;
  }

  void expect(char character) {
    if (take() != character) {
      --position_;
      fail(std::string("'") + character + "' bekleniyordu");
    }
  }

  void skip_ws() {
    while (position_ < text_.size()) {
      const char character = text_[position_];
      if (character == ' ' || character == '\t' || character == '\n' || character == '\r') {
        ++position_;
      } else if (character == '/' && position_ + 1 < text_.size() && text_[position_ + 1] == '/') {
        while (position_ < text_.size() && text_[position_] != '\n') {
          ++position_;
        }
      } else {
        break;
      }
    }
  }

  Value parse_value() {
    skip_ws();
    const char character = peek();
    if (character == '{') {
      return parse_object();
    }
    if (character == '[') {
      return parse_array();
    }
    if (character == '"') {
      return Value::make_string(parse_string());
    }
    if (character == 't') {
      take_literal("true");
      return Value::make_bool(true);
    }
    if (character == 'f') {
      take_literal("false");
      return Value::make_bool(false);
    }
    if (character == 'n') {
      take_literal("null");
      return Value();
    }
    return parse_number();
  }

  void take_literal(const char* literal) {
    const std::size_t length = std::strlen(literal);
    if (text_.compare(position_, length, literal) != 0) {
      fail(std::string("'") + literal + "' bekleniyordu");
    }
    position_ += length;
  }

  Value parse_object() {
    expect('{');
    Object object;
    skip_ws();
    if (peek() == '}') {
      ++position_;
      return Value::make_object(std::move(object));
    }
    while (true) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      Value value = parse_value();
      object.emplace(std::move(key), std::move(value));
      skip_ws();
      const char next = take();
      if (next == '}') {
        break;
      }
      if (next != ',') {
        --position_;
        fail("nesnede ',' veya '}' bekleniyordu");
      }
    }
    return Value::make_object(std::move(object));
  }

  Value parse_array() {
    expect('[');
    Array array;
    skip_ws();
    if (peek() == ']') {
      ++position_;
      return Value::make_array(std::move(array));
    }
    while (true) {
      array.push_back(parse_value());
      skip_ws();
      const char next = take();
      if (next == ']') {
        break;
      }
      if (next != ',') {
        --position_;
        fail("dizide ',' veya ']' bekleniyordu");
      }
    }
    return Value::make_array(std::move(array));
  }

  std::string parse_string() {
    expect('"');
    std::string result;
    while (true) {
      const char character = take();
      if (character == '"') {
        break;
      }
      if (character != '\\') {
        result.push_back(character);
        continue;
      }
      const char escape = take();
      switch (escape) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u': {
          unsigned code = 0;
          for (int digit = 0; digit < 4; ++digit) {
            const char hex = take();
            code <<= 4;
            if (hex >= '0' && hex <= '9') {
              code |= static_cast<unsigned>(hex - '0');
            } else if (hex >= 'a' && hex <= 'f') {
              code |= static_cast<unsigned>(hex - 'a' + 10);
            } else if (hex >= 'A' && hex <= 'F') {
              code |= static_cast<unsigned>(hex - 'A' + 10);
            } else {
              fail("gecersiz \\u kacisi");
            }
          }
          // Basit UTF-8 kodlamasi (vekil ciftleri desteklenmez; yapilandirma ASCII).
          if (code < 0x80) {
            result.push_back(static_cast<char>(code));
          } else if (code < 0x800) {
            result.push_back(static_cast<char>(0xC0 | (code >> 6)));
            result.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          } else {
            result.push_back(static_cast<char>(0xE0 | (code >> 12)));
            result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (code & 0x3F)));
          }
          break;
        }
        default:
          fail("gecersiz kacis karakteri");
      }
    }
    return result;
  }

  Value parse_number() {
    const std::size_t start = position_;
    if (peek() == '-') {
      ++position_;
    }
    while (position_ < text_.size() && (std::isdigit(static_cast<unsigned char>(text_[position_])) != 0 ||
                                        text_[position_] == '.' || text_[position_] == 'e' ||
                                        text_[position_] == 'E' || text_[position_] == '+' || text_[position_] == '-')) {
      ++position_;
    }
    if (position_ == start) {
      fail("sayi bekleniyordu");
    }
    const std::string token = text_.substr(start, position_ - start);
    try {
      return Value::make_number(std::stod(token));
    } catch (const std::exception&) {
      fail("gecersiz sayi: " + token);
    }
  }

  const std::string& text_;
  std::size_t position_ = 0;
};

}  // namespace

void Value::require(Kind expected) const {
  if (kind_ != expected) {
    throw JsonError("nexus json: deger beklenen turde degil");
  }
}

bool Value::has(const std::string& key) const {
  if (kind_ != Kind::Object) {
    return false;
  }
  return object_->find(key) != object_->end();
}

const Value& Value::at(const std::string& key) const {
  require(Kind::Object);
  const auto found = object_->find(key);
  if (found == object_->end()) {
    throw JsonError("nexus json: eksik alan: " + key);
  }
  return found->second;
}

Value Value::make_bool(bool value) {
  Value result;
  result.kind_ = Kind::Bool;
  result.bool_ = value;
  return result;
}

Value Value::make_number(double value) {
  Value result;
  result.kind_ = Kind::Number;
  result.number_ = value;
  return result;
}

Value Value::make_string(std::string value) {
  Value result;
  result.kind_ = Kind::String;
  result.string_ = std::move(value);
  return result;
}

Value Value::make_array(Array value) {
  Value result;
  result.kind_ = Kind::Array;
  result.array_ = std::make_shared<Array>(std::move(value));
  return result;
}

Value Value::make_object(Object value) {
  Value result;
  result.kind_ = Kind::Object;
  result.object_ = std::make_shared<Object>(std::move(value));
  return result;
}

Value parse(const std::string& text) {
  Parser parser(text);
  return parser.parse_document();
}

Value parse_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw JsonError("nexus json: dosya acilamadi: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return parse(buffer.str());
}

double get_number(const Value& object, const std::string& key, double fallback) {
  if (!object.has(key)) {
    return fallback;
  }
  return object.at(key).as_number();
}

std::int64_t get_int(const Value& object, const std::string& key, std::int64_t fallback) {
  if (!object.has(key)) {
    return fallback;
  }
  return static_cast<std::int64_t>(std::llround(object.at(key).as_number()));
}

bool get_bool(const Value& object, const std::string& key, bool fallback) {
  if (!object.has(key)) {
    return fallback;
  }
  return object.at(key).as_bool();
}

std::string get_string(const Value& object, const std::string& key, const std::string& fallback) {
  if (!object.has(key)) {
    return fallback;
  }
  return object.at(key).as_string();
}

double require_number(const Value& object, const std::string& key) { return object.at(key).as_number(); }

std::int64_t require_int(const Value& object, const std::string& key) {
  return static_cast<std::int64_t>(std::llround(object.at(key).as_number()));
}

}  // namespace jsonmini
}  // namespace nexus
