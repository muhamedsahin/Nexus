/**
 * @file json.cpp
 * @brief FAZ 17: dogrulamali kucuk JSON okuyucu.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/json.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace nexus {
namespace interop {

namespace {

class Parser {
 public:
  Parser(std::string_view s, const std::string& what) : s_(s), what_(what) {}
  Json run() {
    Json v = value(0);
    ws();
    if (i_ != s_.size()) fail("fazla veri");
    return v;
  }

 private:
  [[noreturn]] void fail(const char* why) const {
    throw InteropError(what_ + ": JSON " + why + " (bayt " + std::to_string(i_) + ")");
  }
  void ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
  }
  char peek() {
    ws();
    if (i_ >= s_.size()) fail("beklenmedik son");
    return s_[i_];
  }
  Json value(int depth) {
    if (depth > 256) fail("cok derin ic ice");
    const char c = peek();
    Json v;
    if (c == '{') {
      v.kind = Json::Kind::Object;
      ++i_;
      if (peek() == '}') {
        ++i_;
        return v;
      }
      for (;;) {
        if (peek() != '"') fail("anahtar bekleniyordu");
        std::string k = string();
        if (peek() != ':') fail("':' bekleniyordu");
        ++i_;
        v.obj.emplace_back(std::move(k), value(depth + 1));
        const char d = peek();
        ++i_;
        if (d == '}') break;
        if (d != ',') fail("',' ya da '}' bekleniyordu");
      }
    } else if (c == '[') {
      v.kind = Json::Kind::Array;
      ++i_;
      if (peek() == ']') {
        ++i_;
        return v;
      }
      for (;;) {
        v.arr.push_back(value(depth + 1));
        const char d = peek();
        ++i_;
        if (d == ']') break;
        if (d != ',') fail("',' ya da ']' bekleniyordu");
      }
    } else if (c == '"') {
      v.kind = Json::Kind::String;
      v.str = string();
    } else if (s_.substr(i_, 4) == "true") {
      v.kind = Json::Kind::Bool;
      v.b = true;
      i_ += 4;
    } else if (s_.substr(i_, 5) == "false") {
      v.kind = Json::Kind::Bool;
      i_ += 5;
    } else if (s_.substr(i_, 4) == "null") {
      i_ += 4;
    } else if (c == '-' || std::isdigit(static_cast<unsigned char>(c)) || s_.substr(i_, 3) == "NaN" ||
               s_.substr(i_, 8) == "Infinity" || s_.substr(i_, 9) == "-Infinity") {
      v.kind = Json::Kind::Number;
      if (s_.substr(i_, 3) == "NaN") {
        v.num = std::numeric_limits<double>::quiet_NaN();
        i_ += 3;
        return v;
      }
      if (s_.substr(i_, 8) == "Infinity") {
        v.num = std::numeric_limits<double>::infinity();
        i_ += 8;
        return v;
      }
      if (s_.substr(i_, 9) == "-Infinity") {
        v.num = -std::numeric_limits<double>::infinity();
        i_ += 9;
        return v;
      }
      const std::size_t start = i_;
      if (s_[i_] == '-') ++i_;
      bool frac = false;
      while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' || s_[i_] == 'e' ||
                                s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-')) {
        if (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E') frac = true;
        ++i_;
      }
      const std::string tok(s_.substr(start, i_ - start));
      char* end = nullptr;
      v.num = std::strtod(tok.c_str(), &end);
      if (end != tok.c_str() + tok.size()) fail("gecersiz sayi");
      if (!frac) {
        errno = 0;
        const long long iv = std::strtoll(tok.c_str(), &end, 10);
        if (errno == 0 && end == tok.c_str() + tok.size()) {
          v.integer = iv;
          v.is_integer = true;
        }
      }
    } else {
      fail("deger bekleniyordu");
    }
    return v;
  }
  void put_utf8(std::string& o, unsigned cp) {
    if (cp < 0x80) {
      o += static_cast<char>(cp);
    } else if (cp < 0x800) {
      o += static_cast<char>(0xC0 | (cp >> 6));
      o += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      o += static_cast<char>(0xE0 | (cp >> 12));
      o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      o += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      o += static_cast<char>(0xF0 | (cp >> 18));
      o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      o += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }
  unsigned hex4() {
    if (i_ + 4 > s_.size()) fail("kisa \\u");
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      const char h = s_[i_++];
      v <<= 4;
      if (h >= '0' && h <= '9') v |= static_cast<unsigned>(h - '0');
      else if (h >= 'a' && h <= 'f') v |= static_cast<unsigned>(h - 'a' + 10);
      else if (h >= 'A' && h <= 'F') v |= static_cast<unsigned>(h - 'A' + 10);
      else fail("gecersiz \\u");
    }
    return v;
  }
  std::string string() {
    ++i_;
    std::string o;
    for (;;) {
      if (i_ >= s_.size()) fail("kapanmamis dize");
      const char c = s_[i_++];
      if (c == '"') break;
      if (c != '\\') {
        o += c;
        continue;
      }
      if (i_ >= s_.size()) fail("kacis");
      const char e = s_[i_++];
      switch (e) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
            i_ += 2;
            const unsigned lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          put_utf8(o, cp);
          break;
        }
        default: fail("bilinmeyen kacis");
      }
    }
    return o;
  }
  std::string_view s_;
  const std::string& what_;
  std::size_t i_ = 0;
};

}  // namespace

bool Json::has(const std::string& k) const { return get(k) != nullptr; }

const Json* Json::get(const std::string& k) const {
  if (kind != Kind::Object) return nullptr;
  for (const auto& kv : obj)
    if (kv.first == k) return &kv.second;
  return nullptr;
}

const Json& Json::at(const std::string& k) const {
  const Json* v = get(k);
  if (v == nullptr) throw InteropError("JSON: eksik alan '" + k + "'");
  return *v;
}

std::int64_t Json::as_int() const {
  if (kind != Kind::Number) throw InteropError("JSON: sayi bekleniyordu");
  if (is_integer) return integer;
  if (std::floor(num) != num) throw InteropError("JSON: tamsayi bekleniyordu");
  return static_cast<std::int64_t>(num);
}

double Json::as_double() const {
  if (kind != Kind::Number) throw InteropError("JSON: sayi bekleniyordu");
  return num;
}

const std::string& Json::as_string() const {
  if (kind != Kind::String) throw InteropError("JSON: dize bekleniyordu");
  return str;
}

bool Json::as_bool() const {
  if (kind != Kind::Bool) throw InteropError("JSON: bool bekleniyordu");
  return b;
}

Json parse_json(std::string_view text, const std::string& what) { return Parser(text, what).run(); }

std::string json_quote(std::string_view s) {
  std::string o = "\"";
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned char>(c));
          o += b;
        } else {
          o += c;
        }
    }
  }
  return o + "\"";
}

}  // namespace interop
}  // namespace nexus
