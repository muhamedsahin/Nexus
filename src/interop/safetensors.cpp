/**
 * @file safetensors.cpp
 * @brief FAZ 17.1: safetensors okuma (mmap) / yazma, dogrulama.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/safetensors.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nexus {
namespace interop {

std::size_t safetensors_dtype_size(const std::string& d) {
  if (d == "F64" || d == "I64" || d == "U64") return 8;
  if (d == "F32" || d == "I32" || d == "U32") return 4;
  if (d == "F16" || d == "BF16" || d == "I16" || d == "U16") return 2;
  if (d == "I8" || d == "U8" || d == "BOOL" || d == "F8_E4M3" || d == "F8_E5M2") return 1;
  throw InteropError("safetensors: bilinmeyen dtype '" + d + "'");
}

std::int64_t StTensor::numel() const {
  std::int64_t n = 1;
  for (auto s : shape) n *= s;
  return n;
}

struct SafeTensors::Impl {
  std::string path;
  std::vector<StTensor> tensors;
  std::map<std::string, std::string> meta;
  const std::uint8_t* base = nullptr;  // dosya basi
  std::uint64_t size = 0;
  std::uint64_t data_offset = 0;
  std::vector<std::uint8_t> owned;
  bool mapped = false;
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
#else
  int fd = -1;
#endif
  ~Impl() {
#if defined(_WIN32)
    if (mapped && base) UnmapViewOfFile(base);
    if (mapping) CloseHandle(mapping);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
    if (mapped && base) munmap(const_cast<std::uint8_t*>(base), size);
    if (fd >= 0) close(fd);
#endif
  }
};

SafeTensors::SafeTensors() : impl_(std::make_unique<Impl>()) {}
SafeTensors::SafeTensors(SafeTensors&&) noexcept = default;
SafeTensors& SafeTensors::operator=(SafeTensors&&) noexcept = default;
SafeTensors::~SafeTensors() = default;

SafeTensors SafeTensors::open(const std::string& path, bool mmap) {
  SafeTensors st;
  Impl& m = *st.impl_;
  m.path = path;
  if (mmap) {
#if defined(_WIN32)
    const int wl = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wp(static_cast<std::size_t>(wl), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wp.data(), wl);
    m.file = CreateFileW(wp.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (m.file == INVALID_HANDLE_VALUE) throw InteropError("safetensors: acilamadi " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(m.file, &sz);
    m.size = static_cast<std::uint64_t>(sz.QuadPart);
    if (m.size < 8) throw InteropError("safetensors: " + path + " cok kisa");
    m.mapping = CreateFileMappingW(m.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m.mapping) throw InteropError("safetensors: eslenemedi " + path);
    m.base = static_cast<const std::uint8_t*>(MapViewOfFile(m.mapping, FILE_MAP_READ, 0, 0, 0));
    if (!m.base) throw InteropError("safetensors: goruntu acilamadi " + path);
#else
    m.fd = ::open(path.c_str(), O_RDONLY);
    if (m.fd < 0) throw InteropError("safetensors: acilamadi " + path);
    struct stat sb;
    fstat(m.fd, &sb);
    m.size = static_cast<std::uint64_t>(sb.st_size);
    if (m.size < 8) throw InteropError("safetensors: " + path + " cok kisa");
    void* p = ::mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, m.fd, 0);
    if (p == MAP_FAILED) throw InteropError("safetensors: eslenemedi " + path);
    m.base = static_cast<const std::uint8_t*>(p);
#endif
    m.mapped = true;
  } else {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw InteropError("safetensors: acilamadi " + path);
    in.seekg(0, std::ios::end);
    m.size = static_cast<std::uint64_t>(in.tellg());
    in.seekg(0);
    m.owned.resize(m.size);
    in.read(reinterpret_cast<char*>(m.owned.data()), static_cast<std::streamsize>(m.size));
    if (!in || m.size < 8) throw InteropError("safetensors: okunamadi " + path);
    m.base = m.owned.data();
  }
  std::uint64_t hlen = 0;
  std::memcpy(&hlen, m.base, 8);
  if (hlen > m.size - 8 || hlen > (std::uint64_t{100} << 20)) {
    throw InteropError("safetensors: " + path + ": baslik boyu gecersiz (" + std::to_string(hlen) + ")");
  }
  m.data_offset = 8 + hlen;
  const std::uint64_t data_size = m.size - m.data_offset;
  const Json h = parse_json(std::string_view(reinterpret_cast<const char*>(m.base + 8), static_cast<std::size_t>(hlen)), path);
  if (h.kind != Json::Kind::Object) throw InteropError("safetensors: baslik nesne degil");
  std::set<std::string> names;
  for (const auto& [k, v] : h.obj) {
    if (k == "__metadata__") {
      if (v.kind != Json::Kind::Object) throw InteropError("safetensors: __metadata__ nesne degil");
      for (const auto& [mk, mv] : v.obj) m.meta[mk] = mv.as_string();
      continue;
    }
    if (!names.insert(k).second) throw InteropError("safetensors: tekrar eden tensor adi " + k);
    StTensor t;
    t.name = k;
    t.dtype = v.at("dtype").as_string();
    t.elem = safetensors_dtype_size(t.dtype);
    std::uint64_t n = 1;
    for (const Json& s : v.at("shape").arr) {
      const std::int64_t d = s.as_int();
      if (d < 0) throw InteropError("safetensors: " + k + ": negatif boyut");
      t.shape.push_back(d);
      if (d != 0 && n > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(d))
        throw InteropError("safetensors: " + k + ": sekil tasmasi");
      n *= static_cast<std::uint64_t>(d);
    }
    const auto& off = v.at("data_offsets").arr;
    if (off.size() != 2) throw InteropError("safetensors: " + k + ": data_offsets [b, e] olmali");
    const std::int64_t b = off[0].as_int(), e = off[1].as_int();
    if (b < 0 || e < b || static_cast<std::uint64_t>(e) > data_size)
      throw InteropError("safetensors: " + k + ": aralik veri bolgesi disinda");
    t.begin = static_cast<std::uint64_t>(b);
    t.end = static_cast<std::uint64_t>(e);
    if (n > std::numeric_limits<std::uint64_t>::max() / t.elem || n * t.elem != t.end - t.begin)
      throw InteropError("safetensors: " + k + ": sekil x dtype boyutu araliga esit degil");
    m.tensors.push_back(std::move(t));
  }
  std::vector<const StTensor*> order;
  for (const auto& t : m.tensors) order.push_back(&t);
  std::sort(order.begin(), order.end(), [](const StTensor* a, const StTensor* b) { return a->begin < b->begin; });
  std::uint64_t at = 0;
  for (const StTensor* t : order) {
    if (t->begin != at) throw InteropError("safetensors: " + t->name + ": aralik bosluklu ya da ust uste");
    at = t->end;
  }
  if (at != data_size) throw InteropError("safetensors: veri bolgesinin sonunda indekslenmemis bayt");
  return st;
}

const std::vector<StTensor>& SafeTensors::tensors() const noexcept { return impl_->tensors; }
const std::map<std::string, std::string>& SafeTensors::metadata() const noexcept { return impl_->meta; }
bool SafeTensors::mapped() const noexcept { return impl_->mapped; }
std::uint64_t SafeTensors::file_bytes() const noexcept { return impl_->size; }

const StTensor* SafeTensors::find(const std::string& name) const noexcept {
  for (const auto& t : impl_->tensors)
    if (t.name == name) return &t;
  return nullptr;
}

const StTensor& SafeTensors::at(const std::string& name) const {
  const StTensor* t = find(name);
  if (!t) throw InteropError("safetensors: tensor yok '" + name + "' (" + impl_->path + ")");
  return *t;
}

std::span<const std::uint8_t> SafeTensors::bytes(const StTensor& t) const {
  return {impl_->base + impl_->data_offset + t.begin, static_cast<std::size_t>(t.end - t.begin)};
}

namespace {
float half_to_float(std::uint16_t h) {
  const std::uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, f = h & 0x3FF;
  std::uint32_t o;
  if (e == 0) {
    if (f == 0) {
      o = s;
    } else {  // alt normal
      int ee = -1;
      std::uint32_t ff = f;
      do {
        ++ee;
        ff <<= 1;
      } while ((ff & 0x400) == 0);
      o = s | ((127 - 15 - ee) << 23) | ((ff & 0x3FF) << 13);
    }
  } else if (e == 31) {
    o = s | 0x7F800000u | (f << 13);
  } else {
    o = s | ((e + 127 - 15) << 23) | (f << 13);
  }
  float r;
  std::memcpy(&r, &o, 4);
  return r;
}
}  // namespace

std::vector<float> SafeTensors::to_f32(const StTensor& t) const {
  const auto b = bytes(t);
  const std::size_t n = static_cast<std::size_t>(t.numel());
  std::vector<float> out(n);
  if (t.dtype == "F32") {
    std::memcpy(out.data(), b.data(), n * 4);
  } else if (t.dtype == "F16") {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t h;
      std::memcpy(&h, b.data() + 2 * i, 2);
      out[i] = half_to_float(h);
    }
  } else if (t.dtype == "BF16") {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t h;
      std::memcpy(&h, b.data() + 2 * i, 2);
      const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
      std::memcpy(&out[i], &u, 4);
    }
  } else if (t.dtype == "F64") {
    for (std::size_t i = 0; i < n; ++i) {
      double d;
      std::memcpy(&d, b.data() + 8 * i, 8);
      out[i] = static_cast<float>(d);
    }
  } else {
    throw InteropError("safetensors: " + t.name + ": " + t.dtype + " kayan noktaya cevrilemez");
  }
  return out;
}

void write_safetensors(const std::string& path, const std::vector<StWrite>& tensors,
                       const std::map<std::string, std::string>& metadata) {
  std::string h = "{";
  bool first = true;
  if (!metadata.empty()) {
    h += "\"__metadata__\":{";
    bool f2 = true;
    for (const auto& [k, v] : metadata) {
      h += (f2 ? "" : ",") + json_quote(k) + ":" + json_quote(v);
      f2 = false;
    }
    h += "}";
    first = false;
  }
  std::set<std::string> names;
  std::uint64_t at = 0;
  for (const StWrite& t : tensors) {
    if (!names.insert(t.name).second) throw InteropError("safetensors yaz: tekrar eden ad " + t.name);
    std::uint64_t n = safetensors_dtype_size(t.dtype);
    for (auto s : t.shape) n *= static_cast<std::uint64_t>(s);
    if (n != t.bytes) throw InteropError("safetensors yaz: " + t.name + ": sekil x dtype != bayt");
    h += std::string(first ? "" : ",") + json_quote(t.name) + ":{\"dtype\":" + json_quote(t.dtype) + ",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i) h += (i ? "," : "") + std::to_string(t.shape[i]);
    h += "],\"data_offsets\":[" + std::to_string(at) + "," + std::to_string(at + t.bytes) + "]}";
    at += t.bytes;
    first = false;
  }
  h += "}";
  while (h.size() % 8 != 0) h += ' ';
  const std::string part = path + ".part";
  {
    std::ofstream out(part, std::ios::binary | std::ios::trunc);
    if (!out) throw InteropError("safetensors yaz: acilamadi " + part);
    const std::uint64_t hl = h.size();
    out.write(reinterpret_cast<const char*>(&hl), 8);
    out.write(h.data(), static_cast<std::streamsize>(h.size()));
    for (const StWrite& t : tensors) out.write(static_cast<const char*>(t.data), static_cast<std::streamsize>(t.bytes));
    out.flush();
    if (!out) throw InteropError("safetensors yaz: yazma hatasi " + part);
  }
  std::remove(path.c_str());
  if (std::rename(part.c_str(), path.c_str()) != 0) throw InteropError("safetensors yaz: yeniden adlandirilamadi " + path);
}

}  // namespace interop
}  // namespace nexus
