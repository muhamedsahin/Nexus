#pragma once

/**
 * @file safetensors.hpp
 * @brief FAZ 17.1: safetensors okuma (mmap, sifir kopya) ve yazma.
 *
 * Bicim: u64 baslik boyu (LE) | JSON baslik {"ad": {"dtype", "shape", "data_offsets": [b, e]}, "__metadata__": {...}}
 * | veri. Okumada dogrulanan her sey (bozuk/kotu niyetli dosya acik hata, tasma yok):
 *   - baslik boyu <= dosya - 8 ve <= 100 MB; JSON nesne;
 *   - dtype bilinen (F64 F32 F16 BF16 I64 I32 I16 I8 U8 BOOL F8_E4M3 F8_E5M2);
 *   - sekil boyutlari >= 0, eleman * boyut carpimi tasmasiz ve == e - b;
 *   - araliklar veri bolgesi icinde, siralaninca bitisik ve bosluksuz (0'dan veri sonuna kadar), ust uste yok.
 * mmap = true: dosya bellege eslenir (Windows MapViewOfFile / POSIX mmap); bytes() kopyasiz goruntudur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/json.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nexus {
namespace interop {

struct StTensor {
  std::string name;
  std::string dtype;
  std::size_t elem = 0;  // bayt
  std::vector<std::int64_t> shape;
  std::uint64_t begin = 0, end = 0;  // veri bolgesine gore
  [[nodiscard]] std::int64_t numel() const;
};

class SafeTensors {
 public:
  [[nodiscard]] static SafeTensors open(const std::string& path, bool mmap = true);
  SafeTensors(SafeTensors&&) noexcept;
  SafeTensors& operator=(SafeTensors&&) noexcept;
  ~SafeTensors();

  [[nodiscard]] const std::vector<StTensor>& tensors() const noexcept;
  [[nodiscard]] const StTensor* find(const std::string& name) const noexcept;
  [[nodiscard]] const StTensor& at(const std::string& name) const;
  [[nodiscard]] std::span<const std::uint8_t> bytes(const StTensor& t) const;
  /// F32 / F16 / BF16 / F64 -> FP32 kopya.
  [[nodiscard]] std::vector<float> to_f32(const StTensor& t) const;
  [[nodiscard]] const std::map<std::string, std::string>& metadata() const noexcept;
  [[nodiscard]] bool mapped() const noexcept;
  [[nodiscard]] std::uint64_t file_bytes() const noexcept;

  struct Impl;

 private:
  SafeTensors();
  std::unique_ptr<Impl> impl_;
};

struct StWrite {
  std::string name;
  std::string dtype;  // "F32" | "F16" | "BF16" | ...
  std::vector<std::int64_t> shape;
  const void* data = nullptr;
  std::size_t bytes = 0;
};
/// Baslik 8'e hizalanir (bosluk dolgusu, HF ile ayni). Ad tekrari / boyut uyusmazligi acik hata.
void write_safetensors(const std::string& path, const std::vector<StWrite>& tensors,
                       const std::map<std::string, std::string>& metadata = {});

[[nodiscard]] std::size_t safetensors_dtype_size(const std::string& dtype);

}  // namespace interop
}  // namespace nexus
