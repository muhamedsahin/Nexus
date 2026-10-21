#pragma once

/**
 * @file error.hpp
 * @brief Birlesik tensör koprusunun hata tipi.
 *
 * Kopru, boyutu uymayan bir gorunum ya da desteklenmeyen bir cihaz ayirma
 * isteginde sessizce bos bellek dondurmez. Cagiran `nexus::TensorError`
 * yakalayarak nedeni okur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <stdexcept>
#include <string>

namespace nexus {

class TensorError : public std::runtime_error {
 public:
  explicit TensorError(const std::string& message) : std::runtime_error(message) {}
  explicit TensorError(const char* message) : std::runtime_error(message) {}
};

}  // namespace nexus
