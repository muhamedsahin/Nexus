#pragma once

/**
 * @file nvme.hpp
 * @brief FAZ 11.5 ZeRO-Infinity G/C: anahtar basina dosya, asenkron okuma/yazma.
 *
 * Windows: CreateFile(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) + OVERLAPPED ReadFile/WriteFile,
 * tamamlanma GetOverlappedResult ile (IOCP portu kullanilmiyor; olay basina bekleme). Isletim sistemi
 * sayfa onbellegi devre disi: tampon ve uzunluk 4096 bayta hizali olmali (AlignedBuffer).
 * POSIX: ayri is parcaciginda pread/pwrite (io_uring / libaio bu surumde yok).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace nexus {
namespace dist {

/// 4096 bayta hizali, uzunlugu 4096'nin katina yuvarlanmis host tamponu (bounce tamponu).
class AlignedBuffer {
 public:
  static constexpr std::size_t kAlign = 4096;
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t bytes);
  ~AlignedBuffer();
  AlignedBuffer(AlignedBuffer&& o) noexcept;
  AlignedBuffer& operator=(AlignedBuffer&& o) noexcept;
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  [[nodiscard]] void* data() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }      // istenen
  [[nodiscard]] std::size_t capacity() const noexcept { return cap_; }     // hizali
  template <typename T>
  [[nodiscard]] T* as() const noexcept { return static_cast<T*>(ptr_); }
  void reset() noexcept;

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
  std::size_t cap_ = 0;
};

class NvmeStore {
 public:
  /// directory yoksa olusturulur; dosyalar yikicida silinir.
  explicit NvmeStore(std::string directory);
  ~NvmeStore();
  NvmeStore(const NvmeStore&) = delete;
  NvmeStore& operator=(const NvmeStore&) = delete;

  class Request {
   public:
    virtual ~Request() = default;
    virtual void wait() = 0;  // hata: DistError
  };
  using RequestPtr = std::unique_ptr<Request>;

  /// buffer.capacity() bayt yazar (hizali). buffer istek bitene kadar yasamali.
  [[nodiscard]] RequestPtr write(const std::string& key, const AlignedBuffer& buffer);
  /// Onceden yazilmis anahtari buffer'a okur (buffer.capacity() >= yazilan).
  [[nodiscard]] RequestPtr read(const std::string& key, AlignedBuffer& buffer);
  [[nodiscard]] const std::string& directory() const noexcept { return dir_; }
  [[nodiscard]] std::uint64_t bytes_written() const noexcept { return written_; }
  [[nodiscard]] std::uint64_t bytes_read() const noexcept { return read_; }

 private:
  [[nodiscard]] std::string path(const std::string& key) const;
  std::string dir_;
  std::uint64_t written_ = 0;
  std::uint64_t read_ = 0;
};

}  // namespace dist
}  // namespace nexus
