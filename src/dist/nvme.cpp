/**
 * @file nvme.cpp
 * @brief FAZ 11.5: hizali tampon ve asenkron dosya G/C (Windows OVERLAPPED + NO_BUFFERING; POSIX pread/pwrite).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/nvme.hpp"

#include "nexus/dist/process_group.hpp"

#include <cstdlib>
#include <filesystem>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace nexus {
namespace dist {

AlignedBuffer::AlignedBuffer(std::size_t bytes) : bytes_(bytes) {
  cap_ = (bytes + kAlign - 1) / kAlign * kAlign;
  if (cap_ == 0) cap_ = kAlign;
#if defined(_WIN32)
  ptr_ = _aligned_malloc(cap_, kAlign);
#else
  ptr_ = std::aligned_alloc(kAlign, cap_);
#endif
  if (ptr_ == nullptr) throw DistError("Nexus NVMe: hizali tampon ayrilamadi");
}

AlignedBuffer::~AlignedBuffer() { reset(); }

void AlignedBuffer::reset() noexcept {
  if (ptr_ != nullptr) {
#if defined(_WIN32)
    _aligned_free(ptr_);
#else
    std::free(ptr_);
#endif
  }
  ptr_ = nullptr;
  bytes_ = cap_ = 0;
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_), cap_(o.cap_) {
  o.ptr_ = nullptr;
  o.bytes_ = o.cap_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& o) noexcept {
  if (this != &o) {
    reset();
    ptr_ = o.ptr_;
    bytes_ = o.bytes_;
    cap_ = o.cap_;
    o.ptr_ = nullptr;
    o.bytes_ = o.cap_ = 0;
  }
  return *this;
}

NvmeStore::NvmeStore(std::string directory) : dir_(std::move(directory)) {
  std::filesystem::create_directories(dir_);
}

NvmeStore::~NvmeStore() {
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(dir_, ec)) {
    if (e.path().extension() == ".nxswap") std::filesystem::remove(e.path(), ec);
  }
}

std::string NvmeStore::path(const std::string& key) const {
  return (std::filesystem::path(dir_) / (key + ".nxswap")).string();
}

namespace {
#if defined(_WIN32)
class WinRequest final : public NvmeStore::Request {
 public:
  WinRequest(const std::string& path, void* data, std::size_t bytes, bool write) : write_(write) {
    const DWORD access = write ? GENERIC_WRITE : GENERIC_READ;
    const DWORD disp = write ? CREATE_ALWAYS : OPEN_EXISTING;
    file_ = CreateFileA(path.c_str(), access, 0, nullptr, disp, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
      throw DistError("Nexus NVMe: dosya acilamadi " + path + " (hata " + std::to_string(GetLastError()) + ")");
    }
    ov_.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    const BOOL ok = write ? WriteFile(file_, data, static_cast<DWORD>(bytes), nullptr, &ov_)
                          : ReadFile(file_, data, static_cast<DWORD>(bytes), nullptr, &ov_);
    if (!ok && GetLastError() != ERROR_IO_PENDING) {
      const DWORD err = GetLastError();
      cleanup();
      throw DistError("Nexus NVMe: G/C baslatilamadi (hata " + std::to_string(err) + ")");
    }
    expected_ = bytes;
  }
  ~WinRequest() override {
    try {
      wait();
    } catch (...) {
    }
  }
  void wait() override {
    if (file_ == INVALID_HANDLE_VALUE) return;
    DWORD done = 0;
    const BOOL ok = GetOverlappedResult(file_, &ov_, &done, TRUE);
    const DWORD err = GetLastError();
    cleanup();
    if (!ok) throw DistError("Nexus NVMe: G/C basarisiz (hata " + std::to_string(err) + ")");
    if (done != expected_) throw DistError("Nexus NVMe: eksik G/C (" + std::to_string(done) + " bayt)");
  }

 private:
  void cleanup() {
    if (ov_.hEvent != nullptr) CloseHandle(ov_.hEvent);
    ov_.hEvent = nullptr;
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    file_ = INVALID_HANDLE_VALUE;
  }
  HANDLE file_ = INVALID_HANDLE_VALUE;
  OVERLAPPED ov_{};
  std::size_t expected_ = 0;
  bool write_;
};
#else
class PosixRequest final : public NvmeStore::Request {
 public:
  PosixRequest(const std::string& path, void* data, std::size_t bytes, bool write) {
    worker_ = std::thread([this, path, data, bytes, write] {
      const int fd = write ? ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600) : ::open(path.c_str(), O_RDONLY);
      if (fd < 0) {
        error_ = "acilamadi " + path;
        return;
      }
      std::size_t pos = 0;
      while (pos < bytes) {
        const ssize_t n = write ? ::pwrite(fd, static_cast<char*>(data) + pos, bytes - pos, static_cast<off_t>(pos))
                                : ::pread(fd, static_cast<char*>(data) + pos, bytes - pos, static_cast<off_t>(pos));
        if (n <= 0) {
          error_ = "G/C basarisiz";
          break;
        }
        pos += static_cast<std::size_t>(n);
      }
      ::close(fd);
    });
  }
  ~PosixRequest() override {
    if (worker_.joinable()) worker_.join();
  }
  void wait() override {
    if (worker_.joinable()) worker_.join();
    if (!error_.empty()) throw DistError("Nexus NVMe: " + error_);
  }

 private:
  std::thread worker_;
  std::string error_;
};
#endif
}  // namespace

NvmeStore::RequestPtr NvmeStore::write(const std::string& key, const AlignedBuffer& buffer) {
  written_ += buffer.capacity();
#if defined(_WIN32)
  return std::make_unique<WinRequest>(path(key), buffer.data(), buffer.capacity(), true);
#else
  return std::make_unique<PosixRequest>(path(key), buffer.data(), buffer.capacity(), true);
#endif
}

NvmeStore::RequestPtr NvmeStore::read(const std::string& key, AlignedBuffer& buffer) {
  read_ += buffer.capacity();
#if defined(_WIN32)
  return std::make_unique<WinRequest>(path(key), buffer.data(), buffer.capacity(), false);
#else
  return std::make_unique<PosixRequest>(path(key), buffer.data(), buffer.capacity(), false);
#endif
}

}  // namespace dist
}  // namespace nexus
