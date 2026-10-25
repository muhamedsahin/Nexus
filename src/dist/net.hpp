#pragma once

/**
 * @file net.hpp
 * @brief FAZ 10 ic: tasinabilir TCP soket sarmalayicisi (Winsock / POSIX) ve tasima arayuzu.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/process_group.hpp"

#include <chrono>
#include <cstdint>
#include <string>

namespace nexus {
namespace dist {
namespace net {

using Handle = std::intptr_t;
constexpr Handle kInvalid = -1;

void startup();  // WSAStartup (Windows), surec basina bir kez
void close(Handle h) noexcept;
/// port 0: isletim sistemi secer. Donus: dinleyen soket; port gercek degerle yazilir.
[[nodiscard]] Handle listen_on(const std::string& host, int& port);
/// Zaman asimina kadar yeniden dener (sunucu henuz baslamamis olabilir).
[[nodiscard]] Handle connect_to(const std::string& host, int port, std::chrono::milliseconds timeout);
/// Zaman asimli accept (DistError).
[[nodiscard]] Handle accept_from(Handle listener, std::chrono::milliseconds timeout);
void send_all(Handle h, const void* data, std::size_t bytes);
/// Tam bytes okur; zaman asimi ya da kopan baglanti DistError (what: neden).
void recv_all(Handle h, void* data, std::size_t bytes, std::chrono::milliseconds timeout);
[[nodiscard]] std::string local_ip(Handle connected);
void tune(Handle h);  // TCP_NODELAY + buyuk tamponlar

}  // namespace net

/// Noktadan noktaya tasima: her mesaj (grup kimligi, bayt) cercevesiyle gonderilir; alici
/// beklenen grubun cercevesi gelene kadar baska gruplarinkini yan kuyrukta tutar.
class Transport {
 public:
  virtual ~Transport() = default;
  [[nodiscard]] virtual int world_rank() const noexcept = 0;
  [[nodiscard]] virtual int world_size() const noexcept = 0;
  /// Asenkron gonderim baslatir; donus degeri wait_send ile beklenir. Tampon o zamana kadar canli kalmali.
  virtual std::uint64_t isend(int peer, std::uint32_t group, const void* data, std::size_t bytes) = 0;
  virtual void wait_send(int peer, std::uint64_t ticket, std::chrono::milliseconds timeout) = 0;
  virtual void recv(int peer, std::uint32_t group, void* data, std::size_t bytes, std::chrono::milliseconds timeout) = 0;
  [[nodiscard]] virtual const char* name() const noexcept = 0;
};

}  // namespace dist
}  // namespace nexus
