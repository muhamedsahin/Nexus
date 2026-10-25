#pragma once

/**
 * @file store.hpp
 * @brief FAZ 10.2: TCPStore benzeri anahtar-deger deposu (rendezvous). Rank 0 (ya da baslatici)
 *        sunucuyu kendi surecinde calistirir; her rank istemcidir (sunucu sureci de).
 *
 * Komutlar: set, get (anahtar gelene kadar bekler, zaman asimiyla), add (atomik tamsayi),
 * check, wait. barrier(ad) = add + sayac dunya boyuna ulasana kadar get.
 * Tel bicimi: [u8 komut][u32 anahtar uzunlugu][anahtar][u64 deger uzunlugu][deger];
 * yanit [u8 durum][u64 uzunluk][veri]. Istemci baglanirken sunucu hazir degilse zaman asimina
 * kadar yeniden dener (rank'ler herhangi bir sirada baslayabilir).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace dist {

class TcpStore {
 public:
  /// is_server: bu surec sunucuyu da calistirir (port 0: isletim sistemi secer, port() ile okunur).
  TcpStore(const std::string& host, int port, int world_size, bool is_server,
           std::chrono::milliseconds timeout = std::chrono::seconds(60));
  ~TcpStore();
  TcpStore(const TcpStore&) = delete;
  TcpStore& operator=(const TcpStore&) = delete;

  void set(const std::string& key, const std::string& value);
  [[nodiscard]] std::string get(const std::string& key);
  std::int64_t add(const std::string& key, std::int64_t delta);
  [[nodiscard]] bool check(const std::string& key);
  void wait(const std::vector<std::string>& keys);
  void barrier(const std::string& name);

  [[nodiscard]] int port() const noexcept { return port_; }
  [[nodiscard]] int world_size() const noexcept { return world_; }
  /// Bu istemcinin sunucuya baglandigi yerel IP (dugumler arasi adres yayini icin).
  [[nodiscard]] const std::string& local_address() const noexcept { return local_address_; }
  [[nodiscard]] std::chrono::milliseconds timeout() const noexcept { return timeout_; }

  struct Server;

 private:
  std::string request(std::uint8_t cmd, const std::string& key, const std::string& value);

  std::unique_ptr<Server> server_;
  std::intptr_t socket_ = -1;
  int port_ = 0;
  int world_ = 1;
  std::chrono::milliseconds timeout_;
  std::string local_address_;
};

}  // namespace dist
}  // namespace nexus
