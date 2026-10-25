#pragma once

/**
 * @file process_group.hpp
 * @brief FAZ 10.1: ProcessGroup — rank'ler arasi kollektifler (asenkron Work), alt gruplar,
 *        dogrulama modu. Arka uclar ayni algoritmalari bir noktadan noktaya tasima katmani
 *        (Transport) uzerinde kosar:
 *          - "thread": tek surecte rank = is parcacigi, bellek ici posta kutulari (test).
 *          - "tcp": rank = surec, TCPStore ile bulusma, cift basina bir soket (Gloo benzeri).
 *          - "nccl": Windows'ta resmi NCCL yok -> acik hata (sessiz yedek yok). Linux NCCL bu
 *            depoda derlenmiyor ve test edilemiyor (tek GPU, Windows): yine acik hata.
 * Algoritmalar: halka all-reduce (reduce-scatter + all-gather, n parca), halka reduce-scatter,
 * halka all-gather, ikili agac yayin, cift cift all-to-all(v), send/recv, barrier (all-reduce).
 * Her grup bir is parcaciginda islemleri cagri sirasiyla yurutur (NCCL akis sirasi gibi);
 * cagri hemen Work dondurur. Butun rank'ler kollektifleri ayni sirada cagirmalidir.
 *
 * Dogrulama modu (10.6, NEXUS_DIST_DEBUG=1 ya da Options::validate): her kollektiften once
 * (sira no, islem, dtype, sayi, kok) parmak izi tum rank'lerle karsilastirilir; uyusmazlik
 * kilitlenme yerine iki tarafi da adlandiran istisna olur.
 * Indirgeme: F32/F64/I32/I64 dogrudan; F16/BF16 her adimda FP32'de toplanip yuvarlanir (Gloo gibi).
 * Avg yalniz kayan noktada (tamsayida acik hata). Zaman asimi: karsi rank'ten veri gelmezse
 * Work istisna ile biter; grup "bozuk" olur, sonraki her cagri hata verir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/dtype.hpp"

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexus {
namespace dist {

enum class ReduceOp : std::uint8_t { Sum, Avg, Max, Min, Product };
[[nodiscard]] const char* to_string(ReduceOp op) noexcept;

class DistError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Zaman asimi (karsi rank yanit vermedi). Kopan baglanti DistError'dir.
class DistTimeout : public DistError {
 public:
  using DistError::DistError;
};

/// Asenkron islem. wait() hata olduysa istisnayi yeniden firlatir.
class Work {
 public:
  virtual ~Work() = default;
  [[nodiscard]] virtual bool is_completed() const = 0;
  /// timeout < 0: grup zaman asimi. Sure dolarsa DistError.
  virtual void wait(std::chrono::milliseconds timeout = std::chrono::milliseconds(-1)) = 0;
  /// Bitis zamani (steady_clock); ortusme olcumu icin. Bitmeden cagrilirsa bos.
  [[nodiscard]] virtual std::chrono::steady_clock::time_point started() const = 0;
  [[nodiscard]] virtual std::chrono::steady_clock::time_point finished() const = 0;
};
using WorkPtr = std::shared_ptr<Work>;

class Transport;

struct GroupOptions {
  std::chrono::milliseconds timeout{std::chrono::seconds(60)};
  bool validate = false;  // NEXUS_DIST_DEBUG=1 ortam degiskeni de acar
};

class ProcessGroup {
 public:
  ~ProcessGroup();
  ProcessGroup(const ProcessGroup&) = delete;
  ProcessGroup& operator=(const ProcessGroup&) = delete;

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return size_; }
  [[nodiscard]] const std::string& backend() const noexcept { return backend_; }
  /// Bu grubun uyelerinin ust (dunya) rank'leri.
  [[nodiscard]] const std::vector<int>& members() const noexcept { return members_; }

  WorkPtr all_reduce(void* data, std::size_t count, DType dtype, ReduceOp op);
  /// in: [size * count], out: [count] (rank'in parcasi).
  WorkPtr reduce_scatter(const void* in, void* out, std::size_t count, DType dtype, ReduceOp op);
  /// in: [count], out: [size * count].
  WorkPtr all_gather(const void* in, void* out, std::size_t count, DType dtype);
  WorkPtr broadcast(void* data, std::size_t count, DType dtype, int root);
  /// in: rank r'ye send_counts[r] eleman (ardisik); out: rank r'den recv_counts[r] eleman (ardisik).
  WorkPtr all_to_all_v(const void* in, const std::vector<std::size_t>& send_counts, void* out,
                       const std::vector<std::size_t>& recv_counts, DType dtype);
  WorkPtr send(const void* data, std::size_t count, DType dtype, int dst, int tag = 0);
  WorkPtr recv(void* data, std::size_t count, DType dtype, int src, int tag = 0);
  /// FAZ 12: tek islemde dst'ye gonder + src'den al (halka adimi). Gonderim ve alim ayni gorevde
  /// ortusur; tcp'de iki ayri isin karsilikli gonderimde birbirini beklemesini (kilitlenme) onler.
  WorkPtr send_recv(const void* send_data, std::size_t send_count, int dst, void* recv_data, std::size_t recv_count,
                    int src, DType dtype, int tag = 0);
  WorkPtr barrier();
  /// Gorevi tum islerin bitmesini bekler (hata varsa firlatir).
  void synchronize();

  /// Alt grup (bu grubun rank'leriyle). Butun uyeler ayni listeyle cagirmali; uye olmayan
  /// rank'ler de cagirir (kollektif) ve nullptr alir.
  [[nodiscard]] std::shared_ptr<ProcessGroup> new_group(const std::vector<int>& ranks);

  /// FAZ 10.5: hiyerarsik all-reduce — dugum ici reduce-scatter -> dugumler arasi all-reduce
  /// -> dugum ici all-gather. ranks_per_node bu grubun boyutunu bolmeli; ardisik rank'ler ayni
  /// dugumde. count ranks_per_node'a bolunmeli (degilse halka all-reduce'a duser ve bunu soyler:
  /// `last_algorithm()`).
  WorkPtr all_reduce_hierarchical(void* data, std::size_t count, DType dtype, ReduceOp op, int ranks_per_node);
  [[nodiscard]] const char* last_algorithm() const noexcept { return last_algorithm_; }

  [[nodiscard]] const GroupOptions& options() const noexcept { return options_; }
  /// Kuyruga alinan kollektif sayisi (send/recv haric; dogrulama sira numarasi).
  [[nodiscard]] std::uint64_t sequence() const noexcept { return seq_; }

  struct Impl;

 private:
  friend std::shared_ptr<ProcessGroup> make_group(std::shared_ptr<Transport>, std::vector<int>, int, std::string,
                                                  GroupOptions);
  friend std::shared_ptr<ProcessGroup> init_process_group(const std::string&, GroupOptions);
  ProcessGroup(std::shared_ptr<Transport> transport, std::vector<int> members, int rank, std::string backend,
               GroupOptions options);
  /// collective = false: send/recv (iki rank) — dogrulama el sikismasi ve sira sayaci disinda.
  WorkPtr enqueue(const char* op, DType dtype, std::size_t count, int root, std::function<void()> body,
                  bool collective = true);

  std::shared_ptr<Transport> transport_;
  std::vector<int> members_;  // grup rank'i -> dunya rank'i
  int rank_ = 0;
  int size_ = 1;
  std::string backend_;
  GroupOptions options_;
  std::uint64_t seq_ = 0;
  std::uint32_t group_id_ = 0;
  std::uint32_t next_child_ = 1;
  const char* last_algorithm_ = "ring";
  std::unique_ptr<Impl> impl_;
};

using GroupPtr = std::shared_ptr<ProcessGroup>;

// ------------------------------------------------------------------------------------------
// Arka uclar
// ------------------------------------------------------------------------------------------

/// Tek surecte `size` rank: her is parcacigi kendi rank'iyle create_thread_group cagirir.
class ThreadWorld {
 public:
  explicit ThreadWorld(int size);
  ~ThreadWorld();
  [[nodiscard]] int size() const noexcept;
  [[nodiscard]] GroupPtr create_group(int rank, GroupOptions options = {});
  struct State;

 private:
  std::shared_ptr<State> state_;
};

class TcpStore;

/// TCP arka ucu: store uzerinden adres alisverisi, cift basina bir baglanti (dusuk rank dinler).
[[nodiscard]] GroupPtr create_tcp_group(TcpStore& store, int rank, int size, GroupOptions options = {});

/// NCCL: bu derlemede yok — her zaman DistError (Windows'ta resmi NCCL olmadigini soyler).
[[nodiscard]] GroupPtr create_nccl_group(TcpStore& store, int rank, int size, GroupOptions options = {});

/// Ortamdan baslatma (nexus-run / torchrun): RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
/// backend: "tcp" | "nccl". Store rank 0'da sunucudur; donen grup store'u canli tutar.
[[nodiscard]] GroupPtr init_process_group(const std::string& backend, GroupOptions options = {});

[[nodiscard]] int env_int(const char* name, int fallback);

}  // namespace dist
}  // namespace nexus
