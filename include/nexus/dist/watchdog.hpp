#pragma once

/**
 * @file watchdog.hpp
 * @brief FAZ 15.F: bekci — rank ilerleme kalp atisi, takilan rank tespiti, straggler tespiti.
 *
 * Her rank adim sonunda beat(step, step_ms) yayinlar (pano: ayni surecte LocalBoard, surecler arasi StoreBoard =
 * TcpStore anahtarlari). Her rank'in izleme is parcacigi panoyu `poll` araliginda okur:
 *   - Takilma: bir rank'in adimi en ileri rank'ten geride ve son atisi `stall_timeout`tan eski -> Stalled
 *     (rank, takildigi adim, gecikme). Bir rank adim basina bir kez raporlanir.
 *   - Straggler: senkron egitimde herkes yavas rank'i bekledigi icin adim SURELERI esitlenir; bu yuzden olcut adima
 *     VARIS gecikmesidir: tum rank'lerin gordugu her adim icin gecikme_r = varis_r - ortanca varis. Son `window`
 *     adimin ortalama gecikmesi max(ortanca adim suresi * `straggler_fraction`, `straggler_min_lag`) degerini asarsa -> Straggler.
 * Kollektif zaman asimi ayridir: GroupOptions::timeout (Work::wait DistTimeout atar); bekci hangi rank'in
 * takildigini soyler. Raporlar on_event'e ve events()'e gider; bekci egitimi kendisi durdurmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nexus {
namespace dist {

class TcpStore;

struct Beat {
  std::int64_t step = -1;  // tamamlanan son adim (-1: henuz yok)
  double step_ms = 0.0;
  std::int64_t time_ms = 0;  // system_clock (ayni dugumdeki surecler arasi karsilastirilabilir)
};

class HeartbeatBoard {
 public:
  virtual ~HeartbeatBoard() = default;
  virtual void publish(int rank, const Beat& beat) = 0;
  [[nodiscard]] virtual std::vector<Beat> read(int world) = 0;
};

/// Ayni surecteki rank'ler (ThreadWorld).
class LocalBoard final : public HeartbeatBoard {
 public:
  explicit LocalBoard(int world) : beats_(static_cast<std::size_t>(world)) {}
  void publish(int rank, const Beat& beat) override;
  std::vector<Beat> read(int world) override;

 private:
  std::mutex mu_;
  std::vector<Beat> beats_;
};

/// TcpStore anahtarlari "nexus/wd/<rank>" (nexus-run surecleri).
class StoreBoard final : public HeartbeatBoard {
 public:
  explicit StoreBoard(TcpStore& store) : store_(&store) {}
  void publish(int rank, const Beat& beat) override;
  std::vector<Beat> read(int world) override;

 private:
  TcpStore* store_;
};

struct WatchdogOptions {
  std::chrono::milliseconds stall_timeout{std::chrono::seconds(30)};
  std::chrono::milliseconds poll{std::chrono::milliseconds(50)};
  double straggler_fraction = 0.5;
  std::chrono::milliseconds straggler_min_lag{std::chrono::milliseconds(10)};  // jitter siniri: bundan kucuk gecikme sayilmaz
  int straggler_window = 5;
};

struct WatchdogEvent {
  enum class Kind : std::uint8_t { Stalled, Straggler };
  Kind kind = Kind::Stalled;
  int rank = -1;
  std::int64_t step = -1;      // rank'in son tamamlanan adimi
  std::int64_t lead_step = -1; // en ileri rank'in adimi
  double lag_ms = 0.0;         // son atistan beri gecen (Stalled)
  double step_ms = 0.0;        // Straggler: ortalama varis gecikmesi
  double median_ms = 0.0;      // Straggler: ortanca adim suresi
  std::int64_t detected_ms = 0;  // system_clock
  std::string message;
};

class Watchdog {
 public:
  Watchdog(std::shared_ptr<HeartbeatBoard> board, int rank, int world, WatchdogOptions options = {},
           std::function<void(const WatchdogEvent&)> on_event = {});
  ~Watchdog();
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;

  void beat(std::int64_t step, double step_ms);
  [[nodiscard]] std::vector<WatchdogEvent> events() const;
  [[nodiscard]] static std::int64_t now_ms();

 private:
  void monitor();

  std::shared_ptr<HeartbeatBoard> board_;
  int rank_ = 0;
  int world_ = 1;
  WatchdogOptions options_;
  std::function<void(const WatchdogEvent&)> on_event_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::vector<WatchdogEvent> events_;
  std::vector<std::int64_t> reported_stall_;   // rank -> raporlanan adim
  std::vector<bool> reported_straggler_;
  std::vector<std::vector<std::int64_t>> arrival_;  // rank -> adim -> varis (system_clock ms; -1: gorulmedi)
  std::vector<double> step_ms_;                     // rank -> son adim suresi
  std::int64_t scored_step_ = -1;                   // gecikmesi hesaplanan son adim
  std::vector<std::vector<double>> lateness_;       // rank -> son gecikmeler
  std::thread thread_;
};

}  // namespace dist
}  // namespace nexus
