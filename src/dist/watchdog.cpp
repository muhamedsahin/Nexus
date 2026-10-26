/**
 * @file watchdog.cpp
 * @brief FAZ 15.F: bekci — kalp atisi panolari, takilma ve straggler tespiti.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/watchdog.hpp"

#include "nexus/dist/store.hpp"

#include <algorithm>
#include <limits>
#include <cstdio>
#include <sstream>

namespace nexus {
namespace dist {

void LocalBoard::publish(int rank, const Beat& beat) {
  std::lock_guard<std::mutex> lock(mu_);
  beats_.at(static_cast<std::size_t>(rank)) = beat;
}

std::vector<Beat> LocalBoard::read(int world) {
  std::lock_guard<std::mutex> lock(mu_);
  return std::vector<Beat>(beats_.begin(), beats_.begin() + std::min<std::ptrdiff_t>(world, static_cast<std::ptrdiff_t>(beats_.size())));
}

void StoreBoard::publish(int rank, const Beat& beat) {
  std::ostringstream o;
  o.precision(17);
  o << beat.step << ' ' << beat.step_ms << ' ' << beat.time_ms;
  store_->set("nexus/wd/" + std::to_string(rank), o.str());
}

std::vector<Beat> StoreBoard::read(int world) {
  std::vector<Beat> out(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r) {
    const std::string key = "nexus/wd/" + std::to_string(r);
    if (!store_->check(key)) continue;
    std::istringstream in(store_->get(key));
    Beat& b = out[static_cast<std::size_t>(r)];
    in >> b.step >> b.step_ms >> b.time_ms;
  }
  return out;
}

std::int64_t Watchdog::now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

Watchdog::Watchdog(std::shared_ptr<HeartbeatBoard> board, int rank, int world, WatchdogOptions options,
                   std::function<void(const WatchdogEvent&)> on_event)
    : board_(std::move(board)), rank_(rank), world_(world), options_(options), on_event_(std::move(on_event)),
      reported_stall_(static_cast<std::size_t>(world), -2), reported_straggler_(static_cast<std::size_t>(world), false),
      arrival_(static_cast<std::size_t>(world)), step_ms_(static_cast<std::size_t>(world), 0.0),
      lateness_(static_cast<std::size_t>(world)) {
  board_->publish(rank_, Beat{-1, 0.0, now_ms()});
  thread_ = std::thread([this] { monitor(); });
}

Watchdog::~Watchdog() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Watchdog::beat(std::int64_t step, double step_ms) { board_->publish(rank_, Beat{step, step_ms, now_ms()}); }

std::vector<WatchdogEvent> Watchdog::events() const {
  std::lock_guard<std::mutex> lock(mu_);
  return events_;
}

void Watchdog::monitor() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (cv_.wait_for(lock, options_.poll, [&] { return stop_; })) return;
    }
    std::vector<Beat> beats;
    try {
      beats = board_->read(world_);
    } catch (const std::exception&) {
      continue;  // pano gecici olarak okunamadi (store kapanirken)
    }
    const std::int64_t now = now_ms();
    std::int64_t lead = -1;
    for (const Beat& b : beats) lead = std::max(lead, b.step);
    std::vector<WatchdogEvent> fresh;
    // Takilma
    for (int r = 0; r < world_; ++r) {
      const Beat& b = beats[static_cast<std::size_t>(r)];
      const double lag = static_cast<double>(now - b.time_ms);
      if (b.step < lead && lag > static_cast<double>(options_.stall_timeout.count()) &&
          reported_stall_[static_cast<std::size_t>(r)] != b.step) {
        reported_stall_[static_cast<std::size_t>(r)] = b.step;
        WatchdogEvent e;
        e.kind = WatchdogEvent::Kind::Stalled;
        e.rank = r;
        e.step = b.step;
        e.lead_step = lead;
        e.lag_ms = lag;
        e.detected_ms = now;
        std::ostringstream m;
        m << "bekci: rank " << r << " takildi (son adim " << b.step << ", en ileri " << lead << ", " << lag
          << " ms kalp atisi yok)";
        e.message = m.str();
        fresh.push_back(e);
      }
    }
    // Straggler: adima varis gecikmesi (senkron egitimde adim sureleri esit oldugu icin)
    std::int64_t common = std::numeric_limits<std::int64_t>::max();
    for (int r = 0; r < world_; ++r) {
      const Beat& b = beats[static_cast<std::size_t>(r)];
      auto& a = arrival_[static_cast<std::size_t>(r)];
      if (b.step >= 0) {
        if (static_cast<std::int64_t>(a.size()) <= b.step) a.resize(static_cast<std::size_t>(b.step) + 1, -1);
        if (a[static_cast<std::size_t>(b.step)] < 0) a[static_cast<std::size_t>(b.step)] = b.time_ms;
        step_ms_[static_cast<std::size_t>(r)] = b.step_ms;
      }
      common = std::min(common, b.step);
    }
    for (std::int64_t s = scored_step_ + 1; s <= common && world_ >= 2; ++s) {
      std::vector<std::int64_t> t;
      for (int r = 0; r < world_; ++r) t.push_back(arrival_[static_cast<std::size_t>(r)][static_cast<std::size_t>(s)]);
      if (std::find(t.begin(), t.end(), -1) != t.end()) continue;  // bu adimi kacirdik (yoklama arasinda gecti)
      std::vector<std::int64_t> sorted = t;
      std::sort(sorted.begin(), sorted.end());
      const double med = static_cast<double>(sorted[sorted.size() / 2]);
      for (int r = 0; r < world_; ++r) {
        auto& l = lateness_[static_cast<std::size_t>(r)];
        l.push_back(static_cast<double>(t[static_cast<std::size_t>(r)]) - med);
        if (static_cast<int>(l.size()) > options_.straggler_window) l.erase(l.begin());
      }
    }
    if (common >= 0) scored_step_ = std::max(scored_step_, common);
    if (world_ >= 2 && static_cast<int>(lateness_[0].size()) >= options_.straggler_window) {
      std::vector<double> d = step_ms_;
      std::sort(d.begin(), d.end());
      const double median = d[d.size() / 2];
      for (int r = 0; r < world_; ++r) {
        double mean = 0.0;
        for (double v : lateness_[static_cast<std::size_t>(r)]) mean += v;
        mean /= static_cast<double>(lateness_[static_cast<std::size_t>(r)].size());
        const bool slow = mean > std::max(options_.straggler_fraction * median, static_cast<double>(options_.straggler_min_lag.count()));
        if (slow && !reported_straggler_[static_cast<std::size_t>(r)]) {
          WatchdogEvent e;
          e.kind = WatchdogEvent::Kind::Straggler;
          e.rank = r;
          e.step = beats[static_cast<std::size_t>(r)].step;
          e.lead_step = lead;
          e.step_ms = mean;
          e.median_ms = median;
          e.detected_ms = now;
          std::ostringstream m;
          m << "bekci: rank " << r << " yavas (son " << options_.straggler_window << " adimda ortalama " << mean
            << " ms gec varis, ortanca adim " << median << " ms)";
          e.message = m.str();
          fresh.push_back(e);
        }
        reported_straggler_[static_cast<std::size_t>(r)] = slow;
      }
    }
    if (!fresh.empty()) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        events_.insert(events_.end(), fresh.begin(), fresh.end());
      }
      for (const WatchdogEvent& e : fresh) {
        if (on_event_) on_event_(e);
      }
    }
  }
}

}  // namespace dist
}  // namespace nexus
