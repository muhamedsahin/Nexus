#pragma once

/**
 * @file loop_profile.hpp
 * @brief Adım zamanlarının birikmiş dökümü. "Eğitim neden yavaş?" sorusunun cevabı.
 *
 * Süreler saniyedir. Prefetch işe yarıyorsa `data_wait_s`, duvar süresinin küçük
 * bir payıdır: tüketici kuyrukta hazır batch bulur. Darboğaz forward/backward ise
 * veri thread'i boşuna beklemez, kuyruk dolunca üretici durur (backpressure).
 */

#include <cstdint>
#include <sstream>
#include <string>

namespace nexus_train {

struct LoopProfile {
  double data_wait_s = 0.0;
  double forward_s = 0.0;
  double loss_s = 0.0;
  double backward_s = 0.0;
  double optimizer_s = 0.0;
  double callback_s = 0.0;
  double wall_s = 0.0;
  std::uint64_t batches = 0;
  std::uint64_t steps = 0;

  [[nodiscard]] std::string report() const {
    const double wall = wall_s > 0.0 ? wall_s : 1e-12;
    auto pct = [&](double part) {
      std::ostringstream os;
      os.setf(std::ios::fixed);
      os.precision(1);
      os << (100.0 * part / wall);
      return os.str();
    };
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(4);
    out << "batches=" << batches << " steps=" << steps << " wall_s=" << wall_s
        << " data_wait_s=" << data_wait_s << " (" << pct(data_wait_s) << "%)"
        << " forward_s=" << forward_s << " (" << pct(forward_s) << "%)"
        << " loss_s=" << loss_s << " (" << pct(loss_s) << "%)"
        << " backward_s=" << backward_s << " (" << pct(backward_s) << "%)"
        << " optimizer_s=" << optimizer_s << " (" << pct(optimizer_s) << "%)"
        << " callback_s=" << callback_s << " (" << pct(callback_s) << "%)";
    return out.str();
  }
};

}  // namespace nexus_train
