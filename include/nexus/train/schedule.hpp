#pragma once

/**
 * @file schedule.hpp
 * @brief FAZ 9.4: adima bagli ogrenme hizi zamanlayicilari (durumsuz: carpan = f(adim)).
 *
 * Adim 0'dan baslayan optimizer adimidir (mikro adim degil). Carpan taban LR ile carpilir;
 * parametre grubu LR carpani (muP) ayrica uygulanir. Zamanlayici durum tasimadigi icin
 * devam (resume) yalniz adim sayacini kaydeder.
 *
 * Isinma, nanoGPT train.py `get_lr` ile ayni: adim < warmup icin (adim + 1) / (warmup + 1).
 * Kosinus: oran = (adim - warmup) / (max - warmup), carpan = min + 0.5 (1 + cos(pi oran)) (1 - min);
 * adim > max icin min. WSD (warmup-stable-decay, MiniCPM): isinma -> 1 -> son `decay` adimda
 * lineer (ya da 1 - sqrt) inis. Ters karekok: isinmadan sonra sqrt((warmup + 1) / (adim + 1)).
 *
 * NexusOptim scheduler/ siniflari epoch sayacli ve kendi optimizer'inin param_group LR
 * alanina yazar (durumlu `step()`); burada GPU AdamW ve CPU AdamW ayni carpani kullandigi
 * icin formuller durumsuz olarak yeniden yazildi. test_schedule_values: nanoGPT get_lr
 * degerleri ve isinmasiz kosinusun NexusOptim CosineAnnealingLR ile ayni oldugu denetlenir.
 * (NexusOptim LinearWarmup isinmayi (e + 1) / warmup ile yapar; nanoGPT'den farklidir.)
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>

namespace nexus {
namespace train {

struct ConstantLr {
  std::int64_t warmup_steps = 0;
};

struct LinearLr {
  std::int64_t warmup_steps = 0;
  std::int64_t max_steps = 1;
  double min_lr_ratio = 0.0;
};

struct WarmupCosine {
  std::int64_t warmup_steps = 0;
  std::int64_t max_steps = 1;
  double min_lr_ratio = 0.1;
};

struct WSD {
  std::int64_t warmup = 0;
  std::int64_t stable = 0;
  std::int64_t decay = 1;
  double min_lr_ratio = 0.0;
  bool sqrt_decay = false;  // true: carpan = 1 - sqrt(oran) (MiniCPM "1-sqrt")
};

struct InverseSqrt {
  std::int64_t warmup_steps = 1;
};

class LrSchedule {
 public:
  LrSchedule() = default;
  LrSchedule(ConstantLr s) : spec_(s) { check_warmup(s.warmup_steps); }  // NOLINT(google-explicit-constructor)
  LrSchedule(LinearLr s) : spec_(s) {  // NOLINT(google-explicit-constructor)
    check_warmup(s.warmup_steps);
    check_range(s.warmup_steps, s.max_steps, s.min_lr_ratio);
  }
  LrSchedule(WarmupCosine s) : spec_(s) {  // NOLINT(google-explicit-constructor)
    check_warmup(s.warmup_steps);
    check_range(s.warmup_steps, s.max_steps, s.min_lr_ratio);
  }
  LrSchedule(WSD s) : spec_(s) {  // NOLINT(google-explicit-constructor)
    check_warmup(s.warmup);
    if (s.stable < 0 || s.decay < 1 || !(s.min_lr_ratio >= 0.0 && s.min_lr_ratio <= 1.0)) {
      throw std::invalid_argument("WSD: stable >= 0, decay >= 1, 0 <= min_lr_ratio <= 1");
    }
  }
  LrSchedule(InverseSqrt s) : spec_(s) {  // NOLINT(google-explicit-constructor)
    if (s.warmup_steps < 1) {
      throw std::invalid_argument("InverseSqrt: warmup_steps >= 1");
    }
  }

  /// Taban LR carpani (0-tabanli optimizer adimi).
  [[nodiscard]] double factor(std::int64_t step) const {
    if (step < 0) {
      throw std::invalid_argument("LrSchedule: adim >= 0");
    }
    return std::visit([step](const auto& s) { return eval(s, step); }, spec_);
  }

  [[nodiscard]] std::string describe() const {
    return std::visit([](const auto& s) { return name(s); }, spec_);
  }

 private:
  static double warm(std::int64_t step, std::int64_t warmup) {
    return static_cast<double>(step + 1) / static_cast<double>(warmup + 1);
  }
  static double eval(const ConstantLr& s, std::int64_t step) {
    return step < s.warmup_steps ? warm(step, s.warmup_steps) : 1.0;
  }
  static double eval(const LinearLr& s, std::int64_t step) {
    if (step < s.warmup_steps) {
      return warm(step, s.warmup_steps);
    }
    if (step > s.max_steps) {
      return s.min_lr_ratio;
    }
    const double r = static_cast<double>(step - s.warmup_steps) / static_cast<double>(s.max_steps - s.warmup_steps);
    return s.min_lr_ratio + (1.0 - r) * (1.0 - s.min_lr_ratio);
  }
  static double eval(const WarmupCosine& s, std::int64_t step) {
    if (step < s.warmup_steps) {
      return warm(step, s.warmup_steps);
    }
    if (step > s.max_steps) {
      return s.min_lr_ratio;
    }
    const double r = static_cast<double>(step - s.warmup_steps) / static_cast<double>(s.max_steps - s.warmup_steps);
    const double coeff = 0.5 * (1.0 + std::cos(3.14159265358979323846 * r));
    return s.min_lr_ratio + coeff * (1.0 - s.min_lr_ratio);
  }
  static double eval(const WSD& s, std::int64_t step) {
    if (step < s.warmup) {
      return warm(step, s.warmup);
    }
    const std::int64_t decay_begin = s.warmup + s.stable;
    if (step < decay_begin) {
      return 1.0;
    }
    const double r = std::min(1.0, static_cast<double>(step - decay_begin + 1) / static_cast<double>(s.decay));
    const double shape = s.sqrt_decay ? 1.0 - std::sqrt(r) : 1.0 - r;
    return s.min_lr_ratio + shape * (1.0 - s.min_lr_ratio);
  }
  static double eval(const InverseSqrt& s, std::int64_t step) {
    if (step < s.warmup_steps) {
      return warm(step, s.warmup_steps);
    }
    return std::sqrt(static_cast<double>(s.warmup_steps + 1) / static_cast<double>(step + 1));
  }
  static std::string name(const ConstantLr&) { return "constant"; }
  static std::string name(const LinearLr&) { return "linear"; }
  static std::string name(const WarmupCosine&) { return "warmup_cosine"; }
  static std::string name(const WSD&) { return "wsd"; }
  static std::string name(const InverseSqrt&) { return "inverse_sqrt"; }

  static void check_warmup(std::int64_t warmup) {
    if (warmup < 0) {
      throw std::invalid_argument("LrSchedule: warmup >= 0");
    }
  }
  static void check_range(std::int64_t warmup, std::int64_t max_steps, double min_ratio) {
    if (max_steps <= warmup || !(min_ratio >= 0.0 && min_ratio <= 1.0)) {
      throw std::invalid_argument("LrSchedule: max_steps > warmup ve 0 <= min_lr_ratio <= 1");
    }
  }

  std::variant<ConstantLr, LinearLr, WarmupCosine, WSD, InverseSqrt> spec_{ConstantLr{}};
};

}  // namespace train
}  // namespace nexus
