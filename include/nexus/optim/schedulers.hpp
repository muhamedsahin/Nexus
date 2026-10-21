#pragma once

/**
 * @file schedulers.hpp
 * @brief Epoch basina ogrenme hizi. Sonuc, optimizer::set_learning_rate ile yazilir.
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nexus {
namespace optim {

class CosineAnnealing {
 public:
    CosineAnnealing(float base_lr, int t_max, float eta_min = 0.0f) : base_(base_lr), eta_min_(eta_min), t_max_(t_max) {
        if (t_max < 1 || !(base_lr > 0.0f)) {
            throw std::invalid_argument("CosineAnnealing T_max and base lr must be positive");
        }
    }

    float step() {
        const float t = static_cast<float>(std::min(epoch_, t_max_));
        const float cosine = std::cos(3.14159265f * t / static_cast<float>(t_max_));
        ++epoch_;
        return eta_min_ + (base_ - eta_min_) * (1.0f + cosine) * 0.5f;
    }

 private:
    float base_ = 1.0e-3f;
    float eta_min_ = 0.0f;
    int t_max_ = 1;
    int epoch_ = 0;
};

class LinearWarmup {
 public:
    LinearWarmup(float base_lr, int warmup_steps) : base_(base_lr), warmup_(warmup_steps) {
        if (warmup_steps < 1 || !(base_lr > 0.0f)) {
            throw std::invalid_argument("LinearWarmup warmup and base lr must be positive");
        }
    }

    float step() {
        ++step_;
        const float scale = std::min(1.0f, static_cast<float>(step_) / static_cast<float>(warmup_));
        return base_ * scale;
    }

 private:
    float base_ = 1.0e-3f;
    int warmup_ = 1;
    int step_ = 0;
};

class OneCycle {
 public:
    OneCycle(float max_lr, int total_steps, float pct_start = 0.3f) : max_lr_(max_lr), total_(total_steps), pct_start_(pct_start) {
        if (total_steps < 2 || !(max_lr > 0.0f) || !(pct_start > 0.0f) || pct_start >= 1.0f) {
            throw std::invalid_argument("OneCycle total, max lr, or pct_start is invalid");
        }
    }

    float step() {
        const float progress = std::min(1.0f, static_cast<float>(step_) / static_cast<float>(total_));
        ++step_;
        if (progress < pct_start_) {
            return max_lr_ * (progress / pct_start_);
        }
        const float fall = (progress - pct_start_) / (1.0f - pct_start_);
        return max_lr_ * (1.0f - fall);
    }

 private:
    float max_lr_ = 1.0e-3f;
    int total_ = 2;
    float pct_start_ = 0.3f;
    int step_ = 0;
};

}  // namespace optim
}  // namespace nexus
