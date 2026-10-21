#pragma once

/**
 * @file callbacks.hpp
 * @brief Epoch sonu gozlemcileri. EarlyStopping dogrulama kaybi izler.
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nexus {
namespace callbacks {

struct EpochState {
    int epoch = 0;
    float train_loss = 0.0f;
    float val_loss = 0.0f;
    float accuracy = 0.0f;
    float f1 = 0.0f;
};

class Callback {
 public:
    virtual ~Callback() = default;
    virtual void on_epoch_end(const EpochState& state) { (void)state; }
    [[nodiscard]] virtual bool stop() const { return false; }
};

class EarlyStopping : public Callback {
 public:
    explicit EarlyStopping(int patience, float min_delta = 0.0f) : patience_(patience), min_delta_(min_delta) {
        if (patience < 1) {
            throw std::invalid_argument("EarlyStopping patience must be at least 1");
        }
    }

    void on_epoch_end(const EpochState& state) override {
        if (!std::isfinite(state.val_loss)) {
            return;
        }
        if (!std::isfinite(best_) || state.val_loss < best_ - min_delta_) {
            best_ = state.val_loss;
            bad_epochs_ = 0;
            return;
        }
        ++bad_epochs_;
        if (bad_epochs_ >= patience_) {
            stopped_ = true;
        }
    }

    [[nodiscard]] bool stop() const override { return stopped_; }
    [[nodiscard]] float best() const noexcept { return best_; }
    [[nodiscard]] int bad_epochs() const noexcept { return bad_epochs_; }
    void restore(float best, int bad_epochs, bool stopped) {
        best_ = best;
        bad_epochs_ = bad_epochs;
        stopped_ = stopped;
    }

 private:
    int patience_ = 1;
    float min_delta_ = 0.0f;
    float best_ = std::numeric_limits<float>::quiet_NaN();
    int bad_epochs_ = 0;
    bool stopped_ = false;
};

class ProgressLogger : public Callback {
 public:
    void on_epoch_end(const EpochState& state) override { last_ = state; }
    [[nodiscard]] const EpochState& last() const noexcept { return last_; }

 private:
    EpochState last_{};
};

}  // namespace callbacks
}  // namespace nexus
