#pragma once

/**
 * @file callback.hpp
 * @brief Egitim dongusunun disaridan gozlemi. TrainerBuilder Faz 6'dadir;
 * bu liste kullanicinin kendi dongusune takilir.
 */

#include <memory>
#include <utility>
#include <vector>

namespace nexus {

class CustomCallback {
 public:
    virtual ~CustomCallback() = default;
    virtual void on_train_start() {}
    virtual void on_step_end(int step, double loss) {
        (void)step;
        (void)loss;
    }
    virtual void on_train_end(double final_loss) { (void)final_loss; }
};

class CallbackList {
 public:
    void add(std::shared_ptr<CustomCallback> callback) {
        if (!callback) {
            return;
        }
        callbacks_.push_back(std::move(callback));
    }

    void on_train_start() {
        for (const std::shared_ptr<CustomCallback>& callback : callbacks_) {
            callback->on_train_start();
        }
    }

    void on_step_end(int step, double loss) {
        for (const std::shared_ptr<CustomCallback>& callback : callbacks_) {
            callback->on_step_end(step, loss);
        }
    }

    void on_train_end(double final_loss) {
        for (const std::shared_ptr<CustomCallback>& callback : callbacks_) {
            callback->on_train_end(final_loss);
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return callbacks_.size(); }

 private:
    std::vector<std::shared_ptr<CustomCallback>> callbacks_;
};

}  // namespace nexus
