#pragma once

/**
 * @file metric.hpp
 * @brief Toplu metrik. Kullanici fonksiyonu bir batch'in ortalamasini dondurur.
 * `value` batch uzunluklariyla agirlikli genel ortalamadir.
 */

#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace nexus {

class CustomMetric {
 public:
    using Fn = std::function<double(std::span<const float> pred, std::span<const float> target)>;

    CustomMetric(std::string name, Fn fn) : name_(std::move(name)), fn_(std::move(fn)) {
        if (!fn_) {
            throw std::invalid_argument("CustomMetric function is empty");
        }
    }

    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    void reset() {
        weighted_ = 0.0;
        count_ = 0;
    }

    void update(std::span<const float> pred, std::span<const float> target) {
        if (pred.size() != target.size() || pred.empty()) {
            throw std::invalid_argument("CustomMetric pred and target must be the same non-empty length");
        }
        weighted_ += fn_(pred, target) * static_cast<double>(pred.size());
        count_ += pred.size();
    }

    [[nodiscard]] double value() const {
        if (count_ == 0) {
            throw std::logic_error("CustomMetric value before update");
        }
        return weighted_ / static_cast<double>(count_);
    }

    [[nodiscard]] std::size_t count() const noexcept { return count_; }

 private:
    std::string name_;
    Fn fn_;
    double weighted_ = 0.0;
    std::size_t count_ = 0;
};

}  // namespace nexus
