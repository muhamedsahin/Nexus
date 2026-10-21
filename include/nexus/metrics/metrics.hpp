#pragma once

/**
 * @file metrics.hpp
 * @brief Argmax dogrulugu ve makro F1. Logitler sinif ekseninde okunur.
 */

#include <string>
#include <vector>

namespace nexus {
namespace metrics {

class Accuracy {
 public:
    [[nodiscard]] const char* name() const noexcept { return "accuracy"; }
    void reset() {
        correct_ = 0;
        total_ = 0;
    }
    void update(const float* logits, const int* labels, int batch, int classes) {
        for (int row = 0; row < batch; ++row) {
            int best = 0;
            const float* row_logits = logits + static_cast<std::size_t>(row * classes);
            for (int col = 1; col < classes; ++col) {
                if (row_logits[col] > row_logits[best]) {
                    best = col;
                }
            }
            correct_ += best == labels[row] ? 1 : 0;
            ++total_;
        }
    }
    [[nodiscard]] double value() const { return total_ == 0 ? 0.0 : static_cast<double>(correct_) / static_cast<double>(total_); }

 private:
    int correct_ = 0;
    int total_ = 0;
};

class F1Score {
 public:
    [[nodiscard]] const char* name() const noexcept { return "f1"; }
    void reset() {
        predictions_.clear();
        labels_.clear();
        classes_ = 0;
    }
    void update(const float* logits, const int* labels, int batch, int classes) {
        classes_ = classes;
        for (int row = 0; row < batch; ++row) {
            int best = 0;
            const float* row_logits = logits + static_cast<std::size_t>(row * classes);
            for (int col = 1; col < classes; ++col) {
                if (row_logits[col] > row_logits[best]) {
                    best = col;
                }
            }
            predictions_.push_back(best);
            labels_.push_back(labels[row]);
        }
    }
    [[nodiscard]] double value() const {
        if (classes_ <= 0 || labels_.empty()) {
            return 0.0;
        }
        double sum = 0.0;
        int used = 0;
        for (int cls = 0; cls < classes_; ++cls) {
            int tp = 0;
            int fp = 0;
            int fn = 0;
            for (std::size_t i = 0; i < labels_.size(); ++i) {
                const bool predicted = predictions_[i] == cls;
                const bool truth = labels_[i] == cls;
                tp += predicted && truth ? 1 : 0;
                fp += predicted && !truth ? 1 : 0;
                fn += !predicted && truth ? 1 : 0;
            }
            if (tp + fp + fn == 0) {
                continue;
            }
            const double precision = tp + fp == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fp);
            const double recall = tp + fn == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fn);
            const double denom = precision + recall;
            sum += denom == 0.0 ? 0.0 : 2.0 * precision * recall / denom;
            ++used;
        }
        return used == 0 ? 0.0 : sum / static_cast<double>(used);
    }

 private:
    std::vector<int> predictions_;
    std::vector<int> labels_;
    int classes_ = 0;
};

}  // namespace metrics
}  // namespace nexus
