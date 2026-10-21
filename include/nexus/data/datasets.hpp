#pragma once

/**
 * @file datasets.hpp
 * @brief Bellekteki ornekler ve virgulle ayrilmis sayisal CSV.
 * Son sutun sinif etiketidir. Ilk satir harf iceriyorsa baslik sayilir.
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nexus {
namespace data {

struct Batch {
    std::vector<float> features;
    std::vector<int> labels;
    int batch_size = 0;
    int features_per_sample = 0;
};

struct Table {
    std::vector<float> features;
    std::vector<int> labels;
    int features_per_sample = 0;
    [[nodiscard]] int samples() const {
        return features_per_sample == 0 ? 0 : static_cast<int>(labels.size());
    }
};

class DataLoader {
 public:
    DataLoader(std::vector<float> features, std::vector<int> labels, int features_per_sample, int batch_size, bool shuffle = true,
               std::uint32_t seed = 1)
        : features_(std::move(features)), labels_(std::move(labels)), features_per_sample_(features_per_sample), batch_size_(batch_size),
          shuffle_(shuffle), rng_(seed == 0 ? 1u : seed) {
        if (features_per_sample <= 0 || batch_size <= 0) {
            throw std::invalid_argument("DataLoader feature width and batch size must be positive");
        }
        if (labels_.empty() || features_.size() != labels_.size() * static_cast<std::size_t>(features_per_sample)) {
            throw std::invalid_argument("DataLoader features and labels do not match");
        }
        order_.resize(labels_.size());
        reset();
    }

    [[nodiscard]] int sample_count() const { return static_cast<int>(labels_.size()); }
    [[nodiscard]] int features_per_sample() const { return features_per_sample_; }

    void reset() {
        for (std::size_t i = 0; i < order_.size(); ++i) {
            order_[i] = static_cast<int>(i);
        }
        if (shuffle_) {
            for (std::size_t i = order_.size(); i > 1; --i) {
                const std::size_t j = static_cast<std::size_t>(rng_.next() % static_cast<std::uint32_t>(i));
                std::swap(order_[i - 1], order_[j]);
            }
        }
        cursor_ = 0;
    }

    bool next(Batch& batch) {
        if (cursor_ >= static_cast<int>(order_.size())) {
            return false;
        }
        const int count = std::min(batch_size_, static_cast<int>(order_.size()) - cursor_);
        batch.batch_size = count;
        batch.features_per_sample = features_per_sample_;
        batch.features.resize(static_cast<std::size_t>(count * features_per_sample_));
        batch.labels.resize(static_cast<std::size_t>(count));
        for (int row = 0; row < count; ++row) {
            const int index = order_[static_cast<std::size_t>(cursor_ + row)];
            batch.labels[static_cast<std::size_t>(row)] = labels_[static_cast<std::size_t>(index)];
            const float* source = features_.data() + static_cast<std::size_t>(index * features_per_sample_);
            float* destination = batch.features.data() + static_cast<std::size_t>(row * features_per_sample_);
            for (int col = 0; col < features_per_sample_; ++col) {
                destination[col] = source[col];
            }
        }
        cursor_ += count;
        return true;
    }

 private:
    struct Rng {
        explicit Rng(std::uint32_t seed) : state(seed) {}
        std::uint32_t next() {
            state = state * 1664525u + 1013904223u;
            return state;
        }
        std::uint32_t state;
    };

    std::vector<float> features_;
    std::vector<int> labels_;
    std::vector<int> order_;
    int features_per_sample_ = 0;
    int batch_size_ = 1;
    int cursor_ = 0;
    bool shuffle_ = true;
    Rng rng_;
};

inline Table parse_csv(std::string_view text) {
    Table table;
    std::stringstream stream{std::string(text)};
    std::string line;
    bool header_checked = false;
    while (std::getline(stream, line)) {
        if (line.empty()) {
            continue;
        }
        if (!header_checked) {
            header_checked = true;
            bool letter = false;
            for (char character : line) {
                if (std::isalpha(static_cast<unsigned char>(character)) != 0) {
                    letter = true;
                    break;
                }
            }
            if (letter) {
                continue;
            }
        }
        std::vector<float> values;
        std::stringstream row_stream(line);
        std::string cell;
        while (std::getline(row_stream, cell, ',')) {
            values.push_back(std::stof(cell));
        }
        if (values.size() < 2) {
            throw std::invalid_argument("CSV row needs features and a label");
        }
        const int width = static_cast<int>(values.size() - 1);
        if (table.features_per_sample == 0) {
            table.features_per_sample = width;
        }
        if (width != table.features_per_sample) {
            throw std::invalid_argument("CSV rows have different widths");
        }
        table.features.insert(table.features.end(), values.begin(), values.end() - 1);
        table.labels.push_back(static_cast<int>(values.back()));
    }
    if (table.labels.empty()) {
        throw std::invalid_argument("CSV contained no data rows");
    }
    return table;
}

inline Table read_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("CSV file could not be opened: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_csv(buffer.str());
}

inline DataLoader make_loader(const Table& table, int batch_size, bool shuffle = true, std::uint32_t seed = 1) {
    return DataLoader(table.features, table.labels, table.features_per_sample, batch_size, shuffle, seed);
}

}  // namespace data
}  // namespace nexus
