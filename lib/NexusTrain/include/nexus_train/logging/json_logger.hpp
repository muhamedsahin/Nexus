#pragma once

/**
 * @file json_logger.hpp
 * @brief Satır başına bir JSON nesnesi (JSONL). Dış araçlar dosyayı akar okur.
 */

#include "nexus_train/logging/logger_base.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nexus_train {

class JsonLogger final : public LoggerBase {
 public:
  explicit JsonLogger(std::string path) : out_(path, std::ios::out | std::ios::trunc) {
    if (!out_) {
      throw std::runtime_error("NexusTrain: cannot open json log " + path);
    }
  }

  void log_step(const TrainingState& state) override { write("step", state); }
  void log_epoch(const TrainingState& state) override { write("epoch", state); }

 private:
  static std::string escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char ch : text) {
      if (ch == '"' || ch == '\\') {
        out.push_back('\\');
      }
      out.push_back(ch);
    }
    return out;
  }

  void write(const char* kind, const TrainingState& state) {
    std::ostringstream line;
    line.setf(std::ios::fixed);
    line.precision(8);
    line << "{\"kind\":\"" << kind << "\",\"epoch\":" << state.epoch << ",\"step\":" << state.global_step
         << ",\"loss\":" << state.current_loss << ",\"metrics\":{";
    bool first = true;
    for (const auto& [name, value] : state.metrics) {
      if (!first) {
        line << ',';
      }
      first = false;
      line << '"' << escape(name) << "\":" << value;
    }
    line << "}}\n";
    out_ << line.str();
    out_.flush();
  }

  std::ofstream out_;
};

}  // namespace nexus_train
