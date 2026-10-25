/**
 * @file loggers.cpp
 * @brief FAZ 9.6: TensorBoard olay dosyasi (protobuf'suz) ve W&B uyumlu JSONL.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/train/loggers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nexus {
namespace train {
namespace {

std::array<std::uint32_t, 256> make_crc_table() {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) {
      c = (c & 1u) != 0 ? 0x82F63B78u ^ (c >> 1) : c >> 1;
    }
    table[i] = c;
  }
  return table;
}

// ---- protobuf tel bicimi (yalniz gereken alanlar) ----
void varint(std::string& out, std::uint64_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<char>((v & 0x7F) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<char>(v));
}
void key(std::string& out, int field, int wire) { varint(out, (static_cast<std::uint64_t>(field) << 3) | static_cast<unsigned>(wire)); }
void fixed64(std::string& out, double v) {
  char b[8];
  std::memcpy(b, &v, 8);
  out.append(b, 8);
}
void fixed32(std::string& out, float v) {
  char b[4];
  std::memcpy(b, &v, 4);
  out.append(b, 4);
}
void bytes_field(std::string& out, int field, const std::string& value) {
  key(out, field, 2);
  varint(out, value.size());
  out += value;
}

double unix_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

std::string json_number(double v) {
  if (!std::isfinite(v)) {
    return v != v ? "NaN" : (v > 0 ? "Infinity" : "-Infinity");  // W&B JSON ayristiricisi bunlari kabul eder
  }
  std::ostringstream s;
  s << std::setprecision(10) << v;
  return s.str();
}

std::string json_escape(const std::string& text) {
  std::string out;
  for (char ch : text) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
    }
    out.push_back(ch);
  }
  return out;
}

std::vector<std::pair<std::string, double>> sorted_metrics(const nexus_train::TrainingState& state) {
  std::vector<std::pair<std::string, double>> values(state.metrics.begin(), state.metrics.end());
  std::sort(values.begin(), values.end());
  return values;
}

}  // namespace

std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc) noexcept {
  static const std::array<std::uint32_t, 256> table = make_crc_table();
  const auto* p = static_cast<const unsigned char*>(data);
  crc = ~crc;
  for (std::size_t i = 0; i < size; ++i) {
    crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
  }
  return ~crc;
}

std::uint32_t masked_crc32c(const void* data, std::size_t size) noexcept {
  const std::uint32_t c = crc32c(data, size);
  return ((c >> 15) | (c << 17)) + 0xA282EAD8u;
}

TensorBoardLogger::TensorBoardLogger(const std::string& directory) {
  std::filesystem::create_directories(directory);
  std::ostringstream name;
  name << "events.out.tfevents." << static_cast<std::int64_t>(unix_seconds()) << ".nexus";
  path_ = (std::filesystem::path(directory) / name.str()).string();
  out_.open(path_, std::ios::binary | std::ios::trunc);
  if (!out_) {
    throw std::runtime_error("Nexus TensorBoardLogger: dosya acilamadi " + path_);
  }
  std::string event;
  key(event, 1, 1);
  fixed64(event, unix_seconds());
  bytes_field(event, 3, "brain.Event:2");
  write_record(event);
}

void TensorBoardLogger::write_record(const std::string& payload) {
  const std::uint64_t length = payload.size();
  char header[12];
  std::memcpy(header, &length, 8);
  const std::uint32_t lcrc = masked_crc32c(header, 8);
  std::memcpy(header + 8, &lcrc, 4);
  const std::uint32_t dcrc = masked_crc32c(payload.data(), payload.size());
  out_.write(header, 12);
  out_.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  out_.write(reinterpret_cast<const char*>(&dcrc), 4);
  out_.flush();
}

void TensorBoardLogger::write_scalars(const std::vector<std::pair<std::string, double>>& values, std::int64_t step) {
  std::string summary;
  for (const auto& [tag, value] : values) {
    std::string v;
    bytes_field(v, 1, tag);
    key(v, 2, 5);
    fixed32(v, static_cast<float>(value));
    bytes_field(summary, 1, v);
  }
  std::string event;
  key(event, 1, 1);
  fixed64(event, unix_seconds());
  key(event, 2, 0);
  varint(event, static_cast<std::uint64_t>(step));
  bytes_field(event, 5, summary);
  write_record(event);
}

void TensorBoardLogger::scalar(const std::string& tag, double value, std::int64_t step) {
  write_scalars({{tag, value}}, step);
}

void TensorBoardLogger::log_step(const nexus_train::TrainingState& state) {
  std::vector<std::pair<std::string, double>> values = sorted_metrics(state);
  values.insert(values.begin(), {"train/loss", state.current_loss});
  write_scalars(values, state.global_step);
}

void TensorBoardLogger::log_epoch(const nexus_train::TrainingState& state) {
  std::vector<std::pair<std::string, double>> values;
  for (const auto& [name, value] : sorted_metrics(state)) {
    values.emplace_back("epoch/" + name, value);
  }
  write_scalars(values, state.global_step);
}

namespace {
std::uint64_t read_varint(const std::string& s, std::size_t& pos) {
  std::uint64_t v = 0;
  for (int shift = 0; pos < s.size() && shift < 64; shift += 7) {
    const auto b = static_cast<unsigned char>(s[pos++]);
    v |= static_cast<std::uint64_t>(b & 0x7F) << shift;
    if ((b & 0x80) == 0) {
      return v;
    }
  }
  throw std::runtime_error("tfevents: bozuk varint");
}

void skip_field(const std::string& s, std::size_t& pos, int wire) {
  if (wire == 0) {
    (void)read_varint(s, pos);
  } else if (wire == 1) {
    pos += 8;
  } else if (wire == 5) {
    pos += 4;
  } else if (wire == 2) {
    pos += read_varint(s, pos);
  } else {
    throw std::runtime_error("tfevents: desteklenmeyen tel tipi");
  }
}
}  // namespace

std::vector<TensorBoardScalar> read_tensorboard_scalars(const std::string& path, std::string* file_version) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("tfevents: acilamadi " + path);
  }
  std::vector<TensorBoardScalar> out;
  for (;;) {
    char header[12];
    if (!in.read(header, 12)) {
      break;
    }
    std::uint64_t length = 0;
    std::uint32_t lcrc = 0;
    std::memcpy(&length, header, 8);
    std::memcpy(&lcrc, header + 8, 4);
    if (lcrc != masked_crc32c(header, 8)) {
      throw std::runtime_error("tfevents: uzunluk CRC uyusmuyor");
    }
    std::string data(static_cast<std::size_t>(length), '\0');
    std::uint32_t dcrc = 0;
    if (!in.read(data.data(), static_cast<std::streamsize>(length)) || !in.read(reinterpret_cast<char*>(&dcrc), 4)) {
      throw std::runtime_error("tfevents: kayit yarim");
    }
    if (dcrc != masked_crc32c(data.data(), data.size())) {
      throw std::runtime_error("tfevents: veri CRC uyusmuyor");
    }
    std::size_t pos = 0;
    std::int64_t step = 0;
    std::vector<std::pair<std::string, float>> values;
    while (pos < data.size()) {
      const std::uint64_t k = read_varint(data, pos);
      const int field = static_cast<int>(k >> 3);
      const int wire = static_cast<int>(k & 7);
      if (field == 2 && wire == 0) {
        step = static_cast<std::int64_t>(read_varint(data, pos));
      } else if (field == 3 && wire == 2) {
        const std::uint64_t n = read_varint(data, pos);
        if (file_version != nullptr) {
          *file_version = data.substr(pos, n);
        }
        pos += n;
      } else if (field == 5 && wire == 2) {
        const std::size_t end = pos + read_varint(data, pos);
        while (pos < end) {
          const std::uint64_t vk = read_varint(data, pos);
          if ((vk >> 3) != 1 || (vk & 7) != 2) {
            skip_field(data, pos, static_cast<int>(vk & 7));
            continue;
          }
          const std::size_t vend = pos + read_varint(data, pos);
          std::string tag;
          float value = 0.0f;
          while (pos < vend) {
            const std::uint64_t fk = read_varint(data, pos);
            if ((fk >> 3) == 1 && (fk & 7) == 2) {
              const std::uint64_t n = read_varint(data, pos);
              tag = data.substr(pos, n);
              pos += n;
            } else if ((fk >> 3) == 2 && (fk & 7) == 5) {
              std::memcpy(&value, data.data() + pos, 4);
              pos += 4;
            } else {
              skip_field(data, pos, static_cast<int>(fk & 7));
            }
          }
          values.emplace_back(std::move(tag), value);
        }
      } else {
        skip_field(data, pos, wire);
      }
    }
    for (auto& [tag, value] : values) {
      out.push_back(TensorBoardScalar{step, std::move(tag), value});
    }
  }
  return out;
}

WandbJsonlLogger::WandbJsonlLogger(const std::string& path)
    : out_(path, std::ios::out | std::ios::trunc), start_(std::chrono::steady_clock::now()) {
  if (!out_) {
    throw std::runtime_error("Nexus WandbJsonlLogger: dosya acilamadi " + path);
  }
}

void WandbJsonlLogger::write(const nexus_train::TrainingState& state, const char* prefix) {
  const double runtime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  std::ostringstream line;
  line << "{\"_step\":" << state.global_step << ",\"_runtime\":" << json_number(runtime)
       << ",\"_timestamp\":" << json_number(unix_seconds()) << ",\"" << prefix << "loss\":" << json_number(state.current_loss);
  for (const auto& [name, value] : sorted_metrics(state)) {
    line << ",\"" << json_escape(name) << "\":" << json_number(value);  // metrik adlari zaten "train/", "val/" onekli
  }
  line << "}\n";
  out_ << line.str();
  out_.flush();
}

void WandbJsonlLogger::log_step(const nexus_train::TrainingState& state) { write(state, "train/"); }
void WandbJsonlLogger::log_epoch(const nexus_train::TrainingState& state) { write(state, "epoch/"); }

}  // namespace train
}  // namespace nexus
