/**
 * @file prof.cpp
 * @brief Bellek muhasebe defteri, sayac kaydi, profiler ve Chrome iz dokumu.
 *
 * Defter ve sayac kaydi kilit altinda calisir; olcum kodu sicak yolda
 * yalnizca kilit + toplama gorur. Profiler kapaliyken ScopedTimer kayit
 * yazmaz; iz dokumu yalnizca istendiginde dosyaya gider.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/prof/counters.hpp"
#include "nexus/prof/memory_ledger.hpp"
#include "nexus/prof/timer.hpp"
#include "nexus/prof/trace.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <utility>

namespace nexus {
namespace prof {
namespace {

constexpr const char* kLabelNames[static_cast<std::size_t>(MemoryLabel::kCount)] = {
    "parameter", "gradient", "optimizer_state", "activation", "temporary", "comm_buffer", "kv_cache"};

struct LedgerSpin {
  std::atomic_flag& flag;
  explicit LedgerSpin(std::atomic_flag& source) : flag(source) {
    while (flag.test_and_set(std::memory_order_acquire)) {
    }
  }
  ~LedgerSpin() { flag.clear(std::memory_order_release); }
};

thread_local MemoryLabel t_current_label = MemoryLabel::Temporary;

void bump_entry(LedgerEntry& entry, std::int64_t bytes, bool alloc) noexcept {
  if (alloc) {
    entry.current_bytes += bytes;
    if (entry.current_bytes > entry.peak_bytes) {
      entry.peak_bytes = entry.current_bytes;
    }
    entry.live_blocks += 1;
    entry.total_allocs += 1;
  } else {
    entry.current_bytes -= bytes;
    if (entry.live_blocks > 0) {
      entry.live_blocks -= 1;
    }
  }
}

void write_json_string(std::ostringstream& out, std::string_view text) {
  out << '"';
  for (const char character : text) {
    switch (character) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (static_cast<unsigned char>(character) < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(character));
          out << buffer;
        } else {
          out << character;
        }
        break;
    }
  }
  out << '"';
}

std::int64_t process_epoch_ns() noexcept {
  static const std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(epoch.time_since_epoch()).count();
}

}  // namespace

const char* label_name(MemoryLabel label) noexcept {
  const auto index = static_cast<std::size_t>(label);
  if (index >= static_cast<std::size_t>(MemoryLabel::kCount)) {
    return "temporary";
  }
  return kLabelNames[index];
}

void MemoryLedger::record_alloc(MemoryLabel label, LedgerSpace space, std::size_t bytes) noexcept {
  const LedgerSpin guard(flag_);
  const auto label_index = static_cast<std::size_t>(label);
  const auto space_index = static_cast<std::size_t>(space) - 1u;
  const auto signed_bytes = static_cast<std::int64_t>(bytes);
  if (label_index < by_label_.size()) {
    bump_entry(by_label_[label_index], signed_bytes, true);
  }
  if (space_index < by_space_.size()) {
    bump_entry(by_space_[space_index], signed_bytes, true);
  }
  current_bytes_ += signed_bytes;
  if (current_bytes_ > peak_bytes_) {
    peak_bytes_ = current_bytes_;
  }
  live_blocks_ += 1;
  total_allocs_ += 1;
}

void MemoryLedger::record_free(MemoryLabel label, LedgerSpace space, std::size_t bytes) noexcept {
  const LedgerSpin guard(flag_);
  const auto label_index = static_cast<std::size_t>(label);
  const auto space_index = static_cast<std::size_t>(space) - 1u;
  const auto signed_bytes = static_cast<std::int64_t>(bytes);
  if (label_index < by_label_.size()) {
    bump_entry(by_label_[label_index], signed_bytes, false);
  }
  if (space_index < by_space_.size()) {
    bump_entry(by_space_[space_index], signed_bytes, false);
  }
  current_bytes_ -= signed_bytes;
  if (live_blocks_ > 0) {
    live_blocks_ -= 1;
  }
}

LedgerSnapshot MemoryLedger::snapshot() const noexcept {
  const LedgerSpin guard(flag_);
  LedgerSnapshot result;
  result.by_label = by_label_;
  result.by_space = by_space_;
  result.current_bytes = current_bytes_;
  result.peak_bytes = peak_bytes_;
  result.live_blocks = live_blocks_;
  result.total_allocs = total_allocs_;
  return result;
}

std::int64_t MemoryLedger::current_bytes() const noexcept {
  const LedgerSpin guard(flag_);
  return current_bytes_;
}

std::int64_t MemoryLedger::peak_bytes() const noexcept {
  const LedgerSpin guard(flag_);
  return peak_bytes_;
}

void MemoryLedger::reset_peak() noexcept {
  const LedgerSpin guard(flag_);
  for (LedgerEntry& entry : by_label_) {
    entry.peak_bytes = entry.current_bytes;
  }
  for (LedgerEntry& entry : by_space_) {
    entry.peak_bytes = entry.current_bytes;
  }
  peak_bytes_ = current_bytes_;
}

void MemoryLedger::reset_all() noexcept {
  const LedgerSpin guard(flag_);
  by_label_ = {};
  by_space_ = {};
  current_bytes_ = 0;
  peak_bytes_ = 0;
  live_blocks_ = 0;
  total_allocs_ = 0;
}

MemoryLedger& ledger() noexcept {
  // Yorumlayici kapanis sirasi belirsiz; defter kasitli olarak sizdirilir.
  static auto* instance = new MemoryLedger();
  return *instance;
}

MemoryLabel current_allocation_label() noexcept { return t_current_label; }

AllocationLabelScope::AllocationLabelScope(MemoryLabel label) noexcept : previous_(t_current_label) {
  t_current_label = label;
}

AllocationLabelScope::~AllocationLabelScope() noexcept { t_current_label = previous_; }


void CounterRegistry::add_flops(const std::string& name, std::int64_t count) {
  const std::lock_guard<std::mutex> guard(mutex_);
  counters_[name].flops += count;
}

void CounterRegistry::add_bytes(const std::string& name, std::int64_t count) {
  const std::lock_guard<std::mutex> guard(mutex_);
  counters_[name].bytes += count;
}

void CounterRegistry::add_calls(const std::string& name, std::int64_t count) {
  const std::lock_guard<std::mutex> guard(mutex_);
  counters_[name].calls += count;
}

std::map<std::string, CounterValues> CounterRegistry::snapshot() const {
  const std::lock_guard<std::mutex> guard(mutex_);
  return counters_;
}

std::int64_t CounterRegistry::total_flops() const {
  const std::lock_guard<std::mutex> guard(mutex_);
  std::int64_t total = 0;
  for (const auto& [name, values] : counters_) {
    (void)name;
    total += values.flops;
  }
  return total;
}

std::int64_t CounterRegistry::total_bytes() const {
  const std::lock_guard<std::mutex> guard(mutex_);
  std::int64_t total = 0;
  for (const auto& [name, values] : counters_) {
    (void)name;
    total += values.bytes;
  }
  return total;
}

void CounterRegistry::reset() {
  const std::lock_guard<std::mutex> guard(mutex_);
  counters_.clear();
}

CounterRegistry& counters() noexcept {
  static auto* instance = new CounterRegistry();
  return *instance;
}

bool Profiler::enabled() const noexcept {
  // enabled_ yalnizca ana is parcacigindan degistirilir; okuma kilitsizdir.
  return enabled_;
}

void Profiler::set_enabled(bool enabled) noexcept { enabled_ = enabled; }

void Profiler::record(const IntervalRecord& interval) {
  const std::lock_guard<std::mutex> guard(mutex_);
  if (!enabled_) {
    return;
  }
  IntervalTotal& total = totals_[interval.name];
  total.calls += 1;
  total.total_ns += interval.duration_ns;
  if (total.calls == 1 || interval.duration_ns < total.min_ns) {
    total.min_ns = interval.duration_ns;
  }
  if (interval.duration_ns > total.max_ns) {
    total.max_ns = interval.duration_ns;
  }
  intervals_.push_back(interval);
}

std::vector<IntervalRecord> Profiler::intervals() const {
  const std::lock_guard<std::mutex> guard(mutex_);
  return intervals_;
}

std::map<std::string, IntervalTotal> Profiler::totals() const {
  const std::lock_guard<std::mutex> guard(mutex_);
  return totals_;
}

void Profiler::reset() {
  const std::lock_guard<std::mutex> guard(mutex_);
  intervals_.clear();
  totals_.clear();
}

std::int64_t Profiler::now_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

Profiler& profiler() noexcept {
  static auto* instance = new Profiler();
  return *instance;
}

ScopedTimer::ScopedTimer(std::string_view name, std::string_view category) noexcept
    : name_(name), category_(category), start_(std::chrono::steady_clock::now()), record_(profiler().enabled()) {}

ScopedTimer::~ScopedTimer() {
  const auto end = std::chrono::steady_clock::now();
  if (!record_) {
    return;
  }
  IntervalRecord interval;
  interval.name = name_;
  interval.category = category_;
  interval.thread_id = static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  interval.start_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(start_.time_since_epoch()).count() - process_epoch_ns();
  interval.duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
  profiler().record(interval);
}

std::int64_t ScopedTimer::elapsed_ns() const noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_).count();
}

std::string chrome_trace_json() {
  const std::vector<IntervalRecord> intervals = profiler().intervals();
  const std::map<std::string, CounterValues> counter_snapshot = counters().snapshot();
  const std::int64_t origin_ns = process_epoch_ns();

  std::ostringstream out;
  out << "{\"traceEvents\":[";
  bool first = true;
  for (const IntervalRecord& interval : intervals) {
    if (!first) {
      out << ',';
    }
    first = false;
    out << "{\"name\":";
    write_json_string(out, interval.name);
    out << ",\"cat\":";
    write_json_string(out, interval.category);
    out << ",\"ph\":\"X\",\"pid\":1,\"tid\":" << interval.thread_id << ",\"ts\":"
        << static_cast<double>(interval.start_ns) / 1000.0 << ",\"dur\":"
        << static_cast<double>(interval.duration_ns) / 1000.0 << '}';
  }
  if (!counter_snapshot.empty()) {
    out << ",{\"name\":\"nexus_counters\",\"ph\":\"C\",\"pid\":1,\"tid\":0,\"ts\":0,\"args\":{";
    bool first_counter = true;
    for (const auto& [name, values] : counter_snapshot) {
      if (!first_counter) {
        out << ',';
      }
      first_counter = false;
      write_json_string(out, name + "_flops");
      out << ':' << values.flops << ',';
      write_json_string(out, name + "_bytes");
      out << ':' << values.bytes << ',';
      write_json_string(out, name + "_calls");
      out << ':' << values.calls;
    }
    out << "}}";
  }
  out << "],\"originNs\":" << origin_ns << ",\"displayTimeUnit\":\"ns\"}";
  return out.str();
}

bool dump_chrome_trace(std::string_view path) {
  std::ofstream file(std::string(path), std::ios::binary | std::ios::trunc);
  if (!file) {
    return false;
  }
  file << chrome_trace_json();
  file.flush();
  return static_cast<bool>(file);
}

}  // namespace prof
}  // namespace nexus
