/**
 * @file checkpoint_manager.cpp
 * @brief NXT1 dizin biçimi: model.nxm1 + optimizer.nxo + trainer.nxs.
 */

#include "nexus_train/checkpoint/checkpoint_manager.hpp"

#include "nexus_model/core/serialization.hpp"
#include "nexus_train/core/tensor_batch.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace nexus_train {
namespace {

void write_u32(std::ostream& out, std::uint32_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_u64(std::ostream& out, std::uint64_t value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_f64(std::ostream& out, double value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }

std::uint32_t read_u32(std::istream& in) {
  std::uint32_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

std::uint64_t read_u64(std::istream& in) {
  std::uint64_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

double read_f64(std::istream& in) {
  double value = 0.0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  return value;
}

void write_bytes(std::ostream& out, const std::string& bytes) {
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    fail("checkpoint name is too long");
  }
  write_u32(out, static_cast<std::uint32_t>(bytes.size()));
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string read_bytes(std::istream& in) {
  const std::uint32_t n = read_u32(in);
  std::string bytes(n, '\0');
  if (n > 0) {
    in.read(bytes.data(), static_cast<std::streamsize>(n));
  }
  return bytes;
}

void write_optimizer(const std::filesystem::path& path, const nexus_optim::OptimizerState& state) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    fail("cannot open optimizer checkpoint for writing");
  }
  out.write("NXO1", 4);
  write_bytes(out, state.algorithm);
  write_u64(out, state.step);
  write_u32(out, static_cast<std::uint32_t>(state.group_learning_rates.size()));
  for (double lr : state.group_learning_rates) write_f64(out, lr);
  write_u32(out, static_cast<std::uint32_t>(state.blobs.size()));
  for (const auto& blob : state.blobs) {
    write_u64(out, static_cast<std::uint64_t>(blob.size()));
    if (!blob.empty()) {
      out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
    }
  }
  write_u32(out, static_cast<std::uint32_t>(state.scalars.size()));
  for (double value : state.scalars) write_f64(out, value);
  if (!out) {
    fail("failed while writing optimizer checkpoint");
  }
}

nexus_optim::OptimizerState read_optimizer(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    fail("cannot open optimizer checkpoint for reading");
  }
  char magic[4] = {};
  in.read(magic, 4);
  if (std::strncmp(magic, "NXO1", 4) != 0) {
    fail("optimizer checkpoint magic mismatch");
  }
  nexus_optim::OptimizerState state;
  state.algorithm = read_bytes(in);
  state.step = read_u64(in);
  const std::uint32_t n_lr = read_u32(in);
  state.group_learning_rates.resize(n_lr);
  for (std::uint32_t i = 0; i < n_lr; ++i) state.group_learning_rates[i] = read_f64(in);
  const std::uint32_t n_blobs = read_u32(in);
  state.blobs.resize(n_blobs);
  for (std::uint32_t i = 0; i < n_blobs; ++i) {
    const auto n = static_cast<std::size_t>(read_u64(in));
    state.blobs[i].resize(n);
    if (n > 0) {
      in.read(reinterpret_cast<char*>(state.blobs[i].data()), static_cast<std::streamsize>(n));
    }
  }
  const std::uint32_t n_scalars = read_u32(in);
  state.scalars.resize(n_scalars);
  for (std::uint32_t i = 0; i < n_scalars; ++i) state.scalars[i] = read_f64(in);
  if (!in) {
    fail("truncated optimizer checkpoint");
  }
  return state;
}

void write_trainer(const std::filesystem::path& path, const TrainingState& state) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    fail("cannot open trainer checkpoint for writing");
  }
  out.write("NXS1", 4);
  write_u32(out, static_cast<std::uint32_t>(state.epoch));
  write_u32(out, static_cast<std::uint32_t>(state.global_step));
  write_f64(out, state.current_loss);
  write_f64(out, state.best_metric);
  const std::uint8_t stop = state.should_stop ? 1 : 0;
  out.write(reinterpret_cast<const char*>(&stop), 1);
  write_u32(out, static_cast<std::uint32_t>(state.metrics.size()));
  for (const auto& [name, value] : state.metrics) {
    write_bytes(out, name);
    write_f64(out, value);
  }
  if (!out) {
    fail("failed while writing trainer checkpoint");
  }
}

void read_trainer(const std::filesystem::path& path, TrainingState& state) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    fail("cannot open trainer checkpoint for reading");
  }
  char magic[4] = {};
  in.read(magic, 4);
  if (std::strncmp(magic, "NXS1", 4) != 0) {
    fail("trainer checkpoint magic mismatch");
  }
  state.epoch = static_cast<int>(read_u32(in));
  state.global_step = static_cast<int>(read_u32(in));
  state.current_loss = read_f64(in);
  state.best_metric = read_f64(in);
  std::uint8_t stop = 0;
  in.read(reinterpret_cast<char*>(&stop), 1);
  state.should_stop = stop != 0;
  state.metrics.clear();
  const std::uint32_t n = read_u32(in);
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::string name = read_bytes(in);
    state.metrics.emplace(name, read_f64(in));
  }
  if (!in) {
    fail("truncated trainer checkpoint");
  }
}

}  // namespace

CheckpointManager::CheckpointManager(std::string checkpoint_dir, std::string monitor_metric, bool monitor_mode_min)
    : root_(std::move(checkpoint_dir)),
      monitor_metric_(std::move(monitor_metric)),
      monitor_mode_min_(monitor_mode_min),
      best_value_(monitor_mode_min ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity()) {
  std::filesystem::create_directories(root_);
  writer_ = std::thread([this] { writer_main(); });
}

CheckpointManager::~CheckpointManager() {
  try {
    flush();
  } catch (...) {
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  ready_.notify_all();
  if (writer_.joinable()) {
    writer_.join();
  }
}

std::shared_ptr<const CheckpointSnapshot> CheckpointManager::capture(const nexus_model::Module& model,
                                                                    const OptimizerHandle& optimizer,
                                                                    const TrainingState& state) const {
  auto snapshot = std::make_shared<CheckpointSnapshot>();
  snapshot->model = model.state_dict();
  snapshot->optimizer = optimizer.state_dict();
  snapshot->state = state;
  return snapshot;
}

void CheckpointManager::publish(std::shared_ptr<const CheckpointSnapshot> snapshot, const std::filesystem::path& directory,
                                bool is_async) {
  if (!snapshot) {
    fail("null checkpoint snapshot");
  }
  {
    std::unique_lock<std::mutex> lock(mutex_);
    drained_.wait(lock, [&] { return queue_.size() < 2; });
    if (error_) {
      const std::exception_ptr error = error_;
      error_ = nullptr;
      std::rethrow_exception(error);
    }
    queue_.push(Job{std::move(snapshot), directory});
  }
  ready_.notify_one();
  if (!is_async) {
    flush();
  }
}

void CheckpointManager::save(const nexus_model::Module& model, const OptimizerHandle& optimizer, const TrainingState& state,
                             bool is_async) {
  publish(capture(model, optimizer, state), last_directory(), is_async);
}

void CheckpointManager::save(const nexus_model::Module& model, const OptimizerHandle& optimizer, const TrainingState& state,
                             const std::filesystem::path& directory, bool is_async) {
  publish(capture(model, optimizer, state), directory, is_async);
}

void CheckpointManager::load(nexus_model::Module& model, OptimizerHandle& optimizer, TrainingState& state,
                             const std::string& path) {
  flush();
  read_into(std::filesystem::path(path), model, optimizer, state);
}

bool CheckpointManager::is_best(double current_metric) {
  if (!std::isfinite(current_metric)) {
    return false;
  }
  const bool better = monitor_mode_min_ ? current_metric < best_value_ : current_metric > best_value_;
  if (better) {
    best_value_ = current_metric;
  }
  return better;
}

void CheckpointManager::flush() {
  std::unique_lock<std::mutex> lock(mutex_);
  drained_.wait(lock, [&] { return queue_.empty() && !busy_; });
  if (error_) {
    const std::exception_ptr error = error_;
    error_ = nullptr;
    std::rethrow_exception(error);
  }
}

std::filesystem::path CheckpointManager::best_directory() const { return root_ / "best"; }

std::filesystem::path CheckpointManager::last_directory() const { return root_ / "last"; }

void CheckpointManager::writer_main() {
  for (;;) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (queue_.empty()) {
        return;
      }
      job = std::move(queue_.front());
      queue_.pop();
      busy_ = true;
    }
    try {
      write_job(job);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      error_ = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      busy_ = false;
    }
    drained_.notify_all();
  }
}

void CheckpointManager::write_job(const Job& job) {
  std::filesystem::create_directories(job.directory);
  nexus_model::save_state_dict(job.snapshot->model, (job.directory / "model.nxm1").string());
  write_optimizer(job.directory / "optimizer.nxo", job.snapshot->optimizer);
  write_trainer(job.directory / "trainer.nxs", job.snapshot->state);
}

void CheckpointManager::read_into(const std::filesystem::path& directory, nexus_model::Module& model,
                                  OptimizerHandle& optimizer, TrainingState& state) {
  const auto model_state = nexus_model::load_state_dict_file((directory / "model.nxm1").string());
  model.load_state_dict(model_state, true);
  optimizer.load_state_dict(read_optimizer(directory / "optimizer.nxo"));
  read_trainer(directory / "trainer.nxs", state);
}

}  // namespace nexus_train
