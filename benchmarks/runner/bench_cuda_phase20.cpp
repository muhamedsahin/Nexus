/** FAZ 20: real-token scale validation using the existing CUDA model/trainer.
 * Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
 * öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
 */
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/gemm.hpp"
#include "nexus/cuda/lm_backend.hpp"
#include "nexus/train/lm_trainer.hpp"
#include "nexusdata/io/mapped_file.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace nc = nexus::cuda;
namespace nt = nexus::train;
namespace fs = std::filesystem;
namespace {
// Read exactly one batch at a time. No corpus-sized allocation; cursor participates in checkpoints.
class BinarySource final : public nt::TokenBatchSource {
 public:
  BinarySource(const std::string& path, int vocab) : file_(path, std::ios::binary), vocab_(vocab) {
    if (!file_) throw std::runtime_error("cannot open tokens: " + path);
    bytes_ = fs::file_size(path);
    if (!bytes_ || bytes_ % sizeof(std::int32_t)) throw std::runtime_error("invalid int32 token file");
    const nexusdata::MappedFile mapped(path);
    digest_ = nexus::ckpt::sha256_hex(mapped.data(), mapped.size());
  }
  void next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) override {
    const auto count = static_cast<std::uint64_t>(rows) * time;
    if (count > bytes_ / 4 || cursor_ > bytes_ / 4 - count) throw std::runtime_error("token stream exhausted");
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(cursor_ * 4));
    file_.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count * 4));
    if (!file_) throw std::runtime_error("short token read");
    for (std::uint64_t i = 0; i < count; ++i)
      if (dst[i] < 0 || dst[i] >= vocab_) throw std::runtime_error("token outside model vocabulary");
    for (int r = 0; r < rows; ++r) ids[r] = cursor_ + static_cast<std::uint64_t>(r) * time;
    cursor_ += count;
  }
  std::string state() const override { return digest_ + ":" + std::to_string(cursor_); }
  void restore(const std::string& value) override {
    if (value.substr(0, 65) != digest_ + ":") throw std::runtime_error("checkpoint token SHA256 mismatch");
    const auto cursor = value.substr(65);
    std::size_t used = 0;
    const auto n = std::stoull(cursor, &used);
    if (used != cursor.size() || n > bytes_ / 4) throw std::runtime_error("invalid token cursor");
    cursor_ = n;
  }
 private:
  std::ifstream file_;
  int vocab_;
  std::string digest_;
  std::uint64_t cursor_ = 0, bytes_ = 0;
};

void dump_init(nc::GptModel& model, const fs::path& path) {
  const auto values = model.download_parameters();
  std::ofstream bin(path, std::ios::binary);
  bin.exceptions(std::ios::badbit | std::ios::failbit);
  bin.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * 4));
  std::ofstream meta(path.string() + ".json");
  meta.exceptions(std::ios::badbit | std::ios::failbit);
  const auto& c = model.config();
  meta << "{\"layers\":" << c.layers << ",\"heads\":" << c.heads << ",\"dim\":" << c.dim
       << ",\"vocab\":" << c.vocab << ",\"max_seq\":" << c.max_seq << ",\"params\":[";
  bool first = true;
  for (const auto& p : model.parameter_info()) {
    meta << (first ? "" : ",") << "{\"name\":\"" << p.name << "\",\"offset\":" << p.offset << ",\"shape\":[";
    for (std::size_t i = 0; i < p.shape.size(); ++i) meta << (i ? "," : "") << p.shape[i];
    meta << "]}";
    first = false;
  }
  meta << "]}\n";
}

int run(int argc, char** argv) {
  std::map<std::string, std::string> opts;
  const std::set<std::string> allowed{"model", "tokens", "out", "steps", "stop-after", "seq", "micro",
      "precision", "optimizer", "lr", "warmup", "seed", "dump-init", "save", "resume",
      "checkpoint-dir", "checkpoint-every"};
  for (int i = 1; i < argc; i += 2) {
    const std::string key = argv[i];
    if (i + 1 >= argc || key.rfind("--", 0) || !allowed.count(key.substr(2)) || opts.count(key.substr(2)))
      throw std::invalid_argument("unknown, duplicate or incomplete option: " + key);
    opts[key.substr(2)] = argv[i + 1];
  }
  auto str = [&](const std::string& key, const std::string& fallback) { return opts.count(key) ? opts.at(key) : fallback; };
  auto number = [&](const std::string& key, int fallback, int minimum, int maximum) {
    const auto value = str(key, std::to_string(fallback));
    std::size_t end = 0;
    const auto n = std::stoll(value, &end);
    if (end != value.size() || n < minimum || n > maximum) throw std::invalid_argument("invalid --" + key);
    return static_cast<int>(n);
  };
  const std::string name = str("model", "124m"), precision = str("precision", "bf16");
  const std::string optimizer = str("optimizer", "adamw"), output = str("out", "");
  if (output.empty() || str("tokens", "").empty()) throw std::invalid_argument("--tokens and --out required");
  if (fs::exists(output)) throw std::invalid_argument("output already exists; preserve evidence and use a new path");
  if (opts.count("dump-init") && (fs::exists(opts.at("dump-init")) || fs::exists(opts.at("dump-init") + ".json") ||
      fs::weakly_canonical(opts.at("dump-init")) == fs::weakly_canonical(output)))
    throw std::invalid_argument("initial-weight export would overwrite an existing file or the run output");
  const int steps = number("steps", 100, 1, 10000000), stop = number("stop-after", steps, 1, steps);
  const int time = number("seq", 1024, 2, 1024), batch = number("micro", 1, 1, 64);
  const int warmup = number("warmup", 10, 0, steps - 1), seed = number("seed", 1337, 0, 2147483647);
  const int checkpoint_every = number("checkpoint-every", 0, 0, steps);
  if (checkpoint_every && !opts.count("checkpoint-dir")) throw std::invalid_argument("checkpoint-every requires checkpoint-dir");
  std::size_t end = 0;
  const auto lr_text = str("lr", "0.0006");
  const double lr = std::stod(lr_text, &end);
  if (end != lr_text.size() || !std::isfinite(lr) || lr <= 0 || lr > 1) throw std::invalid_argument("invalid --lr");
  nc::GptConfig config;
  if (name == "350m") { config.layers = 24; config.dim = 1024; config.heads = 16; }
  else if (name == "1.3b") { config.layers = 24; config.dim = 2048; config.heads = 16; }
  else if (name == "smoke") { config.layers = 2; config.dim = 64; config.heads = 4; config.vocab = 128; }
  else if (name != "124m") throw std::invalid_argument("model: smoke | 124m | 350m | 1.3b");
  if (precision != "f32" && precision != "bf16") throw std::invalid_argument("precision: f32 | bf16");
  if (optimizer != "adamw" && optimizer != "fused8" && optimizer != "offload")
    throw std::invalid_argument("optimizer: adamw | fused8 | offload");
  const bool fused = optimizer == "fused8";
  if (fused && (precision != "bf16" || opts.count("save") || opts.count("resume") || opts.count("dump-init") || opts.count("checkpoint-dir")))
    throw std::invalid_argument("fused8 requires BF16; checkpoint/parity export use the standard trainer");
  if (nc::device_count() == 0) return 77;
  nc::set_float32_matmul_precision(nc::MatmulPrecision::Highest);
  config.compute_dtype = precision == "f32" ? nexus::DType::F32 : nexus::DType::BF16;
  config.recompute = nc::Recompute::Full;
  config.ce_chunk_rows = 256;
  config.deterministic = config.deterministic_attention = true;
  config.optimizer = fused ? nc::OptimizerKind::AdamW8bitFused : optimizer == "offload" ? nc::OptimizerKind::ZeroOffload : nc::OptimizerKind::AdamW;
  if (fused) { config.master = nc::MasterWeights::NoMaster; config.adam8_embedding_fp32 = false; }
  BinarySource source(str("tokens", ""), config.vocab);
  std::ofstream out(output);
  out.exceptions(std::ios::badbit | std::ios::failbit);
  out << std::setprecision(17);
  auto& allocator = nc::caching_device_allocator();
  allocator.empty_cache();
  allocator.reset_peak_stats();
  auto record = [&](int step, double loss, double ms, double learning_rate) {
    if (!std::isfinite(loss) || !std::isfinite(ms) || ms <= 0) throw std::runtime_error("nonfinite loss or invalid timing");
    const auto s = allocator.stats(0);
    out << "{\"type\":\"step\",\"step\":" << step << ",\"loss\":" << loss << ",\"ms\":" << ms
        << ",\"lr\":" << learning_rate << ",\"target_tokens\":" << batch * (time - 1)
        << ",\"peak_allocated_bytes\":" << s.peak_allocated_bytes << ",\"peak_reserved_bytes\":" << s.peak_reserved_bytes
        << ",\"reserved_bytes\":" << s.reserved_bytes << ",\"inactive_split_bytes\":" << s.inactive_split_bytes << "}\n";
    out.flush();
    if (step % 10 == 0) std::cout << "step " << step << " loss " << loss << " ms " << ms << std::endl;
  };
  auto metadata = [&](const nc::GptModel& model) {
    // Strict FP32 is not Tensor Core TF32: leave its peak unknown instead of using a false denominator.
    const double peak = precision == "bf16" ? nc::tensor_core_peak_flops(0, config.compute_dtype) : 0;
    out << "{\"type\":\"config\",\"model\":\"" << name << "\",\"precision\":\"" << precision
        << "\",\"optimizer\":\"" << optimizer << "\",\"parameters\":" << model.num_parameters()
        << ",\"layers\":" << config.layers << ",\"dim\":" << config.dim << ",\"heads\":" << config.heads
        << ",\"seq\":" << time << ",\"micro\":" << batch << ",\"schedule_steps\":" << steps << ",\"seed\":" << seed
        << ",\"peak_flops\":" << peak << ",\"device_state_bytes\":" << model.device_state_bytes()
        << ",\"host_state_bytes\":" << model.host_state_bytes() << ",\"recompute\":\"full\""
        << ",\"token_sha256\":\"" << source.state().substr(0, 64) << "\",\"lr\":" << lr
        << ",\"warmup\":" << warmup << ",\"grad_clip\":\"" << (fused ? "previous-step-global-norm" : "current-step-global-norm") << "\"}\n";
  };
  if (fused) {
    const nc::Stream stream;  // Borrowed CUDA default stream; no owned handle to leak on failure.
    nc::GptModel model(config, batch, time, seed, stream);
    metadata(model);
    nc::DeviceBuffer device(static_cast<std::size_t>(batch) * time * 4, stream);
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(batch) * time);
    std::vector<std::uint64_t> ids(batch);
    const nt::LrSchedule schedule = nt::WarmupCosine{warmup, steps, 0.1};
    nc::DeviceAdamWConfig adam;
    adam.beta2 = 0.95f; adam.weight_decay = 0.1f;
    // Fused layerwise update cannot implement a full-model pre-update norm clip.
    // Do not label this trajectory as standard AdamW parity.
    adam.max_grad_norm = 1.0f;
    for (int i = 0; i < stop; ++i) {
      const auto start = std::chrono::steady_clock::now();
      source.next(tokens.data(), batch, time, ids.data());
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(device.data(), tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice, stream.handle()));
      adam.lr = static_cast<float>(lr * schedule.factor(i));
      model.set_fused_optimizer(adam);
      model.forward_backward(device.as<std::int32_t>());
      model.optimizer_step(adam);
      const double loss = model.read_loss();
      stream.synchronize();
      record(i, loss, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), adam.lr);
    }
  } else {
    nt::NanOptions nan;
    nan.policy = nt::NanPolicy::Stop;
    nt::LanguageTrainerBuilder builder(nc::gpt_backend(config));
    builder
        .execution_policy(nexus::ExecutionPolicy::cuda(0)).data(source)
        .precision(precision == "f32" ? nt::Precision::F32 : nt::Precision::BF16)
        .recompute(nt::RecomputePolicy::Full).deterministic(true).seed(seed)
        .micro_batch(batch).seq_len(time).global_batch_tokens(static_cast<std::int64_t>(batch) * time)
        .optimizer(nt::AdamW{lr, 0.9, 0.95, 1e-8, 0.1}).schedule(nt::WarmupCosine{warmup, steps, 0.1})
        .grad_clip(1.0).nan_policy(nan).log_console(false);
    if (opts.count("save") || opts.count("resume") || opts.count("checkpoint-dir")) {
      if (opts.count("save") && opts.count("resume") && opts.at("save") != opts.at("resume"))
        throw std::invalid_argument("save and resume must use the same checkpoint directory");
      nexus::ckpt::CheckpointOptions checkpoint;
      checkpoint.dir = str("save", str("resume", str("checkpoint-dir", "")));
      if (opts.count("checkpoint-dir") && opts.at("checkpoint-dir") != checkpoint.dir)
        throw std::invalid_argument("checkpoint directories must agree");
      checkpoint.interval = checkpoint_every;
      checkpoint.keep = 2;
      std::ostringstream contract;
      contract << std::setprecision(17) << "phase20-v1 " << name << ' ' << precision << ' ' << optimizer << ' '
               << steps << ' ' << warmup << ' ' << lr << ' ' << seed << ' ' << batch << ' ' << time;
      const fs::path contract_path = fs::path(checkpoint.dir) / "phase20-contract.txt";
      if (opts.count("resume") || fs::exists(contract_path)) {
        std::ifstream in(contract_path);
        const std::string saved((std::istreambuf_iterator<char>(in)), {});
        if (saved != contract.str()) throw std::runtime_error("checkpoint recipe mismatch or missing contract");
      } else {
        fs::create_directories(checkpoint.dir);
        const auto text = contract.str();
        nexus::ckpt::write_file_atomic(contract_path, text.data(), text.size());
      }
      builder.checkpointing(checkpoint);
    }
    auto trainer = builder.build();
    auto& model = dynamic_cast<nc::GptLmBackend&>(trainer->backend()).model();
    metadata(model);
    if (opts.count("resume") && !trainer->resume_latest()) throw std::runtime_error("no completed checkpoint to resume");
    if (opts.count("dump-init")) dump_init(model, opts.at("dump-init"));
    if (trainer->step() >= stop) throw std::invalid_argument("stop-after must exceed restored step");
    while (trainer->step() < stop) {
      const auto summary = trainer->train(trainer->step() + 1);
      if (summary.stopped || trainer->history().back().skipped) throw std::runtime_error("training stopped or skipped a step");
      const auto& r = trainer->history().back();
      record(static_cast<int>(r.step), r.loss, r.step_ms, r.lr);
    }
    if (opts.count("save")) { trainer->save_checkpoint(); trainer->checkpoints()->wait(); }
  }
  return 0;
}
}  // namespace
int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& e) { std::cerr << "phase20: " << e.what() << '\n'; return 1; }
}
