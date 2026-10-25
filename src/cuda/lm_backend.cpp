/**
 * @file lm_backend.cpp
 * @brief FAZ 9: GptModel -> train::LmBackend (dil modeli egitmeni v2 GPU arka ucu).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/lm_backend.hpp"

#include "nexus/core/error.hpp"

#include <cmath>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cstring>
#include <stdexcept>
#include <string>

namespace nexus {
namespace cuda {
namespace {

GptConfig configure(GptConfig config, const train::LmBackendRequest& r) {
  if (!r.policy.use_cuda()) {
    throw std::invalid_argument("cuda::gpt_backend: execution_policy(ExecutionPolicy::cuda(i)) gerekli");
  }
  switch (r.precision) {
    case train::Precision::F32:
      config.compute_dtype = DType::F32;
      break;
    case train::Precision::BF16:
      config.compute_dtype = DType::BF16;
      break;
    case train::Precision::F16:
      config.compute_dtype = DType::F16;
      break;
  }
  switch (r.master) {
    case train::MasterWeightsPolicy::Fp32:
      config.master = MasterWeights::Fp32Master;
      break;
    case train::MasterWeightsPolicy::None:
      config.master = MasterWeights::NoMaster;
      break;
    case train::MasterWeightsPolicy::CastPerForward:
      config.master = MasterWeights::CastPerForward;
      break;
  }
  switch (r.recompute) {
    case train::RecomputePolicy::None:
      config.recompute = Recompute::None;
      break;
    case train::RecomputePolicy::Selective:
      config.recompute = Recompute::Selective;
      break;
    case train::RecomputePolicy::Full:
      config.recompute = Recompute::Full;
      break;
  }
  config.deterministic = r.deterministic;
  config.deterministic_attention = r.deterministic;
  if (config.optimizer == OptimizerKind::AdamW8bitFused) {
    throw std::invalid_argument(
        "cuda::gpt_backend: AdamW8bitFused guncellemeyi geri geciste yapar; v2 egitmende (biriktirme, mikro "
        "adim agirligi) desteklenmez — GptTrainer kullanin");
  }
  if (config.optimizer == OptimizerKind::ZeroOffload && r.accum_steps > 1) {
    throw std::invalid_argument("cuda::gpt_backend: ZeroOffload ile gradyan biriktirme desteklenmez");
  }
  return config;
}

}  // namespace

namespace {
// NVML azami SM saati (nvidia-smi clocks.max.sm). Surucu kutuphanesi calisma aninda yuklenir: derleme/baglama
// bagimliligi yok. Bulunamazsa 0 doner ve cagiran cudaDevAttrClockRate'e gecer (kaynak raporlanir).
unsigned nvml_max_sm_clock_mhz(int device) {
  using Init = int (*)();
  using ByPci = int (*)(const char*, void**);
  using MaxClock = int (*)(void*, int, unsigned*);
  using Shutdown = int (*)();
#if defined(_WIN32)
  HMODULE lib = LoadLibraryA("nvml.dll");
  auto sym = [&](const char* n) { return lib != nullptr ? reinterpret_cast<void*>(GetProcAddress(lib, n)) : nullptr; };
#else
  void* lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  auto sym = [&](const char* n) { return lib != nullptr ? dlsym(lib, n) : nullptr; };
#endif
  auto init = reinterpret_cast<Init>(sym("nvmlInit_v2"));
  auto by_pci = reinterpret_cast<ByPci>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
  auto max_clock = reinterpret_cast<MaxClock>(sym("nvmlDeviceGetMaxClockInfo"));
  auto shutdown = reinterpret_cast<Shutdown>(sym("nvmlShutdown"));
  unsigned mhz = 0;
  char bus[32] = {};
  if (init != nullptr && by_pci != nullptr && max_clock != nullptr && shutdown != nullptr &&
      cudaDeviceGetPCIBusId(bus, sizeof(bus), device) == cudaSuccess && init() == 0) {
    void* handle = nullptr;
    constexpr int kNvmlClockSm = 1;
    if (by_pci(bus, &handle) != 0 || max_clock(handle, kNvmlClockSm, &mhz) != 0) {
      mhz = 0;
    }
    shutdown();
  }
#if defined(_WIN32)
  if (lib != nullptr) FreeLibrary(lib);
#else
  if (lib != nullptr) dlclose(lib);
#endif
  return mhz;
}
}  // namespace

double tensor_core_peak_flops(int device, DType compute, double* clock_mhz, const char** clock_source) {
  int major = 0;
  int minor = 0;
  int sms = 0;
  int clock_khz = 0;
  NEXUS_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  NEXUS_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  NEXUS_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  NEXUS_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
  const unsigned nvml = nvml_max_sm_clock_mhz(device);
  if (nvml != 0) {
    clock_khz = static_cast<int>(nvml) * 1000;
  }
  if (clock_source != nullptr) {
    *clock_source = nvml != 0 ? "NVML azami SM saati" : "cudaDevAttrClockRate (NVML yok)";
  }
  if (clock_mhz != nullptr) {
    *clock_mhz = clock_khz / 1000.0;
  }
  const bool narrow = compute == DType::BF16 || compute == DType::F16;
  double per_sm_clock = 0.0;  // dense FLOP / saat / SM
  if (major == 8 && (minor == 6 || minor == 9)) {
    per_sm_clock = narrow ? 512.0 : 256.0;  // GA10x/AD10x GeForce: FP32 biriktirmeli yari hiz
  } else if (major == 8 && minor == 0) {
    per_sm_clock = narrow ? 1024.0 : 512.0;  // A100
  }
  return per_sm_clock * sms * clock_khz * 1.0e3;
}

train::BackendFactory gpt_backend(GptConfig config) {
  return [config](const train::LmBackendRequest& request) -> std::unique_ptr<train::LmBackend> {
    return std::make_unique<GptLmBackend>(config, request);
  };
}

GptLmBackend::GptLmBackend(GptConfig config, const train::LmBackendRequest& request)
    : device_(request.policy.device_index()) {
  const GptConfig c = configure(std::move(config), request);
  if (request.accum_steps > kMaxAccum) {
    throw std::invalid_argument("cuda::gpt_backend: biriktirme adimi en fazla 4096");
  }
  NEXUS_CUDA_CHECK(cudaSetDevice(device_));
  stream_ = Stream::create(device_);
  model_ = std::make_unique<GptModel>(c, request.micro_batch, request.seq_len, request.seed, stream_);
  if (c.optimizer == OptimizerKind::AdamW || c.optimizer == OptimizerKind::AdamW8bit) {
    model_->set_nonfinite_guard(true);
  }
  tokens_per_micro_ = static_cast<std::size_t>(request.micro_batch) * request.seq_len;
  for (int i = 0; i < 2; ++i) {
    pinned_tokens_[i] = static_cast<std::int32_t*>(pinned_allocator().allocate(tokens_per_micro_ * sizeof(std::int32_t)));
    device_tokens_[i] = DeviceBuffer(tokens_per_micro_ * sizeof(std::int32_t), stream_);
  }
  pinned_losses_ = static_cast<float*>(pinned_allocator().allocate(kMaxAccum * sizeof(float)));
  pinned_tail_ = static_cast<StepTail*>(pinned_allocator().allocate(sizeof(StepTail)));
  peak_flops_ = tensor_core_peak_flops(device_, c.compute_dtype);
}

GptLmBackend::~GptLmBackend() {
  try {
    stream_.synchronize();
  } catch (...) {
  }
  for (std::int32_t* p : pinned_tokens_) {
    pinned_allocator().deallocate(p);
  }
  pinned_allocator().deallocate(pinned_losses_);
  pinned_allocator().deallocate(pinned_tail_);
  device_tokens_[0].reset();
  device_tokens_[1].reset();
  model_.reset();
  stream_.destroy();
}

std::string GptLmBackend::device() const { return "cuda:" + std::to_string(device_); }

train::LmShape GptLmBackend::shape() const {
  train::LmShape s;
  for (const GptParamInfo& p : model_->parameter_info()) {
    s.params += static_cast<std::int64_t>(p.numel);
    if (p.name != "wpe") {
      s.params_no_pos += static_cast<std::int64_t>(p.numel);
    }
  }
  const GptConfig& c = model_->config();
  s.layers = c.layers;
  s.heads = c.heads;
  s.head_dim = c.dim / c.heads;
  return s;
}

std::vector<train::LmParamInfo> GptLmBackend::parameters() const {
  std::vector<train::LmParamInfo> out;
  const int L = model_->config().layers;
  for (const GptParamInfo& p : model_->parameter_info()) {
    train::LmParamInfo info;
    info.name = p.name;
    info.numel = p.numel;
    // Katmanlarca yigilmis tensorlerde ([L, ...]) katman dilimi rutbesi: ln1_w [L, C] -> 1, qkv_w -> 2.
    const bool layered = p.shape.size() >= 2 && p.shape.front() == L && p.name != "wte" && p.name != "wpe";
    info.rank = static_cast<int>(p.shape.size()) - (layered ? 1 : 0);
    info.embedding = p.name == "wte" || p.name == "wpe";
    out.push_back(std::move(info));
  }
  return out;
}

void GptLmBackend::set_param_scales(const std::vector<float>& weight_decay, const std::vector<float>& lr) {
  if (weight_decay.size() != model_->parameter_info().size() || lr.size() != weight_decay.size()) {
    throw std::invalid_argument("cuda::gpt_backend: carpan sayisi parametre sayisina esit olmali");
  }
  std::vector<GptParamScale> scales(weight_decay.size());
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = GptParamScale{weight_decay[i], lr[i]};
  }
  model_->set_param_scales(scales);
}

void GptLmBackend::reshape(int micro_batch, int seq_len) {
  if (micro_batch != model_->batch() || seq_len != model_->time()) {
    throw std::invalid_argument("cuda::gpt_backend: GptModel tamponlari sabit boyutludur (reshape yok)");
  }
}

const std::int32_t* GptLmBackend::upload(const std::int32_t* tokens) {
  const int s = slot_;
  slot_ ^= 1;
  if (copy_pending_[s]) {
    copy_done_[s].synchronize();  // bu pinned yuvanin onceki H2D'si bitmeden ustune yazma
  }
  std::memcpy(pinned_tokens_[s], tokens, tokens_per_micro_ * sizeof(std::int32_t));
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(device_tokens_[s].data(), pinned_tokens_[s], tokens_per_micro_ * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream_.handle()));
  copy_done_[s].record(stream_);
  copy_pending_[s] = true;
  return device_tokens_[s].as<std::int32_t>();
}

void GptLmBackend::micro_step(const std::int32_t* tokens, float grad_weight, int index, int count) {
  if (count > kMaxAccum || index < 0 || index >= count) {
    throw std::invalid_argument("cuda::gpt_backend: mikro adim indeksi gecersiz");
  }
  const std::int32_t* d = upload(tokens);
  model_->forward_backward(d, grad_weight, index > 0);  // ilk mikro adim sifirlar, sonrakiler ekler
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(pinned_losses_ + index, model_->device_loss(), sizeof(float), cudaMemcpyDeviceToHost,
                                   stream_.handle()));
  count_ = count;
}

void GptLmBackend::optimizer_step(const train::AdamW& opt, double lr, double clip) {
  DeviceAdamWConfig c;
  c.lr = lr;
  c.beta1 = opt.beta1;
  c.beta2 = opt.beta2;
  c.eps = opt.eps;
  c.weight_decay = opt.weight_decay;
  c.max_grad_norm = clip;
  model_->optimizer_step(c);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(&pinned_tail_->grad_norm, model_->device_grad_norm(), sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_.handle()));
  // Kayip olcegi durumu (found_inf, skipped_steps): model ozel tamponu; read_loss_scale senkron oldugu icin
  // finish_step'te okunur (adim zaten bitmis olur).
  step_done_.record(stream_);
}

train::LmStepResult GptLmBackend::finish_step() {
  step_done_.synchronize();
  train::LmStepResult r;
  r.losses.assign(pinned_losses_, pinned_losses_ + count_);
  r.grad_norm = pinned_tail_->grad_norm;
  const DeviceLossScaleState st = model_->read_loss_scale();
  r.skipped = st.found_inf != 0;
  r.skipped_total = st.skipped_steps;
  r.loss_scale = st.scale;
  return r;
}

double GptLmBackend::eval_loss(const std::int32_t* tokens) {
  model_->forward(upload(tokens));
  return model_->read_loss();
}

std::vector<float> GptLmBackend::token_losses(const std::int32_t* tokens) {
  model_->forward(upload(tokens));
  std::vector<float> out(tokens_per_micro_);
  NEXUS_CUDA_CHECK(cudaMemcpyAsync(out.data(), model_->device_token_losses(), out.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_.handle()));
  stream_.synchronize();
  return out;
}

train::LmNonFinite GptLmBackend::diagnose(const std::int32_t* tokens) {
  const GptNonFiniteReport rep = model_->diagnose_nonfinite(upload(tokens));
  train::LmNonFinite out;
  out.first_activation = rep.first_activation;
  out.parameters = rep.parameters;
  out.gradients = rep.gradients;
  out.loss = rep.loss;
  return out;
}

std::vector<unsigned char> GptLmBackend::save_state() const {
  const GptStateSnapshot snap = model_->save_state();
  std::vector<unsigned char> out;
  auto put = [&out](const void* p, std::size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    out.insert(out.end(), b, b + n);
  };
  const std::uint64_t count = snap.blobs.size();
  put(&snap.step, sizeof(snap.step));
  put(&count, sizeof(count));
  for (const auto& blob : snap.blobs) {
    const std::uint64_t n = blob.size();
    put(&n, sizeof(n));
    put(blob.data(), blob.size());
  }
  return out;
}

void GptLmBackend::load_state(const std::vector<unsigned char>& blob) {
  std::size_t pos = 0;
  auto get = [&](void* p, std::size_t n) {
    if (pos + n > blob.size()) {
      throw std::runtime_error("cuda::gpt_backend: durum blobu kisa/bozuk");
    }
    std::memcpy(p, blob.data() + pos, n);
    pos += n;
  };
  GptStateSnapshot snap;
  std::uint64_t count = 0;
  get(&snap.step, sizeof(snap.step));
  get(&count, sizeof(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    std::uint64_t n = 0;
    get(&n, sizeof(n));
    std::vector<unsigned char> b(static_cast<std::size_t>(n));
    get(b.data(), b.size());
    snap.blobs.push_back(std::move(b));
  }
  model_->load_state(snap);
}

std::vector<train::LmNormEntry> GptLmBackend::layer_norms() {
  std::vector<train::LmNormEntry> out;
  for (const GptModel::NormEntry& e : model_->layer_norms()) {
    out.push_back(train::LmNormEntry{e.name, e.weight, e.grad});
  }
  return out;
}

std::uint64_t GptLmBackend::peak_memory_bytes() const { return device_allocator().stats(device_).peak_reserved_bytes; }

void GptLmBackend::reset_peak_memory() { device_allocator().reset_peak_stats(); }

}  // namespace cuda
}  // namespace nexus
