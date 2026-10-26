/**
 * @file lm_trainer.cpp
 * @brief FAZ 9: dil modeli egitmeni v2 — dongu, CPU arka ucu, degerlendirme, teshis, kayit.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/train/lm_trainer.hpp"

#include "nexus/dist/fsdp.hpp"

#include "nexus/fused/training.hpp"
#include "nexus/nn/modules.hpp"
#include "nexus/train/grad_utils.hpp"
#include "nexus/train/loggers.hpp"
#include "nexus_optim/algorithms/adam.hpp"
#include "nexus_train/logging/csv_logger.hpp"
#include "nexus_train/logging/json_logger.hpp"
#include "nexusdata/text/token_shard.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <typeinfo>
#include <utility>

namespace nexus {
namespace train {
namespace {

constexpr std::int32_t kIgnoreIndex = -100;

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---- kucuk ikili yazici/okuyucu (kayit dosyasi ve CPU durum blobu) ----
class Writer {
 public:
  template <typename T>
  void pod(const T& v) {
    const auto* p = reinterpret_cast<const unsigned char*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
  }
  void bytes(const void* data, std::size_t n) {
    pod<std::uint64_t>(n);
    const auto* p = static_cast<const unsigned char*>(data);
    out.insert(out.end(), p, p + n);
  }
  void str(const std::string& s) { bytes(s.data(), s.size()); }
  std::vector<unsigned char> out;
};

class Reader {
 public:
  explicit Reader(const std::vector<unsigned char>& in) : in_(in) {}
  template <typename T>
  T pod() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, in_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }
  std::vector<unsigned char> bytes() {
    const auto n = static_cast<std::size_t>(pod<std::uint64_t>());
    need(n);
    std::vector<unsigned char> v(in_.begin() + static_cast<std::ptrdiff_t>(pos_), in_.begin() + static_cast<std::ptrdiff_t>(pos_ + n));
    pos_ += n;
    return v;
  }
  std::string str() {
    const std::vector<unsigned char> b = bytes();
    return {b.begin(), b.end()};
  }

 private:
  void need(std::size_t n) const {
    if (pos_ + n > in_.size()) {
      throw std::runtime_error("Nexus LM trainer: durum blobu kisa/bozuk");
    }
  }
  const std::vector<unsigned char>& in_;
  std::size_t pos_ = 0;
};

std::string short_type_name(const nn::Module& m) {
  std::string name = typeid(m).name();
  const std::size_t colon = name.rfind("::");
  if (colon != std::string::npos) {
    name = name.substr(colon + 2);
  }
  const std::size_t space = name.rfind(' ');
  if (space != std::string::npos) {
    name = name.substr(space + 1);
  }
  return name;
}

bool finite_tensor(const nexus_model::Tensor& t) {
  const float* p = t.data();
  for (std::size_t i = 0; i < t.numel(); ++i) {
    if (!std::isfinite(p[i])) {
      return false;
    }
  }
  return true;
}

double l2(const nexus_model::Tensor& t) {
  double acc = 0.0;
  const float* p = t.data();
  for (std::size_t i = 0; i < t.numel(); ++i) {
    acc += static_cast<double>(p[i]) * p[i];
  }
  return std::sqrt(acc);
}

// ==========================================================================================
// CPU arka ucu: nn::Module + NexusOptim AdamW
// ==========================================================================================

class CpuModuleBackend final : public LmBackend {
 public:
  CpuModuleBackend(nn::Module& model, const LmBackendRequest& request)
      : model_(&model), batch_(request.micro_batch), time_(request.seq_len) {
    if (request.compile) {
      throw std::invalid_argument(
          "Nexus LM trainer: nn::Module arka ucu trainer.compile desteklemiyor (lm::LanguageModel icin nexus::compile::compile)");
    }
    if (request.policy.use_cuda()) {
      throw std::invalid_argument(
          "Nexus LM trainer: nn::Module CPU arka ucudur; CUDA icin cuda::gpt_backend(config) kullanin");
    }
    if (request.precision != Precision::F32) {
      throw std::invalid_argument("Nexus LM trainer: CPU arka ucu yalniz FP32 (BF16/F16 GPU arka ucunda)");
    }
    if (request.recompute != RecomputePolicy::None) {
      throw std::invalid_argument("Nexus LM trainer: CPU nn::Module arka ucunda yeniden hesaplama yok");
    }
    params_ = model.parameters();
    if (params_.empty()) {
      throw std::invalid_argument("Nexus LM trainer: modelin parametresi yok");
    }
    // Adlar: Sequential ise "<katman>.<Tur>.<j>", degilse "p<i>".
    if (const auto* seq = dynamic_cast<const nn::Sequential*>(&model)) {
      for (std::size_t i = 0; i < seq->size(); ++i) {
        const nn::Module& layer = *seq->layer(i);
        const bool emb = dynamic_cast<const nn::Embedding*>(&layer) != nullptr;
        const bool pos = dynamic_cast<const nn::PositionalEncoding*>(&layer) != nullptr;
        const std::vector<nexus_model::Parameter*> owned = layer.parameters();
        for (std::size_t j = 0; j < owned.size(); ++j) {
          names_.push_back(std::to_string(i) + "." + short_type_name(layer) + "." + std::to_string(j));
          embedding_.push_back(emb || pos);
          positional_.push_back(pos);
        }
      }
    }
    if (names_.size() != params_.size()) {
      names_.clear();
      embedding_.assign(params_.size(), false);
      positional_.assign(params_.size(), false);
      for (std::size_t i = 0; i < params_.size(); ++i) {
        names_.push_back("p" + std::to_string(i));
      }
    }
    wd_.assign(params_.size(), 1.0f);
    lr_.assign(params_.size(), 1.0f);
  }

  std::string device() const override { return "cpu"; }

  LmShape shape() const override {
    LmShape s;
    for (std::size_t i = 0; i < params_.size(); ++i) {
      s.params += static_cast<std::int64_t>(params_[i]->data.numel());
      if (!positional_[i]) {
        s.params_no_pos += static_cast<std::int64_t>(params_[i]->data.numel());
      }
    }
    return s;  // katman/kafa sayisi nn::Module'den okunmaz: MFU dikkat terimi 0
  }

  std::vector<LmParamInfo> parameters() const override {
    std::vector<LmParamInfo> out;
    for (std::size_t i = 0; i < params_.size(); ++i) {
      out.push_back(LmParamInfo{names_[i], params_[i]->data.numel(), static_cast<int>(params_[i]->data.rank()),
                                embedding_[i]});
    }
    return out;
  }

  void set_param_scales(const std::vector<float>& wd, const std::vector<float>& lr) override {
    if (wd.size() != params_.size() || lr.size() != params_.size()) {
      throw std::invalid_argument("Nexus LM trainer: carpan sayisi parametre sayisina esit olmali");
    }
    if (impl_) {
      throw std::logic_error("Nexus LM trainer: parametre gruplari ilk adimdan once kurulur");
    }
    wd_ = wd;
    lr_ = lr;
  }

  int micro_batch() const override { return batch_; }
  int seq_len() const override { return time_; }
  void reshape(int micro_batch, int seq_len) override {
    batch_ = micro_batch;
    time_ = seq_len;
  }

  void attach_process_group(std::shared_ptr<dist::ProcessGroup> group, const dist::DdpOptions& options) override {
    group_ = std::move(group);
    ddp_ = options;
    // Asama = parametreli bir Sequential katmani; ilk tahmin: ters katman sirasi (geri gecis sirasi).
    // Dilimde offset = parametre indeksi (CPU gradyanlari ayri tensorler; kova paketleme indeksle).
    std::vector<dist::Stage> stages;
    if (const auto* seq = dynamic_cast<const nn::Sequential*>(model_)) {
      std::size_t p = 0;
      for (std::size_t i = 0; i < seq->size(); ++i) {
        const std::size_t count = seq->layer(i)->parameters().size();
        dist::Stage st;
        st.id = static_cast<int>(i);
        for (std::size_t j = 0; j < count; ++j, ++p) {
          st.slices.push_back(dist::GradSlice{p, params_[p]->grad.numel()});
        }
        if (count > 0) {
          stages.push_back(std::move(st));
        }
      }
      std::reverse(stages.begin(), stages.end());
      layered_ = true;
    } else {
      dist::Stage st;
      for (std::size_t p = 0; p < params_.size(); ++p) {
        st.slices.push_back(dist::GradSlice{p, params_[p]->grad.numel()});
      }
      stages.push_back(std::move(st));
      layered_ = false;
    }
    stages_ = std::move(stages);
    plan();
  }

  void broadcast_parameters() override {
    require_group();
    std::vector<dist::WorkPtr> works;
    for (nexus_model::Parameter* p : params_) {
      works.push_back(group_->broadcast(p->data.data(), p->data.numel(), DType::F32, 0));
    }
    for (const dist::WorkPtr& w : works) {
      w->wait();
    }
  }

  dist::DdpStepStats ddp_stats() const override { return stats_; }

  void attach_fsdp(std::shared_ptr<dist::ProcessGroup> group, const dist::FsdpOptions& options) override {
    auto* seq = dynamic_cast<nn::Sequential*>(model_);
    if (seq == nullptr) {
      throw dist::DistError("Nexus LM trainer: FSDP nn::Sequential model ister");
    }
    engine_ = std::make_unique<dist::FsdpEngine>(*seq, std::move(group), options);
    engine_->set_param_scales(wd_, lr_);
    engine_->initialize();
  }

  dist::FsdpEngine* fsdp() noexcept override { return engine_.get(); }

  void set_rng(std::uint64_t seed, std::uint64_t offset) override { nn::set_dropout_rng(seed, offset); }

  std::string state_layout() const override {
    if (!engine_) return "replicated";
    return "fsdp:" + std::to_string(static_cast<int>(engine_->options().sharding)) + ":" + std::to_string(engine_->shard_size());
  }

  void load_resharded_state(const std::vector<std::vector<unsigned char>>& blobs) override {
    if (blobs.empty()) throw std::runtime_error("Nexus LM trainer: yeniden parcalama icin kayit yuku yok");
    if (!engine_) {  // cogaltilmis (DDP / tek rank): her rank'in yuku tam durumdur
      load_state(blobs[0]);
      return;
    }
    std::vector<std::vector<unsigned char>> shards;
    std::int64_t skipped = 0;
    for (const auto& b : blobs) {
      Reader r(b);
      shards.push_back(r.bytes());
      skipped = r.pod<std::int64_t>();
    }
    engine_->load_resharded(shards);
    skipped_total_ = skipped;
  }

  void micro_step(const std::int32_t* tokens, float grad_weight, int index, int count) override {
    if (index == 0) {
      if (engine_) {
        engine_->zero_grad();  // FSDP: parcalanmis/serbest gradyanlar dokunulmaz
      } else {
        for (nexus_model::Parameter* p : params_) {
          p->zero_grad();  // eski LanguageTrainer ile ayni sifirlama
        }
      }
      losses_.clear();
    }
    nexus_model::Tensor logits = forward(tokens);
    const int vocab = static_cast<int>(logits.size(2));
    nexus_model::Tensor full = nexus_model::Tensor::zeros(
        {static_cast<std::size_t>(batch_), static_cast<std::size_t>(time_), static_cast<std::size_t>(vocab)});
    const fused::CrossEntropyResult result =
        fused::cross_entropy_shifted(logits.data(), tokens, batch_, time_, vocab, ce_, full.data());
    if (grad_weight != 1.0f) {
      float* g = full.data();
      for (std::size_t i = 0; i < full.numel(); ++i) {
        g[i] *= grad_weight;  // mikro adim agirligi (dlogits)
      }
    }
    if (engine_) {
      engine_->backward(full, index + 1 == count);  // no_sync: yalniz son mikro adim RS
      losses_.push_back(result.loss);
      return;
    }
    const bool sync = group_ != nullptr && index + 1 == count;  // no_sync: yalniz son mikro adim
    if (!sync) {
      model_->backward(full);  // Parameter::grad += (FAZ 2 AccumulateGrad)
    } else {
      sync_backward(full);
    }
    losses_.push_back(result.loss);
  }

  void optimizer_step(const AdamW& opt, double lr, double clip) override {
    if (engine_) {
      const dist::FsdpEngine::StepResult r =
          engine_->step(dist::FsdpAdamW{opt.lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay}, lr, clip);
      norm_ = r.grad_norm;
      skipped_ = r.skipped;
      skipped_total_ += r.skipped ? 1 : 0;
      return;
    }
    ensure_optimizer(opt);
    if (group_ != nullptr && pending_) {
      finish_sync();
    }
    // FAZ 9.8 ortak yardimci (siniflandirma Trainer'i ile ayni): norm, sonluluk, torch kirpma katsayisi.
    const GradClipResult clipped = clip_global_norm(params_, clip);
    norm_ = clipped.norm;
    skipped_ = !clipped.finite;
    if (skipped_) {
      skipped_total_ += 1;
      return;  // inf/NaN: parametre ve momentler degismez
    }
    auto& groups = impl_->param_groups();
    for (std::size_t gi = 0; gi < groups.size(); ++gi) {
      groups[gi].options.learning_rate = lr * static_cast<double>(group_lr_[gi]);
    }
    impl_->step();
  }

  LmStepResult finish_step() override {
    LmStepResult r;
    r.losses = losses_;
    r.grad_norm = norm_;
    r.skipped = skipped_;
    r.skipped_total = skipped_total_;
    return r;
  }

  double eval_loss(const std::int32_t* tokens) override {
    model_->eval();
    nexus_model::Tensor logits = forward(tokens);
    model_->train();
    const fused::CrossEntropyResult result = fused::cross_entropy_shifted(
        logits.data(), tokens, batch_, time_, static_cast<int>(logits.size(2)), ce_, nullptr);
    return result.loss;
  }

  std::vector<float> token_losses(const std::int32_t* tokens) override {
    model_->eval();
    nexus_model::Tensor logits = forward(tokens);
    model_->train();
    const int vocab = static_cast<int>(logits.size(2));
    std::vector<float> out(static_cast<std::size_t>(batch_) * time_, 0.0f);
    for (int b = 0; b < batch_; ++b) {
      for (int t = 0; t + 1 < time_; ++t) {
        const std::size_t row = static_cast<std::size_t>(b) * time_ + t;
        const std::int32_t target = tokens[row + 1];
        if (target == kIgnoreIndex) {
          continue;
        }
        const float* x = logits.data() + row * vocab;
        const float m = *std::max_element(x, x + vocab);
        double z = 0.0;
        for (int j = 0; j < vocab; ++j) {
          z += std::exp(static_cast<double>(x[j] - m));
        }
        out[row] = static_cast<float>(std::log(z) + m - x[target]);
      }
    }
    return out;
  }

  LmNonFinite diagnose(const std::int32_t* tokens) override {
    LmNonFinite rep;
    if (engine_) {  // FSDP: katman katman tarama parcali parametrelerle yapilmaz; model duzeyinde
      const nexus_model::Tensor logits = forward(tokens);
      rep.first_activation = finite_tensor(logits) ? "" : "model";
      rep.loss = fused::cross_entropy_shifted(logits.data(), tokens, batch_, time_, static_cast<int>(logits.size(2)), ce_,
                                              nullptr).loss;
      return rep;
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
      if (!finite_tensor(params_[i]->data)) {
        rep.parameters.push_back(names_[i]);
      }
    }
    nexus_model::Tensor current = token_tensor(tokens);
    if (const auto* seq = dynamic_cast<const nn::Sequential*>(model_)) {
      for (std::size_t i = 0; i < seq->size(); ++i) {
        current = seq->layer(i)->forward(current);
        if (rep.first_activation.empty() && !finite_tensor(current)) {
          rep.first_activation = std::to_string(i) + "." + short_type_name(*seq->layer(i));
        }
      }
    } else {
      current = model_->forward(current);
      if (!finite_tensor(current)) {
        rep.first_activation = "model";
      }
    }
    const int vocab = static_cast<int>(current.size(2));
    nexus_model::Tensor full = nexus_model::Tensor::zeros(
        {static_cast<std::size_t>(batch_), static_cast<std::size_t>(time_), static_cast<std::size_t>(vocab)});
    rep.loss = fused::cross_entropy_shifted(current.data(), tokens, batch_, time_, vocab, ce_, full.data()).loss;
    if (rep.first_activation.empty() && !std::isfinite(rep.loss)) {
      rep.first_activation = "loss";
    }
    for (nexus_model::Parameter* p : params_) {
      p->zero_grad();
    }
    model_->backward(full);
    for (std::size_t k = params_.size(); k-- > 0;) {  // geri gecis sirasi: son katman once
      if (!finite_tensor(params_[k]->grad)) {
        rep.gradients.push_back(names_[k]);
      }
    }
    return rep;
  }

  std::vector<unsigned char> save_state() const override {
    if (engine_) {  // FSDP: bu rank'in parcalari (trainer rank basina dosya yazar)
      Writer w;
      const std::vector<unsigned char> blob = engine_->save_shard();
      w.bytes(blob.data(), blob.size());
      w.pod(skipped_total_);
      return w.out;
    }
    Writer w;
    w.pod<std::uint64_t>(params_.size());
    for (const nexus_model::Parameter* p : params_) {
      w.bytes(p->data.data(), p->data.numel() * sizeof(float));
    }
    w.pod<std::uint8_t>(impl_ ? 1 : 0);
    if (impl_) {
      w.pod(opt_);
      const nexus_optim::OptimizerState st = impl_->state_dict();
      w.str(st.algorithm);
      w.pod(st.step);
      w.pod<std::uint64_t>(st.group_learning_rates.size());
      for (double v : st.group_learning_rates) {
        w.pod(v);
      }
      w.pod<std::uint64_t>(st.blobs.size());
      for (const auto& b : st.blobs) {
        w.bytes(b.data(), b.size());
      }
      w.pod<std::uint64_t>(st.scalars.size());
      for (double v : st.scalars) {
        w.pod(v);
      }
    }
    w.pod(skipped_total_);
    return w.out;
  }

  void load_state(const std::vector<unsigned char>& blob) override {
    if (engine_) {
      Reader r(blob);
      engine_->load_shard(r.bytes());
      skipped_total_ = r.pod<std::int64_t>();
      return;
    }
    Reader r(blob);
    if (r.pod<std::uint64_t>() != params_.size()) {
      throw std::runtime_error("Nexus LM trainer: durum baska bir modelden (parametre sayisi)");
    }
    for (nexus_model::Parameter* p : params_) {
      const std::vector<unsigned char> b = r.bytes();
      if (b.size() != p->data.numel() * sizeof(float)) {
        throw std::runtime_error("Nexus LM trainer: durum baska bir modelden (tensor boyutu)");
      }
      std::memcpy(p->data.data(), b.data(), b.size());
    }
    if (r.pod<std::uint8_t>() != 0) {
      const auto opt = r.pod<AdamW>();
      impl_.reset();
      ensure_optimizer(opt);
      nexus_optim::OptimizerState st;
      st.algorithm = r.str();
      st.step = r.pod<std::uint64_t>();
      st.group_learning_rates.resize(static_cast<std::size_t>(r.pod<std::uint64_t>()));
      for (double& v : st.group_learning_rates) {
        v = r.pod<double>();
      }
      st.blobs.resize(static_cast<std::size_t>(r.pod<std::uint64_t>()));
      for (auto& b : st.blobs) {
        const std::vector<unsigned char> raw = r.bytes();
        b.assign(raw.begin(), raw.end());
      }
      st.scalars.resize(static_cast<std::size_t>(r.pod<std::uint64_t>()));
      for (double& v : st.scalars) {
        v = r.pod<double>();
      }
      impl_->load_state_dict(st);
    }
    skipped_total_ = r.pod<std::int64_t>();
  }

  std::vector<LmNormEntry> layer_norms() override {
    std::vector<LmNormEntry> out;
    if (engine_) {
      return out;  // FSDP: parametre/gradyan tensorleri parcali ya da serbest (katman normu yok)
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
      out.push_back(LmNormEntry{names_[i], l2(params_[i]->data), l2(params_[i]->grad)});
    }
    return out;
  }

  std::uint64_t peak_memory_bytes() const override { return 0; }
  void reset_peak_memory() override {}
  double peak_flops() const override { return 0.0; }

  nn::Module& module() noexcept { return *model_; }

 private:
  void require_group() const {
    if (group_ == nullptr) {
      throw std::logic_error("Nexus LM trainer: once attach_process_group");
    }
  }

  void plan() {
    buckets_ = dist::plan_buckets(stages_, ddp_.bucket_bytes);
    host_.assign(buckets_.size(), dist::HostBucket{});
    for (std::size_t b = 0; b < buckets_.size(); ++b) {
      host_[b].resize(buckets_[b].numel, ddp_.ranks_per_node);
    }
    bucket_of_.clear();
    for (std::size_t b = 0; b < buckets_.size(); ++b) {
      for (int id : buckets_[b].stages) {
        bucket_of_[id] = b;
      }
    }
  }

  void launch(std::size_t b) {
    float* dst = host_[b].data();
    for (const dist::GradSlice& sl : buckets_[b].slices) {
      std::memcpy(dst, params_[sl.offset]->grad.data(), sl.numel * sizeof(float));
      dst += sl.numel;
    }
    host_[b].launch(*group_, ddp_);
    launched_[b] = true;
  }

  // Son mikro adimin geri gecisi: Sequential'da katman katman; her katmanin parametreleri hazir
  // olunca asamasi isaretlenir, kovasinin tum asamalari hazirsa all-reduce baslar (ProcessGroup is
  // parcacigi geri gecisin kalaniyla es zamanli calisir).
  void sync_backward(const nexus_model::Tensor& dlogits) {
    launched_.assign(buckets_.size(), false);
    ready_.assign(buckets_.size(), 0);
    observed_.clear();
    pending_ = true;
    const bool early = layered_ && ddp_.overlap && !ddp_.find_unused_parameters;
    t_bwd0_ = std::chrono::steady_clock::now();
    if (early) {
      const auto* seq = static_cast<const nn::Sequential*>(model_);
      nexus_model::Tensor grad = dlogits;
      for (std::size_t i = seq->size(); i-- > 0;) {
        grad = seq->layer(i)->backward(grad);
        const auto it = bucket_of_.find(static_cast<int>(i));
        if (it == bucket_of_.end()) {
          continue;  // parametresiz katman
        }
        observed_.push_back(static_cast<int>(i));
        const std::size_t b = it->second;
        if (++ready_[b] == static_cast<int>(buckets_[b].stages.size())) {
          launch(b);
        }
      }
    } else {
      model_->backward(dlogits);
    }
    t_bwd1_ = std::chrono::steady_clock::now();
  }

  void finish_sync() {
    for (std::size_t b = 0; b < buckets_.size(); ++b) {
      if (!launched_[b]) {
        launch(b);  // ortusmesiz yol / find_unused_parameters: geri gecisten sonra
      }
    }
    using ms = std::chrono::duration<double, std::milli>;
    stats_ = dist::DdpStepStats{};
    stats_.backward_ms = ms(t_bwd1_ - t_bwd0_).count();
    stats_.buckets = static_cast<int>(buckets_.size());
    for (std::size_t b = 0; b < buckets_.size(); ++b) {
      host_[b].finish();
      const float* src = host_[b].data();
      for (const dist::GradSlice& sl : buckets_[b].slices) {
        std::memcpy(params_[sl.offset]->grad.data(), src, sl.numel * sizeof(float));
        src += sl.numel;
      }
      const auto s0 = host_[b].work()->started();
      const auto s1 = host_[b].work()->finished();
      const double a = ms(s0 - t_bwd0_).count();
      const double e = ms(s1 - t_bwd0_).count();
      stats_.comm_first_start_ms = stats_.comm_first_start_ms < 0 ? a : std::min(stats_.comm_first_start_ms, a);
      stats_.comm_last_end_ms = std::max(stats_.comm_last_end_ms, e);
      stats_.comm_busy_ms += e - a;
      stats_.comm_overlapped_ms += std::max(0.0, std::min(e, stats_.backward_ms) - std::max(a, 0.0));
    }
    pending_ = false;
    // Ilk senkron adim: gercek hazir olma sirasi tahminden farkliysa kovalar yeniden kurulur.
    if (!order_checked_ && !observed_.empty()) {
      order_checked_ = true;
      std::vector<int> planned;
      for (const dist::Stage& st : stages_) {
        planned.push_back(st.id);
      }
      if (planned != observed_) {
        std::vector<dist::Stage> reordered;
        for (int id : observed_) {
          for (const dist::Stage& st : stages_) {
            if (st.id == id) reordered.push_back(st);
          }
        }
        stages_ = std::move(reordered);
        plan();
        stats_.rebuilt = true;
      }
    }
  }

  nexus_model::Tensor token_tensor(const std::int32_t* tokens) const {
    nexus_model::Tensor t =
        nexus_model::Tensor::zeros({static_cast<std::size_t>(batch_), static_cast<std::size_t>(time_)});
    for (std::size_t i = 0; i < t.numel(); ++i) {
      t[i] = static_cast<float>(tokens[i]);
    }
    return t;
  }

  nexus_model::Tensor forward(const std::int32_t* tokens) {
    nexus_model::Tensor logits = engine_ ? engine_->forward(token_tensor(tokens)) : model_->forward(token_tensor(tokens));
    if (logits.rank() != 3 || logits.size(0) != static_cast<std::size_t>(batch_) ||
        logits.size(1) != static_cast<std::size_t>(time_)) {
      throw std::invalid_argument("language logits must be [batch, time, vocab]");
    }
    return logits;
  }

  void ensure_optimizer(const AdamW& opt) {
    if (impl_) {
      if (opt.beta1 != opt_.beta1 || opt.beta2 != opt_.beta2 || opt.eps != opt_.eps ||
          opt.weight_decay != opt_.weight_decay) {
        throw std::logic_error("Nexus LM trainer: AdamW beta/eps/wd egitim sirasinda degistirilemez");
      }
      return;
    }
    opt_ = opt;
    // (wd, lr) carpan ciftine gore gruplar; parametre sirasi grup icinde korunur.
    std::vector<std::pair<float, float>> keys;
    std::vector<std::vector<std::size_t>> members;
    for (std::size_t i = 0; i < params_.size(); ++i) {
      const std::pair<float, float> k{wd_[i], lr_[i]};
      auto it = std::find(keys.begin(), keys.end(), k);
      if (it == keys.end()) {
        keys.push_back(k);
        members.emplace_back();
        it = keys.end() - 1;
      }
      members[static_cast<std::size_t>(it - keys.begin())].push_back(i);
    }
    std::vector<nexus_optim::ParamGroup<float>> groups;
    group_lr_.clear();
    for (std::size_t g = 0; g < keys.size(); ++g) {
      std::vector<float*> values;
      std::vector<float*> grads;
      std::vector<std::size_t> sizes;
      for (const std::size_t i : members[g]) {
        values.push_back(params_[i]->data.data());
        grads.push_back(params_[i]->grad.data());
        sizes.push_back(params_[i]->data.numel());
      }
      nexus_optim::ParamGroupOptions options;
      options.learning_rate = opt.lr * static_cast<double>(keys[g].second);
      options.weight_decay = opt.weight_decay * static_cast<double>(keys[g].first);
      options.weight_decay_override = true;
      groups.push_back(nexus_optim::make_group<float>(std::move(values), std::move(grads), std::move(sizes), options));
      group_lr_.push_back(keys[g].second);
    }
    nexus_optim::AdamW<float>::Options native;
    native.lr = opt.lr;
    native.beta1 = opt.beta1;
    native.beta2 = opt.beta2;
    native.eps = opt.eps;
    native.weight_decay = opt.weight_decay;
    impl_ = std::make_unique<nexus_optim::AdamW<float>>(std::move(groups), native);
  }

  nn::Module* model_;
  int batch_;
  int time_;
  std::vector<nexus_model::Parameter*> params_;
  std::vector<std::string> names_;
  std::vector<bool> embedding_;
  std::vector<bool> positional_;
  std::vector<float> wd_;
  std::vector<float> lr_;
  std::vector<float> group_lr_;
  AdamW opt_{};
  std::unique_ptr<nexus_optim::AdamW<float>> impl_;
  fused::CrossEntropyOptions ce_{};
  std::vector<float> losses_;
  double norm_ = 0.0;
  bool skipped_ = false;
  std::int64_t skipped_total_ = 0;
  std::unique_ptr<dist::FsdpEngine> engine_;  // FAZ 11
  // FAZ 10 DDP
  std::shared_ptr<dist::ProcessGroup> group_;
  dist::DdpOptions ddp_{};
  bool layered_ = false;
  std::vector<dist::Stage> stages_;
  std::vector<dist::Bucket> buckets_;
  std::vector<dist::HostBucket> host_;
  std::map<int, std::size_t> bucket_of_;
  std::vector<bool> launched_;
  std::vector<int> ready_;
  std::vector<int> observed_;
  bool order_checked_ = false;
  bool pending_ = false;
  std::chrono::steady_clock::time_point t_bwd0_{}, t_bwd1_{};
  dist::DdpStepStats stats_{};
};

double median_of(std::vector<double> v) {
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid), v.end());
  double m = v[mid];
  if (v.size() % 2 == 0) {
    m = 0.5 * (m + *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(mid)));
  }
  return m;
}

}  // namespace

void LmBackend::attach_process_group(std::shared_ptr<dist::ProcessGroup>, const dist::DdpOptions&) {
  throw dist::DistError("Nexus LM trainer: bu arka uc veri paralelligini desteklemiyor");
}

void LmBackend::broadcast_parameters() {
  throw dist::DistError("Nexus LM trainer: bu arka uc veri paralelligini desteklemiyor");
}

void LmBackend::attach_fsdp(std::shared_ptr<dist::ProcessGroup>, const dist::FsdpOptions&) {
  throw dist::DistError("Nexus LM trainer: bu arka uc ZeRO/FSDP desteklemiyor (yalniz CPU nn::Sequential)");
}

void LmBackend::load_resharded_state(const std::vector<std::vector<unsigned char>>&) {
  throw std::runtime_error("Nexus LM trainer: bu arka uc farkli rank sayisinda kaydedilmis durumu yukleyemez (yeniden parcalama yok)");
}

BackendFactory cpu_backend(nn::Module& model) {
  return [&model](const LmBackendRequest& request) -> std::unique_ptr<LmBackend> {
    return std::make_unique<CpuModuleBackend>(model, request);
  };
}

// ==========================================================================================
// Veri kaynaklari
// ==========================================================================================

StreamSource::StreamSource(nexusdata::text::TokenStream& stream) : stream_(&stream) {}

void StreamSource::next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) {
  if (stream_->window() + 1 < time) {
    throw std::invalid_argument("Nexus LM trainer: TokenStream penceresi + 1 >= seq_len olmali");
  }
  nexusdata::text::PackedSample sample;
  for (int r = 0; r < rows; ++r) {
    if (ids != nullptr) {
      ids[r] = stream_->state().global_index;
    }
    stream_->next(sample);
    std::memcpy(dst + static_cast<std::size_t>(r) * time, sample.tokens.data(), static_cast<std::size_t>(time) * 4);
  }
}

std::string StreamSource::state() const { return stream_->state().serialize(); }

void StreamSource::restore(const std::string& state) {
  stream_->restore(nexusdata::text::SamplerState::deserialize(state));
}

FixedRowsSource::FixedRowsSource(std::vector<std::int32_t> tokens, int rows, int time) {
  set(std::move(tokens), rows, time);
}

void FixedRowsSource::set(std::vector<std::int32_t> tokens, int rows, int time) {
  if (rows < 1 || time < 2 || tokens.size() != static_cast<std::size_t>(rows) * time) {
    throw std::invalid_argument("FixedRowsSource: tokens.size() == rows * time, time >= 2");
  }
  tokens_ = std::move(tokens);
  rows_ = rows;
  time_ = time;
}

void FixedRowsSource::next(std::int32_t* dst, int rows, int time, std::uint64_t* ids) {
  if (time != time_ || rows % rows_ != 0) {
    throw std::invalid_argument("FixedRowsSource: istenen yigin sabit satir sekline uymuyor");
  }
  for (int r = 0; r < rows; r += rows_) {
    std::memcpy(dst + static_cast<std::size_t>(r) * time, tokens_.data(), tokens_.size() * 4);
  }
  if (ids != nullptr) {
    for (int r = 0; r < rows; ++r) {
      ids[r] = static_cast<std::uint64_t>(r % rows_);
    }
  }
  ++calls_;
}

// ==========================================================================================
// Builder
// ==========================================================================================

LanguageTrainerBuilder::LanguageTrainerBuilder(nn::Module& model) : factory_(cpu_backend(model)), cpu_module_(true) {}

LanguageTrainerBuilder::LanguageTrainerBuilder(const lm::LanguageModel& model)
    : factory_(parallel_backend(model, nullptr, {})), lm_model_(&model) {}

LanguageTrainerBuilder::LanguageTrainerBuilder(BackendFactory factory) : factory_(std::move(factory)) {
  if (!factory_) {
    throw std::invalid_argument("LanguageTrainerBuilder: bos arka uc fabrikasi");
  }
}

LanguageTrainerBuilder& LanguageTrainerBuilder::data(TokenBatchSource& train, TokenBatchSource* validation) {
  owned_train_.reset();
  owned_val_.reset();
  train_ = &train;
  val_ = validation;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::data(nexusdata::text::TokenStream& train,
                                                     nexusdata::text::TokenStream* validation) {
  owned_train_ = std::make_unique<StreamSource>(train);
  owned_val_ = validation != nullptr ? std::make_unique<StreamSource>(*validation) : nullptr;
  train_ = owned_train_.get();
  val_ = owned_val_.get();
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::optimizer(AdamW config, ParamGroupPolicy groups) {
  if (!(config.lr > 0.0) || !(config.beta1 >= 0.0 && config.beta1 < 1.0) || !(config.beta2 >= 0.0 && config.beta2 < 1.0) ||
      !(config.eps > 0.0) || !(config.weight_decay >= 0.0)) {
    throw std::invalid_argument("LanguageTrainerBuilder: AdamW lr > 0, 0 <= beta < 1, eps > 0, wd >= 0");
  }
  adamw_ = config;
  groups_ = std::move(groups);
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::schedule(LrSchedule schedule) {
  schedule_ = std::move(schedule);
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::global_batch_tokens(std::int64_t tokens) {
  if (tokens < 1) {
    throw std::invalid_argument("LanguageTrainerBuilder: global_batch_tokens >= 1");
  }
  global_batch_tokens_ = tokens;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::micro_batch(int rows) {
  if (rows < 1) {
    throw std::invalid_argument("LanguageTrainerBuilder: micro_batch >= 1");
  }
  request_.micro_batch = rows;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::seq_len(int time) {
  if (time < 2) {
    throw std::invalid_argument("LanguageTrainerBuilder: seq_len >= 2");
  }
  request_.seq_len = time;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::grad_clip(double max_norm) {
  if (!(max_norm >= 0.0)) {
    throw std::invalid_argument("LanguageTrainerBuilder: grad_clip >= 0 (0: kirpma yok)");
  }
  clip_ = max_norm;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::precision(Precision precision, MasterWeightsPolicy master) {
  request_.precision = precision;
  request_.master = master;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::recompute(RecomputePolicy policy) {
  request_.recompute = policy;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::eval_every(std::int64_t steps, std::int64_t eval_tokens) {
  if (steps < 0 || eval_tokens < 0 || (steps > 0 && eval_tokens < 1)) {
    throw std::invalid_argument("LanguageTrainerBuilder: eval_every(k > 0, eval_tokens > 0) ya da (0, 0)");
  }
  eval_every_ = steps;
  eval_tokens_ = eval_tokens;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::log_every(std::int64_t steps) {
  log_every_ = std::max<std::int64_t>(1, steps);
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::execution_policy(ExecutionPolicy policy) {
  if (policy.hybrid()) {
    throw std::invalid_argument("LanguageTrainerBuilder: hibrit politika dil modeli egitmeninde yok");
  }
  request_.policy = policy;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::seed(std::uint64_t seed) {
  request_.seed = seed;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::deterministic(bool enabled) {
  request_.deterministic = enabled;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::nan_policy(NanOptions options) {
  if (options.snapshot_every < 1 || !(options.rollback_lr_factor > 0.0 && options.rollback_lr_factor <= 1.0) ||
      options.max_rollbacks < 0 || options.max_consecutive_skips < 1) {
    throw std::invalid_argument("LanguageTrainerBuilder: NanOptions gecersiz");
  }
  nan_ = options;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::spike_detector(SpikeOptions options) {
  if (options.enabled && (options.window < 4 || !(options.k > 0.0))) {
    throw std::invalid_argument("LanguageTrainerBuilder: sicrama penceresi >= 4, k > 0");
  }
  spike_ = options;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::histogram_every(std::int64_t steps) {
  histogram_every_ = std::max<std::int64_t>(0, steps);
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::logger(std::shared_ptr<nexus_train::LoggerBase> sink) {
  if (!sink) {
    throw std::invalid_argument("LanguageTrainerBuilder: bos gunlukcu");
  }
  loggers_.push_back(std::move(sink));
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::log_console(bool enabled) {
  console_ = enabled;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::log_csv(const std::string& path) {
  return logger(std::make_shared<nexus_train::CsvLogger>(path));
}

LanguageTrainerBuilder& LanguageTrainerBuilder::log_jsonl(const std::string& path) {
  return logger(std::make_shared<nexus_train::JsonLogger>(path));
}

LanguageTrainerBuilder& LanguageTrainerBuilder::log_tensorboard(const std::string& directory) {
  return logger(std::make_shared<TensorBoardLogger>(directory));
}

LanguageTrainerBuilder& LanguageTrainerBuilder::log_wandb(const std::string& path) {
  return logger(std::make_shared<WandbJsonlLogger>(path));
}

LanguageTrainerBuilder& LanguageTrainerBuilder::after_backward(std::function<void(LmBackend&, std::int64_t)> hook) {
  after_backward_ = std::move(hook);
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::process_group(std::shared_ptr<dist::ProcessGroup> group,
                                                              dist::DdpOptions options) {
  if (!group) {
    throw std::invalid_argument("LanguageTrainerBuilder: bos ProcessGroup");
  }
  group_ = std::move(group);
  ddp_ = options;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::fsdp(std::shared_ptr<dist::ProcessGroup> group, dist::FsdpOptions options) {
  if (!group) {
    throw std::invalid_argument("LanguageTrainerBuilder: bos ProcessGroup");
  }
  group_ = std::move(group);
  use_fsdp_ = true;
  fsdp_ = options;
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::compile(bool enabled, std::string mode) {
  if (mode != "default" && mode != "reduce-overhead" && mode != "max-autotune") {
    throw std::invalid_argument("LanguageTrainerBuilder::compile: mode 'default' | 'reduce-overhead' | 'max-autotune'");
  }
  request_.compile = enabled;
  request_.compile_mode = std::move(mode);
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::parallel(std::shared_ptr<dist::ProcessGroup> group, Parallel config) {
  if (lm_model_ != nullptr) {
    // FAZ 12: lm::LanguageModel -> ParallelLm (DP x PP x CP x TP); grup = world.
    if (!group) {
      throw std::invalid_argument("LanguageTrainerBuilder: bos ProcessGroup");
    }
    dist::ParallelConfig c;
    c.dp = config.dp;
    c.tp = config.tp;
    c.pp = config.pp;
    c.cp = config.cp;
    c.zero = config.zero < 0 ? 0 : config.zero;
    c.sp = config.sp;
    c.async_tp = config.async_tp;
    c.pp_schedule = dist::parse_pipe_schedule(config.pp_schedule);
    c.virtual_stages = config.virtual_stages;
    if (config.cp_algorithm == "ring") {
      c.cp_algorithm = dist::CpAlgorithm::Ring;
    } else if (config.cp_algorithm == "ulysses") {
      c.cp_algorithm = dist::CpAlgorithm::Ulysses;
    } else {
      throw std::invalid_argument("LanguageTrainerBuilder: cp_algorithm 'ring' | 'ulysses'");
    }
    if (config.cp_layout == "zigzag") {
      c.cp_layout = dist::CpLayout::Zigzag;
    } else if (config.cp_layout == "contiguous") {
      c.cp_layout = dist::CpLayout::Contiguous;
    } else {
      throw std::invalid_argument("LanguageTrainerBuilder: cp_layout 'zigzag' | 'contiguous'");
    }
    parallel_ = c;
    world_ = group;
    factory_ = parallel_backend(*lm_model_, std::move(group), c);
    return *this;
  }
  if (config.tp != 1 || config.pp != 1 || config.cp != 1) {
    throw std::invalid_argument("LanguageTrainerBuilder: tp/pp/cp > 1 builder'in lm::LanguageModel ile kurulmasini ister");
  }
  if (config.zero < 0) {
    return process_group(std::move(group), config.ddp);
  }
  config.fsdp.sharding = dist::zero_stage(config.zero);
  return fsdp(std::move(group), config.fsdp);
}

LanguageTrainerBuilder& LanguageTrainerBuilder::checkpointing(ckpt::CheckpointOptions options) {
  if (options.dir.empty()) {
    throw std::invalid_argument("LanguageTrainerBuilder::checkpointing: dizin bos");
  }
  ckpt_ = std::move(options);
  return *this;
}

LanguageTrainerBuilder& LanguageTrainerBuilder::watchdog(std::shared_ptr<dist::Watchdog> watchdog) {
  watchdog_ = std::move(watchdog);
  return *this;
}

std::unique_ptr<LanguageModelTrainer> LanguageTrainerBuilder::build() {
  return std::unique_ptr<LanguageModelTrainer>(new LanguageModelTrainer(*this));
}

// ==========================================================================================
// Egitmen
// ==========================================================================================

LanguageModelTrainer::LanguageModelTrainer(LanguageTrainerBuilder& b)
    : train_(b.train_), val_(b.val_), owned_train_(std::move(b.owned_train_)), owned_val_(std::move(b.owned_val_)),
      adamw_(b.adamw_), schedule_(b.schedule_), micro_batch_(b.request_.micro_batch), seq_len_(b.request_.seq_len),
      clip_(b.clip_), eval_every_(b.eval_every_), log_every_(b.log_every_), nan_(b.nan_), spike_(b.spike_),
      histogram_every_(b.histogram_every_), console_(b.console_), loggers_(b.loggers_),
      after_backward_(b.after_backward_), group_(b.group_) {
  if (group_) {
    dp_ = group_->size();
    rank_ = group_->rank();
  }
  if (b.lm_model_ != nullptr && b.world_) {
    dp_ = b.parallel_.dp;  // FAZ 12: dp grubu arka ucun cihaz aginda; gunluk dunya rank 0'da
    rank_ = b.world_->rank();
    multi_rank_ = b.world_->size() > 1;
  }
  multi_rank_ = multi_rank_ || dp_ > 1;
  if (rank_ != 0) {
    console_ = false;  // gunluk yalniz rank 0'da
    loggers_.clear();
  }
  if (train_ == nullptr) {
    throw std::invalid_argument("LanguageTrainerBuilder: .data(train, ...) gerekli");
  }
  const std::int64_t per_micro = static_cast<std::int64_t>(micro_batch_) * seq_len_;
  if (b.global_batch_tokens_ > 0) {
    const std::int64_t per_step = per_micro * dp_;
    if (b.global_batch_tokens_ % per_step != 0) {
      throw std::invalid_argument(
          "LanguageTrainerBuilder: global_batch_tokens, micro_batch * seq_len * dp'nin kati olmali (" +
          std::to_string(b.global_batch_tokens_) + " % " + std::to_string(per_step) + ")");
    }
    accum_ = static_cast<int>(b.global_batch_tokens_ / per_step);  // global = micro * accum * dp
  }
  LmBackendRequest request = b.request_;
  request.accum_steps = accum_;
  global_batch_tokens_ = per_micro * accum_ * dp_;
  rng_seed_ = request.seed;
  watchdog_ = b.watchdog_;
  backend_ = b.factory_(request);
  if (b.ckpt_) {  // FAZ 15: kollektiflerden (yayin / parcalama) once — rank 0 yarim kayitlari temizler
    const int world = b.world_ ? b.world_->size() : (group_ ? group_->size() : 1);
    ckpt_ = std::make_unique<ckpt::CheckpointManager>(*b.ckpt_, rank_, world);
  }
  if (b.lm_model_ != nullptr && dp_ > 1) {
    group_ = backend_->data_parallel_group();
  }
  if (backend_->micro_batch() != micro_batch_ || backend_->seq_len() != seq_len_) {
    throw std::logic_error("LanguageTrainerBuilder: arka uc yigin sekli istenenle ayni degil");
  }
  // FAZ 9.3 parametre gruplari.
  const std::vector<LmParamInfo> params = backend_->parameters();
  std::vector<float> wd(params.size());
  std::vector<float> lr(params.size(), 1.0f);
  for (std::size_t i = 0; i < params.size(); ++i) {
    const LmParamInfo& p = params[i];
    bool decay = false;
    switch (b.groups_.decay) {
      case ParamGroupPolicy::Decay::All:
        decay = true;
        break;
      case ParamGroupPolicy::Decay::None:
        decay = false;
        break;
      case ParamGroupPolicy::Decay::MatricesOnly:
        decay = p.rank >= 2 && (!p.embedding || b.groups_.decay_embeddings);
        break;
    }
    wd[i] = decay ? 1.0f : 0.0f;
    for (const ParamGroupRule& rule : b.groups_.rules) {
      if (p.name.rfind(rule.prefix, 0) == 0) {
        lr[i] = rule.lr_scale;
        if (rule.weight_decay_scale >= 0.0f) {
          wd[i] = rule.weight_decay_scale;
        }
      }
    }
  }
  backend_->set_param_scales(wd, lr);
  if (group_ && b.use_fsdp_) {
    backend_->attach_fsdp(group_, b.fsdp_);  // rank 0'dan yayin + parcalama icinde
  } else if (group_ && dp_ > 1 && b.lm_model_ == nullptr) {
    backend_->attach_process_group(group_, b.ddp_);
    backend_->broadcast_parameters();
  }
  if (eval_every_ > 0) {
    if (val_ == nullptr) {
      throw std::invalid_argument("LanguageTrainerBuilder: eval_every icin dogrulama kaynagi gerekli");
    }
  }
  if (val_ != nullptr) {
    const std::int64_t tokens = b.eval_tokens_ > 0 ? b.eval_tokens_ : per_micro;
    eval_batches_ = std::max<std::int64_t>(1, (tokens + per_micro - 1) / per_micro);
    val_origin_ = val_->state();
  }
}

LanguageModelTrainer::~LanguageModelTrainer() = default;

std::int64_t LanguageModelTrainer::tokens_per_step() const noexcept {
  return static_cast<std::int64_t>(accum_) * micro_batch_ * seq_len_ * dp_;  // kuresel
}

std::string LanguageModelTrainer::rank_path(const std::string& path) const {
  return multi_rank_ ? path + ".rank" + std::to_string(rank_) : path;
}

double LanguageModelTrainer::flops_per_token() const {
  const LmShape s = backend_->shape();
  return 6.0 * static_cast<double>(s.params_no_pos) +
         12.0 * s.layers * s.heads * static_cast<double>(s.head_dim) * seq_len_;
}

void LanguageModelTrainer::reshape(int micro_batch, int seq_len) {
  if (micro_batch < 1 || seq_len < 2) {
    throw std::invalid_argument("LanguageModelTrainer::reshape: micro_batch >= 1, seq_len >= 2");
  }
  backend_->reshape(micro_batch, seq_len);
  micro_batch_ = micro_batch;
  seq_len_ = seq_len;
  prefetched_ = false;
}

void LanguageModelTrainer::fetch(std::vector<std::int32_t>& tokens, std::vector<std::uint64_t>& ids) {
  const int rows = accum_ * micro_batch_;
  tokens.resize(static_cast<std::size_t>(rows) * seq_len_);
  ids.resize(static_cast<std::size_t>(rows));
  train_->next(tokens.data(), rows, seq_len_, ids.data());
}

void LanguageModelTrainer::take_snapshot() {
  snapshot_.backend = backend_->save_state();
  snapshot_.step = step_;
}

bool LanguageModelTrainer::detect_spike(double loss) {
  bool spike = false;
  if (static_cast<std::int64_t>(recent_.size()) >= std::max<std::int64_t>(spike_.min_steps, 4)) {
    const std::vector<double> window(recent_.begin(), recent_.end());
    const double med = median_of(window);
    std::vector<double> dev;
    dev.reserve(window.size());
    for (double v : window) {
      dev.push_back(std::fabs(v - med));
    }
    const double mad = std::max(median_of(dev), 1.0e-3 * std::fabs(med));
    if (loss > med + spike_.k * 1.4826 * mad) {
      SpikeEvent e;
      e.step = step_;
      e.loss = loss;
      e.median = med;
      e.mad = mad;
      e.sample_ids = ids_;
      spikes_.push_back(std::move(e));
      spike = true;
    }
  }
  recent_.push_back(loss);
  while (static_cast<int>(recent_.size()) > spike_.window) {
    recent_.pop_front();
  }
  return spike;
}

TrainSummary LanguageModelTrainer::train(std::int64_t max_steps) {
  TrainSummary summary;
  if (nan_.policy == NanPolicy::Rollback && snapshot_.step < 0) {
    take_snapshot();
  }
  const std::size_t micro_tokens = static_cast<std::size_t>(micro_batch_) * seq_len_;
  std::vector<double> weights(static_cast<std::size_t>(accum_));
  std::string state_before_prefetch;
  while (step_ < max_steps) {
    const auto t0 = std::chrono::steady_clock::now();
    double data_ms = 0.0;
    if (prefetched_) {
      std::swap(tokens_, next_tokens_);
      std::swap(ids_, next_ids_);
      data_ms = next_data_ms_;
      prefetched_ = false;
    } else {
      fetch(tokens_, ids_);
      data_ms = ms_since(t0);
    }
    // Token agirligi: w_i = gecerli_i / toplam (hedef tokens[b, t+1] != ignore).
    std::int64_t total_valid = 0;
    std::vector<std::int64_t> valid(static_cast<std::size_t>(accum_), 0);
    for (int i = 0; i < accum_; ++i) {
      const std::int32_t* mb = tokens_.data() + static_cast<std::size_t>(i) * micro_tokens;
      for (int r = 0; r < micro_batch_; ++r) {
        for (int t = 1; t < seq_len_; ++t) {
          valid[static_cast<std::size_t>(i)] += mb[static_cast<std::size_t>(r) * seq_len_ + t] != kIgnoreIndex ? 1 : 0;
        }
      }
      total_valid += valid[static_cast<std::size_t>(i)];
    }
    if (dp_ > 1) {
      group_->all_reduce(&total_valid, 1, DType::I64, dist::ReduceOp::Sum)->wait();  // kuresel token agirligi
    }
    if (total_valid == 0) {
      throw std::runtime_error("Nexus LM trainer: adimda gecerli hedef token yok");
    }
    for (int i = 0; i < accum_; ++i) {
      weights[static_cast<std::size_t>(i)] =
          static_cast<double>(valid[static_cast<std::size_t>(i)]) / static_cast<double>(total_valid);
      backend_->set_rng(rng_seed_, rng_offset_++);
      backend_->micro_step(tokens_.data() + static_cast<std::size_t>(i) * micro_tokens,
                           static_cast<float>(weights[static_cast<std::size_t>(i)]), i, accum_);
    }
    if (after_backward_) {
      after_backward_(*backend_, step_);
    }
    const double lr = adamw_.lr * schedule_.factor(step_) * lr_factor_;
    backend_->optimizer_step(adamw_, lr, clip_);
    // Aygit calisirken sonraki adimin verisi (son adimda degil: kaynak durumu bozulmasin).
    if (step_ + 1 < max_steps) {
      const auto tp = std::chrono::steady_clock::now();
      state_before_prefetch = train_->state();
      fetch(next_tokens_, next_ids_);
      next_data_ms_ = ms_since(tp);
      prefetched_ = true;
      prefetch_state_ = state_before_prefetch;
    }
    const LmStepResult result = backend_->finish_step();
    const double step_ms = ms_since(t0);
    if (watchdog_) {
      watchdog_->beat(step_, step_ms);  // bu adim (0 tabanli) tamamlandi
    }

    StepRecord rec;
    rec.step = step_;
    rec.lr = lr;
    rec.grad_norm = result.grad_norm;
    rec.skipped = result.skipped;
    rec.skipped_total = result.skipped_total;
    rec.step_ms = step_ms;
    rec.data_ms = data_ms;
    rec.val_loss = std::numeric_limits<double>::quiet_NaN();
    double loss = 0.0;
    for (std::size_t i = 0; i < result.losses.size() && i < weights.size(); ++i) {
      loss += weights[i] * static_cast<double>(result.losses[i]);
    }
    if (dp_ > 1) {
      group_->all_reduce(&loss, 1, DType::F64, dist::ReduceOp::Sum)->wait();  // kuresel token agirlikli kayip
      rec.ddp = backend_->ddp_stats();
    }
    rec.loss = loss;
    rec.tokens_per_s = static_cast<double>(tokens_per_step()) / (step_ms / 1000.0);
    const double peak = backend_->peak_flops();
    rec.mfu = peak > 0.0 ? flops_per_token() * rec.tokens_per_s / (peak * dp_) : std::numeric_limits<double>::quiet_NaN();
    rec.peak_memory = backend_->peak_memory_bytes();

    const bool anomaly = !std::isfinite(loss) || (!backend_->loss_scaled() && (result.skipped || !std::isfinite(result.grad_norm)));
    if (anomaly) {
      NanEvent ev;
      ev.step = step_;
      ev.loss = loss;
      ev.grad_norm = result.grad_norm;
      if (nan_.diagnose) {
        for (int i = 0; i < accum_; ++i) {
          LmNonFinite rep = backend_->diagnose(tokens_.data() + static_cast<std::size_t>(i) * micro_tokens);
          const bool bad = !rep.first_activation.empty() || !rep.gradients.empty() || !rep.parameters.empty() ||
                           !std::isfinite(rep.loss);
          if (bad || i == 0) {
            rep.micro_batch = bad ? i : -1;
            rep.reproduced = bad;
            ev.report = std::move(rep);
          }
          if (bad) {
            break;
          }
        }
      }
      history_.push_back(rec);
      if (nan_.policy == NanPolicy::Stop) {
        ev.action = "stop";
        nan_events_.push_back(ev);
        step_ += 1;
        summary.stopped = true;
        summary.reason = "NaN/Inf adim " + std::to_string(ev.step) +
                         (ev.report.first_activation.empty() ? std::string() : ", ilk: " + ev.report.first_activation);
        break;
      }
      if (nan_.policy == NanPolicy::Rollback) {
        if (rollbacks_ >= nan_.max_rollbacks) {
          ev.action = "stop (geri donus siniri)";
          nan_events_.push_back(ev);
          step_ += 1;
          summary.stopped = true;
          summary.reason = "NaN/Inf: geri donus siniri asildi";
          break;
        }
        backend_->load_state(snapshot_.backend);
        lr_factor_ *= nan_.rollback_lr_factor;
        rollbacks_ += 1;
        std::ostringstream act;
        act << "rollback -> adim " << snapshot_.step << ", lr x" << lr_factor_;
        ev.action = act.str();
        nan_events_.push_back(ev);
        step_ = snapshot_.step;  // veri ileri devam eder: kotu yigin yeniden gorulmez
        recent_.clear();
        continue;
      }
      ev.action = "skip";
      nan_events_.push_back(ev);
      step_ += 1;
      consecutive_skips_ += 1;
      if (consecutive_skips_ > nan_.max_consecutive_skips) {
        summary.stopped = true;
        summary.reason = "NaN/Inf: art arda " + std::to_string(consecutive_skips_) + " adim atlandi";
        break;
      }
      continue;
    }
    consecutive_skips_ = 0;
    if (spike_.enabled) {
      detect_spike(loss);
    }
    step_ += 1;
    if (nan_.policy == NanPolicy::Rollback && step_ % nan_.snapshot_every == 0) {
      take_snapshot();
    }
    std::vector<LmNormEntry> norms;
    const bool want_norms = histogram_every_ > 0 && step_ % histogram_every_ == 0;
    if (want_norms) {
      norms = backend_->layer_norms();  // bu adimin (kirpma sonrasi) gradyani ve guncel agirlik
      norms_.push_back(norms);
    }
    const bool want_eval = eval_every_ > 0 && step_ % eval_every_ == 0;
    if (want_eval) {
      rec.val_loss = evaluate().loss;
    }
    history_.push_back(rec);
    summary.last_loss = loss;
    loss_scale_ = result.loss_scale;
    if (step_ % log_every_ == 0 || want_eval || want_norms) {
      log(rec, want_norms ? &norms : nullptr);
    }
    if (ckpt_ && ckpt_->due(step_)) {
      save_checkpoint();
    }
  }
  if (ckpt_) {
    ckpt_->wait();  // eszamansiz yazimlar train() donmeden biter (hata varsa burada firlar)
  }
  summary.steps = step_;
  if (!history_.empty() && !summary.stopped) {
    summary.last_loss = history_.back().loss;
  }
  return summary;
}

void LanguageModelTrainer::log(const StepRecord& rec, const std::vector<LmNormEntry>* norms) {
  nexus_train::TrainingState state;
  state.global_step = static_cast<int>(step_);
  state.current_loss = rec.loss;
  state.metrics["train/lr"] = rec.lr;
  state.metrics["train/grad_norm"] = rec.grad_norm;
  state.metrics["train/skipped_steps"] = static_cast<double>(rec.skipped_total);
  state.metrics["perf/tokens_per_s"] = rec.tokens_per_s;
  state.metrics["perf/step_ms"] = rec.step_ms;
  state.metrics["perf/data_ms"] = rec.data_ms;
  if (std::isfinite(rec.mfu)) {
    state.metrics["perf/mfu"] = rec.mfu;
  }
  if (rec.peak_memory > 0) {
    state.metrics["perf/peak_memory_gb"] = static_cast<double>(rec.peak_memory) / 1.0e9;
  }
  const bool has_val = std::isfinite(rec.val_loss);
  if (has_val) {
    state.metrics["val/loss"] = rec.val_loss;
    state.metrics["val/ppl"] = std::exp(rec.val_loss);
  }
  if (norms != nullptr) {
    for (const LmNormEntry& n : *norms) {
      state.metrics["norm/weight/" + n.name] = n.weight;
      state.metrics["norm/grad/" + n.name] = n.grad;
    }
  }
  for (const auto& sink : loggers_) {
    sink->log_step(state);
  }
  if (console_) {
    std::printf("adim %6lld | kayip %.4f | lr %.3e | norm %.3f | %.0f tok/s", static_cast<long long>(step_), rec.loss,
                rec.lr, rec.grad_norm, rec.tokens_per_s);
    if (std::isfinite(rec.mfu)) {
      std::printf(" | MFU %.1f%%", 100.0 * rec.mfu);
    }
    if (rec.peak_memory > 0) {
      std::printf(" | bellek %.2f GB", static_cast<double>(rec.peak_memory) / 1.0e9);
    }
    std::printf(" | %.0f ms", rec.step_ms);
    if (has_val) {
      std::printf(" | dogrulama %.4f (ppl %.2f)", rec.val_loss, std::exp(rec.val_loss));
    }
    std::printf("\n");
    std::fflush(stdout);
  }
}

EvalResult LanguageModelTrainer::evaluate() {
  if (val_ == nullptr) {
    throw std::logic_error("LanguageModelTrainer::evaluate: dogrulama kaynagi yok");
  }
  const std::string saved = val_->state();
  val_->restore(val_origin_);  // her degerlendirmede ayni token kumesi
  std::vector<std::int32_t> buf(static_cast<std::size_t>(micro_batch_) * seq_len_);
  double sum = 0.0;
  std::int64_t tokens = 0;
  for (std::int64_t k = 0; k < eval_batches_; ++k) {
    val_->next(buf.data(), micro_batch_, seq_len_, nullptr);
    std::int64_t valid = 0;
    for (int r = 0; r < micro_batch_; ++r) {
      for (int t = 1; t < seq_len_; ++t) {
        valid += buf[static_cast<std::size_t>(r) * seq_len_ + t] != kIgnoreIndex ? 1 : 0;
      }
    }
    sum += backend_->eval_loss(buf.data()) * static_cast<double>(valid);
    tokens += valid;
  }
  val_->restore(saved);
  if (dp_ > 1) {  // her rank kendi dogrulama parcasini degerlendirir
    group_->all_reduce(&sum, 1, DType::F64, dist::ReduceOp::Sum)->wait();
    group_->all_reduce(&tokens, 1, DType::I64, dist::ReduceOp::Sum)->wait();
  }
  EvalResult r;
  r.tokens = tokens;
  r.loss = tokens > 0 ? sum / static_cast<double>(tokens) : std::numeric_limits<double>::quiet_NaN();
  r.perplexity = std::exp(r.loss);
  return r;
}

std::vector<double> LanguageModelTrainer::score_choices(const MultipleChoiceItem& item) {
  if (item.choices.empty()) {
    throw std::invalid_argument("score_choices: en az bir secenek");
  }
  std::vector<double> scores;
  std::vector<std::int32_t> batch(static_cast<std::size_t>(micro_batch_) * seq_len_, 0);
  std::vector<std::pair<int, int>> spans;  // satir basina (secenek basi, secenek uzunlugu)
  auto flush = [&]() {
    if (spans.empty()) {
      return;
    }
    const std::vector<float> losses = backend_->token_losses(batch.data());
    for (std::size_t r = 0; r < spans.size(); ++r) {
      double logp = 0.0;
      for (int j = spans[r].first; j < spans[r].first + spans[r].second; ++j) {
        logp -= losses[r * static_cast<std::size_t>(seq_len_) + static_cast<std::size_t>(j - 1)];  // j'yi tahmin eden konum
      }
      scores.push_back(logp);
    }
    spans.clear();
  };
  for (const std::vector<std::int32_t>& choice : item.choices) {
    if (choice.empty() || static_cast<int>(choice.size()) >= seq_len_) {
      throw std::invalid_argument("score_choices: secenek bos olamaz ve seq_len'den kisa olmali");
    }
    // Baglam soldan kirpilir; satir = baglam + secenek, kalan konumlar 0 (nedensel: etkisiz).
    const int keep = std::min(static_cast<int>(item.context.size()), seq_len_ - static_cast<int>(choice.size()));
    if (keep < 1) {
      throw std::invalid_argument("score_choices: en az bir baglam tokeni gerekli");
    }
    const std::size_t row = spans.size();
    std::int32_t* dst = batch.data() + row * static_cast<std::size_t>(seq_len_);
    std::fill(dst, dst + seq_len_, 0);
    std::copy(item.context.end() - keep, item.context.end(), dst);
    std::copy(choice.begin(), choice.end(), dst + keep);
    spans.emplace_back(keep, static_cast<int>(choice.size()));
    if (static_cast<int>(spans.size()) == micro_batch_) {
      flush();
    }
  }
  flush();
  return scores;
}

namespace {
constexpr char kMagic[8] = {'N', 'X', 'L', 'M', 'T', 'R', '9', '\0'};
}

void LanguageModelTrainer::save(const std::string& path) const {
  Writer w;
  w.out.insert(w.out.end(), kMagic, kMagic + 8);
  w.pod<std::uint32_t>(1);
  w.pod(step_);
  w.pod(lr_factor_);
  w.pod<std::int32_t>(rollbacks_);
  w.pod<std::int32_t>(accum_);
  w.pod<std::int32_t>(micro_batch_);
  w.pod<std::int32_t>(seq_len_);
  // Onceden alinmis (prefetch) yigin henuz kullanilmadi: kaynak durumu ondan onceki hal.
  w.str(prefetched_ ? prefetch_state_ : train_->state());
  w.str(val_ != nullptr ? val_->state() : std::string());
  const std::vector<unsigned char> blob = backend_->save_state();
  w.bytes(blob.data(), blob.size());
  std::ofstream out(rank_path(path), std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(w.out.data()), static_cast<std::streamsize>(w.out.size()));
  if (!out) {
    throw std::runtime_error("LanguageModelTrainer::save: yazilamadi " + rank_path(path));
  }
}

void LanguageModelTrainer::resume(const std::string& path) {
  std::ifstream in(rank_path(path), std::ios::binary);
  if (!in) {
    throw std::runtime_error("LanguageModelTrainer::resume: acilamadi " + rank_path(path));
  }
  std::vector<unsigned char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (data.size() < 8 || std::memcmp(data.data(), kMagic, 8) != 0) {
    throw std::runtime_error("LanguageModelTrainer::resume: Nexus LM egitmen kaydi degil");
  }
  std::vector<unsigned char> body(data.begin() + 8, data.end());
  Reader r(body);
  if (r.pod<std::uint32_t>() != 1) {
    throw std::runtime_error("LanguageModelTrainer::resume: desteklenmeyen surum");
  }
  const auto step = r.pod<std::int64_t>();
  const auto lr_factor = r.pod<double>();
  const auto rollbacks = r.pod<std::int32_t>();
  const auto accum = r.pod<std::int32_t>();
  const auto mb = r.pod<std::int32_t>();
  const auto sl = r.pod<std::int32_t>();
  if (accum != accum_ || mb != micro_batch_ || sl != seq_len_) {
    throw std::runtime_error("LanguageModelTrainer::resume: kayit farkli bir yigin yapilandirmasindan");
  }
  const std::string train_state = r.str();
  const std::string val_state = r.str();
  backend_->load_state(r.bytes());
  train_->restore(train_state);
  if (val_ != nullptr && !val_state.empty()) {
    val_->restore(val_state);
  }
  step_ = step;
  lr_factor_ = lr_factor;
  rollbacks_ = rollbacks;
  prefetched_ = false;
  recent_.clear();
}

ckpt::TrainerState LanguageModelTrainer::trainer_state() const {
  ckpt::TrainerState st;
  st.step = step_;
  st.global_sample_index = static_cast<std::uint64_t>(step_) * static_cast<std::uint64_t>(accum_ * micro_batch_ * dp_);
  st.train_cursor = prefetched_ ? prefetch_state_ : train_->state();  // onceden alinan yigin henuz kullanilmadi
  st.val_cursor = val_ != nullptr ? val_->state() : std::string();
  st.lr_factor = lr_factor_;
  st.schedule_step = step_;
  st.loss_scale = loss_scale_;
  st.rng = ckpt::RngState{rng_seed_, rng_offset_};
  st.accum_counter = 0;
  st.accum_steps = accum_;
  st.micro_batch = micro_batch_;
  st.seq_len = seq_len_;
  st.global_batch_tokens = global_batch_tokens_;
  st.rollbacks = rollbacks_;
  st.world = dp_;
  st.rank = rank_;
  return st;
}

void LanguageModelTrainer::save_checkpoint() {
  if (!ckpt_) {
    throw std::logic_error("LanguageModelTrainer::save_checkpoint: builder.checkpointing() yok");
  }
  const ckpt::TrainerState st = trainer_state();
  // Ilk tampon: arka uc durumu burada (egitim is parcaciginda) kopyalanir; yazma eszamansiz kipte arka planda.
  ckpt_->save(step_, st, backend_->save_state(), backend_->state_layout());
}

bool LanguageModelTrainer::resume_latest() {
  if (!ckpt_) {
    throw std::logic_error("LanguageModelTrainer::resume_latest: builder.checkpointing() yok");
  }
  std::optional<ckpt::LoadedCheckpoint> lc = ckpt_->load_latest();
  if (!lc) {
    return false;
  }
  const ckpt::TrainerState& st = lc->states.front();
  // Yeniden parcalamada rank basina micro_batch degisebilir; dizi uzunlugu ve kuresel yigin ayni olmali.
  if ((!lc->resharded && st.micro_batch != micro_batch_) || st.seq_len != seq_len_ ||
      st.global_batch_tokens != global_batch_tokens_) {
    throw std::runtime_error("LanguageModelTrainer::resume_latest: kayit farkli yigin yapilandirmasindan (micro " +
                             std::to_string(st.micro_batch) + ", T " + std::to_string(st.seq_len) + ", kuresel token " +
                             std::to_string(st.global_batch_tokens) + ")");
  }
  if (!lc->resharded) {
    backend_->load_state(lc->payloads.front());
    train_->restore(st.train_cursor);
    if (val_ != nullptr && !st.val_cursor.empty()) {
      val_->restore(st.val_cursor);
    }
  } else {
    if (!train_->global_cursor()) {
      throw std::runtime_error("LanguageModelTrainer::resume_latest: kayit " + std::to_string(lc->manifest.world) +
                               " rank'te alinmis; veri kaynagi kuresel imlec desteklemiyor (global_cursor)");
    }
    for (const ckpt::TrainerState& o : lc->states) {
      if (o.train_cursor != st.train_cursor || o.step != st.step) {
        throw std::runtime_error("LanguageModelTrainer::resume_latest: rank'lerin veri imlecleri/adimlari farkli");
      }
    }
    backend_->load_resharded_state(lc->payloads);
    train_->restore(st.train_cursor);
    if (val_ != nullptr && !st.val_cursor.empty()) {
      val_->restore(st.val_cursor);
    }
  }
  step_ = st.step;
  lr_factor_ = st.lr_factor;
  rollbacks_ = st.rollbacks;
  loss_scale_ = st.loss_scale;
  rng_seed_ = st.rng.seed;
  rng_offset_ = st.rng.offset;
  prefetched_ = false;
  recent_.clear();
  return true;
}

}  // namespace train
}  // namespace nexus
