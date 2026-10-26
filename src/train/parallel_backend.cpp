/**
 * @file parallel_backend.cpp
 * @brief FAZ 12.8: LanguageModelTrainer icin dist::ParallelLm arka ucu (lm::LanguageModel, DP x PP x CP x TP).
 *
 * Biriktirme mikro adimlari pipeline mikro yiginlaridir: micro_step tokenlari tamponlar, son mikro adimda
 * tum zamanlama (ileri + geri + indirgemeler) kosar. Desteklenmeyenler (reshape, token_losses, diagnose,
 * layer_norms) acik hata verir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/parallel_lm.hpp"
#include "nexus/lm/model.hpp"
#include "nexus/train/lm_trainer.hpp"

#include <stdexcept>

namespace nexus {
namespace train {

namespace {

class ParallelLmBackend final : public LmBackend {
 public:
  ParallelLmBackend(const lm::LanguageModel& model, std::shared_ptr<dist::ProcessGroup> world,
                    const dist::ParallelConfig& config, const LmBackendRequest& request)
      : model_(&model), request_(request) {
    if (request.precision != Precision::F32) {
      throw dist::DistError("Nexus paralel arka uc: yalniz FP32 (CPU)");
    }
    if (request.compile) {
      throw dist::DistError("Nexus paralel arka uc: trainer.compile desteklenmiyor (CPU derleyici: nexus::compile::compile)");
    }
    if (!world) {
      single_ = std::make_unique<dist::ThreadWorld>(1);
      world = single_->create_group(0);
    }
    pl_ = std::make_unique<dist::ParallelLm>(model, std::move(world), config);
  }

  std::string device() const override {
    const dist::ParallelConfig& c = pl_->config();
    return "cpu-parallel(dp" + std::to_string(c.dp) + " pp" + std::to_string(c.pp) + " cp" + std::to_string(c.cp) +
           " tp" + std::to_string(c.tp) + ")";
  }

  LmShape shape() const override {
    const lm::LmConfig& c = model_->config();
    LmShape s;
    s.params = model_->num_parameters();
    s.params_no_pos = s.params;
    if (const lm::NamedParam* p = model_->find("wpe")) s.params_no_pos -= p->value.numel();
    s.layers = c.n_layer;
    s.heads = c.n_head;
    s.head_dim = c.hdim();
    return s;
  }

  std::vector<LmParamInfo> parameters() const override {
    std::vector<LmParamInfo> out;
    for (const dist::ParallelLm::Param& p : pl_->parameters()) {
      LmParamInfo i;
      i.name = p.name;
      i.numel = p.value.size();
      i.rank = static_cast<int>(p.full_shape.size());
      i.embedding = p.name == "wte" || p.name == "wpe";
      out.push_back(i);
    }
    return out;
  }

  void set_param_scales(const std::vector<float>& weight_decay, const std::vector<float>& lr) override {
    pl_->set_param_scales(weight_decay, lr);
  }

  int micro_batch() const override { return request_.micro_batch; }
  int seq_len() const override { return request_.seq_len; }

  void reshape(int, int) override { throw dist::DistError("Nexus paralel arka uc: reshape desteklenmiyor"); }

  void micro_step(const std::int32_t* tokens, float grad_weight, int index, int count) override {
    const std::size_t n = static_cast<std::size_t>(request_.micro_batch) * request_.seq_len;
    if (index == 0) {
      buffers_.assign(static_cast<std::size_t>(count), std::vector<std::int32_t>(n));
      weights_.assign(static_cast<std::size_t>(count), 0.0f);
    }
    std::copy(tokens, tokens + n, buffers_[static_cast<std::size_t>(index)].begin());
    weights_[static_cast<std::size_t>(index)] = grad_weight;
    if (index + 1 == count) {
      std::vector<const std::int32_t*> micro;
      for (const auto& b : buffers_) micro.push_back(b.data());
      losses_ = pl_->forward_backward(micro, weights_, request_.micro_batch, request_.seq_len);
    }
  }

  void optimizer_step(const AdamW& opt, double lr, double clip) override {
    const dist::ParallelLm::StepResult r =
        pl_->step(dist::FsdpAdamW{opt.lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay}, lr, clip);
    norm_ = r.grad_norm;
    skipped_ = r.skipped;
    skipped_total_ += r.skipped ? 1 : 0;
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
    return pl_->eval_loss(tokens, request_.micro_batch, request_.seq_len);
  }

  std::vector<float> token_losses(const std::int32_t*) override {
    throw dist::DistError("Nexus paralel arka uc: token_losses desteklenmiyor (score_choices icin tek rank CPU arka ucu)");
  }

  LmNonFinite diagnose(const std::int32_t*) override {
    throw dist::DistError("Nexus paralel arka uc: NaN teshisi (diagnose) desteklenmiyor; NanOptions::diagnose = false");
  }

  std::vector<unsigned char> save_state() const override { return pl_->save_state(); }
  void load_state(const std::vector<unsigned char>& blob) override { pl_->load_state(blob); }

  std::vector<LmNormEntry> layer_norms() override {
    throw dist::DistError("Nexus paralel arka uc: katman norm histogrami desteklenmiyor (histogram_every = 0)");
  }

  std::uint64_t peak_memory_bytes() const override { return 0; }
  void reset_peak_memory() override {}
  double peak_flops() const override { return 0.0; }

  std::shared_ptr<dist::ProcessGroup> data_parallel_group() const override { return pl_->data_parallel_group(); }
  dist::ParallelLm* parallel_model() noexcept override { return pl_.get(); }

 private:
  const lm::LanguageModel* model_;
  LmBackendRequest request_;
  std::unique_ptr<dist::ThreadWorld> single_;
  std::unique_ptr<dist::ParallelLm> pl_;
  std::vector<std::vector<std::int32_t>> buffers_;
  std::vector<float> weights_;
  std::vector<float> losses_;
  double norm_ = 0.0;
  bool skipped_ = false;
  std::int64_t skipped_total_ = 0;
};

}  // namespace

BackendFactory parallel_backend(const lm::LanguageModel& model, std::shared_ptr<dist::ProcessGroup> world,
                                dist::ParallelConfig config) {
  return [&model, world, config](const LmBackendRequest& request) -> std::unique_ptr<LmBackend> {
    return std::make_unique<ParallelLmBackend>(model, world, config, request);
  };
}

}  // namespace train
}  // namespace nexus
