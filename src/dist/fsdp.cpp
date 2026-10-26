/**
 * @file fsdp.cpp
 * @brief FAZ 11: FsdpEngine — duz birimler, ZeRO-0/1/2/3, HSDP, onceden getirme, NVMe takasi, parcali kayit.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/dist/fsdp.hpp"

#include "nexus/core/half.hpp"
#include "nexus/nn/modules.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>

namespace nexus {
namespace dist {

const char* to_string(ShardingStrategy s) noexcept {
  switch (s) {
    case ShardingStrategy::NoShard: return "NO_SHARD (ZeRO-0)";
    case ShardingStrategy::ShardOptimizer: return "ZeRO-1";
    case ShardingStrategy::ShardGradOp: return "SHARD_GRAD_OP (ZeRO-2)";
    case ShardingStrategy::FullShard: return "FULL_SHARD (ZeRO-3)";
    case ShardingStrategy::Hybrid: return "HYBRID_SHARD";
  }
  return "?";
}

ShardingStrategy zero_stage(int stage) {
  switch (stage) {
    case 0: return ShardingStrategy::NoShard;
    case 1: return ShardingStrategy::ShardOptimizer;
    case 2: return ShardingStrategy::ShardGradOp;
    case 3: return ShardingStrategy::FullShard;
    default: throw DistError("Nexus FSDP: ZeRO asamasi 0..3");
  }
}

struct FsdpEngine::Unit {
  std::vector<float> master;       // FP32 ana kopya parcasi [S] (ZeRO-1/2/3); NoShard'da bos
  std::vector<float> m, v;         // [S] (NoShard: [P])
  std::vector<float> grad_shard;   // [S] RS sonucu
  std::vector<float> full;         // [P] gecici: AG cikisi / NoShard paket
  std::vector<std::uint16_t> half_in, half_out;
  std::vector<float> reduce_in;    // [P] gecici RS girdisi
  WorkPtr gather_work, reduce_work;
  bool gathered = true;            // parametre tensorleri bellekte mi
  bool gathering = false;
  bool reduce_pending = false;
  bool full_grads = false;         // NoShard/ZeRO-1: 'full' tam indirgenmis gradyani tutuyor
  struct Segment {
    std::size_t begin, end;
    float wd, lr;
  };
  std::vector<Segment> segments;   // duz uzayda [begin, end) -> carpanlar
  AlignedBuffer swap_m, swap_v, swap_master;
  bool m_on_disk = false, master_on_disk = false;
};

namespace {
nexus_model::Tensor empty_tensor() { return nexus_model::Tensor(); }

void to_half(const float* src, std::uint16_t* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = float_to_bfloat16_bits(src[i]);
}
void from_half(const std::uint16_t* src, float* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = bfloat16_bits_to_float(src[i]);
}
}  // namespace

FsdpEngine::FsdpEngine(nn::Sequential& model, GroupPtr group, FsdpOptions options)
    : model_(&model), group_(std::move(group)), options_(options) {
  if (!group_) throw DistError("Nexus FSDP: ProcessGroup gerekli");
  const DType pd = options_.mixed_precision.param;
  const DType rd = options_.mixed_precision.reduce;
  if ((pd != DType::F32 && pd != DType::BF16) || (rd != DType::F32 && rd != DType::BF16)) {
    throw DistError("Nexus FSDP: param/reduce tipi F32 ya da BF16");
  }
  if (options_.offload.params_nvme && !sharded_params()) {
    throw DistError("Nexus FSDP: parametre NVMe takasi yalniz FULL_SHARD / HYBRID ile");
  }
  // Parca grubu.
  shard_group_ = group_;
  if (options_.sharding == ShardingStrategy::Hybrid) {
    const int k = options_.ranks_per_node > 0 ? options_.ranks_per_node : env_int("LOCAL_WORLD_SIZE", group_->size());
    if (k < 1 || group_->size() % k != 0) throw DistError("Nexus FSDP: HSDP icin ranks_per_node dunya boyunu bolmeli");
    const int nodes = group_->size() / k;
    GroupPtr intra, inter;
    for (int node = 0; node < nodes; ++node) {
      std::vector<int> r;
      for (int j = 0; j < k; ++j) r.push_back(node * k + j);
      if (GroupPtr g = group_->new_group(r)) intra = g;
    }
    for (int local = 0; local < k; ++local) {
      std::vector<int> r;
      for (int node = 0; node < nodes; ++node) r.push_back(node * k + local);
      if (GroupPtr g = group_->new_group(r)) inter = g;
    }
    shard_group_ = intra;
    replica_group_ = nodes > 1 ? inter : nullptr;
  }
  shard_world_ = shard_group_->size();
  shard_rank_ = shard_group_->rank();
  // Birimler (sarmalama politikasi).
  params_ = model.parameters();
  for (nexus_model::Parameter* p : params_) shapes_.push_back(p->data.shape_vec());
  std::size_t pi = 0;
  auto add_unit = [&](const std::string& name, int layer, std::size_t count) {
    FsdpUnit u;
    u.name = name;
    u.layer = layer;
    for (std::size_t j = 0; j < count; ++j, ++pi) {
      u.params.push_back(pi);
      u.begin.push_back(u.numel);
      u.numel += params_[pi]->data.numel();
    }
    u.shard = (u.numel + static_cast<std::size_t>(shard_world_) - 1) / static_cast<std::size_t>(shard_world_);
    units_.push_back(std::move(u));
  };
  if (options_.wrap == FsdpOptions::Wrap::Whole) {
    add_unit("model", -1, params_.size());
  } else {
    for (std::size_t i = 0; i < model.size(); ++i) {
      const std::size_t count = model.layer(i)->parameters().size();
      if (count > 0) add_unit(std::to_string(i), static_cast<int>(i), count);
    }
  }
  for (std::size_t u = 0; u < units_.size(); ++u) state_.push_back(std::make_unique<Unit>());
  wd_.assign(params_.size(), 1.0f);
  lr_.assign(params_.size(), 1.0f);
  if (options_.offload.optimizer_nvme || options_.offload.params_nvme) {
    std::string dir = options_.offload.directory;
    if (dir.empty()) dir = (std::filesystem::temp_directory_path() / "nexus_zero_infinity").string();
    dir += "/rank" + std::to_string(group_->rank());
    nvme_ = std::make_unique<NvmeStore>(dir);
  }
}

FsdpEngine::~FsdpEngine() = default;

bool FsdpEngine::sharded_params() const noexcept {
  return options_.sharding == ShardingStrategy::FullShard || options_.sharding == ShardingStrategy::Hybrid;
}

void FsdpEngine::set_param_scales(const std::vector<float>& weight_decay, const std::vector<float>& lr) {
  if (weight_decay.size() != params_.size() || lr.size() != params_.size()) {
    throw DistError("Nexus FSDP: carpan sayisi parametre sayisina esit olmali");
  }
  wd_ = weight_decay;
  lr_ = lr;
}

void FsdpEngine::initialize() {
  std::vector<WorkPtr> works;
  for (nexus_model::Parameter* p : params_) works.push_back(group_->broadcast(p->data.data(), p->data.numel(), DType::F32, 0));
  for (const WorkPtr& w : works) w->wait();
  for (std::size_t u = 0; u < units_.size(); ++u) {
    FsdpUnit& fu = units_[u];
    Unit& st = *state_[u];
    const std::size_t P = fu.shard * static_cast<std::size_t>(shard_world_);
    st.segments.clear();
    for (std::size_t j = 0; j < fu.params.size(); ++j) {
      const std::size_t p = fu.params[j];
      st.segments.push_back({fu.begin[j], fu.begin[j] + params_[p]->data.numel(), wd_[p], lr_[p]});
    }
    if (options_.sharding == ShardingStrategy::NoShard) {
      st.m.assign(P, 0.0f);
      st.v.assign(P, 0.0f);
    } else {
      std::vector<float> flat(P, 0.0f);
      for (std::size_t j = 0; j < fu.params.size(); ++j) {
        const nexus_model::Parameter* p = params_[fu.params[j]];
        std::memcpy(flat.data() + fu.begin[j], p->data.data(), p->data.numel() * 4);
      }
      const std::size_t off = static_cast<std::size_t>(shard_rank_) * fu.shard;
      st.master.assign(flat.begin() + static_cast<std::ptrdiff_t>(off), flat.begin() + static_cast<std::ptrdiff_t>(off + fu.shard));
      st.m.assign(fu.shard, 0.0f);
      st.v.assign(fu.shard, 0.0f);
    }
    st.grad_shard.assign(fu.shard, 0.0f);
    if (sharded_params()) {
      release_params(u);
      for (std::size_t p : fu.params) params_[p]->grad = empty_tensor();
    }
    if (options_.offload.params_nvme) store_master(u);
    if (options_.offload.optimizer_nvme) store_optimizer(u);
  }
}

std::size_t FsdpEngine::shard_offset() const noexcept { return static_cast<std::size_t>(shard_rank_); }

// ------------------------------------------------------------------------------ NVMe takasi
void FsdpEngine::store_master(std::size_t u) {
  Unit& st = *state_[u];
  st.swap_master = AlignedBuffer(st.master.size() * 4);
  std::memcpy(st.swap_master.data(), st.master.data(), st.master.size() * 4);
  nvme_->write("u" + std::to_string(u) + "_master", st.swap_master)->wait();
  st.swap_master.reset();
  st.master.clear();
  st.master.shrink_to_fit();
  st.master_on_disk = true;
}

void FsdpEngine::load_master(std::size_t u) {
  Unit& st = *state_[u];
  if (!st.master_on_disk) return;
  const std::size_t n = units_[u].shard;
  st.swap_master = AlignedBuffer(n * 4);
  nvme_->read("u" + std::to_string(u) + "_master", st.swap_master)->wait();
  st.master.assign(st.swap_master.as<float>(), st.swap_master.as<float>() + n);
  st.swap_master.reset();
  st.master_on_disk = false;
}

void FsdpEngine::store_optimizer(std::size_t u) {
  Unit& st = *state_[u];
  st.swap_m = AlignedBuffer(st.m.size() * 4);
  st.swap_v = AlignedBuffer(st.v.size() * 4);
  std::memcpy(st.swap_m.data(), st.m.data(), st.m.size() * 4);
  std::memcpy(st.swap_v.data(), st.v.data(), st.v.size() * 4);
  auto a = nvme_->write("u" + std::to_string(u) + "_m", st.swap_m);
  auto b = nvme_->write("u" + std::to_string(u) + "_v", st.swap_v);
  a->wait();
  b->wait();
  st.swap_m.reset();
  st.swap_v.reset();
  st.m.clear();
  st.m.shrink_to_fit();
  st.v.clear();
  st.v.shrink_to_fit();
  st.m_on_disk = true;
}

void FsdpEngine::load_optimizer(std::size_t u) {
  Unit& st = *state_[u];
  if (!st.m_on_disk) return;
  const std::size_t n = options_.sharding == ShardingStrategy::NoShard
                            ? units_[u].shard * static_cast<std::size_t>(shard_world_)
                            : units_[u].shard;
  st.swap_m = AlignedBuffer(n * 4);
  st.swap_v = AlignedBuffer(n * 4);
  auto a = nvme_->read("u" + std::to_string(u) + "_m", st.swap_m);
  auto b = nvme_->read("u" + std::to_string(u) + "_v", st.swap_v);  // iki istek ayni anda ucusta
  a->wait();
  b->wait();
  st.m.assign(st.swap_m.as<float>(), st.swap_m.as<float>() + n);
  st.v.assign(st.swap_v.as<float>(), st.swap_v.as<float>() + n);
  st.swap_m.reset();
  st.swap_v.reset();
  st.m_on_disk = false;
}

// ------------------------------------------------------------------------------ parametre AG
void FsdpEngine::gather(std::size_t u) {
  Unit& st = *state_[u];
  if (st.gathered || st.gathering) return;
  load_master(u);
  const FsdpUnit& fu = units_[u];
  const std::size_t P = fu.shard * static_cast<std::size_t>(shard_world_);
  if (options_.mixed_precision.param == DType::BF16) {
    st.half_in.resize(fu.shard);
    st.half_out.resize(P);
    to_half(st.master.data(), st.half_in.data(), fu.shard);
    st.gather_work = shard_group_->all_gather(st.half_in.data(), st.half_out.data(), fu.shard, DType::BF16);
  } else {
    st.full.resize(P);
    st.gather_work = shard_group_->all_gather(st.master.data(), st.full.data(), fu.shard, DType::F32);
  }
  st.gathering = true;
  ++inflight_;
  max_inflight_ = std::max(max_inflight_, inflight_);
  trace_.push_back("AG " + fu.name);
}

void FsdpEngine::wait_gathered(std::size_t u) {
  Unit& st = *state_[u];
  if (st.gathered) return;
  if (!st.gathering) gather(u);
  st.gather_work->wait();
  st.gathering = false;
  --inflight_;
  const FsdpUnit& fu = units_[u];
  if (options_.mixed_precision.param == DType::BF16) {
    st.full.resize(st.half_out.size());
    from_half(st.half_out.data(), st.full.data(), st.half_out.size());
    st.half_in.clear();
    st.half_out.clear();
  }
  for (std::size_t j = 0; j < fu.params.size(); ++j) {
    nexus_model::Parameter* p = params_[fu.params[j]];
    p->data = nexus_model::Tensor::zeros(shapes_[fu.params[j]]);
    std::memcpy(p->data.data(), st.full.data() + fu.begin[j], p->data.numel() * 4);
  }
  st.full.clear();
  st.full.shrink_to_fit();
  if (options_.offload.params_nvme) {  // parca yeniden diske (adimda tekrar okunur)
    st.master.clear();
    st.master.shrink_to_fit();
    st.master_on_disk = true;
  }
  st.gathered = true;
}

void FsdpEngine::release_params(std::size_t u) {
  if (!sharded_params()) return;
  Unit& st = *state_[u];
  for (std::size_t p : units_[u].params) params_[p]->data = empty_tensor();
  st.gathered = false;
  trace_.push_back("FREE " + units_[u].name);
}

void FsdpEngine::prefetch_after(std::size_t position, bool backward) {
  if (!options_.prefetch || !sharded_params() || !order_recorded_) return;
  const std::size_t n = order_.size();
  for (std::size_t k = 1; k < n; ++k) {
    if (options_.limit_all_gathers > 0 && inflight_ >= options_.limit_all_gathers) break;
    std::size_t idx;
    if (backward) {
      if (position < k) break;
      idx = position - k;
    } else {
      idx = position + k;
      if (idx >= n) break;
    }
    const std::size_t u = static_cast<std::size_t>(order_[idx]);
    Unit& st = *state_[u];
    if (!st.gathered && !st.gathering) gather(u);
  }
}

nexus_model::Tensor FsdpEngine::forward(const nexus_model::Tensor& input) {
  std::map<int, std::size_t> unit_of_layer;
  for (std::size_t u = 0; u < units_.size(); ++u) unit_of_layer[units_[u].layer] = u;
  if (options_.wrap == FsdpOptions::Wrap::Whole) {
    if (sharded_params()) wait_gathered(0);
    trace_.push_back("F model");
    nexus_model::Tensor out = model_->forward(input);
    if (!order_recorded_) {
      order_ = {0};
      order_recorded_ = true;
    }
    return out;
  }
  const bool record = !order_recorded_;
  nexus_model::Tensor current = input;
  std::size_t position = 0;
  for (std::size_t i = 0; i < model_->size(); ++i) {
    const auto it = unit_of_layer.find(static_cast<int>(i));
    if (it == unit_of_layer.end()) {
      current = model_->layer(i)->forward(current);
      continue;
    }
    const std::size_t u = it->second;
    if (record) order_.push_back(static_cast<int>(u));
    if (sharded_params()) {
      wait_gathered(u);
      prefetch_after(position, false);  // sonraki birimin AG'si bu birimin hesabiyla ortusur
    }
    trace_.push_back("F " + units_[u].name);
    current = model_->layer(i)->forward(current);
    release_params(u);
    ++position;
  }
  if (record) order_recorded_ = true;
  return current;
}

void FsdpEngine::ensure_grads(std::size_t u) {
  for (std::size_t p : units_[u].params) {
    nexus_model::Parameter* q = params_[p];
    if (q->grad.numel() != q->data.numel()) q->grad = nexus_model::Tensor::zeros(shapes_[p]);
  }
}

void FsdpEngine::zero_grad() {
  for (nexus_model::Parameter* p : params_) {
    if (p->grad.numel() != 0) p->grad.zero();
  }
}

void FsdpEngine::launch_reduce(std::size_t u) {
  Unit& st = *state_[u];
  const FsdpUnit& fu = units_[u];
  const std::size_t P = fu.shard * static_cast<std::size_t>(shard_world_);
  st.reduce_in.assign(P, 0.0f);
  for (std::size_t j = 0; j < fu.params.size(); ++j) {
    const nexus_model::Parameter* p = params_[fu.params[j]];
    std::memcpy(st.reduce_in.data() + fu.begin[j], p->grad.data(), p->grad.numel() * 4);
  }
  if (options_.sharding == ShardingStrategy::ShardGradOp || sharded_params()) {
    for (std::size_t p : fu.params) params_[p]->grad = empty_tensor();  // tam gradyan paketlendi: serbest
  }
  if (options_.mixed_precision.reduce == DType::BF16) {
    st.half_in.resize(P);
    st.half_out.resize(fu.shard);
    to_half(st.reduce_in.data(), st.half_in.data(), P);
    st.reduce_work = shard_group_->reduce_scatter(st.half_in.data(), st.half_out.data(), fu.shard, DType::BF16, ReduceOp::Sum);
  } else {
    st.reduce_work = shard_group_->reduce_scatter(st.reduce_in.data(), st.grad_shard.data(), fu.shard, DType::F32, ReduceOp::Sum);
  }
  st.reduce_pending = true;
  trace_.push_back("RS " + fu.name);
}

void FsdpEngine::finish_reduce(std::size_t u) {
  Unit& st = *state_[u];
  if (!st.reduce_pending) return;
  st.reduce_work->wait();
  st.reduce_pending = false;
  const FsdpUnit& fu = units_[u];
  if (options_.mixed_precision.reduce == DType::BF16) {
    from_half(st.half_out.data(), st.grad_shard.data(), fu.shard);
    st.half_in.clear();
    st.half_out.clear();
  }
  st.reduce_in.clear();
  st.reduce_in.shrink_to_fit();
  if (replica_group_) {  // HSDP: dugumler arasi replika toplami
    replica_group_->all_reduce(st.grad_shard.data(), fu.shard, DType::F32, ReduceOp::Sum)->wait();
  }
  if (options_.sharding == ShardingStrategy::NoShard || options_.sharding == ShardingStrategy::ShardOptimizer) {
    // all-reduce = RS + AG: tam indirgenmis gradyan ('full' tamponunda ve gradyan tensorlerinde).
    const std::size_t P = fu.shard * static_cast<std::size_t>(shard_world_);
    st.full.resize(P);
    shard_group_->all_gather(st.grad_shard.data(), st.full.data(), fu.shard, DType::F32)->wait();
    for (std::size_t j = 0; j < fu.params.size(); ++j) {
      nexus_model::Parameter* p = params_[fu.params[j]];
      std::memcpy(p->grad.data(), st.full.data() + fu.begin[j], p->grad.numel() * 4);
    }
    st.full_grads = true;
  }
}

void FsdpEngine::backward(const nexus_model::Tensor& grad_output, bool sync) {
  std::map<int, std::size_t> unit_of_layer;
  for (std::size_t u = 0; u < units_.size(); ++u) unit_of_layer[units_[u].layer] = u;
  if (options_.wrap == FsdpOptions::Wrap::Whole) {
    if (sharded_params()) wait_gathered(0);
    ensure_grads(0);
    trace_.push_back("B model");
    model_->backward(grad_output);
    if (sync) launch_reduce(0);
    release_params(0);
    return;
  }
  nexus_model::Tensor grad = grad_output;
  std::size_t position = order_.size();
  for (std::size_t i = model_->size(); i-- > 0;) {
    const auto it = unit_of_layer.find(static_cast<int>(i));
    if (it == unit_of_layer.end()) {
      grad = model_->layer(i)->backward(grad);
      continue;
    }
    const std::size_t u = it->second;
    --position;
    if (sharded_params()) {
      wait_gathered(u);
      prefetch_after(position, true);  // onceki birimin AG'si bu birimin geri gecisiyle ortusur
    }
    ensure_grads(u);
    trace_.push_back("B " + units_[u].name);
    grad = model_->layer(i)->backward(grad);
    if (sync) launch_reduce(u);  // RS asenkron: kalan geri gecisle ortusur
    release_params(u);
  }
}

FsdpEngine::StepResult FsdpEngine::step(const FsdpAdamW& opt, double lr, double clip) {
  for (std::size_t u = 0; u < units_.size(); ++u) finish_reduce(u);
  // Global norm parca gradyanlarindan (tum asamalarda ayni toplama sirasi).
  double sq = 0.0;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    for (float g : state_[u]->grad_shard) sq += static_cast<double>(g) * g;
  }
  shard_group_->all_reduce(&sq, 1, DType::F64, ReduceOp::Sum)->wait();
  StepResult r;
  r.grad_norm = std::sqrt(sq);
  r.skipped = !std::isfinite(r.grad_norm);
  auto clear_grads = [&] {
    for (std::size_t u = 0; u < units_.size(); ++u) {
      state_[u]->full.clear();
      state_[u]->full_grads = false;
    }
  };
  if (r.skipped) {
    clear_grads();
    return r;
  }
  const float coef = clip > 0.0 ? static_cast<float>(std::min(1.0, clip / (r.grad_norm + 1.0e-6))) : 1.0f;
  ++step_;
  const float b1 = static_cast<float>(opt.beta1), b2 = static_cast<float>(opt.beta2);
  const float ob1 = static_cast<float>(1.0 - opt.beta1), ob2 = static_cast<float>(1.0 - opt.beta2);
  const float inv1 = static_cast<float>(1.0 / (1.0 - std::pow(opt.beta1, static_cast<double>(step_))));
  const float inv2 = static_cast<float>(1.0 / (1.0 - std::pow(opt.beta2, static_cast<double>(step_))));
  const float eps = static_cast<float>(opt.eps);
  for (std::size_t u = 0; u < units_.size(); ++u) {
    const FsdpUnit& fu = units_[u];
    Unit& st = *state_[u];
    load_optimizer(u);
    load_master(u);
    const bool full = options_.sharding == ShardingStrategy::NoShard;
    const std::size_t P = fu.shard * static_cast<std::size_t>(shard_world_);
    const std::size_t lo = full ? 0 : static_cast<std::size_t>(shard_rank_) * fu.shard;
    const std::size_t hi = full ? P : lo + fu.shard;
    std::vector<float> packed;
    float* param = nullptr;
    const float* grad = nullptr;
    if (full) {  // NoShard: parametreler yerinde (paketlenip guncellenir, sonra acilir)
      packed.assign(P, 0.0f);
      for (std::size_t j = 0; j < fu.params.size(); ++j) {
        const nexus_model::Parameter* p = params_[fu.params[j]];
        std::memcpy(packed.data() + fu.begin[j], p->data.data(), p->data.numel() * 4);
      }
      param = packed.data();
      grad = st.full.data();
    } else {
      param = st.master.data();
      grad = st.grad_shard.data();
    }
    for (const Unit::Segment& seg : st.segments) {
      const std::size_t a = std::max(seg.begin, lo);
      const std::size_t b = std::min(seg.end, hi);
      if (a >= b) continue;
      const float lr_s = static_cast<float>(lr) * seg.lr;
      const float lr_wd = lr_s * static_cast<float>(opt.weight_decay) * seg.wd;
      for (std::size_t g = a; g < b; ++g) {
        const std::size_t i = g - lo;
        float p = param[i];
        const float gi = grad[i] * coef;
        if (lr_wd != 0.0f) p = p - lr_wd * p;
        const float mt = b1 * st.m[i] + ob1 * gi;
        const float vt = b2 * st.v[i] + ob2 * gi * gi;
        st.m[i] = mt;
        st.v[i] = vt;
        p = p - lr_s * (mt * inv1) / (std::sqrt(vt * inv2) + eps);
        param[i] = p;
      }
    }
    if (full) {
      for (std::size_t j = 0; j < fu.params.size(); ++j) {
        nexus_model::Parameter* p = params_[fu.params[j]];
        std::memcpy(p->data.data(), packed.data() + fu.begin[j], p->data.numel() * 4);
      }
    } else if (!sharded_params()) {  // ZeRO-1/2: guncel parcalari topla
      st.full.resize(P);
      shard_group_->all_gather(st.master.data(), st.full.data(), fu.shard, DType::F32)->wait();
      for (std::size_t j = 0; j < fu.params.size(); ++j) {
        nexus_model::Parameter* p = params_[fu.params[j]];
        std::memcpy(p->data.data(), st.full.data() + fu.begin[j], p->data.numel() * 4);
      }
    }
    if (options_.offload.optimizer_nvme) store_optimizer(u);
    if (options_.offload.params_nvme) store_master(u);
  }
  clear_grads();
  for (std::size_t u = 0; u < units_.size(); ++u) {
    state_[u]->full.clear();
    state_[u]->full.shrink_to_fit();
  }
  return r;
}

FsdpMemory FsdpEngine::persistent_memory() const {
  FsdpMemory m;
  for (const nexus_model::Parameter* p : params_) {
    m.param_bytes += p->data.numel() * 4;
    m.grad_bytes += p->grad.numel() * 4;
  }
  for (const auto& st : state_) {
    m.param_bytes += st->master.size() * 4;
    m.grad_bytes += st->grad_shard.size() * 4;
    m.optimizer_bytes += (st->m.size() + st->v.size()) * 4;
  }
  for (std::size_t u = 0; u < units_.size(); ++u) {
    const Unit& st = *state_[u];
    if (st.m_on_disk) m.nvme_bytes += 2 * units_[u].shard * 4;
    if (st.master_on_disk) m.nvme_bytes += units_[u].shard * 4;
  }
  return m;
}

// ------------------------------------------------------------------------------ tam durum ve kayit
std::vector<float> FsdpEngine::full_parameters() {
  std::vector<float> out;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    const FsdpUnit& fu = units_[u];
    Unit& st = *state_[u];
    std::vector<float> flat(fu.shard * static_cast<std::size_t>(shard_world_), 0.0f);
    if (options_.sharding == ShardingStrategy::NoShard) {
      for (std::size_t j = 0; j < fu.params.size(); ++j) {
        const nexus_model::Parameter* p = params_[fu.params[j]];
        std::memcpy(flat.data() + fu.begin[j], p->data.data(), p->data.numel() * 4);
      }
    } else {
      load_master(u);
      shard_group_->all_gather(st.master.data(), flat.data(), fu.shard, DType::F32)->wait();
      if (options_.offload.params_nvme) store_master(u);
    }
    out.insert(out.end(), flat.begin(), flat.begin() + static_cast<std::ptrdiff_t>(fu.numel));
  }
  return out;
}

std::vector<float> FsdpEngine::full_optimizer_state(int which) {
  std::vector<float> out;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    const FsdpUnit& fu = units_[u];
    Unit& st = *state_[u];
    load_optimizer(u);
    const std::vector<float>& src = which == 0 ? st.m : st.v;
    std::vector<float> flat;
    if (options_.sharding == ShardingStrategy::NoShard) {
      flat = src;
    } else {
      flat.assign(fu.shard * static_cast<std::size_t>(shard_world_), 0.0f);
      shard_group_->all_gather(src.data(), flat.data(), fu.shard, DType::F32)->wait();
    }
    if (options_.offload.optimizer_nvme) store_optimizer(u);
    out.insert(out.end(), flat.begin(), flat.begin() + static_cast<std::ptrdiff_t>(fu.numel));
  }
  return out;
}

namespace {
template <typename T>
void put(std::vector<unsigned char>& b, const T& v) {
  const auto* p = reinterpret_cast<const unsigned char*>(&v);
  b.insert(b.end(), p, p + sizeof(T));
}
void put_floats(std::vector<unsigned char>& b, const float* p, std::size_t n) {
  put<std::uint64_t>(b, n);
  const auto* c = reinterpret_cast<const unsigned char*>(p);
  b.insert(b.end(), c, c + n * 4);
}
struct In {
  const std::vector<unsigned char>& b;
  std::size_t pos = 0;
  template <typename T>
  T get() {
    if (pos + sizeof(T) > b.size()) throw DistError("Nexus FSDP: parcali kayit kisa/bozuk");
    T v;
    std::memcpy(&v, b.data() + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }
  std::vector<float> floats() {
    const auto n = static_cast<std::size_t>(get<std::uint64_t>());
    if (pos + n * 4 > b.size()) throw DistError("Nexus FSDP: parcali kayit kisa/bozuk");
    std::vector<float> v(n);
    std::memcpy(v.data(), b.data() + pos, n * 4);
    pos += n * 4;
    return v;
  }
};
constexpr std::uint32_t kShardMagic = 0x3144534Eu;  // "NSD1"
}  // namespace

std::vector<unsigned char> FsdpEngine::save_shard() {
  // Bicim: sihir, dunya (parca grubu), rank, adim, birim sayisi; birim basina N, kapsanan duz
  // aralik [a, b), FP32 parametre, m, v (ayni aralik).
  std::vector<unsigned char> b;
  put(b, kShardMagic);
  put<std::int32_t>(b, shard_world_);
  put<std::int32_t>(b, shard_rank_);
  put<std::int64_t>(b, step_);
  put<std::uint64_t>(b, units_.size());
  for (std::size_t u = 0; u < units_.size(); ++u) {
    const FsdpUnit& fu = units_[u];
    Unit& st = *state_[u];
    load_master(u);
    load_optimizer(u);
    const bool full = options_.sharding == ShardingStrategy::NoShard;
    const std::size_t a = full ? 0 : static_cast<std::size_t>(shard_rank_) * fu.shard;
    const std::size_t e = std::min(full ? fu.numel : a + fu.shard, fu.numel);
    const std::size_t n = e > a ? e - a : 0;
    put<std::uint64_t>(b, fu.numel);
    put<std::uint64_t>(b, a);
    std::vector<float> p(n);
    if (full) {
      for (std::size_t j = 0; j < fu.params.size(); ++j) {
        const nexus_model::Parameter* q = params_[fu.params[j]];
        std::memcpy(p.data() + fu.begin[j], q->data.data(), q->data.numel() * 4);
      }
    } else if (n > 0) {
      std::memcpy(p.data(), st.master.data(), n * 4);
    }
    put_floats(b, p.data(), n);
    put_floats(b, st.m.data(), n);
    put_floats(b, st.v.data(), n);
    if (options_.offload.params_nvme) store_master(u);
    if (options_.offload.optimizer_nvme) store_optimizer(u);
  }
  return b;
}

void FsdpEngine::load_shard(const std::vector<unsigned char>& blob) {
  // FAZ 15.3: ayni parca duzeni -> yalniz bu rank'in kendi parcasi okunur (diger rank dosyalari gerekmez).
  // NoShard (tam kopya) kaydi her rank'te tam uzayi kapsar: genel yol.
  if (options_.sharding == ShardingStrategy::NoShard) {
    load_resharded({blob});
    return;
  }
  In in{blob};
  if (in.get<std::uint32_t>() != kShardMagic) throw DistError("Nexus FSDP: parcali kayit degil");
  const auto world = in.get<std::int32_t>();
  const auto rank = in.get<std::int32_t>();
  if (world != shard_world_ || rank != shard_rank_) {
    throw DistError("Nexus FSDP: kayit parca " + std::to_string(rank) + "/" + std::to_string(world) + ", bu rank " +
                    std::to_string(shard_rank_) + "/" + std::to_string(shard_world_) + " (load_resharded kullanin)");
  }
  const auto step = in.get<std::int64_t>();
  if (in.get<std::uint64_t>() != units_.size()) throw DistError("Nexus FSDP: kayit farkli bir birim yapisindan");
  for (std::size_t u = 0; u < units_.size(); ++u) {
    const FsdpUnit& fu = units_[u];
    Unit& st = *state_[u];
    if (in.get<std::uint64_t>() != fu.numel) throw DistError("Nexus FSDP: birim boyutu uyusmuyor");
    const auto a = static_cast<std::size_t>(in.get<std::uint64_t>());
    if (a != static_cast<std::size_t>(shard_rank_) * fu.shard) throw DistError("Nexus FSDP: parca araligi uyusmuyor");
    std::vector<float> p = in.floats(), m = in.floats(), v = in.floats();
    p.resize(fu.shard, 0.0f);  // son rank'in dolgusu
    m.resize(fu.shard, 0.0f);
    v.resize(fu.shard, 0.0f);
    st.master = std::move(p);
    st.m = std::move(m);
    st.v = std::move(v);
    st.master_on_disk = false;
    st.m_on_disk = false;
    if (!sharded_params()) {  // ZeRO-1/2: parametreler tam kopya -> parcalardan topla
      const std::size_t P = fu.shard * static_cast<std::size_t>(shard_world_);
      st.full.resize(P);
      shard_group_->all_gather(st.master.data(), st.full.data(), fu.shard, DType::F32)->wait();
      for (std::size_t j = 0; j < fu.params.size(); ++j) {
        nexus_model::Parameter* q = params_[fu.params[j]];
        std::memcpy(q->data.data(), st.full.data() + fu.begin[j], q->data.numel() * 4);
      }
    }
    if (options_.offload.params_nvme) store_master(u);
    if (options_.offload.optimizer_nvme) store_optimizer(u);
  }
  step_ = step;
}

void FsdpEngine::load_resharded(const std::vector<std::vector<unsigned char>>& blobs) {
  // Tum kayitlardan her birimin tam duz uzayini kur, sonra bu rank'in yeni parcasini al.
  std::vector<std::vector<float>> P(units_.size()), M(units_.size()), V(units_.size());
  for (std::size_t u = 0; u < units_.size(); ++u) {
    P[u].assign(units_[u].numel, 0.0f);
    M[u].assign(units_[u].numel, 0.0f);
    V[u].assign(units_[u].numel, 0.0f);
  }
  std::int64_t step = 0;
  std::vector<std::vector<bool>> covered(units_.size());
  for (std::size_t u = 0; u < units_.size(); ++u) covered[u].assign(units_[u].numel, false);
  for (const auto& blob : blobs) {
    In in{blob};
    if (in.get<std::uint32_t>() != kShardMagic) throw DistError("Nexus FSDP: parcali kayit degil");
    (void)in.get<std::int32_t>();
    (void)in.get<std::int32_t>();
    step = in.get<std::int64_t>();
    if (in.get<std::uint64_t>() != units_.size()) throw DistError("Nexus FSDP: kayit farkli bir birim yapisindan");
    for (std::size_t u = 0; u < units_.size(); ++u) {
      if (in.get<std::uint64_t>() != units_[u].numel) throw DistError("Nexus FSDP: birim boyutu uyusmuyor");
      const auto a = static_cast<std::size_t>(in.get<std::uint64_t>());
      const std::vector<float> p = in.floats(), m = in.floats(), v = in.floats();
      std::copy(p.begin(), p.end(), P[u].begin() + static_cast<std::ptrdiff_t>(a));
      std::copy(m.begin(), m.end(), M[u].begin() + static_cast<std::ptrdiff_t>(a));
      std::copy(v.begin(), v.end(), V[u].begin() + static_cast<std::ptrdiff_t>(a));
      for (std::size_t k = 0; k < p.size(); ++k) covered[u][a + k] = true;
    }
  }
  for (std::size_t u = 0; u < units_.size(); ++u) {
    if (std::find(covered[u].begin(), covered[u].end(), false) != covered[u].end()) {
      throw DistError("Nexus FSDP: kayitlar birim " + units_[u].name + " uzayini kapsamiyor (eksik rank dosyasi?)");
    }
  }
  step_ = step;
  for (std::size_t u = 0; u < units_.size(); ++u) {
    const FsdpUnit& fu = units_[u];
    Unit& st = *state_[u];
    const std::size_t Pz = fu.shard * static_cast<std::size_t>(shard_world_);
    std::vector<float> fp(Pz, 0.0f), fm(Pz, 0.0f), fv(Pz, 0.0f);
    std::copy(P[u].begin(), P[u].end(), fp.begin());
    std::copy(M[u].begin(), M[u].end(), fm.begin());
    std::copy(V[u].begin(), V[u].end(), fv.begin());
    if (options_.sharding == ShardingStrategy::NoShard) {
      st.m = fm;
      st.v = fv;
    } else {
      const std::size_t off = static_cast<std::size_t>(shard_rank_) * fu.shard;
      st.master.assign(fp.begin() + static_cast<std::ptrdiff_t>(off), fp.begin() + static_cast<std::ptrdiff_t>(off + fu.shard));
      st.m.assign(fm.begin() + static_cast<std::ptrdiff_t>(off), fm.begin() + static_cast<std::ptrdiff_t>(off + fu.shard));
      st.v.assign(fv.begin() + static_cast<std::ptrdiff_t>(off), fv.begin() + static_cast<std::ptrdiff_t>(off + fu.shard));
      st.master_on_disk = false;
    }
    st.m_on_disk = false;
    if (!sharded_params()) {
      for (std::size_t j = 0; j < fu.params.size(); ++j) {
        nexus_model::Parameter* p = params_[fu.params[j]];
        std::memcpy(p->data.data(), fp.data() + fu.begin[j], p->data.numel() * 4);
      }
    }
    if (options_.offload.params_nvme) store_master(u);
    if (options_.offload.optimizer_nvme) store_optimizer(u);
  }
}

}  // namespace dist
}  // namespace nexus
