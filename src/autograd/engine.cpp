/**
 * @file engine.cpp
 * @brief FAZ 2 autograd motoru: Variable, mod denetimi, SavedTensor, AccumulateGrad, backward.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace nexus {
namespace autograd {
namespace {

thread_local bool t_grad_enabled = true;
thread_local bool t_anomaly_mode = false;

std::mutex g_hooks_mutex;
SavedPackFn g_saved_pack;
SavedUnpackFn g_saved_unpack;

constexpr float kPi = 3.14159265358979323846f;

[[nodiscard]] std::shared_ptr<AutogradMeta> make_meta(bool requires_grad, bool is_leaf) {
  auto meta = std::make_shared<AutogradMeta>();
  meta->requires_grad = requires_grad;
  meta->is_leaf = is_leaf;
  if (requires_grad && is_leaf) {
    meta->grad_fn = std::make_shared<AccumulateGrad>(meta);
  }
  return meta;
}

// Grafik uretmeyen eleman-eleman toplama (motor icinde gradyan birikimi).
[[nodiscard]] Variable add_plain(const Variable& a, const Variable& b) {
  const auto& av = a.values();
  const auto& bv = b.values();
  std::vector<float> out(av.size());
  for (std::size_t i = 0; i < av.size(); ++i) {
    out[i] = av[i] + bv[i];
  }
  return Variable::make(std::move(out), a.shape(), false);
}

}  // namespace

// ---- Mod denetimi ----------------------------------------------------------------
bool grad_enabled() noexcept { return t_grad_enabled; }
void set_grad_enabled(bool enabled) noexcept { t_grad_enabled = enabled; }

NoGradGuard::NoGradGuard() noexcept : previous_(grad_enabled()) { set_grad_enabled(false); }
NoGradGuard::~NoGradGuard() { set_grad_enabled(previous_); }
EnableGradGuard::EnableGradGuard() noexcept : previous_(grad_enabled()) { set_grad_enabled(true); }
EnableGradGuard::~EnableGradGuard() { set_grad_enabled(previous_); }

bool anomaly_mode() noexcept { return t_anomaly_mode; }
void set_anomaly_mode(bool enabled) noexcept { t_anomaly_mode = enabled; }

AnomalyModeGuard::AnomalyModeGuard(bool enabled) noexcept : previous_(anomaly_mode()) { set_anomaly_mode(enabled); }
AnomalyModeGuard::~AnomalyModeGuard() { set_anomaly_mode(previous_); }

std::uint64_t next_sequence_nr() noexcept {
  static std::atomic<std::uint64_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

// ---- Kaydedilmis degisken kancalari ------------------------------------------------
void set_saved_tensors_hooks(SavedPackFn pack, SavedUnpackFn unpack) {
  if (static_cast<bool>(pack) != static_cast<bool>(unpack)) {
    throw std::invalid_argument("Nexus autograd: pack ve unpack kancalari birlikte verilmeli");
  }
  std::lock_guard<std::mutex> guard(g_hooks_mutex);
  g_saved_pack = std::move(pack);
  g_saved_unpack = std::move(unpack);
}

void clear_saved_tensors_hooks() {
  std::lock_guard<std::mutex> guard(g_hooks_mutex);
  g_saved_pack = {};
  g_saved_unpack = {};
}

SavedTensorsHooksGuard::SavedTensorsHooksGuard(SavedPackFn pack, SavedUnpackFn unpack) {
  {
    std::lock_guard<std::mutex> guard(g_hooks_mutex);
    previous_pack_ = g_saved_pack;
    previous_unpack_ = g_saved_unpack;
  }
  set_saved_tensors_hooks(std::move(pack), std::move(unpack));
}

SavedTensorsHooksGuard::~SavedTensorsHooksGuard() {
  std::lock_guard<std::mutex> guard(g_hooks_mutex);
  g_saved_pack = std::move(previous_pack_);
  g_saved_unpack = std::move(previous_unpack_);
}

Variable SavedTensor::unpack(const char* op_name) const {
  const auto modified = [op_name]() {
    return std::runtime_error(std::string("Nexus autograd: kaydedilmis tensör yerinde degistirildi (op: ") + op_name +
                              ")");
  };
  if (unpack_fn_) {
    if (had_meta_) {
      // Ozgun tensör hala yasiyorsa yerinde yazma denetlenir; olmusse artik degisemez.
      if (const std::shared_ptr<AutogradMeta> meta = meta_.lock(); meta && meta->version != version_) {
        throw modified();
      }
    }
    std::shared_ptr<void> packed = packed_;  // unpack tek seferlik degil: retain_graph ile tekrar okunabilir
    Variable restored = unpack_fn_(std::move(packed));
    if (!restored.has_data() || restored.shape().dims != shape_.dims) {
      throw std::runtime_error(std::string("Nexus autograd: unpack kancasi kaydedilen sekli geri vermedi (op: ") +
                               op_name + ")");
    }
    return restored;
  }
  if (!value_) {
    throw std::runtime_error(std::string("Nexus autograd: kaydedilmis tensör bos (op: ") + op_name + ")");
  }
  if (value_->version() != version_) {
    throw modified();
  }
  return *value_;
}

SavedTensor save_variable(const Variable& value) {
  SavedPackFn pack;
  SavedUnpackFn unpack;
  {
    std::lock_guard<std::mutex> guard(g_hooks_mutex);
    pack = g_saved_pack;
    unpack = g_saved_unpack;
  }
  SavedTensor saved;
  saved.version_ = value.version();
  saved.shape_ = value.shape();
  if (pack) {
    saved.packed_ = pack(value);
    saved.unpack_fn_ = std::move(unpack);
    saved.meta_ = value.meta();
    saved.had_meta_ = static_cast<bool>(value.meta());
  } else {
    saved.value_ = std::make_shared<const Variable>(value);
  }
  return saved;
}

Variable restore_variable(const SavedTensor& saved, const char* op_name) { return saved.unpack(op_name); }

// ---- Variable -----------------------------------------------------------------------
Variable Variable::make(std::vector<float> values, Shape shape, bool requires_grad) {
  if (static_cast<std::int64_t>(values.size()) != shape.numel()) {
    throw std::invalid_argument("Nexus autograd: veri boyutu sekil ile uyusmuyor");
  }
  Variable out;
  out.data_ = std::make_shared<std::vector<float>>(std::move(values));
  out.shape_ = std::move(shape);
  out.meta_ = make_meta(requires_grad, /*is_leaf=*/true);
  return out;
}

Variable Variable::zeros(Shape shape, bool requires_grad) {
  // numel once hesaplanir: arguman degerlendirme sirasi belirsizdir (MSVC sagdan sola),
  // std::move(shape) once calisirsa bos sekilden 1 elemanli vektor uretilirdi.
  const auto count = static_cast<std::size_t>(shape.numel());
  return make(std::vector<float>(count, 0.0f), std::move(shape), requires_grad);
}

Variable Variable::ones(Shape shape, bool requires_grad) {
  // numel once hesaplanir: arguman degerlendirme sirasi belirsizdir (MSVC sagdan sola),
  // std::move(shape) once calisirsa bos sekilden 1 elemanli vektor uretilirdi.
  const auto count = static_cast<std::size_t>(shape.numel());
  return make(std::vector<float>(count, 1.0f), std::move(shape), requires_grad);
}

Variable Variable::full(Shape shape, float value, bool requires_grad) {
  // numel once hesaplanir: arguman degerlendirme sirasi belirsizdir (MSVC sagdan sola),
  // std::move(shape) once calisirsa bos sekilden 1 elemanli vektor uretilirdi.
  const auto count = static_cast<std::size_t>(shape.numel());
  return make(std::vector<float>(count, value), std::move(shape), requires_grad);
}

Variable Variable::randn(Shape shape, std::uint64_t seed, bool requires_grad) {
  std::vector<float> out(static_cast<std::size_t>(shape.numel()));
  std::uint64_t state = seed == 0 ? 0x9E3779B97F4A7C15ULL : seed;
  auto next_uniform = [&state]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state >> 11) & 0x1FFFFFFFFFFFFFULL) / static_cast<float>(1ULL << 53);
  };
  for (float& value : out) {
    const float u1 = std::fmax(1.0e-9f, next_uniform());
    const float u2 = next_uniform();
    value = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * kPi * u2);
  }
  return make(std::move(out), std::move(shape), requires_grad);
}

Variable Variable::from_data(std::shared_ptr<std::vector<float>> data, Shape shape, std::shared_ptr<AutogradMeta> meta) {
  Variable out;
  out.data_ = std::move(data);
  out.shape_ = std::move(shape);
  out.meta_ = std::move(meta);
  return out;
}

const std::vector<float>& Variable::values() const noexcept { return *data_; }

std::vector<float>& Variable::mutable_values() {
  if (meta_ && meta_->requires_grad) {
    meta_->version += 1;
  }
  return *data_;
}

const Shape& Variable::shape() const noexcept { return shape_; }
std::int64_t Variable::numel() const noexcept { return shape_.numel(); }
float Variable::operator[](std::size_t index) const { return (*data_)[index]; }

float& Variable::at(std::size_t index) {
  if (meta_ && meta_->requires_grad) {
    meta_->version += 1;
  }
  return (*data_)[index];
}

bool Variable::requires_grad() const noexcept { return meta_ && meta_->requires_grad; }
bool Variable::is_leaf() const noexcept { return meta_ && meta_->is_leaf; }
std::uint64_t Variable::version() const noexcept { return meta_ ? meta_->version : 0; }

const std::shared_ptr<Node>& Variable::grad_fn() const noexcept {
  static const std::shared_ptr<Node> empty;
  return meta_ ? meta_->grad_fn : empty;
}

Variable& Variable::grad() const {
  if (!meta_ || !meta_->grad) {
    throw std::runtime_error("Nexus autograd: gradyan yok (once backward cagirin)");
  }
  return *meta_->grad;
}

void Variable::set_grad(Variable gradient) {
  if (!meta_) {
    throw std::runtime_error("Nexus autograd: meta yok, gradyan kurulamaz");
  }
  meta_->grad = std::make_shared<Variable>(std::move(gradient));
}

void Variable::zero_grad() {
  if (meta_) {
    meta_->grad.reset();
  }
}

void Variable::retain_grad() {
  if (meta_) {
    meta_->retains_grad = true;
  }
}

std::uint64_t Variable::register_hook(std::function<Variable(const Variable&)> hook) {
  if (!meta_) {
    throw std::runtime_error("Nexus autograd: meta yok");
  }
  const std::uint64_t id = meta_->next_hook_id++;
  meta_->grad_hooks.emplace_back(id, std::move(hook));
  return id;
}

bool Variable::remove_hook(std::uint64_t hook_id) {
  if (!meta_) {
    return false;
  }
  auto& hooks = meta_->grad_hooks;
  for (auto it = hooks.begin(); it != hooks.end(); ++it) {
    if (it->first == hook_id) {
      hooks.erase(it);
      return true;
    }
  }
  return false;
}

void Variable::register_post_accumulate_hook(std::function<void()> hook) {
  if (!meta_) {
    throw std::runtime_error("Nexus autograd: meta yok");
  }
  meta_->post_accumulate_hooks.push_back(std::move(hook));
}

void Variable::backward() const {
  if (!requires_grad()) {
    throw std::runtime_error("Nexus autograd: requires_grad=false tensörde backward");
  }
  autograd::backward(*this, Variable::full(shape_, 1.0f, false));
}

void Variable::backward(const Variable& gradient, bool retain_graph) const {
  if (!requires_grad()) {
    throw std::runtime_error("Nexus autograd: requires_grad=false tensörde backward");
  }
  if (gradient.shape().dims != shape_.dims) {
    throw std::invalid_argument("Nexus autograd: backward gradyani cikti sekliyle eslesmiyor");
  }
  autograd::backward(std::vector<Variable>{*this}, std::vector<Variable>{gradient}, retain_graph, false);
}

Variable Variable::detach() const {
  Variable out;
  out.data_ = data_;
  out.shape_ = shape_;
  out.meta_ = make_meta(false, true);
  return out;
}

void Variable::add_(const Variable& other) {
  if (meta_ && meta_->grad_fn && !meta_->is_leaf) {
    throw std::runtime_error("Nexus autograd: yaprak olmayan tensör yerinde degistirilemez");
  }
  auto& mine = mutable_values();
  const auto& theirs = other.values();
  if (mine.size() != theirs.size()) {
    throw std::invalid_argument("Nexus autograd: add_ sekil uyusmazligi");
  }
  for (std::size_t i = 0; i < mine.size(); ++i) {
    mine[i] += theirs[i];
  }
}

void Variable::mul_(float scalar) {
  if (meta_ && meta_->grad_fn && !meta_->is_leaf) {
    throw std::runtime_error("Nexus autograd: yaprak olmayan tensör yerinde degistirilemez");
  }
  auto& mine = mutable_values();
  for (float& value : mine) {
    value *= scalar;
  }
}

void Variable::fill_(float value) {
  if (meta_ && meta_->grad_fn && !meta_->is_leaf) {
    throw std::runtime_error("Nexus autograd: yaprak olmayan tensör yerinde degistirilemez");
  }
  auto& mine = mutable_values();
  std::fill(mine.begin(), mine.end(), value);
}

Variable make_op_output(std::vector<float> values, const Shape& shape, std::shared_ptr<Node> node,
                        std::vector<Edge> edges) {
  if (!node) {
    throw std::invalid_argument("Nexus autograd: make_op_output dugum bekler");
  }
  node->next_edges = std::move(edges);
  node->sequence_nr = next_sequence_nr();
  if (anomaly_mode()) {
    node->forward_context = node->name + " (ileri sira " + std::to_string(node->sequence_nr) + ", sekil [";
    for (std::size_t i = 0; i < shape.dims.size(); ++i) {
      node->forward_context += (i ? "," : "") + std::to_string(shape.dims[i]);
    }
    node->forward_context += "])";
  }
  auto meta = std::make_shared<AutogradMeta>();
  meta->requires_grad = true;
  meta->is_leaf = false;
  node->output_meta = meta;
  meta->grad_fn = std::move(node);
  return Variable::from_data(std::make_shared<std::vector<float>>(std::move(values)), shape, std::move(meta));
}

// ---- AccumulateGrad -------------------------------------------------------------------
std::vector<Variable> AccumulateGrad::apply(std::vector<Variable>&& grad_outputs) {
  if (grad_outputs.size() != 1) {
    throw std::runtime_error("Nexus autograd: AccumulateGrad tek gradyan bekler");
  }
  std::shared_ptr<AutogradMeta> meta = meta_.lock();
  if (!meta) {
    return {};  // yaprak yasamiyor; gradyan bos
  }
  Variable grad = std::move(grad_outputs[0]);
  for (const auto& entry : meta->grad_hooks) {
    grad = entry.second(grad);
  }
  if (!meta->grad) {
    meta->grad = std::make_shared<Variable>(Variable::make(std::vector<float>(grad.values()), grad.shape(), false));
  } else {
    auto& accumulated = meta->grad->mutable_values();
    const auto& incoming = grad.values();
    if (accumulated.size() != incoming.size()) {
      throw std::runtime_error("Nexus autograd: gradyan sekil uyusmazligi");
    }
    for (std::size_t i = 0; i < incoming.size(); ++i) {
      accumulated[i] += incoming[i];
    }
  }
  for (const auto& hook : meta->post_accumulate_hooks) {
    hook();
  }
  return {};
}

// ---- Motor -------------------------------------------------------------------------------
void backward(const std::vector<Variable>& roots, const std::vector<Variable>& grad_roots, bool retain_graph,
              bool create_graph) {
  (void)create_graph;  // FAZ 2: cift backward altyapi disi.
  if (roots.empty()) {
    return;
  }
  if (roots.size() != grad_roots.size()) {
    throw std::invalid_argument("Nexus autograd: roots ve grad_roots boyutlari eslesmiyor");
  }

  std::unordered_map<Node*, Variable> grads;
  std::unordered_map<Node*, int> deps;
  std::vector<Node*> root_nodes;
  // Motor dugumleri calisirken grafik serbest kalmasin (retain_graph=false kenarlari siler).
  std::vector<std::shared_ptr<Node>> keep_alive;

  for (std::size_t i = 0; i < roots.size(); ++i) {
    const std::shared_ptr<Node>& fn = roots[i].grad_fn();
    if (!fn) {
      continue;
    }
    if (fn->released) {
      throw std::runtime_error("Nexus autograd: grafik ikinci kez geri yurutuluyor; ilk backward'ta "
                               "retain_graph=true verin (dugum: " + fn->name + ")");
    }
    Node* key = fn.get();
    if (grads.count(key) == 0) {
      grads[key] = grad_roots[i];
      root_nodes.push_back(key);
      keep_alive.push_back(fn);
    } else {
      grads[key] = add_plain(grads[key], grad_roots[i]);
    }
  }

  // Bagimlilik sayimi: tum kokler birlikte; bir kok digerinin atasi olabilir.
  {
    std::unordered_set<Node*> seen(root_nodes.begin(), root_nodes.end());
    std::deque<Node*> queue(root_nodes.begin(), root_nodes.end());
    while (!queue.empty()) {
      Node* node = queue.front();
      queue.pop_front();
      for (const Edge& edge : node->next_edges) {
        Node* child = edge.function.get();
        if (!child) {
          continue;
        }
        if (child->released) {
          throw std::runtime_error("Nexus autograd: grafik ikinci kez geri yurutuluyor; ilk backward'ta "
                                   "retain_graph=true verin (dugum: " + child->name + ")");
        }
        deps[child] += 1;
        if (seen.insert(child).second) {
          queue.push_back(child);
          keep_alive.push_back(edge.function);
        }
      }
    }
  }

  std::deque<Node*> ready;
  for (Node* node : root_nodes) {
    if (deps[node] == 0) {
      ready.push_back(node);
    }
  }

  const bool anomaly = anomaly_mode();
  while (!ready.empty()) {
    Node* node = ready.front();
    ready.pop_front();
    Variable incoming = std::move(grads[node]);
    grads.erase(node);

    // Ara (yaprak olmayan) cikti: kancalar ve retain_grad.
    if (const std::shared_ptr<AutogradMeta> out_meta = node->output_meta.lock()) {
      for (const auto& entry : out_meta->grad_hooks) {
        incoming = entry.second(incoming);
      }
      if (out_meta->retains_grad) {
        if (!out_meta->grad) {
          out_meta->grad =
              std::make_shared<Variable>(Variable::make(std::vector<float>(incoming.values()), incoming.shape(), false));
        } else {
          *out_meta->grad = add_plain(*out_meta->grad, incoming);
        }
      }
    }

    std::vector<Variable> inputs;
    inputs.push_back(std::move(incoming));
    std::vector<Variable> grad_inputs;
    try {
      grad_inputs = node->apply(std::move(inputs));
    } catch (const std::invalid_argument& error) {
      throw std::invalid_argument(std::string(error.what()) + " [geri dugum: " + node->name + "]");
    } catch (const std::runtime_error& error) {
      const std::string what = error.what();
      if (what.find("[geri dugum:") != std::string::npos) {
        throw;
      }
      throw std::runtime_error(what + " [geri dugum: " + node->name + "]");
    }
    if (grad_inputs.size() != node->next_edges.size()) {
      throw std::runtime_error(std::string("Nexus autograd: ") + node->name + " yanlis gradyan sayisi dondurdu");
    }
    if (anomaly) {
      for (const Variable& g : grad_inputs) {
        for (const float value : g.values()) {
          if (!std::isfinite(value)) {
            std::string message = std::string("Nexus autograd anomali: '") + node->name +
                                  "' geri geçisi NaN/Inf gradyan uretti (sira=" + std::to_string(node->sequence_nr) + ")";
            if (!node->forward_context.empty()) {
              message += "; ileri geçis: " + node->forward_context;
            }
            throw std::runtime_error(message);
          }
        }
      }
    }
    for (std::size_t i = 0; i < node->next_edges.size(); ++i) {
      Node* child = node->next_edges[i].function.get();
      if (!child) {
        continue;
      }
      if (grads.count(child) == 0) {
        grads[child] = std::move(grad_inputs[i]);
      } else {
        grads[child] = add_plain(grads[child], grad_inputs[i]);
      }
      int& remaining = deps[child];
      remaining -= 1;
      if (remaining == 0) {
        ready.push_back(child);
      }
    }
    if (!retain_graph && !node->next_edges.empty()) {
      node->next_edges.clear();
      node->released = true;
    }
  }
}

// ---- Gradcheck ----------------------------------------------------------------------------
GradcheckResult gradcheck(const std::function<Variable(const std::vector<Variable>&)>& fn,
                          const std::vector<Variable>& inputs, float eps, double atol, double rtol) {
  GradcheckResult result;
  std::vector<Variable> leaves;
  leaves.reserve(inputs.size());
  for (const Variable& input : inputs) {
    leaves.push_back(Variable::make(std::vector<float>(input.values()), input.shape(), input.requires_grad()));
  }
  {
    EnableGradGuard enable;
    Variable loss = fn(leaves);
    if (loss.numel() != 1) {
      throw std::invalid_argument("Nexus gradcheck: fn skaler kayip dondurmeli");
    }
    loss.backward();
  }
  result.ok = true;
  NoGradGuard no_grad;
  for (std::size_t k = 0; k < leaves.size(); ++k) {
    if (!leaves[k].requires_grad()) {
      continue;
    }
    std::vector<float> analytic(static_cast<std::size_t>(leaves[k].numel()), 0.0f);
    if (leaves[k].meta()->grad) {
      analytic = leaves[k].grad().values();
    }
    std::vector<Variable> probe;
    for (const Variable& leaf : leaves) {
      probe.push_back(Variable::make(std::vector<float>(leaf.values()), leaf.shape(), false));
    }
    for (std::size_t i = 0; i < analytic.size(); ++i) {
      const float original = probe[k].values()[i];
      probe[k].mutable_values()[i] = original + eps;
      const double plus = fn(probe).values()[0];
      probe[k].mutable_values()[i] = original - eps;
      const double minus = fn(probe).values()[0];
      probe[k].mutable_values()[i] = original;
      const double numeric = (plus - minus) / (2.0 * static_cast<double>(eps));
      const double abs_err = std::fabs(numeric - static_cast<double>(analytic[i]));
      const double rel_err = abs_err / std::max(1.0e-12, std::fabs(numeric));
      if (abs_err > result.max_abs_error) {
        result.max_abs_error = abs_err;
        result.worst_input = k;
        result.worst_index = i;
      }
      result.max_rel_error = std::max(result.max_rel_error, abs_err > atol ? rel_err : 0.0);
      if (abs_err > atol + rtol * std::fabs(numeric)) {
        result.ok = false;
      }
    }
  }
  return result;
}

void backward(const Variable& root, const Variable& gradient, bool retain_graph, bool create_graph) {
  backward(std::vector<Variable>{root}, std::vector<Variable>{gradient}, retain_graph, create_graph);
}

}  // namespace autograd
}  // namespace nexus
