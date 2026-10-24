/**
 * @file training.cpp
 * @brief FAZ 3 egitim yolu cekirdekleri: kaydirilmis CE, RMSNorm, gomme+konum,
 *        Philox dropout, foreach AdamW + global norm kirpma.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/fused/training.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace nexus {
namespace fused {
namespace {

using autograd::Edge;
using autograd::Node;
using autograd::SavedTensor;
using autograd::Shape;
using autograd::Variable;
using autograd::grad_enabled;

class TrainingNode : public Node {
 public:
  std::function<std::vector<Variable>(std::vector<Variable>&&)> backward_fn;
  std::vector<Variable> apply(std::vector<Variable>&& grads) override { return backward_fn(std::move(grads)); }
};

[[nodiscard]] Edge edge_of(const Variable& v) { return Edge{v.grad_fn(), 0}; }

void validate_ce(const CrossEntropyOptions& options, int batch, int time, int vocab) {
  if (batch < 1 || time < 2 || vocab < 2) {
    throw std::invalid_argument("Nexus cross_entropy_shifted: batch>=1, time>=2, vocab>=2 olmali");
  }
  if (!(options.label_smoothing >= 0.0f && options.label_smoothing < 1.0f)) {
    throw std::invalid_argument("Nexus cross_entropy_shifted: label_smoothing [0,1) araliginda olmali");
  }
  if (!(options.z_loss >= 0.0f)) {
    throw std::invalid_argument("Nexus cross_entropy_shifted: z_loss negatif olamaz");
  }
}

}  // namespace

// ---- Kaydirilmis CE ------------------------------------------------------------------------
CrossEntropyResult cross_entropy_shifted(const float* logits, const std::int32_t* tokens, int batch, int time,
                                         int vocab, const CrossEntropyOptions& options, float* dlogits) {
  validate_ce(options, batch, time, vocab);
  const auto v = static_cast<std::size_t>(vocab);
  // Once gecerli token sayisi: dlogits olcegi 1/valid oldugundan tek geçiste yazilabilsin.
  std::int64_t valid = 0;
  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t + 1 < time; ++t) {
      const std::int32_t target = tokens[static_cast<std::size_t>(b) * static_cast<std::size_t>(time) +
                                         static_cast<std::size_t>(t) + 1u];
      if (target == options.ignore_index) {
        continue;
      }
      if (target < 0 || target >= vocab) {
        throw std::out_of_range("Nexus cross_entropy_shifted: hedef token sozluk disinda (" + std::to_string(target) +
                                ")");
      }
      ++valid;
    }
  }
  CrossEntropyResult result;
  result.valid = valid;
  const double inv_valid = valid > 0 ? 1.0 / static_cast<double>(valid) : 0.0;
  const float scale = static_cast<float>(inv_valid);
  const float smooth = options.label_smoothing;
  const float on_target = 1.0f - smooth;
  const float uniform = smooth / static_cast<float>(vocab);
  double loss_sum = 0.0;
  double z_sum = 0.0;

  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < time; ++t) {
      const std::size_t row_index = static_cast<std::size_t>(b) * static_cast<std::size_t>(time) + static_cast<std::size_t>(t);
      const float* row = logits + row_index * v;
      float* drow = dlogits != nullptr ? dlogits + row_index * v : nullptr;
      const bool last = t + 1 == time;
      const std::int32_t target = last ? options.ignore_index : tokens[row_index + 1u];
      if (last || target == options.ignore_index) {
        if (drow != nullptr) {
          std::fill(drow, drow + v, 0.0f);
        }
        continue;
      }
      // Satir max + log-sum-exp (FP64 toplam: V=50304'te FP32 birikim hatasini onler).
      float maxv = row[0];
      for (std::size_t j = 1; j < v; ++j) {
        maxv = std::max(maxv, row[j]);
      }
      double total = 0.0;
      double logit_sum = 0.0;
      for (std::size_t j = 0; j < v; ++j) {
        total += std::exp(static_cast<double>(row[j] - maxv));
        logit_sum += static_cast<double>(row[j]);
      }
      const double lse = static_cast<double>(maxv) + std::log(total);
      const double nll = lse - static_cast<double>(row[static_cast<std::size_t>(target)]);
      double row_loss = static_cast<double>(on_target) * nll;
      if (smooth > 0.0f) {
        // -mean_j log p_j = lse - mean_j logit_j
        row_loss += static_cast<double>(smooth) * (lse - logit_sum / static_cast<double>(vocab));
      }
      const double z = static_cast<double>(options.z_loss) * lse * lse;
      loss_sum += row_loss + z;
      z_sum += z;
      if (drow != nullptr) {
        // softmax yeniden: p_j = exp(x_j - lse). Satir once tamamen okundu; yerinde yazmak guvenli.
        const float z_grad = static_cast<float>(2.0 * static_cast<double>(options.z_loss) * lse);
        const float lse_f = static_cast<float>(lse);
        for (std::size_t j = 0; j < v; ++j) {
          const float p = std::exp(row[j] - lse_f);
          float g = p - uniform + z_grad * p;
          if (j == static_cast<std::size_t>(target)) {
            g -= on_target;
          }
          drow[j] = g * scale;
        }
      }
    }
  }
  result.loss = static_cast<float>(loss_sum * inv_valid);
  result.z_loss = static_cast<float>(z_sum * inv_valid);
  return result;
}

Variable cross_entropy_shifted(const Variable& logits, const std::vector<std::int32_t>& tokens, int batch, int time,
                               const CrossEntropyOptions& options) {
  if (logits.shape().rank() != 2 || logits.shape().dims[0] != static_cast<std::int64_t>(batch) * time) {
    throw std::invalid_argument("Nexus cross_entropy_shifted: logits [batch*time, vocab] olmali");
  }
  if (static_cast<std::int64_t>(tokens.size()) != static_cast<std::int64_t>(batch) * time) {
    throw std::invalid_argument("Nexus cross_entropy_shifted: token sayisi batch*time olmali");
  }
  const int vocab = static_cast<int>(logits.shape().dims[1]);
  const CrossEntropyResult forward =
      cross_entropy_shifted(logits.values().data(), tokens.data(), batch, time, vocab, options, nullptr);
  if (!(grad_enabled() && logits.requires_grad())) {
    return Variable::make({forward.loss}, Shape{}, false);
  }
  auto node = std::make_shared<TrainingNode>();
  node->name = "CrossEntropyShiftedBackward";
  // Olasiliklar saklanmaz: logits kaydedilir, geri geçiste tek geçiste yeniden hesaplanir.
  SavedTensor saved = autograd::save_variable(logits);
  node->backward_fn = [saved, tokens, batch, time, vocab, options](std::vector<Variable>&& grads) {
    const Variable in = autograd::restore_variable(saved, "cross_entropy_shifted");
    std::vector<float> d(in.values().size());
    (void)cross_entropy_shifted(in.values().data(), tokens.data(), batch, time, vocab, options, d.data());
    const float upstream = grads[0].values()[0];
    if (upstream != 1.0f) {
      for (float& value : d) {
        value *= upstream;
      }
    }
    return std::vector<Variable>{Variable::make(std::move(d), in.shape(), false)};
  };
  return autograd::make_op_output({forward.loss}, Shape{}, std::move(node), {edge_of(logits)});
}

// ---- RMSNorm ------------------------------------------------------------------------------------
Variable rms_norm(const Variable& x, const Variable& gamma, float eps) {
  if (x.shape().rank() != 2 || gamma.shape().rank() != 1 || gamma.shape().dims[0] != x.shape().dims[1]) {
    throw std::invalid_argument("Nexus rms_norm: x [M,N], gamma [N] olmali");
  }
  if (!(eps > 0.0f)) {
    throw std::invalid_argument("Nexus rms_norm: eps pozitif olmali");
  }
  const std::int64_t m = x.shape().dims[0];
  const std::int64_t n = x.shape().dims[1];
  const auto& xv = x.values();
  const auto& gv = gamma.values();
  std::vector<float> out(xv.size());
  std::vector<float> rstd(static_cast<std::size_t>(m));
  for (std::int64_t i = 0; i < m; ++i) {
    double sq = 0.0;
    for (std::int64_t j = 0; j < n; ++j) {
      const double value = xv[static_cast<std::size_t>(i * n + j)];
      sq += value * value;
    }
    const float r = static_cast<float>(1.0 / std::sqrt(sq / static_cast<double>(n) + static_cast<double>(eps)));
    rstd[static_cast<std::size_t>(i)] = r;
    for (std::int64_t j = 0; j < n; ++j) {
      const std::size_t k = static_cast<std::size_t>(i * n + j);
      out[k] = xv[k] * r * gv[static_cast<std::size_t>(j)];
    }
  }
  const bool x_req = x.requires_grad();
  const bool g_req = gamma.requires_grad();
  if (!(grad_enabled() && (x_req || g_req))) {
    return Variable::make(std::move(out), x.shape(), false);
  }
  auto node = std::make_shared<TrainingNode>();
  node->name = "RMSNormBackward";
  std::vector<Edge> edges;
  if (x_req) {
    edges.push_back(edge_of(x));
  }
  if (g_req) {
    edges.push_back(edge_of(gamma));
  }
  SavedTensor saved_x = autograd::save_variable(x);
  SavedTensor saved_g = autograd::save_variable(gamma);
  node->backward_fn = [saved_x, saved_g, rstd, m, n, x_req, g_req](std::vector<Variable>&& grads) {
    const Variable xin = autograd::restore_variable(saved_x, "rms_norm");
    const Variable gin = autograd::restore_variable(saved_g, "rms_norm");
    const auto& g = grads[0].values();
    const auto& xs = xin.values();
    const auto& gm = gin.values();
    std::vector<float> dx(xs.size(), 0.0f);
    std::vector<float> dgamma(static_cast<std::size_t>(n), 0.0f);
    for (std::int64_t i = 0; i < m; ++i) {
      const float r = rstd[static_cast<std::size_t>(i)];
      double dot = 0.0;  // sum_j g_j * gamma_j * x_j
      for (std::int64_t j = 0; j < n; ++j) {
        const std::size_t k = static_cast<std::size_t>(i * n + j);
        dot += static_cast<double>(g[k]) * gm[static_cast<std::size_t>(j)] * xs[k];
        dgamma[static_cast<std::size_t>(j)] += g[k] * xs[k] * r;
      }
      const float coef = static_cast<float>(dot / static_cast<double>(n)) * r * r * r;
      for (std::int64_t j = 0; j < n; ++j) {
        const std::size_t k = static_cast<std::size_t>(i * n + j);
        dx[k] = g[k] * gm[static_cast<std::size_t>(j)] * r - xs[k] * coef;
      }
    }
    std::vector<Variable> result;
    if (x_req) {
      result.push_back(Variable::make(std::move(dx), Shape{{m, n}}, false));
    }
    if (g_req) {
      result.push_back(Variable::make(std::move(dgamma), Shape{{n}}, false));
    }
    return result;
  };
  return autograd::make_op_output(std::move(out), x.shape(), std::move(node), std::move(edges));
}

// ---- Gomme + konum --------------------------------------------------------------------------------
Variable embedding_pos(const Variable& wte, const Variable& wpe, const std::vector<std::int32_t>& tokens, int batch,
                       int time) {
  if (wte.shape().rank() != 2 || wpe.shape().rank() != 2 || wte.shape().dims[1] != wpe.shape().dims[1]) {
    throw std::invalid_argument("Nexus embedding_pos: wte [V,D], wpe [P,D] olmali");
  }
  if (batch < 1 || time < 1 || wpe.shape().dims[0] < time) {
    throw std::invalid_argument("Nexus embedding_pos: wpe en az time satir olmali");
  }
  if (static_cast<std::int64_t>(tokens.size()) != static_cast<std::int64_t>(batch) * time) {
    throw std::invalid_argument("Nexus embedding_pos: token sayisi batch*time olmali");
  }
  const std::int64_t vocab = wte.shape().dims[0];
  const std::int64_t positions = wpe.shape().dims[0];
  const std::int64_t dim = wte.shape().dims[1];
  const auto& tv = wte.values();
  const auto& pv = wpe.values();
  const std::int64_t rows = static_cast<std::int64_t>(batch) * time;
  std::vector<float> out(static_cast<std::size_t>(rows * dim));
  for (std::int64_t r = 0; r < rows; ++r) {
    const std::int64_t token = tokens[static_cast<std::size_t>(r)];
    if (token < 0 || token >= vocab) {
      throw std::out_of_range("Nexus embedding_pos: token sozluk disinda");
    }
    const std::int64_t pos = r % time;
    for (std::int64_t j = 0; j < dim; ++j) {
      out[static_cast<std::size_t>(r * dim + j)] =
          tv[static_cast<std::size_t>(token * dim + j)] + pv[static_cast<std::size_t>(pos * dim + j)];
    }
  }
  const bool te_req = wte.requires_grad();
  const bool pe_req = wpe.requires_grad();
  const Shape shape{{rows, dim}};
  if (!(grad_enabled() && (te_req || pe_req))) {
    return Variable::make(std::move(out), shape, false);
  }
  auto node = std::make_shared<TrainingNode>();
  node->name = "EmbeddingPosBackward";
  std::vector<Edge> edges;
  if (te_req) {
    edges.push_back(edge_of(wte));
  }
  if (pe_req) {
    edges.push_back(edge_of(wpe));
  }
  node->backward_fn = [tokens, vocab, positions, dim, rows, time, te_req, pe_req](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<Variable> result;
    if (te_req) {
      std::vector<float> d(static_cast<std::size_t>(vocab * dim), 0.0f);
      for (std::int64_t r = 0; r < rows; ++r) {
        const std::int64_t token = tokens[static_cast<std::size_t>(r)];
        for (std::int64_t j = 0; j < dim; ++j) {
          d[static_cast<std::size_t>(token * dim + j)] += g[static_cast<std::size_t>(r * dim + j)];
        }
      }
      result.push_back(Variable::make(std::move(d), Shape{{vocab, dim}}, false));
    }
    if (pe_req) {
      std::vector<float> d(static_cast<std::size_t>(positions * dim), 0.0f);
      for (std::int64_t r = 0; r < rows; ++r) {
        const std::int64_t pos = r % time;
        for (std::int64_t j = 0; j < dim; ++j) {
          d[static_cast<std::size_t>(pos * dim + j)] += g[static_cast<std::size_t>(r * dim + j)];
        }
      }
      result.push_back(Variable::make(std::move(d), Shape{{positions, dim}}, false));
    }
    return result;
  };
  return autograd::make_op_output(std::move(out), shape, std::move(node), std::move(edges));
}

// ---- Philox ---------------------------------------------------------------------------------------
std::array<std::uint32_t, 4> philox4x32_10(std::array<std::uint32_t, 4> counter,
                                           std::array<std::uint32_t, 2> key) noexcept {
  constexpr std::uint32_t kMul0 = 0xD2511F53u;
  constexpr std::uint32_t kMul1 = 0xCD9E8D57u;
  constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;
  constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;
  for (int round = 0; round < 10; ++round) {
    const std::uint64_t p0 = static_cast<std::uint64_t>(kMul0) * counter[0];
    const std::uint64_t p1 = static_cast<std::uint64_t>(kMul1) * counter[2];
    const auto hi0 = static_cast<std::uint32_t>(p0 >> 32);
    const auto lo0 = static_cast<std::uint32_t>(p0);
    const auto hi1 = static_cast<std::uint32_t>(p1 >> 32);
    const auto lo1 = static_cast<std::uint32_t>(p1);
    counter = {hi1 ^ counter[1] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0};
    key[0] += kWeyl0;
    key[1] += kWeyl1;
  }
  return counter;
}

float philox_uniform(std::uint64_t seed, std::uint64_t offset, std::uint64_t index) noexcept {
  const std::uint64_t block = offset + index / 4u;
  const std::array<std::uint32_t, 4> counter = {static_cast<std::uint32_t>(block),
                                                static_cast<std::uint32_t>(block >> 32), 0u, 0u};
  const std::array<std::uint32_t, 2> key = {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};
  const std::uint32_t bits = philox4x32_10(counter, key)[index % 4u];
  // Ust 24 bit -> [0,1) (FP32'de tam temsil edilir; 1.0'a yuvarlanmaz).
  return static_cast<float>(bits >> 8) * (1.0f / 16777216.0f);
}

Variable dropout_philox(const Variable& x, float p, std::uint64_t seed, std::uint64_t offset, bool training) {
  if (!(p >= 0.0f && p < 1.0f)) {
    throw std::invalid_argument("Nexus dropout_philox: p [0,1) araliginda olmali");
  }
  if (!training || p == 0.0f) {
    return x;
  }
  const float scale = 1.0f / (1.0f - p);
  const auto& xv = x.values();
  std::vector<float> out(xv.size());
  for (std::size_t i = 0; i < xv.size(); ++i) {
    out[i] = philox_uniform(seed, offset, i) < p ? 0.0f : xv[i] * scale;
  }
  if (!(grad_enabled() && x.requires_grad())) {
    return Variable::make(std::move(out), x.shape(), false);
  }
  auto node = std::make_shared<TrainingNode>();
  node->name = "DropoutPhiloxBackward";
  node->backward_fn = [p, seed, offset, scale](std::vector<Variable>&& grads) {
    const auto& g = grads[0].values();
    std::vector<float> d(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      d[i] = philox_uniform(seed, offset, i) < p ? 0.0f : g[i] * scale;
    }
    return std::vector<Variable>{Variable::make(std::move(d), grads[0].shape(), false)};
  };
  return autograd::make_op_output(std::move(out), x.shape(), std::move(node), {edge_of(x)});
}

// ---- Foreach AdamW --------------------------------------------------------------------------------
namespace {

struct AdamWCoeffs {
  float b1, ob1, b2, ob2, inv1, inv2, lr, lr_wd, eps, clip;
};

// Takma ad yok (restrict) ve dal yok: derleyici donguyu SIMD'e acar. Islem sirasi
// NexusOptim skaler cekirdegiyle birebir ayni (bit-bit esitlik testte dogrulanir).
template <bool kDecay>
void adamw_kernel(float* __restrict param, const float* __restrict grad, float* __restrict m, float* __restrict v,
                  std::size_t n, const AdamWCoeffs c) {
  for (std::size_t i = 0; i < n; ++i) {
    float p = param[i];
    const float g = grad[i] * c.clip;
    if constexpr (kDecay) {
      p -= c.lr_wd * p;
    }
    const float mt = c.b1 * m[i] + c.ob1 * g;
    const float vt = c.b2 * v[i] + c.ob2 * g * g;
    m[i] = mt;
    v[i] = vt;
    param[i] = p - c.lr * (mt * c.inv1) / (std::sqrt(vt * c.inv2) + c.eps);
  }
}

}  // namespace

double global_grad_norm(std::span<const ParamSlot> slots) noexcept {
  double total = 0.0;
  for (const ParamSlot& slot : slots) {
    for (std::size_t i = 0; i < slot.numel; ++i) {
      const double g = slot.grad[i];
      total += g * g;
    }
  }
  return std::sqrt(total);
}

AdamWStepInfo adamw_multi_tensor(std::span<const ParamSlot> slots, const AdamWConfig& config, std::int64_t step) {
  if (step < 1) {
    throw std::invalid_argument("Nexus adamw_multi_tensor: step 1'den baslar");
  }
  for (const ParamSlot& slot : slots) {
    if (slot.numel != 0 && (slot.param == nullptr || slot.grad == nullptr || slot.exp_avg == nullptr ||
                            slot.exp_avg_sq == nullptr)) {
      throw std::invalid_argument("Nexus adamw_multi_tensor: bos tampon isaretcisi");
    }
  }
  AdamWStepInfo info;
  if (config.max_grad_norm > 0.0) {
    info.grad_norm = global_grad_norm(slots);
    // PyTorch clip_grad_norm_: coef = max_norm / (norm + 1e-6), en fazla 1.
    info.clip_coef = static_cast<float>(std::min(1.0, config.max_grad_norm / (info.grad_norm + 1.0e-6)));
  }
  // NexusOptim AdamW ile ayni katsayilar: double'da hesap, FP32'ye tek yuvarlama.
  const double bc1 = 1.0 - std::pow(config.beta1, static_cast<double>(step));
  const double bc2 = 1.0 - std::pow(config.beta2, static_cast<double>(step));
  const auto inv1 = static_cast<float>(1.0 / bc1);
  const auto inv2 = static_cast<float>(1.0 / bc2);
  const auto b1 = static_cast<float>(config.beta1);
  const auto ob1 = static_cast<float>(1.0 - config.beta1);
  const auto b2 = static_cast<float>(config.beta2);
  const auto ob2 = static_cast<float>(1.0 - config.beta2);
  const auto lr = static_cast<float>(config.lr);
  const auto wd = static_cast<float>(config.weight_decay);
  const float lr_wd = lr * wd;
  const auto eps = static_cast<float>(config.eps);
  const AdamWCoeffs c{b1, ob1, b2, ob2, inv1, inv2, lr, lr_wd, eps, info.clip_coef};
  for (const ParamSlot& slot : slots) {
    // g * 1.0f == g (IEEE), bu yuzden kirpma carpimi kosulsuzdur: dal yok, dongu vektorlesir.
    if (wd != 0.0f) {
      adamw_kernel<true>(slot.param, slot.grad, slot.exp_avg, slot.exp_avg_sq, slot.numel, c);
    } else {
      adamw_kernel<false>(slot.param, slot.grad, slot.exp_avg, slot.exp_avg_sq, slot.numel, c);
    }
  }
  return info;
}

}  // namespace fused
}  // namespace nexus
