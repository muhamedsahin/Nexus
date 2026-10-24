/**
 * @file estimate.cpp
 * @brief Tahmin formullerinin gerceklemesi (kaynaklar estimate.hpp basliginda).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "estimate.hpp"

#include <stdexcept>
#include <string>

namespace nexus {
namespace estimate {
namespace {

void validate(const ModelShape& shape) {
  if (shape.n_layer <= 0 || shape.d_model <= 0 || shape.n_head <= 0 || shape.n_kv_head <= 0 || shape.d_ff <= 0 ||
      shape.vocab <= 0 || shape.seq_len <= 0 || shape.micro_batch <= 0 || shape.gradient_accum <= 0) {
    throw std::invalid_argument("nexus estimate: mimari alanlari pozitif olmali");
  }
  if (shape.d_model % shape.n_head != 0) {
    throw std::invalid_argument("nexus estimate: d_model % n_head == 0 olmali");
  }
  if (shape.n_head % shape.n_kv_head != 0) {
    throw std::invalid_argument("nexus estimate: n_head % n_kv_head == 0 olmali (GQA gruplama)");
  }
}

}  // namespace

ParamBreakdown count_parameters(const ModelShape& shape) {
  validate(shape);
  const std::int64_t d = shape.d_model;
  const std::int64_t d_head = d / shape.n_head;
  const std::int64_t d_kv = d_head * shape.n_kv_head;

  ParamBreakdown result;
  result.token_embed = shape.vocab * d;
  result.pos_embed = shape.learned_pos * d;

  // Dikkat: q (d·d), o (d·d), k ve v (d·d_kv). Bias: q,k,v,o = 2d + 2d_kv.
  const std::int64_t attn_weights = d * d + d * d + 2 * d * d_kv;
  const std::int64_t attn_bias = shape.bias ? (2 * d + 2 * d_kv) : 0;
  result.attention = shape.n_layer * (attn_weights + attn_bias);

  // FFN: GELU (d·d_ff + d_ff·d) ya da SwiGLU (3·d·d_ff). Bias: GELU d_ff+d, SwiGLU 2d_ff+d.
  const std::int64_t ffn_weights = shape.ffn_swiglu ? 3 * d * shape.d_ff : 2 * d * shape.d_ff;
  const std::int64_t ffn_bias = shape.bias ? (shape.ffn_swiglu ? 2 * shape.d_ff + d : shape.d_ff + d) : 0;
  result.ffn = shape.n_layer * (ffn_weights + ffn_bias);

  // Normlar: katman basina iki norm; LayerNorm 2d, RMSNorm d parametre. + son norm.
  const std::int64_t norm_unit = shape.rmsnorm ? d : 2 * d;
  result.norms = shape.n_layer * 2 * norm_unit + norm_unit;

  // Head: bagliysa 0 (embed ile ayni agirlik), ayriksa vocab·d (LM head'i bias'siz).
  result.head = shape.tie_head ? 0 : shape.vocab * d;

  result.total = result.token_embed + result.pos_embed + result.attention + result.ffn + result.norms + result.head;
  return result;
}


MemoryEstimate estimate_memory(const ModelShape& shape, const DTypePlan& dtype, const Plan& plan) {
  validate(shape);
  if (plan.dp <= 0 || plan.tp <= 0 || plan.pp <= 0 || plan.zero_stage < 0 || plan.zero_stage > 3) {
    throw std::invalid_argument("nexus estimate: gecersiz paralellik plani");
  }
  const ParamBreakdown params = count_parameters(shape);

  MemoryEstimate result;
  result.budget_bytes = 8ll << 30;
  result.param_bytes = params.total * dtype.param_bytes;
  result.grad_bytes = params.total * dtype.grad_bytes;
  result.optim_bytes = params.total * dtype.optim_bytes;
  result.master_bytes = params.total * dtype.master_bytes;

  // ZeRO bolusumu (Rajbhandari ve ark.): optim evre>=1, grad evre>=2, param evre>=3.
  const std::int64_t model_parallel = plan.tp * plan.pp;
  const std::int64_t optim_div = model_parallel * (plan.zero_stage >= 1 ? plan.dp : 1);
  const std::int64_t grad_div = model_parallel * (plan.zero_stage >= 2 ? plan.dp : 1);
  const std::int64_t param_div = model_parallel * (plan.zero_stage >= 3 ? plan.dp : 1);
  result.static_sharded = result.param_bytes / param_div + result.grad_bytes / grad_div +
                          result.optim_bytes / optim_div + result.master_bytes / param_div;

  // Aktivasyon (Korthikanti ve ark.): katman basina s·b·h·(34 + 5·a·s/h) bayt,
  // 2 bayt/eleman varsayimiyla; burada activation_elem_bytes ile olceklenir.
  // Tam yeniden hesaplamada katman basina yalnizca 2·s·b·h saklanir.
  const double s = static_cast<double>(shape.seq_len);
  const double b = static_cast<double>(shape.micro_batch);
  const double h = static_cast<double>(shape.d_model);
  const double a = static_cast<double>(shape.n_head);
  const double scale = static_cast<double>(dtype.activation_elem_bytes) / 2.0;
  double per_layer;
  if (plan.full_recompute) {
    per_layer = 2.0 * s * b * h * scale;
  } else {
    per_layer = s * b * h * (34.0 + 5.0 * a * s / h) * scale;
  }
  result.activation_bytes =
      static_cast<std::int64_t>(per_layer * static_cast<double>(shape.n_layer / plan.pp));

  // Logit katmani: b·s·V·elem; ileri cikti + geri gradyan kopyasi (2x).
  result.logits_bytes = 2 * shape.micro_batch * shape.seq_len * shape.vocab * dtype.activation_elem_bytes;

  result.per_gpu_bytes = result.static_sharded + result.activation_bytes + result.logits_bytes;
  result.fits_budget = result.per_gpu_bytes <= result.budget_bytes;
  return result;
}


FlopEstimate estimate_flops(const ModelShape& shape, const Plan& plan) {
  validate(shape);
  if (plan.dp <= 0) {
    throw std::invalid_argument("nexus estimate: dp pozitif olmali");
  }
  const ParamBreakdown params = count_parameters(shape);

  FlopEstimate result;
  // 6N'deki N: matmul agirliklari. Embedding lookup FLOP uretmez; bagli head'de
  // embedding agirliklari head matmulunde zaten sayilir, ayrik head'de yalnizca
  // head sayilir (embedding cikarilir). Ogrenilen konum embedding'i de lookup'tir.
  result.matmul_params = params.total - params.pos_embed - (shape.tie_head ? 0 : params.token_embed);

  result.tokens_per_step = shape.micro_batch * shape.seq_len * plan.dp * shape.gradient_accum;

  // Kaplan/Chinchilla: 6·N·T. PaLM Ek B dikkat terimi: token basina 12·L·d·s.
  const double per_token = 6.0 * static_cast<double>(result.matmul_params) +
                           12.0 * static_cast<double>(shape.n_layer) * static_cast<double>(shape.d_model) *
                               static_cast<double>(shape.seq_len);
  result.flops_per_token = per_token;
  result.flops_per_step = per_token * static_cast<double>(result.tokens_per_step);
  return result;
}

Roofline roofline_bound(double flops_per_step, double bytes_per_step, double peak_flops_per_s,
                        double mem_bytes_per_s) {
  if (bytes_per_step <= 0.0 || peak_flops_per_s <= 0.0 || mem_bytes_per_s <= 0.0) {
    throw std::invalid_argument("nexus estimate: roofline girdileri pozitif olmali");
  }
  Roofline result;
  result.intensity = flops_per_step / bytes_per_step;
  const double memory_bound = result.intensity * mem_bytes_per_s;
  result.ceiling_flops_per_s = peak_flops_per_s < memory_bound ? peak_flops_per_s : memory_bound;
  return result;
}

}  // namespace estimate
}  // namespace nexus
