/**
 * @file autoparallel.cpp
 * @brief FAZ 12.G: otomatik paralellik onericisi (kapali bicim tahmin; varsayimlar autoparallel.hpp'de).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "autoparallel.hpp"

#include <algorithm>
#include <stdexcept>

namespace nexus {
namespace estimate {

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

double ring_allreduce_s(double bytes, int n, double gbps) {
  if (n <= 1 || bytes <= 0.0) return 0.0;
  return 2.0 * (n - 1) / n * bytes / (gbps * 1e9);
}

}  // namespace

Candidate evaluate_plan(const ModelShape& shape, const DTypePlan& dtype, const Cluster& cl, const AutoOptions& opt, int dp,
                        int tp, int pp, int cp, int zero, int mb, bool recompute) {
  Candidate c;
  c.dp = dp;
  c.tp = tp;
  c.pp = pp;
  c.cp = cp;
  c.zero = zero;
  c.micro_batch = mb;
  c.recompute = recompute;
  const ParamBreakdown params = count_parameters(shape);
  const double P = static_cast<double>(params.total);
  const double s = static_cast<double>(shape.seq_len) / cp;
  const double b = mb;
  const double h = static_cast<double>(shape.d_model);
  const double a = static_cast<double>(shape.n_head);
  const double L = static_cast<double>(shape.n_layer);
  const double t = tp;
  const double ae = static_cast<double>(dtype.activation_elem_bytes);
  const std::int64_t seqs = opt.global_batch_tokens / shape.seq_len;
  c.grad_accum = seqs / (static_cast<std::int64_t>(dp) * mb);

  // ---- bellek ----
  const double mp = static_cast<double>(tp) * pp;
  const double dz = static_cast<double>(dp) * cp;  // ZeRO grubu (FAZ 12: dp x cp)
  const double optim_div = mp * (zero >= 1 ? dz : 1.0);
  const double grad_div = mp * (zero >= 2 ? dz : 1.0);
  const double param_div = mp * (zero >= 3 ? dz : 1.0);
  const double stat = P * dtype.param_bytes / param_div + P * dtype.grad_bytes / grad_div + P * dtype.optim_bytes / optim_div +
                      P * dtype.master_bytes / param_div;
  double per_layer = 0.0;  // bayt (2 bayt/eleman tablosu, ae ile olceklenir)
  const double scale = ae / 2.0;
  if (recompute) {
    per_layer = 2.0 * s * b * h / (opt.sequence_parallel ? t : 1.0) * scale;
  } else if (opt.sequence_parallel || tp == 1) {
    per_layer = s * b * h * (34.0 / t + 5.0 * a * s / (h * t)) * scale;
  } else {
    per_layer = s * b * h * (10.0 + 24.0 / t + 5.0 * a * s / (h * t)) * scale;
  }
  // 1F1B: ilk asama min(pp, m) mikro yigin x L/pp katman
  const double inflight = static_cast<double>(std::min<std::int64_t>(pp, std::max<std::int64_t>(c.grad_accum, 1)));
  const double act = per_layer * (L / pp) * inflight;
  const double logits = 2.0 * b * s * static_cast<double>(shape.vocab) / t * ae;
  c.static_gib = stat / kGiB;
  c.activation_gib = act / kGiB;
  c.logits_gib = logits / kGiB;
  c.memory_gib = c.static_gib + c.activation_gib + c.logits_gib + cl.reserve_gib;
  c.fits = c.memory_gib <= cl.mem_gib;

  // ---- zaman ----
  const double tokens = static_cast<double>(opt.global_batch_tokens);
  const double flops_token = 6.0 * static_cast<double>(params.total - params.pos_embed - (shape.tie_head ? 0 : params.token_embed)) +
                             12.0 * L * h * static_cast<double>(shape.seq_len);
  const double model_flops = flops_token * tokens;
  const double hw_flops = model_flops * (recompute ? 4.0 / 3.0 : 1.0);
  const double peak = cl.peak_tflops * 1e12;
  // GEMM verimi parca kuculdukce duser: satir (mikro yigin tokeni, cp'ye bolunmus) ve sutun (h / tp) icin
  // x / (x + 128) doygunluk carpani (varsayim; olcumle kalibre edilmedi).
  const double rows = b * s;
  const double cols = h / t;
  const double eff = cl.mfu * rows / (rows + 128.0) * cols / (cols + 128.0);
  c.compute_s = hw_flops / (cl.gpus * peak * eff);
  const double m = static_cast<double>(std::max<std::int64_t>(c.grad_accum, 1));
  const double tp_bw = tp <= cl.gpus_per_node ? cl.intra_gbps : cl.inter_gbps;
  const double act_bytes = b * s * h * ae;
  c.tp_comm_s = tp > 1 ? (L / pp) * m * 4.0 * ring_allreduce_s(act_bytes, tp, tp_bw) : 0.0;
  const double kv = static_cast<double>(shape.n_kv_head) / shape.n_head * h;
  const int span_cp = tp * cp;
  const double cp_bw = span_cp <= cl.gpus_per_node ? cl.intra_gbps : cl.inter_gbps;
  c.cp_comm_s = cp > 1 ? 0.5 * (L / pp) * m * 3.0 * (cp - 1) * 2.0 * b * s * kv / t * ae / (cp_bw * 1e9) : 0.0;
  const bool dp_inter = static_cast<std::int64_t>(dp) * cp * tp * pp > cl.gpus_per_node;
  const double dp_bw = dp_inter ? cl.inter_gbps : cl.intra_gbps;
  const double grad_bytes = P / mp * dtype.grad_bytes * (zero >= 3 ? 1.5 : 1.0);
  c.dp_comm_s = ring_allreduce_s(grad_bytes, dp * cp, dp_bw) * (1.0 - cl.dp_overlap);
  c.bubble = pp > 1 ? (pp - 1.0) / (m + pp - 1.0) : 0.0;
  c.step_s = (c.compute_s + c.tp_comm_s + c.cp_comm_s) / (1.0 - c.bubble) + c.dp_comm_s;
  c.mfu = model_flops / (c.step_s * cl.gpus * peak);
  return c;
}

AutoResult auto_parallel(const ModelShape& shape, const DTypePlan& dtype, const Cluster& cl, const AutoOptions& opt) {
  if (cl.gpus < 1 || cl.gpus_per_node < 1 || cl.mem_gib <= 0.0 || cl.peak_tflops <= 0.0 || cl.mfu <= 0.0) {
    throw std::invalid_argument("nexus auto-parallel: gecersiz kume tanimi");
  }
  AutoResult res;
  std::vector<Candidate> all;
  const std::int64_t seqs = opt.global_batch_tokens / shape.seq_len;
  if (seqs < 1) throw std::invalid_argument("nexus auto-parallel: global_batch_tokens < seq_len");
  for (int tp = 1; tp <= cl.gpus; tp *= 2) {
    if (cl.gpus % tp != 0 || shape.n_head % tp != 0) continue;
    if (shape.n_kv_head % tp != 0 && tp % shape.n_kv_head != 0) continue;
    for (int pp = 1; pp <= cl.gpus / tp; ++pp) {
      if ((cl.gpus / tp) % pp != 0 || shape.n_layer % pp != 0) continue;
      for (int cp = 1; cp <= std::min(opt.max_cp, cl.gpus / (tp * pp)); cp *= 2) {
        if ((cl.gpus / (tp * pp)) % cp != 0 || shape.seq_len % (2 * cp) != 0) continue;
        const int dp = cl.gpus / (tp * pp * cp);
        for (int mb : opt.micro_batches) {
          if (seqs % (static_cast<std::int64_t>(dp) * mb) != 0) continue;
          for (int zero : opt.zero_stages) {
            if (zero > 0 && dp * cp == 1) continue;  // tek kopya: ZeRO anlamsiz
            for (int rc = 0; rc <= (opt.try_recompute ? 1 : 0); ++rc) {
              Candidate c = evaluate_plan(shape, dtype, cl, opt, dp, tp, pp, cp, zero, mb, rc == 1);
              res.evaluated += 1;
              if (!c.fits) continue;
              res.feasible += 1;
              all.push_back(c);
            }
          }
        }
      }
    }
  }
  std::sort(all.begin(), all.end(), [](const Candidate& x, const Candidate& y) {
    if (x.step_s != y.step_s) return x.step_s < y.step_s;
    return x.memory_gib < y.memory_gib;
  });
  for (const Candidate& c : all) {
    if (static_cast<int>(res.best.size()) >= opt.top) break;
    res.best.push_back(c);
  }
  if (res.best.empty()) {
    res.note = "hicbir (dp, tp, pp, cp, mikro-yigin, recompute, zero) kombinasyonu " + std::to_string(cl.mem_gib) +
               " GiB'a sigmiyor (" + std::to_string(res.evaluated) + " denendi)";
  }
  return res;
}

}  // namespace estimate
}  // namespace nexus
