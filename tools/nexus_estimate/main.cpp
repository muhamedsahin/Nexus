/**
 * @file main.cpp
 * @brief nexus_estimate araci: kapali bicim parametre/bellek/FLOP tahmini.
 *
 * Kullanim:
 *   nexus_estimate --preset gpt2-124m
 *   nexus_estimate --config tahmin.json --json cikti.json
 *   nexus_estimate --layers 12 --d-model 768 ... --peak-tflops 312 --mem-gbps 2039
 *
 * Cikti: insanlar icin tablo + --json verilirse makinayla okunur JSON.
 * Cati/parcalanma orani FAZ 0'da "n/a" raporlanir: k64 havuz ayiricisi
 * (CachingAllocator) FAZ 1'de gelir ve oran reserved/current'tan olculur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "autoparallel.hpp"
#include "estimate.hpp"
#include "json_mini.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using nexus::estimate::DTypePlan;
using nexus::estimate::ModelShape;
using nexus::estimate::Plan;
using nexus::estimate::AutoOptions;
using nexus::estimate::AutoResult;
using nexus::estimate::Candidate;
using nexus::estimate::Cluster;

void usage() {
  std::cout <<
      "nexus_estimate (FAZ 0)\n"
      "  --preset <gpt2-124m|llama-7b>   hazir mimari\n"
      "  --config <dosya.json>           model/dtype/plan/device alanlari\n"
      "  --layers N --d-model N --heads N --kv-heads N --d-ff N --vocab N\n"
      "  --seq N --batch N --learned-pos N --grad-accum N\n"
      "  --tie|--no-tie --swiglu|--no-swiglu --bias|--no-bias --rmsnorm|--no-rmsnorm\n"
      "  --dp N --tp N --pp N --zero <0..3> --recompute\n"
      "  --param-bytes N --grad-bytes N --optim-bytes N --master-bytes N --act-bytes N\n"
      "  --peak-tflops X --mem-gbps X    roofline ust siniri icin (ops.)\n"
      "  --json <dosya>                  JSON cikti yolu (ops.)\n"
      "  --auto-parallel                 FAZ 12.G: (dp,tp,pp,cp,mikro-yigin,recompute,zero) onericisi\n"
      "    --gpus N --gpus-per-node N --gpu-mem-gib X --peak-tflops X --intra-gbps X --inter-gbps X\n"
      "    --mfu X --dp-overlap X --global-batch-tokens N --top N\n"
      "  --help\n";
}

ModelShape preset_shape(const std::string& name) {
  ModelShape shape;
  if (name == "gpt2-124m") {
    shape.n_layer = 12;
    shape.d_model = 768;
    shape.n_head = 12;
    shape.n_kv_head = 12;
    shape.d_ff = 3072;
    shape.vocab = 50257;
    shape.seq_len = 1024;
    shape.micro_batch = 1;
    shape.learned_pos = 1024;
    shape.tie_head = true;
    shape.ffn_swiglu = false;
    shape.bias = true;
    shape.rmsnorm = false;
  } else if (name == "llama-7b") {
    shape.n_layer = 32;
    shape.d_model = 4096;
    shape.n_head = 32;
    shape.n_kv_head = 32;
    shape.d_ff = 11008;
    shape.vocab = 32000;
    shape.seq_len = 2048;
    shape.micro_batch = 1;
    shape.learned_pos = 0;
    shape.tie_head = false;
    shape.ffn_swiglu = true;
    shape.bias = false;
    shape.rmsnorm = true;
  } else {
    throw std::runtime_error("bilinmeyen preset: " + name);
  }
  return shape;
}

void apply_config(const nexus::jsonmini::Value& config, ModelShape& shape, DTypePlan& dtype, Plan& plan,
                  double& peak_tflops, double& mem_gbps) {
  using namespace nexus::jsonmini;
  if (config.has("model")) {
    const Value& m = config.at("model");
    shape.n_layer = get_int(m, "n_layer", shape.n_layer);
    shape.d_model = get_int(m, "d_model", shape.d_model);
    shape.n_head = get_int(m, "n_head", shape.n_head);
    shape.n_kv_head = get_int(m, "n_kv_head", shape.n_kv_head);
    shape.d_ff = get_int(m, "d_ff", shape.d_ff);
    shape.vocab = get_int(m, "vocab", shape.vocab);
    shape.seq_len = get_int(m, "seq_len", shape.seq_len);
    shape.micro_batch = get_int(m, "micro_batch", shape.micro_batch);
    shape.learned_pos = get_int(m, "learned_pos", shape.learned_pos);
    shape.gradient_accum = get_int(m, "gradient_accum", shape.gradient_accum);
    shape.tie_head = get_bool(m, "tie_head", shape.tie_head);
    shape.ffn_swiglu = get_bool(m, "ffn_swiglu", shape.ffn_swiglu);
    shape.bias = get_bool(m, "bias", shape.bias);
    shape.rmsnorm = get_bool(m, "rmsnorm", shape.rmsnorm);
  }
  if (config.has("dtype")) {
    const Value& d = config.at("dtype");
    dtype.param_bytes = get_int(d, "param_bytes", dtype.param_bytes);
    dtype.grad_bytes = get_int(d, "grad_bytes", dtype.grad_bytes);
    dtype.optim_bytes = get_int(d, "optim_bytes", dtype.optim_bytes);
    dtype.master_bytes = get_int(d, "master_bytes", dtype.master_bytes);
    dtype.activation_elem_bytes = get_int(d, "activation_elem_bytes", dtype.activation_elem_bytes);
  }
  if (config.has("plan")) {
    const Value& p = config.at("plan");
    plan.dp = get_int(p, "dp", plan.dp);
    plan.tp = get_int(p, "tp", plan.tp);
    plan.pp = get_int(p, "pp", plan.pp);
    plan.zero_stage = get_int(p, "zero_stage", plan.zero_stage);
    plan.full_recompute = get_bool(p, "full_recompute", plan.full_recompute);
  }
  if (config.has("device")) {
    const Value& d = config.at("device");
    peak_tflops = get_number(d, "peak_tflops", peak_tflops);
    mem_gbps = get_number(d, "mem_gbps", mem_gbps);
  }
}


std::string json_output(const ModelShape& shape, const DTypePlan& dtype, const Plan& plan, double peak_tflops,
                        double mem_gbps) {
  const auto params = nexus::estimate::count_parameters(shape);
  const auto memory = nexus::estimate::estimate_memory(shape, dtype, plan);
  const auto flops = nexus::estimate::estimate_flops(shape, plan);

  std::ostringstream out;
  out.precision(17);  // Python aynasiyla birebir ayni cift duyarlilik temsili
  out << "{\n";
  out << "  \"tool\": \"nexus_estimate\",\n  \"phase\": 0,\n";
  out << "  \"input\": {\n    \"model\": {";
  out << "\"n_layer\": " << shape.n_layer << ", \"d_model\": " << shape.d_model << ", \"n_head\": " << shape.n_head
      << ", \"n_kv_head\": " << shape.n_kv_head << ", \"d_ff\": " << shape.d_ff << ", \"vocab\": " << shape.vocab
      << ", \"seq_len\": " << shape.seq_len << ", \"micro_batch\": " << shape.micro_batch
      << ", \"learned_pos\": " << shape.learned_pos << ", \"gradient_accum\": " << shape.gradient_accum
      << ", \"tie_head\": " << (shape.tie_head ? "true" : "false")
      << ", \"ffn_swiglu\": " << (shape.ffn_swiglu ? "true" : "false")
      << ", \"bias\": " << (shape.bias ? "true" : "false") << ", \"rmsnorm\": " << (shape.rmsnorm ? "true" : "false")
      << "},\n";
  out << "    \"dtype\": {\"param_bytes\": " << dtype.param_bytes << ", \"grad_bytes\": " << dtype.grad_bytes
      << ", \"optim_bytes\": " << dtype.optim_bytes << ", \"master_bytes\": " << dtype.master_bytes
      << ", \"activation_elem_bytes\": " << dtype.activation_elem_bytes << "},\n";
  out << "    \"plan\": {\"dp\": " << plan.dp << ", \"tp\": " << plan.tp << ", \"pp\": " << plan.pp
      << ", \"zero_stage\": " << plan.zero_stage
      << ", \"full_recompute\": " << (plan.full_recompute ? "true" : "false") << "}\n  },\n";
  out << "  \"params\": {\"token_embed\": " << params.token_embed << ", \"pos_embed\": " << params.pos_embed
      << ", \"attention\": " << params.attention << ", \"ffn\": " << params.ffn << ", \"norms\": " << params.norms
      << ", \"head\": " << params.head << ", \"total\": " << params.total << "},\n";
  out << "  \"memory\": {\"param_bytes\": " << memory.param_bytes << ", \"grad_bytes\": " << memory.grad_bytes
      << ", \"optim_bytes\": " << memory.optim_bytes << ", \"master_bytes\": " << memory.master_bytes
      << ", \"static_sharded\": " << memory.static_sharded << ", \"activation_bytes\": " << memory.activation_bytes
      << ", \"logits_bytes\": " << memory.logits_bytes << ", \"per_gpu_bytes\": " << memory.per_gpu_bytes
      << ", \"budget_bytes\": " << memory.budget_bytes
      << ", \"fits_budget\": " << (memory.fits_budget ? "true" : "false") << "},\n";
  out << "  \"flops\": {\"matmul_params\": " << flops.matmul_params
      << ", \"tokens_per_step\": " << flops.tokens_per_step << ", \"flops_per_token\": " << flops.flops_per_token
      << ", \"flops_per_step\": " << flops.flops_per_step << "},\n";
  if (peak_tflops > 0.0 && mem_gbps > 0.0) {
    // Adim basina en az hareket: statik dokunma (param okuma + grad yazma + optim okuma/yazma ~ 3x)
    // + aktivasyon + logit. Kesin deger FAZ 1 izlerinden olculur; bu alt sinirdir.
    const double bytes_per_step = static_cast<double>(memory.activation_bytes + memory.logits_bytes) +
                                  3.0 * static_cast<double>(flops.matmul_params) *
                                      static_cast<double>(dtype.param_bytes);
    const auto roof = nexus::estimate::roofline_bound(flops.flops_per_step, bytes_per_step, peak_tflops * 1e12,
                                                      mem_gbps * 1e9);
    out << "  \"roofline\": {\"bytes_per_step\": " << bytes_per_step << ", \"intensity\": " << roof.intensity
        << ", \"ceiling_flops_per_s\": " << roof.ceiling_flops_per_s << "},\n";
  } else {
    out << "  \"roofline\": null,\n";
  }
  out << "  \"fragmentation\": {\"ratio\": \"n/a\", \"note\": \"FAZ 1 k64 havuzu reserved/current ile olcecek\"}\n";
  out << "}\n";
  return out.str();
}

void print_table(const ModelShape& shape, const DTypePlan& dtype, const Plan& plan) {
  const auto params = nexus::estimate::count_parameters(shape);
  const auto memory = nexus::estimate::estimate_memory(shape, dtype, plan);
  const auto flops = nexus::estimate::estimate_flops(shape, plan);
  std::printf("== nexus_estimate (FAZ 0) ==\n");
  std::printf("Parametreler: toplam %lld (embed %lld + konum %lld + dikkat %lld + ffn %lld + norm %lld + head %lld)\n",
              static_cast<long long>(params.total), static_cast<long long>(params.token_embed),
              static_cast<long long>(params.pos_embed), static_cast<long long>(params.attention),
              static_cast<long long>(params.ffn), static_cast<long long>(params.norms),
              static_cast<long long>(params.head));
  std::printf("Bellek/aygit: statik %lld + aktivasyon %lld + logit %lld = %lld bayt (butce %lld, sigar: %s)\n",
              static_cast<long long>(memory.static_sharded), static_cast<long long>(memory.activation_bytes),
              static_cast<long long>(memory.logits_bytes), static_cast<long long>(memory.per_gpu_bytes),
              static_cast<long long>(memory.budget_bytes), memory.fits_budget ? "evet" : "hayir");
  std::printf("FLOP/adim: %.3e (token/adim %lld, token basina %.3e)\n", flops.flops_per_step,
              static_cast<long long>(flops.tokens_per_step), flops.flops_per_token);
  std::printf("Catilma orani: n/a (FAZ 1 k64 havuzu reserved/current ile olcecek)\n");
}


}  // namespace

int main(int argc, char** argv) {
  ModelShape shape;
  DTypePlan dtype;
  Plan plan;
  double peak_tflops = 0.0;
  double mem_gbps = 0.0;
  std::string json_path;
  bool preset_applied = false;
  bool auto_mode = false;
  Cluster cluster;
  AutoOptions auto_opt;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      auto next_value = [&]() -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error("eksik deger: " + argument);
        }
        return argv[++i];
      };
      if (argument == "--help") {
        usage();
        return 0;
      } else if (argument == "--preset") {
        shape = preset_shape(next_value());
        preset_applied = true;
      } else if (argument == "--config") {
        apply_config(nexus::jsonmini::parse_file(next_value()), shape, dtype, plan, peak_tflops, mem_gbps);
      } else if (argument == "--json") {
        json_path = next_value();
      } else if (argument == "--layers") {
        shape.n_layer = std::stoll(next_value());
      } else if (argument == "--d-model") {
        shape.d_model = std::stoll(next_value());
      } else if (argument == "--heads") {
        shape.n_head = std::stoll(next_value());
      } else if (argument == "--kv-heads") {
        shape.n_kv_head = std::stoll(next_value());
      } else if (argument == "--d-ff") {
        shape.d_ff = std::stoll(next_value());
      } else if (argument == "--vocab") {
        shape.vocab = std::stoll(next_value());
      } else if (argument == "--seq") {
        shape.seq_len = std::stoll(next_value());
      } else if (argument == "--batch") {
        shape.micro_batch = std::stoll(next_value());
      } else if (argument == "--learned-pos") {
        shape.learned_pos = std::stoll(next_value());
      } else if (argument == "--grad-accum") {
        shape.gradient_accum = std::stoll(next_value());
      } else if (argument == "--tie") {
        shape.tie_head = true;
      } else if (argument == "--no-tie") {
        shape.tie_head = false;
      } else if (argument == "--swiglu") {
        shape.ffn_swiglu = true;
      } else if (argument == "--no-swiglu") {
        shape.ffn_swiglu = false;
      } else if (argument == "--bias") {
        shape.bias = true;
      } else if (argument == "--no-bias") {
        shape.bias = false;
      } else if (argument == "--rmsnorm") {
        shape.rmsnorm = true;
      } else if (argument == "--no-rmsnorm") {
        shape.rmsnorm = false;
      } else if (argument == "--dp") {
        plan.dp = std::stoll(next_value());
      } else if (argument == "--tp") {
        plan.tp = std::stoll(next_value());
      } else if (argument == "--pp") {
        plan.pp = std::stoll(next_value());
      } else if (argument == "--zero") {
        plan.zero_stage = std::stoll(next_value());
      } else if (argument == "--recompute") {
        plan.full_recompute = true;
      } else if (argument == "--param-bytes") {
        dtype.param_bytes = std::stoll(next_value());
      } else if (argument == "--grad-bytes") {
        dtype.grad_bytes = std::stoll(next_value());
      } else if (argument == "--optim-bytes") {
        dtype.optim_bytes = std::stoll(next_value());
      } else if (argument == "--master-bytes") {
        dtype.master_bytes = std::stoll(next_value());
      } else if (argument == "--act-bytes") {
        dtype.activation_elem_bytes = std::stoll(next_value());
      } else if (argument == "--peak-tflops") {
        peak_tflops = std::stod(next_value());
        cluster.peak_tflops = peak_tflops;
      } else if (argument == "--auto-parallel") {
        auto_mode = true;
      } else if (argument == "--gpus") {
        cluster.gpus = std::stoi(next_value());
      } else if (argument == "--gpus-per-node") {
        cluster.gpus_per_node = std::stoi(next_value());
      } else if (argument == "--gpu-mem-gib") {
        cluster.mem_gib = std::stod(next_value());
      } else if (argument == "--intra-gbps") {
        cluster.intra_gbps = std::stod(next_value());
      } else if (argument == "--inter-gbps") {
        cluster.inter_gbps = std::stod(next_value());
      } else if (argument == "--mfu") {
        cluster.mfu = std::stod(next_value());
      } else if (argument == "--dp-overlap") {
        cluster.dp_overlap = std::stod(next_value());
      } else if (argument == "--global-batch-tokens") {
        auto_opt.global_batch_tokens = std::stoll(next_value());
      } else if (argument == "--top") {
        auto_opt.top = std::stoi(next_value());
      } else if (argument == "--mem-gbps") {
        mem_gbps = std::stod(next_value());
      } else {
        std::cerr << "bilinmeyen arguman: " << argument << "\n";
        usage();
        return 2;
      }
    }


    if (!preset_applied && argc <= 1) {
      // Varsayilan: GPT-2 124M kalibrasyon sekli (tablo dogrulamasi kolay).
      shape = preset_shape("gpt2-124m");
    }

    if (auto_mode) {
      const AutoResult r = nexus::estimate::auto_parallel(shape, dtype, cluster, auto_opt);
      std::printf("Otomatik paralellik (TAHMIN, olcum degil): %d GPU (%d/dugum), %.0f GiB, %.0f TFLOPS, mfu %.2f\n",
                  cluster.gpus, cluster.gpus_per_node, cluster.mem_gib, cluster.peak_tflops, cluster.mfu);
      std::printf("Denenen %lld, sigan %lld kombinasyon\n", static_cast<long long>(r.evaluated),
                  static_cast<long long>(r.feasible));
      std::string js = "{\"evaluated\": " + std::to_string(r.evaluated) + ", \"feasible\": " + std::to_string(r.feasible) +
                       ", \"best\": [";
      int rank = 1;
      for (const Candidate& c : r.best) {
        std::printf("#%d dp=%d tp=%d pp=%d cp=%d zero=%d mb=%d accum=%lld recompute=%s | bellek %.1f GiB (statik %.1f, akt %.1f, "
                    "logit %.1f) | adim %.3f s (hesap %.3f, tp %.3f, cp %.3f, dp %.3f, kabarcik %.1f%%) | MFU %.1f%%\n",
                    rank, c.dp, c.tp, c.pp, c.cp, c.zero, c.micro_batch, static_cast<long long>(c.grad_accum),
                    c.recompute ? "evet" : "hayir", c.memory_gib, c.static_gib, c.activation_gib, c.logits_gib, c.step_s,
                    c.compute_s, c.tp_comm_s, c.cp_comm_s, c.dp_comm_s, 100.0 * c.bubble, 100.0 * c.mfu);
        js += std::string(rank > 1 ? ", " : "") + "{\"dp\": " + std::to_string(c.dp) + ", \"tp\": " + std::to_string(c.tp) +
              ", \"pp\": " + std::to_string(c.pp) + ", \"cp\": " + std::to_string(c.cp) + ", \"zero\": " + std::to_string(c.zero) +
              ", \"micro_batch\": " + std::to_string(c.micro_batch) + ", \"recompute\": " + (c.recompute ? "true" : "false") +
              ", \"memory_gib\": " + std::to_string(c.memory_gib) + ", \"step_s\": " + std::to_string(c.step_s) +
              ", \"mfu\": " + std::to_string(c.mfu) + "}";
        ++rank;
      }
      js += "], \"note\": \"" + r.note + "\"}\n";
      if (!r.note.empty()) std::printf("%s\n", r.note.c_str());
      if (!json_path.empty()) {
        std::ofstream file(json_path, std::ios::binary | std::ios::trunc);
        file << js;
      }
      return r.best.empty() ? 3 : 0;
    }
    print_table(shape, dtype, plan);
    const std::string json = json_output(shape, dtype, plan, peak_tflops, mem_gbps);
    if (!json_path.empty()) {
      std::ofstream file(json_path, std::ios::binary | std::ios::trunc);
      if (!file) {
        std::cerr << "json dosyasi acilamadi: " << json_path << "\n";
        return 1;
      }
      file << json;
      std::printf("JSON yazildi: %s\n", json_path.c_str());
    } else {
      std::cout << json;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "nexus_estimate hatasi: " << error.what() << "\n";
    return 1;
  }
}
