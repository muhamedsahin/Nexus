#pragma once

/**
 * @file compile.hpp
 * @brief FAZ 13.8: nexus::compile — lm::LanguageModel'i izleyip ileri + geri birlesik grafigi derler (CPU).
 *
 *   auto compiled = nexus::compile::compile(model, {.mode = "default" | "reduce-overhead" | "max-autotune",
 *                                                   .dynamic = true, .fullgraph = true});
 *   float loss = compiled.forward_backward(tokens, B, T, &grads);
 *
 * Boru hatti: trace_lm -> rewrite_patterns -> cse -> constant_fold -> build_joint -> cse -> min_cut_recompute
 * -> fuse_elementwise -> schedule_communication -> plan_memory (baglama basina) -> CPU JIT (fuzyon kumeleri).
 * dynamic = true: B ve T sembolik; degisince yeniden derleme yok (yalniz bellek plani yeniden kurulur).
 * dynamic = false: her yeni (B, T) yeniden izleme sayilir (onbellek yine derleyiciyi atlar).
 * fullgraph: Nexus izlemesi C++'ta oldugu icin grafik kirilmasi yoktur; false'ta da tek grafik.
 * "max-autotune": fuzyon kumeleri icin iki varyant (AVX2 + skaler) olculur, hizlisi secilir.
 * Hata ayiklama: dump_dir verilirse her gecisten sonra IR metni (.txt) + Graphviz (.dot) ve uretilen kaynak.
 * Sayisal karsilastirma: CompileOptions{patterns=false, fusion=false, recompute=false} "derleyici kapali"
 * yoludur (ayni yurutucu, ayristirilmis op'lar, geri gecis yine autodiff ile; dikkat ayristirilmis
 * oldugunda geri gecis yoktur -> yalniz ileri).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/codegen.hpp"
#include "nexus/compile/ir.hpp"
#include "nexus/compile/passes.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace lm {
class LanguageModel;
}
namespace dist {
class ProcessGroup;
}

namespace compile {

struct CompileOptions {
  std::string mode = "default";
  bool dynamic = true;
  bool fullgraph = true;
  bool patterns = true;
  unsigned pattern_mask = ir::kPatAll;  // ir::kPatAttention: yalniz SDPA (geri kalani eleman eleman fuzyonla)
  bool fusion = true;
  bool recompute = true;
  int ce_chunk_rows = 1024;
  int attention_threads = 0;  // 0: donanim
  std::string dump_dir;
  std::string cache_dir;      // bos: ir::default_cache_dir()
};

struct CompileStats {
  double trace_ms = 0.0, passes_ms = 0.0, codegen_ms = 0.0;
  int compiler_invocations = 0;
  int cache_hits = 0;
  int recompiles = 0;  // ilk derleme haric
  int kernels_eager = 0;     // IR'de yalniz SDPA desenli ileri+geri grafik cekirdek (dugum) sayisi
  int kernels_compiled = 0;  // desen + fuzyon sonrasi
  int fused_clusters = 0;
  ir::PatternStats patterns;
  ir::EpilogueStats epilogues;
  ir::RecomputeStats recompute;
  std::int64_t arena_bytes = 0;  // son baglama
  std::int64_t naive_bytes = 0;
  int autotuned = 0;
};

/// Genel grafik yurutucusu (CPU). Param degerleri disaridan (ad -> isaretci), Input'lar ad -> isaretci.
class Executable {
 public:
  Executable(ir::Graph graph, ir::CpuJit& jit, const CompileOptions& options);
  ~Executable();
  void set_group(std::shared_ptr<dist::ProcessGroup> group);
  /// Ciktilar graph().outputs sirasiyla.
  std::vector<std::vector<float>> run(const ir::Bindings& bindings, const std::map<std::string, const float*>& params,
                                      const std::map<std::string, const void*>& inputs);
  [[nodiscard]] const ir::Graph& graph() const noexcept { return graph_; }
  [[nodiscard]] std::int64_t arena_bytes() const noexcept { return last_arena_; }
  [[nodiscard]] std::int64_t naive_bytes() const noexcept { return last_naive_; }
  [[nodiscard]] int autotuned() const noexcept { return autotuned_; }
  struct Impl;

 private:
  ir::Graph graph_;
  std::unique_ptr<Impl> impl_;
  std::int64_t last_arena_ = 0, last_naive_ = 0;
  int autotuned_ = 0;
};

class CompiledLm {
 public:
  CompiledLm(const lm::LanguageModel& model, CompileOptions options);
  ~CompiledLm();
  CompiledLm(CompiledLm&&) noexcept;
  /// Kayip + gradyanlar (model.parameters() sirasiyla; grads bos degilse doldurulur).
  float forward_backward(const std::vector<std::int32_t>& tokens, int batch, int time,
                         std::vector<std::vector<float>>* grads);
  /// Yalniz ileri kayip (ayri ileri grafik).
  float loss(const std::vector<std::int32_t>& tokens, int batch, int time);
  [[nodiscard]] const CompileStats& stats() const noexcept { return stats_; }
  [[nodiscard]] const ir::Graph& forward_graph() const noexcept { return forward_; }
  [[nodiscard]] const ir::Graph& joint_graph() const;
  /// FAZ 13.7: ileri grafigin StableHLO (MLIR metin) disa aktarimi.
  [[nodiscard]] std::string stablehlo() const;

 private:
  void build(int batch, int time, bool train);
  const lm::LanguageModel* model_;
  CompileOptions options_;
  CompileStats stats_;
  ir::Graph forward_;
  std::unique_ptr<ir::CpuJit> jit_;
  std::unique_ptr<Executable> train_, eval_;
  std::vector<std::string> param_names_;
  int built_b_ = -1, built_t_ = -1;
};

[[nodiscard]] CompiledLm compile(const lm::LanguageModel& model, CompileOptions options = {});

/// FAZ 13.7: StableHLO MLIR metni (func.func @main). Ozel cekirdekler stablehlo.custom_call olarak.
[[nodiscard]] std::string export_stablehlo(const ir::Graph& g, const ir::Bindings& bindings);
/// Klasik HLO metni (yalniz matmul/eleman eleman alt kumesi; MatrixFlash compile_hlo ile gidis-donus testi).
[[nodiscard]] std::string export_hlo_text(const ir::Graph& g, const ir::Bindings& bindings);

}  // namespace compile
}  // namespace nexus
