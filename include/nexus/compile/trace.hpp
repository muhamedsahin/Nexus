#pragma once

/**
 * @file trace.hpp
 * @brief FAZ 13.2: izleme — meta tensorler (veri yok, yalniz sembolik sekil + dtype) ile op grafigi kurulur.
 *
 * `Builder` her islemde sekli cikarir ve dugumu ekler; deger = dugum kimligi (int). `trace_lm` bir
 * lm::LmConfig modelinin ileri gecisini (lm::LanguageModel ile ayni matematik ve parametre adlari)
 * semboller B, T ile izler: tokens Input [B*T] -> kaydirilmis CE kaybi. Dikkat ayristirilmis bicimde
 * (SplitHeads / BMMNT / Scale / CausalMask / Softmax / BMM / MergeHeads) izlenir; desen eslemesi (13.3)
 * bunu Attention'a donusturur. Desteklenmeyen lm alanlari (muP, son-norm, paralel blok, pencere, softcap,
 * dropout, qk_norm, ALiBi, sinuzoidal, RoPE olcekleme, recompute) acik hata.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/ir.hpp"
#include "nexus/lm/config.hpp"

namespace nexus {
namespace ir {

class Builder {
 public:
  explicit Builder(Graph& g) : g_(g) {}
  [[nodiscard]] Graph& graph() noexcept { return g_; }
  [[nodiscard]] const Shape& shape(int v) const { return g_.nodes.at(static_cast<std::size_t>(v)).shape; }

  int input(const std::string& name, Shape s, DType t);
  int param(const std::string& name, Shape s);
  int constant(double value, Shape s);
  int positions(int sym_b, int sym_t);  // [B*T] i32, satir r -> r % T

  int unary(Op op, int x, double a = 0.0);
  int binary(Op op, int x, int y);
  int bias_add(int x, int b);
  int sum_rows(int x);
  int matmul(int a, int b);
  int matmul_nt(int a, int b);
  int matmul_tn(int a, int b);
  int slice_cols(int x, std::int64_t begin, std::int64_t len);
  int concat_cols(const std::vector<int>& xs);
  int split_heads(int x, std::int64_t heads, std::int64_t d, int sym_b, int sym_t);
  int repeat_kv(int x, std::int64_t groups);
  int bmm_nt(int a, int b);
  int bmm(int a, int b);
  int causal_mask(int x);
  int softmax(int x);
  int merge_heads(int x, std::int64_t heads, std::int64_t d, int sym_b, int sym_t);
  int layer_norm(int x, int w, int b, double eps);  // b < 0: biassiz
  int rms_norm(int x, int w, double eps);
  int rope(int x, std::int64_t heads, std::int64_t d, std::int64_t rot, double theta, int sym_t);
  int embedding(int table, int ids);
  int cross_entropy(int logits, int tokens, std::int64_t valid_vocab, int sym_t);
  /// Genel: sekli verilen dugum.
  int node(Op op, std::vector<int> in, Shape s, DType t = DType::F32, double a = 0.0, std::vector<std::int64_t> ints = {});
  int output(int of, int index, Shape s, DType t = DType::F32);

 private:
  Graph& g_;
};

struct TracedLm {
  Graph graph;
  int sym_b = 0, sym_t = 1;
  int tokens = 0;  // Input dugumu
  int loss = 0;
};

[[nodiscard]] TracedLm trace_lm(const lm::LmConfig& config);

}  // namespace ir
}  // namespace nexus
