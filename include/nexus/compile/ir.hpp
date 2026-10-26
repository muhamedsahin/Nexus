#pragma once

/**
 * @file ir.hpp
 * @brief FAZ 13.1: nexus::ir — SSA tabanli, tensor duzeyi op grafigi.
 *
 * Dugum = op; her dugumun bir birincil ciktisi vardir, ek ciktilar (LSE, artik toplami, agirlik gradyani)
 * `Output(i)` dugumleriyle adlandirilir (ayri hesap yok, gorunum). Deger kimligi = dugum indeksi.
 * Dugumler eklenis sirasinda topolojiktir (girdiler hep onceki dugumler); gecisler bu degismezi korur.
 *
 * Sembolik sekiller: boyut = katsayi x sembol carpimi (orn. N = 1 * B * T, 3C = 3 * {}). Ornek boyutlar
 * (B, T) calisma aninda baglanir; boylece B ve T degisince yeniden derleme yok (13.2 koruyucu: yalniz
 * rank/dtype/sembol kumesi degisirse yeniden izleme).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nexus {
namespace ir {

/// coef * prod(syms). syms artan sembol kimlikleri (tekrar olabilir).
struct Dim {
  std::int64_t coef = 1;
  std::vector<int> syms;
  [[nodiscard]] static Dim constant(std::int64_t v) { return {v, {}}; }
  [[nodiscard]] static Dim symbol(int s) { return {1, {s}}; }
  [[nodiscard]] bool is_constant() const noexcept { return syms.empty(); }
  friend bool operator==(const Dim& a, const Dim& b) noexcept { return a.coef == b.coef && a.syms == b.syms; }
  friend bool operator!=(const Dim& a, const Dim& b) noexcept { return !(a == b); }
};
[[nodiscard]] Dim operator*(const Dim& a, const Dim& b);
using Shape = std::vector<Dim>;

using Bindings = std::map<int, std::int64_t>;  // sembol -> deger
[[nodiscard]] std::int64_t eval(const Dim& d, const Bindings& b);
[[nodiscard]] std::int64_t numel(const Shape& s, const Bindings& b);

enum class DType : std::uint8_t { F32, I32 };

enum class Op : std::uint16_t {
  // yapraklar
  Input, Param, Const, Positions,
  // eleman eleman (ayni sekil)
  Add, Sub, Mul, Div, Neg, Exp, Log, Tanh, Sigmoid, Gelu, GeluTanh, Silu, Relu, Square, Sqrt, Rsqrt, Scale, AddScalar,
  GeluBwd, GeluTanhBwd, SiluBwd, SigmoidBwd, TanhBwd,
  // yayinli eleman eleman: x [R, C] (op) b [C]
  BiasAdd,
  // indirgeme
  SumRows,  // [R, C] -> [C]
  // dogrusal cebir
  MatMul,    // a [M, K] @ b [K, N]
  MatMulNT,  // a [M, K] @ b[N, K]^T
  MatMulTN,  // a[K, M]^T @ b [K, N]
  // Epilog bayraklari (ints[0], matmul ailesi): 1 = son girdi biriktirilir (beta = 1), 2 = MatMulTN bias
  // gradyani Output(1) (BGRAD), 4 = MatMulNT ciktisi gelu'(pre) ile carpilir (DGELU; pre girdi 2).
  MatMulBias,  // x [M, K] @ w [K, N] + b [N]
  // sekil
  SliceCols, ConcatCols,
  // dikkat (ayristirilmis; desen eslemesiyle Attention'a donusur)
  SplitHeads,  // [N, h*D] -> [B*h, T, D]
  RepeatKV,    // [B*hk, T, D] -> [B*h, T, D]
  BMMNT,       // [G, T, D] x [G, T, D]^T -> [G, T, T]
  BMM,         // [G, T, T] x [G, T, D] -> [G, T, D]
  CausalMask,  // ust ucgen -inf
  Softmax,     // son eksen
  MergeHeads,  // [B*h, T, D] -> [N, h*D]
  // ozel cekirdekler (desen eslemesi hedefleri ve geri gecisleri)
  LayerNorm, LayerNormBwdX, LayerNormBwdW,  // w/b gradyani: BwdW ([2, C]: dw | db)
  RmsNorm, RmsNormBwdX, RmsNormBwdW,
  AddNorm,       // (a + b) -> norm; Output(1) = a + b
  Rope, RopeBwd,
  Attention,     // q [N, H D], k/v [N, Hk D] -> o; Output(1) = lse
  AttentionBwd,  // (do, q, k, v, o, lse) -> dq; Output(1) = dk, Output(2) = dv
  Embedding, EmbeddingBwd,
  CrossEntropy, CrossEntropyBwd,          // logits [N, V], tokens -> skaler (kaydirilmis, ortalama)
  FusedLinearCE, FusedLinearCEBwd,        // (h, w [V, C], tokens) -> skaler; Bwd -> dh, Output(1) = dw
  SwiGlu, SwiGluBwd,                      // [N, 2F] -> [N, F]
  LinearBiasGelu,                         // (x, w, b) -> gelu(x w + b); Output(1) = on-aktivasyon
  AttentionQKV, AttentionQKVBwd,          // paketli qkv [N, (H + 2Hk) D] (dilim yok); Bwd -> dqkv
  // iletisim (13.G)
  AllGather, ReduceScatter, AllReduce,
  // coklu cikti / fuzyon
  Output,
  Fused,
};
[[nodiscard]] const char* to_string(Op op) noexcept;
[[nodiscard]] bool is_elementwise(Op op) noexcept;  // ayni sekil cikti, oge basina (BiasAdd dahil)

struct FusedOp;

struct Node {
  Op op = Op::Const;
  std::vector<int> in;
  Shape shape;
  DType dtype = DType::F32;
  std::string name;       // Param/Input adi, hata ayiklama
  double a = 0.0;          // Const degeri, Scale/AddScalar katsayisi, eps, theta
  std::vector<std::int64_t> ints;  // ozniteliler (SliceCols: begin,len; SplitHeads: h, D; Attention: H, Hk, D ...)
  int index = 0;           // Output(i): cikti indeksi
  std::shared_ptr<FusedOp> fused;  // Op::Fused
};

/// Fuzyon kumesi: alt program (kume ici degerler; girdiler `in`, ciktilar `outputs`).
struct FusedOp {
  struct Inst {
    Op op = Op::Add;
    std::vector<int> args;  // < 0: -(girdi + 1), >= 0: kume ici talimat
    double a = 0.0;
  };
  std::vector<Inst> insts;
  std::vector<int> outputs;        // talimat indeksleri (birincil + Output(i))
  std::vector<bool> input_is_row;  // BiasAdd icin [C] girdisi (satir yayini)
  std::string key;                 // kaynak hash anahtari
};

struct Graph {
  std::vector<Node> nodes;
  std::vector<int> outputs;              // grafik ciktilari (kayip, gradyanlar ...)
  std::vector<std::string> output_names;
  std::vector<std::string> symbols;      // sembol adlari ("B", "T")

  int add(Node n);
  [[nodiscard]] int symbol(const std::string& name);
  [[nodiscard]] int find_param(const std::string& name) const;
  /// Kullanici sayilari (ciktilar dahil).
  [[nodiscard]] std::vector<int> use_counts() const;
  /// Metin dokumu (bir satir / dugum) ve Graphviz.
  [[nodiscard]] std::string to_text() const;
  [[nodiscard]] std::string to_dot() const;
  /// Yapisal hash (derleme onbellegi anahtari; sembolik sekiller dahil, ornek degerler haric).
  [[nodiscard]] std::string fingerprint() const;
  /// "Cekirdek baslatma" sayisi: Input/Param/Const/Output disindaki dugumler.
  [[nodiscard]] int kernel_count() const;
};

[[nodiscard]] std::string shape_str(const Shape& s, const Graph& g);

}  // namespace ir
}  // namespace nexus
