#pragma once

/**
 * @file passes.hpp
 * @brief FAZ 13.2-13.3: autodiff (ileri + geri birlesik grafik), gecisler ve planlama.
 *
 *  - dce / constant_fold / cse: klasik temizlik (topolojik sira korunur).
 *  - rewrite_patterns: (softmax(QK^T)V) -> Attention, silu(a)*b -> SwiGlu, gelu(x W + b) -> LinearBiasGelu,
 *    norm(a + b) -> AddNorm (+ artik Output), CE(h W^T) -> FusedLinearCE (logit tamponu yok, parcali).
 *  - build_joint: ters mod autodiff; ciktilar: kayip + "grad:<param>". Geri dugumler ileri dugumlerden
 *    sonra eklenir (joint.backward_begin). VJP kurali olmayan op acik hata.
 *  - fuse_elementwise: dikey (uretici -> tuketici) ve yatay (ayni girdiyi okuyan) eleman eleman
 *    kumeler -> Op::Fused. Guvenlik kurali: kumeye eklenen dugumun kume disi girdileri kumenin ilk
 *    dugumunden once tanimli olmali (dongu yok); Fused dugumu ilk uyenin yerine konur.
 *  - min_cut_recompute: AOTAutograd bolucusu gibi — ileri degerlerin saklanmasi icin akis agi
 *    (kapasite = bayt, ornek baglamayla), pahali dugumler (matmul, dikkat, gomme, CE) zorunlu kaynak;
 *    minimum kesim saklanacak kumeyi verir; kalan ileri degerler geri geciste kopyalanip yeniden hesaplanir.
 *  - plan_memory: yasam araliklari -> statik arena ofsetleri (buyukten kucuge, ilk uyan; ornek baglamayla).
 *  - schedule_communication: AllGather/ReduceScatter/AllReduce dugumleri girdileri hazir olur olmaz
 *    (oncelikli liste zamanlama), tuketicileri olabildigince gec; yurutucu iletisimi asenkron baslatir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/compile/ir.hpp"

#include <string>
#include <vector>

namespace nexus {
namespace ir {

struct PatternStats {
  int attention = 0, swiglu = 0, linear_gelu = 0, add_norm = 0, linear_ce = 0, linear_bias = 0, packed_qkv = 0;
};

struct EpilogueStats {
  int dgelu = 0, bgrad = 0, accumulate = 0, norm_backward = 0;
};
/// Geri gecis epilog fuzyonlari (cuBLASLt karsiliklari): MatMulNT -> GeluBwd => DGELU epilogu; MatMulTN(x, g) +
/// SumRows(g) => BGRAD (bias gradyani Output 1); Add(c, matmul) => beta = 1 biriktirme.
[[nodiscard]] Graph fuse_epilogues(const Graph& g, EpilogueStats* stats = nullptr);

[[nodiscard]] Graph dce(const Graph& g);
[[nodiscard]] Graph constant_fold(const Graph& g, int* folded = nullptr);
[[nodiscard]] Graph cse(const Graph& g, int* merged = nullptr);
/// mask bitleri: kPatAttention | kPatSwiGlu | kPatLinearGelu | kPatAddNorm | kPatLinearCE.
inline constexpr unsigned kPatAttention = 1, kPatSwiGlu = 2, kPatLinearGelu = 4, kPatAddNorm = 8, kPatLinearCE = 16,
                          kPatLinearBias = 32, kPatPackedQKV = 64, kPatAll = 127;
[[nodiscard]] Graph rewrite_patterns(const Graph& g, PatternStats* stats = nullptr, int ce_chunk_rows = 1024,
                                     unsigned mask = kPatAll);

struct Joint {
  Graph graph;
  int backward_begin = 0;
  std::vector<std::string> param_names;  // grad ciktilari sirasi (ciktilar: [loss, grad...])
};
[[nodiscard]] Joint build_joint(const Graph& forward);

struct FusionStats {
  int clusters = 0;
  int fused_nodes = 0;
};
[[nodiscard]] Graph fuse_elementwise(const Graph& g, FusionStats* stats = nullptr);

struct RecomputeStats {
  int saved = 0;        // geri gecise saklanan ileri deger
  int recomputed = 0;   // geri geciste yeniden hesaplanan dugum
  std::int64_t saved_bytes = 0;  // ornek baglamayla
  std::int64_t save_all_bytes = 0;
};
/// joint.backward_begin gerekir; ornek baglama (B, T) bayt maliyeti icin.
[[nodiscard]] Joint min_cut_recompute(const Joint& joint, const Bindings& example, RecomputeStats* stats = nullptr);

struct MemoryPlan {
  std::vector<std::int64_t> offset;  // deger basina bayt ofseti (-1: arenada degil: Param/Input/Output gorunumu)
  std::vector<std::int64_t> bytes;   // deger basina bayt (Output: ust dugumun ek ciktisi)
  std::vector<std::int64_t> aux_offset;  // Output(i) dugumleri icin ofset (ek cikti tamponu)
  std::int64_t arena_bytes = 0;
  std::int64_t naive_bytes = 0;      // her degere ayri tampon
};
[[nodiscard]] MemoryPlan plan_memory(const Graph& g, const Bindings& bindings);

[[nodiscard]] Graph schedule_communication(const Graph& g, int* moved = nullptr);

}  // namespace ir
}  // namespace nexus
