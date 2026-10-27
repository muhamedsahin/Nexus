#pragma once

/**
 * @file onnx.hpp
 * @brief FAZ 17.4: FAZ 13 IR'si (ir::trace_lm ileri grafigi) -> ONNX (ModelProto, el yazimi protobuf kodlayici).
 *
 * Girdi: "tokens" int64 [batch, seq] (dim_param: dinamik eksenler). Cikti: "logits" float [batch, seq, vocab].
 * IR dugumleri standart ONNX opset 17 / 18 islemlerine acilir (ozel op yok):
 *   Embedding -> Gather; Positions -> Range % T; MatMul / BiasAdd -> MatMul / Add; MatMulNT -> Gemm(transB = 1);
 *   RmsNorm -> x / sqrt(ReduceMean(x^2) + eps) * w (opset 18'de ReduceMean axes GIRDI, 17'de oznitelik);
 *   LayerNorm -> LayerNormalization; Gelu -> 0.5 x (1 + Erf(x / sqrt 2)); Silu -> x * Sigmoid(x);
 *   Rope -> Gather(cos/sin tablolari, konum) + yarim dondurme; SplitHeads / MergeHeads / RepeatKV -> Reshape +
 *   Transpose (+ Unsqueeze / Expand); BMMNT -> MatMul(a, Transpose(b)); CausalMask -> Where(j > i, -inf, x).
 * Dis veri (external_data): esikten buyuk agirliklar <ad>.data dosyasina (location / offset / length, 4096 hizali).
 * Dogrulama: bu depoda onnxruntime yok; test, bagimsiz Python protobuf cozucu + yorumlayici ile yapilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/model.hpp"

#include <cstdint>
#include <string>

namespace nexus {
namespace interop {

struct OnnxOptions {
  int opset = 17;                         // 17 | 18
  bool external_data = false;
  std::size_t external_threshold = 1024;  // bayt
};

struct OnnxStats {
  std::uint64_t model_bytes = 0;
  std::uint64_t data_bytes = 0;
  int nodes = 0;
  int initializers = 0;
  int ir_nodes = 0;
  double ms = 0.0;
};

void export_onnx(const lm::LanguageModel& model, const std::string& path, const OnnxOptions& options = {},
                 OnnxStats* stats = nullptr);

}  // namespace interop
}  // namespace nexus
