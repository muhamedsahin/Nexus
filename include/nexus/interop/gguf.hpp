#pragma once

/**
 * @file gguf.hpp
 * @brief FAZ 17.3: GGUF (v3) disa aktarma + okuma — llama.cpp "llama" mimari kurallari.
 *
 * Dosya: "GGUF" | u32 3 | u64 tensor sayisi | u64 KV sayisi | KV'ler | tensor bilgileri (ad, boyutlar ne[0] en hizli,
 * tip, veri ofseti) | 32 bayt hizali veri. Meta veri: general.* , llama.* (context_length, embedding_length,
 * block_count, feed_forward_length, rope.dimension_count, rope.freq_base, attention.head_count(_kv),
 * attention.layer_norm_rms_epsilon), tokenizer.ggml.* (tokenizer.json verilirse: tokens / scores / token_type,
 * bos / eos / unk). Tensor adlari llama.cpp ile ayni (token_embd, blk.N.attn_{norm,q,k,v,output},
 * blk.N.ffn_{norm,gate,up,down}, output_norm, output). Q ve K agirliklari llama.cpp convert_hf_to_gguf
 * `permute`u ile yeniden siralanir (rotate_half -> GGML NORM RoPE'un bitisik ciftleri).
 * Tipler: F32 (0), F16 (1), Q4_0 (2: blok 32, f16 olcek + 16 bayt; d = max / -8), Q8_0 (8: blok 32, f16 olcek
 * + 32 int8; d = amax / 127). 1-boyutlu tensorler (norm) her zaman F32.
 * Dogrulama: bu depoda llama.cpp yok (indirilmedi); test, bagimsiz Python okuyucu + ileri gecis ile yapilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/interop/json.hpp"
#include "nexus/lm/model.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nexus {
namespace interop {

enum class GgufType : std::uint32_t { F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8 };
[[nodiscard]] const char* to_string(GgufType t) noexcept;

struct GgufOptions {
  GgufType type = GgufType::F16;   // 2-boyutlu agirliklar
  std::string tokenizer_json;      // bos: tokenizer meta verisi yazilmaz
  std::string name = "nexus";
};

struct GgufStats {
  std::uint64_t bytes = 0;
  int tensors = 0;
  int kv = 0;
  double ms = 0.0;
  std::map<std::string, int> types;  // tip -> tensor sayisi
};

void export_gguf(const lm::LanguageModel& model, const std::string& path, const GgufOptions& options = {},
                 GgufStats* stats = nullptr);

/// Okuma (dogrulama / gidis-donus): KV'ler (dize olarak) + tensor tablosu + FP32'ye acilmis veri.
struct GgufTensor {
  std::string name;
  std::vector<std::uint64_t> ne;
  GgufType type = GgufType::F32;
  std::uint64_t offset = 0;
  std::vector<float> data;  // acilmis (ne sirasiyla, satir-ana: ne[0] en hizli)
};
struct GgufFile {
  std::uint32_t version = 0;
  std::map<std::string, std::string> kv;  // skalerler metin; diziler "[n]"
  std::vector<std::string> tokens;
  std::vector<GgufTensor> tensors;
};
[[nodiscard]] GgufFile read_gguf(const std::string& path);
/// GGUF (llama) -> lm::LanguageModel (agirliklar FP32'ye acilir, Q/K ters permutasyon). Gidis-donus dogrulamasi icin.
[[nodiscard]] lm::LanguageModel import_gguf(const std::string& path);

/// Blok nicemleme yardimcilari (test / olcum): n % 32 == 0.
[[nodiscard]] std::vector<std::uint8_t> quantize_q8_0(const float* x, std::size_t n);
[[nodiscard]] std::vector<std::uint8_t> quantize_q4_0(const float* x, std::size_t n);
[[nodiscard]] std::vector<float> dequantize(GgufType t, const std::uint8_t* p, std::size_t n);

}  // namespace interop
}  // namespace nexus
