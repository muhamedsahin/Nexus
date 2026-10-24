/**
 * @file test_estimate_parity.cpp
 * @brief FAZ 0: nexus_estimate kalibrasyonu ve C++/Python birebir paritesi.
 *
 * Kalibrasyon kaynaklari:
 *  - GPT-2 124M: 12L/768d/12h/3072ff/50257vocab/1024 konum, bias+LN+bagli head.
 *    Formulden 124.439.808 parametre; nominal 124M'e gore ~%0,35 (<= %0,5).
 *  - LLaMA-7B: 32L/4096d/32h/11008ff/32000vocab, bias yok, SwiGLU, RMSNorm,
 *    ayrik head. Formulden 6.738.415.616; yayimlanan ~6,74B'ye ~%0,02 (<= %0,5).
 * Parite: CLI (nexus_estimate) ve Python aynasi (estimate.py) ayni girdilerle
 * kosulur; JSON ciktilari alan alan karsilastirilir. Python bulunamazsa parite
 * bolumu atlanir (raporlanir), kalibrasyon bolumu her kosuda zorunludur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "estimate.hpp"
#include "json_mini.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

using nexus::estimate::DTypePlan;
using nexus::estimate::ModelShape;
using nexus::estimate::Plan;

[[nodiscard]] ModelShape gpt2_124m() {
  ModelShape shape;
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
  return shape;
}

[[nodiscard]] ModelShape llama_7b() {
  ModelShape shape;
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
  return shape;
}

// Windows'ta cmd /c, komut tirnakla basliyor ve icinde baska tirnak varsa dis tirnaklari
// soyar; tum komutu bir kat daha tirnaga almak bunu engeller.
[[nodiscard]] bool run_command(const std::string& command) {
#if defined(_WIN32)
  const std::string wrapped = "\"" + command + "\"";
  return std::system(wrapped.c_str()) == 0;
#else
  return std::system(command.c_str()) == 0;
#endif
}

[[nodiscard]] std::string find_python() {
#if defined(_WIN32)
  const char* candidates[] = {"python", "python3", "py -3"};
  const std::string silencer = " >nul 2>&1";
#else
  const char* candidates[] = {"python3", "python"};
  const std::string silencer = " >/dev/null 2>&1";
#endif
  for (const char* candidate : candidates) {
    if (run_command(std::string(candidate) + " --version" + silencer)) {
      return candidate;
    }
  }
  return {};
}

// JSON degerlerini alan alan karsilastirir; ilk uyusmazligi error'a yazar.
bool compare_values(const nexus::jsonmini::Value& lhs, const nexus::jsonmini::Value& rhs, const std::string& path,
                    std::string& error) {
  using nexus::jsonmini::Kind;
  if (lhs.kind() != rhs.kind()) {
    error = path + ": tur uyusmazligi";
    return false;
  }
  switch (lhs.kind()) {
    case Kind::Null:
      return true;
    case Kind::Bool:
      if (lhs.as_bool() != rhs.as_bool()) {
        error = path + ": bool uyusmazligi";
        return false;
      }
      return true;
    case Kind::Number: {
      const double a = lhs.as_number();
      const double b = rhs.as_number();
      const double magnitude = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
      if (std::fabs(a - b) > 1.0e-9 * magnitude) {
        error = path + ": sayi uyusmazligi cpp=" + std::to_string(a) + " py=" + std::to_string(b);
        return false;
      }
      return true;
    }
    case Kind::String:
      if (lhs.as_string() != rhs.as_string()) {
        error = path + ": dize uyusmazligi";
        return false;
      }
      return true;
    case Kind::Array: {
      const auto& left = lhs.as_array();
      const auto& right = rhs.as_array();
      if (left.size() != right.size()) {
        error = path + ": dizi uzunlugu uyusmazligi";
        return false;
      }
      for (std::size_t i = 0; i < left.size(); ++i) {
        if (!compare_values(left[i], right[i], path + "[" + std::to_string(i) + "]", error)) {
          return false;
        }
      }
      return true;
    }
    case Kind::Object: {
      const auto& left = lhs.as_object();
      const auto& right = rhs.as_object();
      if (left.size() != right.size()) {
        error = path + ": nesne alan sayisi uyusmazligi";
        return false;
      }
      for (const auto& [key, value] : left) {
        if (!right.count(key)) {
          error = path + "." + key + ": python cikisinda alan yok";
          return false;
        }
        if (!compare_values(value, right.at(key), path + "." + key, error)) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

void run_parity_case(const std::string& python, const std::string& arguments, const std::string& case_name) {
  const std::filesystem::path cpp_path =
      std::filesystem::temp_directory_path() / ("nexus_est_cpp_" + case_name + ".json");
  const std::filesystem::path py_path =
      std::filesystem::temp_directory_path() / ("nexus_est_py_" + case_name + ".json");

  const std::string cpp_command =
      std::string("\"") + NEXUS_ESTIMATE_EXE + "\" " + arguments + " --json \"" + cpp_path.string() + "\"";
  nexus_test::expect(run_command(cpp_command), ("CLI kosuldu: " + case_name).c_str());

  const std::string py_command =
      python + " \"" + NEXUS_ESTIMATE_PY + "\" " + arguments + " --json \"" + py_path.string() + "\"";
  nexus_test::expect(run_command(py_command), ("python kosuldu: " + case_name).c_str());

  try {
    const nexus::jsonmini::Value cpp_json = nexus::jsonmini::parse_file(cpp_path.string());
    const nexus::jsonmini::Value py_json = nexus::jsonmini::parse_file(py_path.string());
    std::string error;
    const bool equal = compare_values(cpp_json, py_json, "$", error);
    nexus_test::expect(equal, ("parite: " + case_name + (equal ? "" : " [" + error + "]")).c_str());
  } catch (const std::exception& exception) {
    nexus_test::expect(false, ("parite json okuma: " + case_name + " [" + exception.what() + "]").c_str());
  }
}

}  // namespace

int main() {
  nexus_test::section("kalibrasyon: parametre sayimi");
  {
    const auto gpt2 = nexus::estimate::count_parameters(gpt2_124m());
    nexus_test::expect(gpt2.total == 124439808, "GPT-2 124M tam sayim");
    nexus_test::expect(gpt2.token_embed == 38597376 && gpt2.pos_embed == 786432, "GPT-2 embed+konum");
    nexus_test::expect(gpt2.attention == 28348416 && gpt2.ffn == 56669184, "GPT-2 dikkat+ffn");
    nexus_test::expect(gpt2.norms == 38400 && gpt2.head == 0, "GPT-2 normlar+bagli head");
    const double nominal_ratio = std::fabs(static_cast<double>(gpt2.total) - 124.0e6) / 124.0e6;
    nexus_test::info("gpt2_nominal_fark", nominal_ratio);
    nexus_test::expect(nominal_ratio <= 0.005, "GPT-2 nominal 124M'e <= %0,5");

    const auto llama = nexus::estimate::count_parameters(llama_7b());
    nexus_test::expect(llama.total == 6738415616, "LLaMA-7B tam sayim");
    nexus_test::expect(llama.attention == 2147483648 && llama.ffn == 4328521728, "LLaMA dikkat+swiglu");
    const double llama_ratio = std::fabs(static_cast<double>(llama.total) - 6.74e9) / 6.74e9;
    nexus_test::info("llama_nominal_fark", llama_ratio);
    nexus_test::expect(llama_ratio <= 0.005, "LLaMA ~6,74B'ye <= %0,5");
  }

  nexus_test::section("flop formulu capraz kontrol");
  {
    const Plan plan;
    const auto flops = nexus::estimate::estimate_flops(gpt2_124m(), plan);
    const double expected_per_token = 6.0 * 123653376.0 + 12.0 * 12.0 * 768.0 * 1024.0;
    nexus_test::expect(flops.matmul_params == 123653376, "bagli head'de N = toplam - konum");
    nexus_test::expect(flops.tokens_per_step == 1024, "token/adim = b*s*dp*accum");
    nexus_test::expect(std::fabs(flops.flops_per_token - expected_per_token) < 1.0, "token basina 6N + 12Lds");
    nexus_test::expect(std::fabs(flops.flops_per_step - expected_per_token * 1024.0) < 1024.0, "adim floplari");

    const auto llama_flops = nexus::estimate::estimate_flops(llama_7b(), plan);
    nexus_test::expect(llama_flops.matmul_params == 6607343616, "ayrik head'de N = toplam - embed");
  }

  nexus_test::section("gqa ve gecersiz girdiler");
  {
    ModelShape gqa = llama_7b();
    gqa.n_kv_head = 8;  // d_kv = 128*8 = 1024
    const auto params = nexus::estimate::count_parameters(gqa);
    // Katman basina: 2*d^2 + 2*d*d_kv = 33.554.432 + 8.388.608
    nexus_test::expect(params.attention == 32LL * (2LL * 4096 * 4096 + 2LL * 4096 * 1024), "GQA k/v daralir");

    ModelShape bad_heads = gpt2_124m();
    bad_heads.n_head = 5;  // 768 % 5 != 0
    bool threw = false;
    try {
      (void)nexus::estimate::count_parameters(bad_heads);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    nexus_test::expect(threw, "d_model % n_head != 0 firlatilir");

    ModelShape bad_kv = gpt2_124m();
    bad_kv.n_kv_head = 5;  // 12 % 5 != 0
    threw = false;
    try {
      (void)nexus::estimate::count_parameters(bad_kv);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    nexus_test::expect(threw, "n_head % n_kv_head != 0 firlatilir");

    Plan bad_zero;
    bad_zero.zero_stage = 4;
    threw = false;
    try {
      (void)nexus::estimate::estimate_memory(gpt2_124m(), DTypePlan{}, bad_zero);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    nexus_test::expect(threw, "zero_stage > 3 firlatilir");
  }



  nexus_test::section("zero bolusumu ve yeniden hesaplama");
  {
    Plan plan;
    plan.dp = 8;
    const DTypePlan dtype;
    std::int64_t previous = 0;
    bool monotonic = true;
    for (std::int64_t stage = 0; stage <= 3; ++stage) {
      plan.zero_stage = stage;
      const auto memory = nexus::estimate::estimate_memory(gpt2_124m(), dtype, plan);
      if (stage > 0 && memory.static_sharded >= previous) {
        monotonic = false;
      }
      previous = memory.static_sharded;
    }
    nexus_test::expect(monotonic, "ZeRO 0>1>2>3 statik bellegi daraltir");

    plan.zero_stage = 0;
    const auto full = nexus::estimate::estimate_memory(gpt2_124m(), dtype, plan);
    // s*b*h*(34+5*a*s/h)*scale*L; FP32 scale=2: 1024*768*114*2*12
    nexus_test::expect(full.activation_bytes == 2151677952, "Korthikanti aktivasyon formulu");
    nexus_test::expect(full.logits_bytes == 411705344, "logit 2*b*s*V*4");

    Plan recompute = plan;
    recompute.full_recompute = true;
    const auto compact = nexus::estimate::estimate_memory(gpt2_124m(), dtype, recompute);
    // 2*s*b*h*scale*L = 2*1024*768*2*12
    nexus_test::expect(compact.activation_bytes == 37748736, "tam yeniden hesaplamada 2sbh*k*L");
    nexus_test::expect(compact.activation_bytes < full.activation_bytes, "yeniden hesaplama bellegi kucultur");
    // GPT-2 124M, mikro-batch 1, T=1024, FP32 + AdamW: ~4.6 GiB -> 8 GiB karta sigar
    // (FAZ 3 kabul kriteri bunu gerektirir). GPT-2 XL 1.5B ise sigmaz.
    nexus_test::expect(full.fits_budget, "GPT-2 124M FP32 mb=1 tek 8GiB aygita sigar");
    ModelShape xl = gpt2_124m();
    xl.n_layer = 48;
    xl.d_model = 1600;
    xl.n_head = 25;
    xl.n_kv_head = 25;
    xl.d_ff = 6400;
    const auto xl_memory = nexus::estimate::estimate_memory(xl, dtype, plan);
    nexus_test::expect(!xl_memory.fits_budget, "GPT-2 XL FP32 tek 8GiB aygita sigmaz (durust rapor)");
  }

  nexus_test::section("roofline siniri");
  {
    const auto memory_bound = nexus::estimate::roofline_bound(1.0e9, 1.0e6, 1.0e15, 1.0e9);
    nexus_test::expect(std::fabs(memory_bound.intensity - 1.0e3) < 1.0e-6, "yogunluk = flop/bayt");
    nexus_test::expect(std::fabs(memory_bound.ceiling_flops_per_s - 1.0e12) < 1.0, "bellek siniri: yogunluk*bant");

    const auto compute_bound = nexus::estimate::roofline_bound(1.0e15, 1.0e3, 1.0e15, 1.0e9);
    nexus_test::expect(std::fabs(compute_bound.ceiling_flops_per_s - 1.0e15) < 1.0e3, "hesap siniri: zirve");

    bool threw = false;
    try {
      (void)nexus::estimate::roofline_bound(1.0, 0.0, 1.0, 1.0);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    nexus_test::expect(threw, "sifir bayt roofline firlatilir");
  }

  nexus_test::section("CLI ve python aynasi paritesi");
  {
    const std::string python = find_python();
    if (python.empty()) {
      nexus_test::info("python_yok_parite_atlandi", 1);
    } else {
      run_parity_case(python, "--preset gpt2-124m", "gpt2");
      run_parity_case(python, "--preset llama-7b --peak-tflops 312 --mem-gbps 2039", "llama");
      run_parity_case(python,
                      "--preset llama-7b --layers 6 --d-model 512 --heads 8 --kv-heads 2 --d-ff 1376 "
                      "--vocab 9000 --seq 256 --batch 4 --learned-pos 0 --grad-accum 2 --no-tie --swiglu "
                      "--no-bias --rmsnorm --dp 2 --tp 2 --zero 1 --recompute --act-bytes 2 --master-bytes 4",
                      "ozel");
      // Tablo cikti yolu da kosulur (JSON dosyasi olmadan).
      nexus_test::expect(run_command(std::string("\"") + NEXUS_ESTIMATE_EXE + "\" --preset gpt2-124m >nul"),
                         "CLI tablo cikisi");
    }
  }

  return nexus_test::finish("faz0-tahmin-paritesi");
}
