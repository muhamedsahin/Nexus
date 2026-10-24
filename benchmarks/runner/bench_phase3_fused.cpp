/**
 * @file bench_phase3_fused.cpp
 * @brief FAZ 3: füzyonlu çekirdeklerin ileri+geri verimi (füzyonsuz ile karsilastirma).
 *
 * Olculenler: fused_linear (füzyonlu) ve unfused (matmul+add_bias+gelu) ileri+geri
 * islem/sn; fused_layernorm ve fused_softmax ileri+geri verimi. Füzyon bellek
 * trafigini azaltir, sayisal sonuc degistirmez.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus/fused/fused.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

using Clock = std::chrono::steady_clock;
using nexus::autograd::Shape;
using nexus::autograd::Variable;
using namespace nexus::fused;

int g_failed = 0;

void expect(bool condition, std::string_view name) {
  if (condition) {
    std::printf("  PASS  %.*s\n", static_cast<int>(name.size()), name.data());
  } else {
    std::printf("  FAIL  %.*s\n", static_cast<int>(name.size()), name.data());
    ++g_failed;
  }
}

void info(std::string_view key, double value) {
  std::printf("  info  %.*s=%.6g\n", static_cast<int>(key.size()), key.data(), value);
}

[[nodiscard]] int finish(std::string_view suite) {
  if (g_failed != 0) {
    std::printf("  FAIL  %.*s: %d basarisiz denetim\n", static_cast<int>(suite.size()), suite.data(), g_failed);
    return 1;
  }
  std::printf("  OK    %.*s tamamlandi\n", static_cast<int>(suite.size()), suite.data());
  return 0;
}

[[nodiscard]] double ns_since(const Clock::time_point& start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count() * 1.0;
}

}  // namespace

int main() {
  std::printf("== faz3 füzyon benchmarki ==\n");
  constexpr int kIterations = 2000;
  const std::int64_t d = 64;

  // --- fused_linear vs unfused ----------------------------------------------------
  double fused_ns = 0.0;
  double unfused_ns = 0.0;
  {
    Variable x = Variable::randn(Shape{{d, d}}, 1, true);
    Variable w = Variable::randn(Shape{{d, d}}, 2, true);
    Variable b = Variable::zeros(Shape{{d}}, true);

    auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      Variable y = fused_linear(x, w, b, Activation::Gelu);
      Variable loss = nexus::autograd::sum(y);
      loss.backward();
      x.zero_grad();
      w.zero_grad();
      b.zero_grad();
    }
    fused_ns = ns_since(start) / static_cast<double>(kIterations);

    start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      Variable base = nexus::autograd::add_bias(nexus::autograd::matmul(x, w), b);
      Variable y = nexus::autograd::gelu(base);
      Variable loss = nexus::autograd::sum(y);
      loss.backward();
      x.zero_grad();
      w.zero_grad();
      b.zero_grad();
    }
    unfused_ns = ns_since(start) / static_cast<double>(kIterations);
    info("fused_linear_ns_islem", fused_ns);
    info("unfused_linear_ns_islem", unfused_ns);
    expect(fused_ns > 0.0 && unfused_ns > 0.0, "linear adimlari olculdu");
  }

  // --- fused_layernorm ve fused_softmax -------------------------------------------------
  double layernorm_ns = 0.0;
  double softmax_ns = 0.0;
  {
    Variable x = Variable::randn(Shape{{d, d}}, 3, true);
    Variable g = Variable::ones(Shape{{d}}, true);
    Variable b = Variable::zeros(Shape{{d}}, true);

    auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      Variable y = fused_layernorm(x, g, b);
      Variable loss = nexus::autograd::sum(y);
      loss.backward();
      x.zero_grad();
      g.zero_grad();
      b.zero_grad();
    }
    layernorm_ns = ns_since(start) / static_cast<double>(kIterations);

    start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      Variable y = fused_softmax(x);
      Variable loss = nexus::autograd::sum(y);
      loss.backward();
      x.zero_grad();
    }
    softmax_ns = ns_since(start) / static_cast<double>(kIterations);
    info("fused_layernorm_ns_islem", layernorm_ns);
    info("fused_softmax_ns_islem", softmax_ns);
    expect(layernorm_ns > 0.0 && softmax_ns > 0.0, "layernorm/softmax olculdu");
  }

  // --- Makinayla okunur JSON ------------------------------------------------------------
  {
    const std::filesystem::path docs_dir(NEXUS_FAZ3_DOCS_DIR);
    std::error_code ec;
    std::filesystem::create_directories(docs_dir, ec);
    const std::filesystem::path json_path = docs_dir / "bench_phase3_fused.json";
    std::ofstream out(json_path, std::ios::binary | std::ios::trunc);
    if (out) {
      out.precision(17);
      out << "{\n  \"bench\": \"bench_phase3_fused\",\n  \"phase\": 3,\n";
      out << "  \"results\": {\n";
      out << "    \"fused_linear_ns_per_op\": " << fused_ns << ",\n";
      out << "    \"unfused_linear_ns_per_op\": " << unfused_ns << ",\n";
      out << "    \"fused_layernorm_ns_per_op\": " << layernorm_ns << ",\n";
      out << "    \"fused_softmax_ns_per_op\": " << softmax_ns << "\n  },\n";
      out << "  \"notes\": {\n";
      out << "    \"fusion\": \"ileri+geri tek dugumde; ara aktivasyon ayri tensor dolasmaz\",\n";
      out << "    \"flash_attention\": \"sonraki faz; fused_softmax onun yapi tasidir\"\n  }\n";
      out << "}\n";
      out.flush();
      expect(static_cast<bool>(out), "bench JSON yazildi");
    } else {
      expect(false, "bench JSON yazilamadi");
    }
  }

  return finish("bench-faz3-füzyon");
}
