/**
 * @file bench_phase2.cpp
 * @brief FAZ 2: autograd motorunun ileri+geri verimi ve mod maliyeti.
 *
 * Olculenler (gercek steady_clock):
 *  - matmul ileri+geri: N yinelemeli [64,64]@[64,64], islem/sn.
 *  - Grad modu maliyeti: gradyan acik vs NoGradGuard (ileri) ns/islem.
 *  - Kucuk MLP adim verimi: ileri+geri islem/sn.
 * Sonuclar docs/benchmarks/2026-10-23-faz2/bench_phase2.json'a yazilir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

using Clock = std::chrono::steady_clock;
using nexus::autograd::Variable;
using nexus::autograd::Shape;

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
  std::printf("== faz2 autograd benchmarki ==\n");

  // --- matmul ileri+geri verimi -------------------------------------------------
  double matmul_ops_per_sec = 0.0;
  {
    constexpr int kIterations = 5000;
    Variable a = Variable::randn(Shape{{64, 64}}, 1, true);
    Variable b = Variable::randn(Shape{{64, 64}}, 2, true);
    const auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      Variable c = nexus::autograd::matmul(a, b);
      Variable loss = nexus::autograd::sum(c);
      loss.backward();
      a.zero_grad();
      b.zero_grad();
    }
    const double elapsed_s = ns_since(start) / 1.0e9;
    matmul_ops_per_sec = static_cast<double>(kIterations) / elapsed_s;
    info("matmul_ileri_geri_islem_sn", matmul_ops_per_sec);
    expect(matmul_ops_per_sec > 0.0, "matmul adimi olculdu");
  }

  // --- Grad modu maliyeti ----------------------------------------------------------
  double grad_ns = 0.0;
  double nograd_ns = 0.0;
  {
    constexpr int kIterations = 200000;
    Variable x = Variable::make({1.0f, 2.0f, 3.0f, 4.0f}, Shape{{4}}, true);
    auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      (void)nexus::autograd::mul_scalar(x, 2.0f);
    }
    grad_ns = ns_since(start) / static_cast<double>(kIterations);

    nexus::autograd::NoGradGuard guard;
    start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      (void)nexus::autograd::mul_scalar(x, 2.0f);
    }
    nograd_ns = ns_since(start) / static_cast<double>(kIterations);
    info("grad_acik_ns_islem", grad_ns);
    info("nograd_ns_islem", nograd_ns);
    expect(grad_ns > 0.0 && nograd_ns > 0.0, "mod maliyetleri olculdu");
  }

  // --- Kucuk MLP adim verimi ---------------------------------------------------------
  double mlp_steps_per_sec = 0.0;
  {
    constexpr int kIterations = 2000;
    const Variable w1 = Variable::randn(Shape{{4, 8}}, 11, true);
    const Variable b1 = Variable::zeros(Shape{{8}}, true);
    const Variable w2 = Variable::randn(Shape{{8, 3}}, 22, true);
    const Variable b2 = Variable::zeros(Shape{{3}}, true);
    const Variable input = Variable::randn(Shape{{1, 4}}, 33, true);
    const auto start = Clock::now();
    for (int i = 0; i < kIterations; ++i) {
      const Variable h = nexus::autograd::relu(nexus::autograd::add_bias(nexus::autograd::matmul(input, w1), b1));
      const Variable logits = nexus::autograd::add_bias(nexus::autograd::matmul(h, w2), b2);
      const Variable loss = nexus::autograd::sum(logits);
      loss.backward();
    }
    const double elapsed_s = ns_since(start) / 1.0e9;
    mlp_steps_per_sec = static_cast<double>(kIterations) / elapsed_s;
    info("mlp_adim_sn", mlp_steps_per_sec);
    expect(mlp_steps_per_sec > 0.0, "mlp adimi olculdu");
  }

  // --- Makinayla okunur JSON ----------------------------------------------------------
  {
    const std::filesystem::path docs_dir(NEXUS_FAZ2_DOCS_DIR);
    std::error_code ec;
    std::filesystem::create_directories(docs_dir, ec);
    const std::filesystem::path json_path = docs_dir / "bench_phase2.json";
    std::ofstream out(json_path, std::ios::binary | std::ios::trunc);
    if (out) {
      out.precision(17);
      out << "{\n  \"bench\": \"bench_phase2\",\n  \"phase\": 2,\n";
      out << "  \"results\": {\n";
      out << "    \"matmul_fwd_bwd_ops_per_sec\": " << matmul_ops_per_sec << ",\n";
      out << "    \"grad_enabled_ns_per_op\": " << grad_ns << ",\n";
      out << "    \"nograd_ns_per_op\": " << nograd_ns << ",\n";
      out << "    \"mlp_fwd_bwd_steps_per_sec\": " << mlp_steps_per_sec << "\n  },\n";
      out << "  \"notes\": {\n";
      out << "    \"engine\": \"bagimlilik sayaci + topolojik geri; reentrant per-cagri\",\n";
      out << "    \"double_backward\": \"FAZ 2'de altyapi disi (create_graph kabul edilir, kullanilmaz)\"\n  }\n";
      out << "}\n";
      out.flush();
      expect(static_cast<bool>(out), "bench JSON yazildi");
    } else {
      expect(false, "bench JSON yazilamadi");
    }
  }

  return finish("bench-faz2-autograd");
}
