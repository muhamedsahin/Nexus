/**
 * @file bench_phase1_bridge.cpp
 * @brief Faz 1: sifir kopya bant genisligi, aile basina backward_into ve ayna kopyasi.
 *
 * Her kayip ailesinden bir temsilci isitilir, sonra `backward()` (vektor ayirir)
 * ile `backward_into` (cagiranin tamponu) ayni girdide olculur. Hiz orani
 * ayirma maliyetini gosterir. CUDA acikken MirroredTensor H2D+D2H turu
 * sayfa kilitli, sayfalanabilir ve pin_host sonrasi bellek icin ayrica olculur.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/mirrored_tensor.hpp"
#include "nexus/core/tensor.hpp"
#include "nexusloss/losses/classification.hpp"
#include "nexusloss/losses/detection.hpp"
#include "nexusloss/losses/distillation.hpp"
#include "nexusloss/losses/generative.hpp"
#include "nexusloss/losses/metric.hpp"
#include "nexusloss/losses/point_cloud.hpp"
#include "nexusloss/losses/ranking.hpp"
#include "nexusloss/losses/regression.hpp"
#include "nexusloss/losses/reinforcement.hpp"
#include "nexusloss/losses/segmentation.hpp"
#include "nexusloss/losses/self_supervised.hpp"
#include "nexusloss/losses/sequence.hpp"
#include "nexusloss/losses/survival.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Sample {
  double seconds = 0.0;
};

Sample time_call(int warmup, int repeats, const auto& fn) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  const Clock::time_point start = Clock::now();
  for (int i = 0; i < repeats; ++i) {
    fn();
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
  return Sample{seconds / static_cast<double>(repeats)};
}

void print_row(const std::string& name, double seconds, double bytes, double flops, const std::string& note) {
  const double gbps = seconds > 0.0 ? bytes / seconds / 1.0e9 : 0.0;
  const double gflops = seconds > 0.0 ? flops / seconds / 1.0e9 : 0.0;
  const double usec = seconds * 1.0e6;
  const double per_sec = seconds > 0.0 ? 1.0 / seconds : 0.0;
  std::cout << "| " << name;
  const int pad = 42 - static_cast<int>(name.size());
  for (int i = 0; i < pad; ++i) {
    std::cout << ' ';
  }
  std::cout << " | " << usec << " us | " << per_sec << " /s | " << gbps << " GB/s | " << gflops
            << " GFLOPS | " << note << " |\n";
}

std::string speed_note(double inplace_seconds, double allocating_seconds) {
  const double speedup = inplace_seconds > 0.0 ? allocating_seconds / inplace_seconds : 0.0;
  const double ns = inplace_seconds * 1.0e9;
  std::ostringstream note;
  note.setf(std::ios::fixed);
  note.precision(1);
  note << ns << " ns/cagri, speedup " << speedup << "x";
  return note.str();
}

bool finite_span(std::span<const float> values) {
  for (const float value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

template <typename Loss, typename Forward>
bool bench_cached(const std::string& name, Loss& loss, Forward&& forward, std::size_t gradient_n, double bytes,
                  double flops) {
  forward();
  std::vector<float> gradient(gradient_n, 0.0f);
  volatile float sink = 0.0f;
  const Sample allocating = time_call(2, 6, [&] {
    const std::vector<float> out = loss.backward();
    if (!out.empty()) {
      sink = out[0];
    }
  });
  const Sample inplace = time_call(2, 12, [&] { loss.backward_into(std::span<float>(gradient)); });
  if (!gradient.empty()) {
    sink += gradient[gradient_n / 2];
  }
  print_row(name + " backward()", allocating.seconds, bytes, flops, "vektor ayirir");
  print_row(name + " into", inplace.seconds, bytes, flops, speed_note(inplace.seconds, allocating.seconds));
  return std::isfinite(sink) && finite_span(gradient);
}

void fill_pattern(std::vector<float>& values, float scale, float bias) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = bias + scale * static_cast<float>(static_cast<int>(i % 17) - 8);
  }
}

#if defined(NEXUS_ENABLE_CUDA)

bool bench_mirror_round(const std::string& name, nexus::MirroredTensor& mirror) {
  const double bytes = static_cast<double>(mirror.bytes()) * 2.0;
  mirror.modify_host();
  mirror.sync_device(true);
  mirror.modify_device();
  mirror.sync_host();
  const Sample round = time_call(2, 6, [&] {
    mirror.modify_host();
    mirror.sync_device(true);
    mirror.modify_device();
    mirror.sync_host();
  });
  const double one_way = round.seconds * 0.5;
  print_row(name, one_way, bytes * 0.5, 0.0, "tek yon; tur H2D+D2H");
  volatile float sink = mirror.host_data()[0];
  return std::isfinite(sink) && mirror.bytes_to_device() > 0 && mirror.bytes_to_host() > 0;
}

bool bench_mirrors() {
  constexpr std::size_t kCount = 8u << 20;  // 32 MiB
  bool ok = true;
  {
    nexus::MirroredTensor pinned(nexus::Shape{kCount}, nexus::DeviceRef::cuda(0));
    pinned.host_data()[0] = 1.0f;
    ok = bench_mirror_round("H2D pinned", pinned) && ok;
  }
  {
    std::vector<float> host(kCount, 1.0f);
    nexus::MirroredTensor pageable = nexus::MirroredTensor::wrap_host(host.data(), nexus::Shape{kCount},
                                                                       nexus::DeviceRef::cuda(0));
    ok = bench_mirror_round("H2D pageable", pageable) && ok;
    if (!pageable.pin_host()) {
      print_row("H2D pin_host", 0.0, 0.0, 0.0, "kayit basarisiz");
      return false;
    }
    ok = bench_mirror_round("H2D pin_host sonrasi", pageable) && ok;
  }
  return ok;
}

#endif

}  // namespace

int main() {
  constexpr std::size_t kTriad = 1u << 22;
  constexpr std::size_t kGrad = 1u << 20;
  bool ok = true;

  nexus::Tensor a = nexus::Tensor::full(nexus::Shape{kTriad}, 1.0f);
  nexus::Tensor b = nexus::Tensor::full(nexus::Shape{kTriad}, 2.0f);
  nexus::Tensor c = nexus::Tensor::zeros(nexus::Shape{kTriad});
  nexus::TensorView<float> view_a = a.view();
  nexus::TensorView<float> view_b = b.view();
  nexus::TensorView<float> view_c = c.view();
  const Sample triad = time_call(3, 8, [&] {
    float* dst = view_c.data();
    const float* lhs = view_a.data();
    const float* rhs = view_b.data();
    for (std::size_t i = 0; i < kTriad; ++i) {
      dst[i] = lhs[i] + rhs[i];
    }
  });
  volatile float triad_sink = view_c.data()[kTriad / 2];

  std::vector<float> pred(kGrad);
  std::vector<float> target(kGrad, 0.25f);
  std::vector<float> probability(kGrad);
  std::vector<float> mask(kGrad);
  fill_pattern(pred, 0.1f, 0.0f);
  for (std::size_t i = 0; i < kGrad; ++i) {
    probability[i] = 0.05f + 0.9f * static_cast<float>(i % 19) / 18.0f;
    mask[i] = (i % 2) == 0 ? 1.0f : 0.0f;
  }
  const double element_bytes = static_cast<double>(kGrad) * 3.0 * static_cast<double>(sizeof(float));
  const double element_flops = static_cast<double>(kGrad) * 4.0;

  std::cout << "Nexus Faz 1 benchmark\n";
  std::cout << "triad N=" << kTriad << "  grad N=" << kGrad << "  sink=" << triad_sink << "\n";
  std::cout << "| islem                                      | sure      | hiz          | bant        | hesap        | not |\n";
  std::cout << "|--------------------------------------------|-----------|--------------|-------------|--------------|-----|\n";
  print_row("view triad y=a+b", triad.seconds,
            static_cast<double>(kTriad) * 3.0 * static_cast<double>(sizeof(float)), static_cast<double>(kTriad),
            "sifir kopya gorunum");

  {
    nexusloss::MSELoss<float> loss;
    loss.forward(pred, target);
    ok = bench_cached("MSE", loss, [] {}, kGrad, element_bytes, element_flops) && ok;
  }
  {
    nexusloss::HuberLoss<float> loss(1.0f);
    loss.forward(pred, target);
    ok = bench_cached("Huber", loss, [] {}, kGrad, element_bytes, element_flops) && ok;
  }
  {
    nexusloss::BCEWithLogitsLoss<float> loss;
    loss.forward(pred, mask);
    ok = bench_cached("BCEWithLogits", loss, [] {}, kGrad, element_bytes, element_flops * 2.0) && ok;
  }
  {
    nexusloss::FocalLoss<float> loss(0.25f, 2.0f);
    loss.forward(probability, mask);
    ok = bench_cached("Focal", loss, [] {}, kGrad, element_bytes, element_flops * 2.0) && ok;
  }
  {
    nexusloss::classification::CrossEntropyLoss<float> loss(0.0f);
    loss.forward(pred, 2);
    ok = bench_cached("CrossEntropy", loss, [] {}, kGrad, element_bytes, element_flops * 3.0) && ok;
  }
  {
    nexusloss::classification::KLDivLoss<float> loss;
    loss.forward(probability, target);
    ok = bench_cached("KLDiv", loss, [] {}, kGrad, element_bytes, element_flops * 2.0) && ok;
  }
  {
    nexusloss::segmentation::DiceLoss<float> loss(1.0f);
    loss.forward(probability, mask);
    ok = bench_cached("Dice", loss, [] {}, kGrad, element_bytes, element_flops) && ok;
  }
  {
    std::vector<float> boxes(kGrad);
    std::vector<float> goals(kGrad);
    for (std::size_t i = 0; i < kGrad; i += 4) {
      boxes[i] = 0.1f;
      boxes[i + 1] = 0.2f;
      boxes[i + 2] = 2.3f;
      boxes[i + 3] = 1.9f;
      goals[i] = 0.4f;
      goals[i + 1] = 0.0f;
      goals[i + 2] = 2.0f;
      goals[i + 3] = 2.4f;
    }
    nexusloss::detection::IoULoss<float> loss;
    loss.forward(boxes, goals);
    ok = bench_cached("IoU kutu", loss, [] {}, kGrad, element_bytes, static_cast<double>(kGrad)) && ok;
  }
  {
    constexpr std::size_t kDim = 1u << 16;
    std::vector<float> anchor(kDim);
    std::vector<float> positive(kDim);
    std::vector<float> negative(kDim);
    fill_pattern(anchor, 0.05f, 0.1f);
    fill_pattern(positive, 0.05f, 0.2f);
    fill_pattern(negative, 0.05f, 0.8f);
    nexusloss::metric::TripletLoss<float> loss(1.0f);
    loss.forward(anchor, positive, negative);
    const double bytes = static_cast<double>(kDim) * 6.0 * static_cast<double>(sizeof(float));
    ok = bench_cached("Triplet", loss, [] {}, kDim * 3, bytes, static_cast<double>(kDim) * 8.0) && ok;
  }
  {
    constexpr std::size_t kList = 1u << 14;
    std::vector<float> scores(kList);
    std::vector<float> relevance(kList);
    fill_pattern(scores, 0.1f, 0.0f);
    for (std::size_t i = 0; i < kList; ++i) {
      relevance[i] = static_cast<float>(i);
    }
    nexusloss::ranking::ListMLELoss<float> loss;
    loss.forward(scores, relevance);
    const double bytes = static_cast<double>(kList) * 3.0 * static_cast<double>(sizeof(float));
    ok = bench_cached("ListMLE", loss, [] {}, kList, bytes, static_cast<double>(kList) * 8.0) && ok;
  }
  {
    nexusloss::generative::AdversarialLoss<float> loss(false);
    std::vector<float> gradient(kGrad, 0.0f);
    volatile float sink = 0.0f;
    const Sample allocating = time_call(2, 6, [&] {
      const std::vector<float> out = loss.generator_backward(pred);
      sink = out[0];
    });
    const Sample inplace = time_call(2, 12, [&] { loss.generator_backward_into(pred, gradient); });
    sink += gradient[kGrad / 2];
    print_row("GAN G backward()", allocating.seconds, element_bytes, element_flops, "vektor ayirir");
    print_row("GAN G into", inplace.seconds, element_bytes, element_flops,
              speed_note(inplace.seconds, allocating.seconds));
    ok = std::isfinite(sink) && finite_span(gradient) && ok;
  }
  {
    nexusloss::distillation::KnowledgeDistillationLoss<float> loss(2.0f);
    loss.forward(pred, target);
    ok = bench_cached("KnowledgeDistillation", loss, [] {}, kGrad, element_bytes, element_flops * 4.0) && ok;
  }
  {
    constexpr std::size_t kSamples = 128;
    constexpr std::size_t kWidth = 64;
    std::vector<float> left_view(kSamples * kWidth);
    std::vector<float> right_view(kSamples * kWidth);
    fill_pattern(left_view, 0.1f, 0.2f);
    fill_pattern(right_view, 0.1f, -0.1f);
    nexusloss::self_supervised::VICRegLoss<float> loss;
    loss.forward(left_view, right_view, kSamples, kWidth);
    const double bytes = static_cast<double>(left_view.size()) * 4.0 * static_cast<double>(sizeof(float));
    ok = bench_cached("VICReg", loss, [] {}, left_view.size() * 2, bytes,
                      static_cast<double>(kSamples) * static_cast<double>(kWidth) * static_cast<double>(kWidth)) &&
         ok;
  }
  {
    constexpr std::size_t kTime = 32;
    constexpr std::size_t kClass = 16;
    std::vector<float> logits(kTime * kClass, 0.1f);
    const std::vector<std::size_t> labels{1, 2, 3, 4, 5};
    nexusloss::sequence::CTCLoss<float> loss(true);
    loss.forward(logits, kTime, kClass, labels, 0);
    const double bytes = static_cast<double>(logits.size()) * 2.0 * static_cast<double>(sizeof(float));
    ok = bench_cached("CTC T32 C16", loss, [] {}, logits.size(), bytes, static_cast<double>(logits.size()) * 20.0) &&
         ok;
  }
  {
    constexpr std::size_t kRisk = 4096;
    std::vector<float> log_risk(kRisk);
    std::vector<float> duration(kRisk);
    const std::unique_ptr<bool[]> observed(new bool[kRisk]);
    for (std::size_t i = 0; i < kRisk; ++i) {
      observed[i] = true;
    }
    fill_pattern(log_risk, 0.05f, 0.0f);
    for (std::size_t i = 0; i < kRisk; ++i) {
      duration[i] = static_cast<float>(i + 1);
    }
    nexusloss::survival::CoxPHLoss<float> loss;
    loss.forward(log_risk, duration, std::span<const bool>(observed.get(), kRisk));
    const double bytes = static_cast<double>(kRisk) * 3.0 * static_cast<double>(sizeof(float));
    ok = bench_cached("CoxPH", loss, [] {}, kRisk, bytes, static_cast<double>(kRisk) * static_cast<double>(kRisk)) &&
         ok;
  }
  {
    constexpr std::size_t kPoints = 256;
    std::vector<float> cloud_a(kPoints * 3);
    std::vector<float> cloud_b(kPoints * 3);
    fill_pattern(cloud_a, 0.02f, 0.0f);
    fill_pattern(cloud_b, 0.02f, 0.5f);
    nexusloss::point_cloud::ChamferDistanceLoss<float> loss(3);
    loss.forward(cloud_a, cloud_b);
    const double bytes = static_cast<double>(cloud_a.size() + cloud_b.size()) * 2.0 * static_cast<double>(sizeof(float));
    ok = bench_cached("Chamfer 256x3", loss, [] {}, cloud_a.size() + cloud_b.size(), bytes,
                      static_cast<double>(kPoints) * static_cast<double>(kPoints) * 6.0) &&
         ok;
  }
  {
    std::vector<float> newer(kGrad);
    std::vector<float> older(kGrad);
    std::vector<float> advantage(kGrad, 0.15f);
    fill_pattern(newer, 0.01f, -0.4f);
    fill_pattern(older, 0.01f, -0.5f);
    nexusloss::reinforcement::PPOClipLoss<float> loss(0.2f);
    loss.forward(newer, older, advantage);
    ok = bench_cached("PPOClip", loss, [] {}, kGrad, element_bytes, element_flops * 2.0) && ok;
  }

#if defined(NEXUS_ENABLE_CUDA)
  try {
    ok = bench_mirrors() && ok;
  } catch (const std::exception& error) {
    std::cout << "ayna olcumu basarisiz: " << error.what() << "\n";
    ok = false;
  }
#else
  print_row("H2D/D2H ayna", 0.0, 0.0, 0.0, "CUDA kapali, olcum yok");
#endif

  if (!ok || !std::isfinite(triad_sink)) {
    std::cout << "sonuc sayisal degil\n";
    return 1;
  }
  return 0;
}
