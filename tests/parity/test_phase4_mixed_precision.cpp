/** FAZ 4 host: autocast, FP8 RNE/amax ve dinamik olcek politikasi. */
#include "nexus/amp/autocast.hpp"
#include "nexus/core/fp8.hpp"
#include "nexus_train/mixed_precision/loss_scale_policy.hpp"
#include "nexus_test.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {
void autocast_tests() {
  using nexus::DType;
  using nexus::amp::AutocastGuard;
  using nexus::amp::CastPolicy;
  using nexus::amp::OpClass;
  nexus_test::section("FAZ 4.1 autocast op politikasi");
  constexpr std::array<CastPolicy, static_cast<int>(OpClass::kCount)> expected = {
      CastPolicy::Lower, CastPolicy::Lower, CastPolicy::Lower, CastPolicy::Lower, CastPolicy::Lower,
      CastPolicy::Fp32, CastPolicy::Fp32, CastPolicy::Fp32, CastPolicy::Fp32, CastPolicy::Fp32,
      CastPolicy::Fp32, CastPolicy::Fp32, CastPolicy::Fp32, CastPolicy::Fp32, CastPolicy::Fp32,
      CastPolicy::Fp32, CastPolicy::Promote, CastPolicy::Promote, CastPolicy::Promote,
      CastPolicy::Keep, CastPolicy::Keep, CastPolicy::Keep};
  AutocastGuard cpu(nexus::DeviceKind::Cpu, DType::BF16);
  for (int i = 0; i < static_cast<int>(OpClass::kCount); ++i) {
    const auto op = static_cast<OpClass>(i);
    const auto policy = expected[static_cast<std::size_t>(i)];
    nexus_test::expect(nexus::amp::policy_for(op) == policy,
                       std::string("politika tablosu: ") + nexus::amp::to_string(op));
    const DType lhs = policy == CastPolicy::Fp32 || policy == CastPolicy::Keep ? DType::F16 : DType::F32;
    const DType rhs = policy == CastPolicy::Promote || policy == CastPolicy::Keep ? DType::F32 : lhs;
    DType want = policy == CastPolicy::Lower ? DType::BF16 : DType::F32;
    if (policy == CastPolicy::Keep) want = lhs;
    nexus_test::expect(nexus::amp::compute_dtype(nexus::DeviceKind::Cpu, op, {lhs, rhs}) == want,
                       std::string("autocast dtype: ") + nexus::amp::to_string(op));
  }
  nexus_test::expect(nexus::amp::compute_dtype(nexus::DeviceKind::Cpu, OpClass::Matmul, {DType::I32}) == DType::I32,
                     "tamsayi matmul dar float'a cevrilmez");
  {
    AutocastGuard fp16(nexus::DeviceKind::Cpu, DType::F16);
    nexus_test::expect(nexus::amp::compute_dtype(nexus::DeviceKind::Cpu, OpClass::Linear, {DType::F32}) == DType::F16,
                       "ic ice guard FP16");
    nexus_test::expect(!nexus::amp::autocast_state(nexus::DeviceKind::Cuda).enabled, "aygit durumlari bagimsiz");
    {
      AutocastGuard off(nexus::DeviceKind::Cpu, DType::BF16, false);
      nexus_test::expect(nexus::amp::compute_dtype(nexus::DeviceKind::Cpu, OpClass::Add, {DType::F16, DType::F32}) ==
                             DType::F32, "kapali guard normal promote");
    }
    nexus_test::expect(nexus::amp::autocast_state(nexus::DeviceKind::Cpu).dtype == DType::F16,
                       "ic guard durumu geri yukler");
  }
  nexus_test::expect(nexus::amp::autocast_state(nexus::DeviceKind::Cpu).dtype == DType::BF16,
                     "dis guard durumu geri yukler");
  bool threw = false;
  try { AutocastGuard invalid(nexus::DeviceKind::Cpu, DType::F32); } catch (const nexus::TensorError&) { threw = true; }
  nexus_test::expect(threw, "F32 autocast tipi reddedilir");
}
void fp8_tests() {
  nexus_test::section("FAZ 4.5 host FP8 RNE ve amax");
  for (const nexus::Fp8Format format : {nexus::Fp8Format::E4M3, nexus::Fp8Format::E5M2}) {
    const std::uint8_t max_code = format == nexus::Fp8Format::E4M3 ? 0x7E : 0x7B;
    for (int code = 0; code <= 0xFF; ++code) {
      const auto bits = static_cast<std::uint8_t>(code);
      const float value = nexus::fp8_to_float(bits, format);
      if (std::isfinite(value)) {
        nexus_test::expect(nexus::float_to_fp8(value, format) == bits, "FP8 decode->encode bit dongusu");
      }
    }
    std::vector<std::pair<float, std::uint8_t>> cases = {{0.0f, 0}, {-0.0f, 0x80}};
    for (int code = 0; code < max_code; ++code) {
      const float a = nexus::fp8_to_float(static_cast<std::uint8_t>(code), format);
      const float b = nexus::fp8_to_float(static_cast<std::uint8_t>(code + 1), format);
      cases.push_back({(a + b) * 0.5f, static_cast<std::uint8_t>(code + (code & 1))});
    }
    cases.push_back({std::numeric_limits<float>::infinity(), max_code});
    cases.push_back({-std::numeric_limits<float>::infinity(), static_cast<std::uint8_t>(0x80 | max_code)});
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
      nexus_test::expect(std::fesetround(mode) == 0, "fesetround destekleniyor");
      for (const auto& [value, want] : cases) {
        nexus_test::expect(nexus::float_to_fp8(value, format) == want, "FP8 RNE host rounding mode'dan bagimsiz");
      }
    }
    std::fesetround(FE_TONEAREST);
    nexus_test::expect(nexus::fp8_to_float(nexus::float_to_fp8(static_cast<float>(nexus::fp8_max(format)), format), format) ==
                           nexus::fp8_max(format), "FP8 maksimum sonlu");
  }
  nexus::Fp8DelayedScaling scaling(nexus::Fp8Format::E4M3, 3);
  scaling.push_amax(1.0f);
  scaling.push_amax(2.0f);
  scaling.push_amax(8.0f);
  scaling.push_amax(16.0f);
  nexus_test::expect(scaling.history().size() == 3 && scaling.history().front() == 2.0f, "amax sabit uzunluklu pencere");
  nexus_test::expect(scaling.scale() == 28.0f, "delayed scaling amax maksimumu");
  nexus::Fp8BlockScaling blocks(nexus::Fp8Format::E4M3, 65, 32);
  blocks.push_amax(0, 2.0f);
  blocks.push_amax(2, 8.0f);
  nexus_test::expect(blocks.block_count() == 3 && blocks.block_index(64) == 2, "MX blok indeksleme");
  nexus_test::expect(blocks.scale(0) == 224.0f && blocks.scale(1) == 1.0f && blocks.scale(2) == 56.0f,
                     "MX blok bazli bagimsiz delayed scaling");
}
void loss_scale_tests() {
  nexus_test::section("FAZ 4.3 host loss-scale durum makinesi");
  nexus_train::LossScalePolicy policy;
  policy.init_scale = 1024.0;
  policy.max_scale = 2048.0;
  policy.growth_interval = 2;
  nexus_train::LossScaleState state(policy);
  nexus_test::expect(!state.update(false) && state.growth_tracker() == 1, "temiz adim buyume sayaci");
  nexus_test::expect(!state.update(false) && state.scale() == 2048.0 && state.growth_tracker() == 0,
                     "iki temiz adimda buyume ve ust sinir");
  nexus_test::expect(state.update(true) && state.scale() == 1024.0 && state.skipped_steps() == 1,
                     "overflow backoff ve atlama");
  nexus_test::expect(state.steps() == 3, "deneme sayaci tum adimlari icerir");
  for (const int mutate : {0, 1, 2, 3}) {
    nexus_train::LossScalePolicy bad;
    if (mutate == 0) bad.init_scale = std::numeric_limits<double>::infinity();
    if (mutate == 1) bad.init_scale = bad.max_scale * 2.0;
    if (mutate == 2) bad.min_scale = std::numeric_limits<double>::denorm_min();
    if (mutate == 3) bad.max_scale = std::numeric_limits<double>::infinity();
    bool threw = false;
    try { bad.validate(); } catch (const std::invalid_argument&) { threw = true; }
    nexus_test::expect(threw, "gecersiz loss-scale politikasi reddedilir");
  }
}

}  // namespace
int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "all";
  try {
    if (mode == "all" || mode == "autocast") autocast_tests();
    if (mode == "all" || mode == "fp8") fp8_tests();
    if (mode == "all" || mode == "loss") loss_scale_tests();
    return nexus_test::finish("phase4_mixed_precision_host");
  } catch (const std::exception& error) {
    nexus_test::expect(false, error.what());
    return nexus_test::finish("phase4_mixed_precision_host_exception");
  }
}

