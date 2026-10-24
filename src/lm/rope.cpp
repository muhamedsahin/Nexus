/**
 * @file rope.cpp
 * @brief FAZ 6.2: RoPE frekanslari (HF modeling_rope_utils ile ayni formuller) ve dondurme.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/rope.hpp"

#include "nexus/core/error.hpp"

#include <algorithm>
#include <cmath>

namespace nexus {
namespace lm {

namespace {

constexpr double kPi = 3.14159265358979323846;

int original_length(const RopeScaling& s, int max_position) {
  return s.original_max_position > 0 ? s.original_max_position : max_position;
}

// HF: 1 / base^(arange(0, dim, 2) / dim), FP32.
std::vector<float> base_freq(int dim, double base) {
  std::vector<float> f(static_cast<std::size_t>(dim / 2));
  for (int i = 0; i < dim / 2; ++i) {
    f[static_cast<std::size_t>(i)] =
        static_cast<float>(1.0 / std::pow(base, static_cast<double>(2 * i) / static_cast<double>(dim)));
  }
  return f;
}

}  // namespace

float rope_attention_scale(const RopeScaling& s) {
  if (s.type != RopeScaling::Type::Yarn) {
    return 1.0f;
  }
  if (s.attention_factor > 0.0f) {
    return s.attention_factor;
  }
  return s.factor <= 1.0f ? 1.0f : static_cast<float>(0.1 * std::log(static_cast<double>(s.factor)) + 1.0);
}

std::vector<float> rope_inv_freq(int dim, float theta, const RopeScaling& s, int max_position, int seq_len) {
  if (dim < 2 || dim % 2 != 0) {
    throw TensorError("Nexus RoPE: rot_dim cift ve >= 2 olmali");
  }
  using T = RopeScaling::Type;
  const double factor = s.factor;
  if (s.type != T::None && !(factor >= 1.0)) {
    throw TensorError("Nexus RoPE: rope_scaling.factor >= 1 olmali");
  }
  const double exponent = static_cast<double>(dim) / static_cast<double>(dim - 2);
  switch (s.type) {
    case T::None:
      return base_freq(dim, theta);
    case T::Linear: {
      std::vector<float> f = base_freq(dim, theta);
      for (float& x : f) {
        x = static_cast<float>(x / factor);
      }
      return f;
    }
    case T::Ntk:
      return base_freq(dim, theta * std::pow(factor, exponent));
    case T::DynamicNtk: {
      const double L0 = original_length(s, max_position);
      const double n = std::max(static_cast<double>(seq_len), L0);
      return base_freq(dim, theta * std::pow(factor * n / L0 - (factor - 1.0), exponent));
    }
    case T::Yarn: {
      const double L0 = original_length(s, max_position);
      auto correction_dim = [&](double rotations) {
        return (dim * std::log(L0 / (rotations * 2.0 * kPi))) / (2.0 * std::log(static_cast<double>(theta)));
      };
      double low = std::floor(correction_dim(s.beta_fast));
      double high = std::ceil(correction_dim(s.beta_slow));
      low = std::max(low, 0.0);
      high = std::min(high, static_cast<double>(dim - 1));
      if (low == high) {
        high += 0.001;  // HF linear_ramp_factor: tekilligi onle
      }
      std::vector<float> f(static_cast<std::size_t>(dim / 2));
      for (int i = 0; i < dim / 2; ++i) {
        const float pos_freq = static_cast<float>(std::pow(static_cast<double>(theta), 2.0 * i / dim));
        const float extrapolation = 1.0f / pos_freq;
        const float interpolation = 1.0f / (static_cast<float>(factor) * pos_freq);
        const float ramp = static_cast<float>(std::clamp((i - low) / (high - low), 0.0, 1.0));
        const float extra_factor = 1.0f - ramp;
        f[static_cast<std::size_t>(i)] = interpolation * (1.0f - extra_factor) + extrapolation * extra_factor;
      }
      return f;
    }
    case T::Llama3: {
      std::vector<float> f = base_freq(dim, theta);
      const double L0 = original_length(s, max_position);
      const double low_wavelen = L0 / s.low_freq_factor;
      const double high_wavelen = L0 / s.high_freq_factor;
      for (float& x : f) {
        const double wavelen = 2.0 * kPi / x;
        if (wavelen > low_wavelen) {
          x = static_cast<float>(x / factor);
        } else if (wavelen >= high_wavelen) {
          const double smooth = (L0 / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
          x = static_cast<float>((1.0 - smooth) * x / factor + smooth * x);
        }
      }
      return f;
    }
  }
  return base_freq(dim, theta);
}

RopeCache::RopeCache(int rot_dim, float theta, RopeScaling scaling, int max_position)
    : rot_(rot_dim), theta_(theta), scaling_(scaling), max_position_(max_position) {
  if (rot_dim < 2 || rot_dim % 2 != 0) {
    throw TensorError("Nexus RoPE: rot_dim cift ve >= 2 olmali");
  }
}

void RopeCache::ensure(int length) {
  if (length < 1) {
    return;
  }
  const bool dynamic = scaling_.type == RopeScaling::Type::DynamicNtk;
  const int effective = dynamic ? std::max(length, original_length(scaling_, max_position_)) : 0;
  if (length <= length_ && (!dynamic || effective == built_for_)) {
    return;
  }
  if (dynamic && effective != built_for_) {
    length_ = 0;  // frekanslar degisti: tablo bastan
  }
  if (length_ == 0) {
    inv_freq_ = rope_inv_freq(rot_, theta_, scaling_, max_position_, effective);
    built_for_ = effective;
  }
  const int target = std::max(length, length_ * 2);  // tembel buyume
  const std::size_t h = half();
  cos_.resize(static_cast<std::size_t>(target) * h);
  sin_.resize(cos_.size());
  const float a = rope_attention_scale(scaling_);
  for (int p = length_; p < target; ++p) {
    for (std::size_t i = 0; i < h; ++i) {
      // HF: freqs = pos (FP32) * inv_freq (FP32), cos/sin FP32 (burada aci FP32 carpim, cos FP64->FP32).
      const float angle = static_cast<float>(p) * inv_freq_[i];
      cos_[static_cast<std::size_t>(p) * h + i] = static_cast<float>(std::cos(static_cast<double>(angle))) * a;
      sin_[static_cast<std::size_t>(p) * h + i] = static_cast<float>(std::sin(static_cast<double>(angle))) * a;
    }
  }
  length_ = target;
}

void rope_apply(float* x, int rows, int heads, int head_dim, const int* positions, const RopeCache& cache,
                bool transpose) {
  const int half = cache.rot_dim() / 2;
  if (cache.rot_dim() > head_dim) {
    throw TensorError("Nexus RoPE: rot_dim > head_dim");
  }
  const float sign = transpose ? -1.0f : 1.0f;
  for (int r = 0; r < rows; ++r) {
    const int pos = positions[r];
    if (pos < 0 || pos >= cache.length()) {
      throw TensorError("Nexus RoPE: konum onbellek disinda (ensure cagrilmadi)");
    }
    const float* c = cache.cos(pos);
    const float* s = cache.sin(pos);
    for (int h = 0; h < heads; ++h) {
      float* v = x + (static_cast<std::size_t>(r) * heads + h) * head_dim;
      for (int i = 0; i < half; ++i) {
        const float x1 = v[i];
        const float x2 = v[i + half];
        const float si = sign * s[i];
        v[i] = x1 * c[i] - x2 * si;
        v[i + half] = x2 * c[i] + x1 * si;
      }
    }
  }
}

}  // namespace lm
}  // namespace nexus
