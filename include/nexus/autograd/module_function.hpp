#pragma once

/**
 * @file module_function.hpp
 * @brief Klasik nexus_model katmanini tek bir autograd dugumu olarak sarar — FAZ 2.E.
 *
 * Elle optimize edilmis `forward`/`backward` ciftleri kaybolmaz: ileri geçis
 * katmanin kendi `forward`'u ile hesaplanir, geri geçiste katmanin `backward`'u
 * cagrilir. Giris gradyani motora doner; parametre gradyanlari klasik sozlesmeyle
 * katmanin `Parameter::grad` alanlarinda birikir (optimizer bunlari okur).
 *
 * Kisit (klasik katman sozlesmesi): katman ara durumunu kendinde tutar, bu yuzden
 * ayni katman icin bir sonraki `forward`, onceki cagrinin `backward`'undan once
 * gelirse kaydedilen durum ezilir. Her sarmalama icin bir forward + bir backward.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"
#include "nexus_model/core/module.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace nexus {
namespace autograd {

namespace detail {

[[nodiscard]] inline nexus_model::Tensor to_model_tensor(const Variable& value) {
  std::vector<std::size_t> dims;
  for (const std::int64_t d : value.shape().dims) {
    dims.push_back(static_cast<std::size_t>(d));
  }
  if (dims.empty()) {
    dims.push_back(1);
  }
  nexus_model::Tensor tensor = nexus_model::Tensor::zeros(dims);
  std::copy(value.values().begin(), value.values().end(), tensor.data());
  return tensor;
}

[[nodiscard]] inline Variable from_model_tensor(const nexus_model::Tensor& tensor) {
  Shape shape;
  for (const std::size_t d : tensor.shape_vec()) {
    shape.dims.push_back(static_cast<std::int64_t>(d));
  }
  return Variable::make(std::vector<float>(tensor.data(), tensor.data() + tensor.numel()), std::move(shape), false);
}

}  // namespace detail

/// `layer.forward(x)`'i bir CustomFunction dugumu olarak teybe yazar.
inline Variable module_function(nexus_model::Module& layer, const Variable& x, std::string name = "ModuleFunction") {
  const nexus_model::Tensor output = layer.forward(detail::to_model_tensor(x));
  Variable out = detail::from_model_tensor(output);
  // Giris gradyan istemese bile parametre gradyanlari icin dugum gerekir: giris
  // ayni veriyle gradyan isteyen gecici bir yapraga cevrilir (dx atilir).
  const Variable input = x.requires_grad() ? x : Variable::make(std::vector<float>(x.values()), x.shape(), true);
  nexus_model::Module* target = &layer;
  return custom_function(std::move(name), {input}, std::vector<float>(out.values()), out.shape(),
                         [target](const Variable& grad_output) {
                           const nexus_model::Tensor dx = target->backward(detail::to_model_tensor(grad_output));
                           return std::vector<Variable>{detail::from_model_tensor(dx)};
                         });
}

}  // namespace autograd
}  // namespace nexus
