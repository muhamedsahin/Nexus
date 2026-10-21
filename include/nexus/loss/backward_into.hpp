#pragma once

/**
 * @file backward_into.hpp
 * @brief Kayıp turevini cagiranin tamponuna yazar.
 *
 * `backward()` sonuc vektorunu kendisi ayirir. `backward_into` ayirmaz:
 * eleman turevi dogrudan `grad_out` icine yazilir, reduction olcegi ayni
 * tamponda uygulanir. NexusLoss siniflarinin uye fonksiyonu asil islemidir;
 * bu sablonlar cagriyi `namespace nexus` altina tasir ve her aileyi ayni
 * kavramla (`InplaceBackward`) kisitlar.
 *
 * Vektor alan asiri yukleme tamponu gerekirse buyutur. `std::vector` kapasitesi
 * kuculmedigi icin sabit boyutlu bir egitim dongusunde ilk adimdan sonra
 * hicbir ayirma olmaz.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include <concepts>
#include <cstddef>
#include <span>
#include <vector>

namespace nexus {

/// `loss.backward_into(span<T>)` cagrisi gecerli olan her NexusLoss sinifi.
template <typename Loss, typename T>
concept InplaceBackward = requires(const Loss& loss, std::span<T> gradient) { loss.backward_into(gradient); };

/// Son forward'un turev uzunlugunu bildiren kayip.
template <typename Loss>
concept SizedGradient = requires(const Loss& loss) {
  { loss.gradient_size() } -> std::convertible_to<std::size_t>;
};

template <typename Loss, typename T>
  requires InplaceBackward<Loss, T>
void backward_into(const Loss& loss, std::span<T> gradient) {
  loss.backward_into(gradient);
}

/// Tamponu `gradient_size()` boyutuna getirir, turevi yazar ve span olarak dondurur.
template <typename Loss, typename T>
  requires InplaceBackward<Loss, T> && SizedGradient<Loss>
std::span<T> backward_into(const Loss& loss, std::vector<T>& buffer) {
  const std::size_t size = static_cast<std::size_t>(loss.gradient_size());
  if (buffer.size() != size) {
    buffer.resize(size);
  }
  const std::span<T> gradient(buffer.data(), size);
  loss.backward_into(gradient);
  return gradient;
}

}  // namespace nexus
