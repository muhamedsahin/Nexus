#pragma once

/**
 * @file custom.hpp
 * @brief Kullanici genisletmeleri: aktivasyon, kayip, katman, optimizator, metrik, callback.
 *
 * AVX2 serit baglantisi `activation_simd.hpp` icindedir ve ayri dahil edilir.
 */

#include "nexus/custom/activation.hpp"
#include "nexus/custom/callback.hpp"
#include "nexus/custom/loss.hpp"
#include "nexus/custom/metric.hpp"
#include "nexus/custom/module.hpp"
#include "nexus/custom/optimizer.hpp"
