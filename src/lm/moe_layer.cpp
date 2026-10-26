/**
 * @file moe_layer.cpp
 * @brief FAZ 14.C: lm::moe_ffn — nexus::moe cekirdeginin otograd sarmalayicisi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/lm/moe_layer.hpp"

#include <memory>
#include <stdexcept>

namespace nexus {
namespace lm {

namespace ag = autograd;
using ag::Shape;
using ag::Variable;

Variable moe_ffn(const Variable& x, const Variable& router, const Variable& noise, const Variable& up, const Variable& down,
                 const moe::Spec& s, bool training, std::uint64_t seed, Variable* router_loss, MoeLayerStats* stats) {
  const Shape& xs = x.shape();
  if (xs.rank() != 2 || xs.dims[1] != s.d_model) throw std::runtime_error("Nexus lm::moe_ffn: x [N, d] bekler");
  const std::int64_t N = xs.dims[0];
  const int d = s.d_model, E = s.experts, F = s.d_ff;
  if (up.numel() != static_cast<std::int64_t>(E) * d * 2 * F || down.numel() != static_cast<std::int64_t>(E) * F * d ||
      router.numel() != static_cast<std::int64_t>(d) * E) {
    throw std::runtime_error("Nexus lm::moe_ffn: agirlik sekilleri (router [d, E], up [E, d, 2F], down [E, F, d])");
  }
  const bool has_noise = noise.has_data();
  auto cache = std::make_shared<moe::LayerCache>();
  std::vector<float> y(static_cast<std::size_t>(N) * d);
  moe::layer_forward(x.values().data(), N, router.values().data(), has_noise ? noise.values().data() : nullptr,
                     up.values().data(), down.values().data(), s, training, seed, y.data(), cache.get());
  const moe::Routing& r = cache->routing;
  if (stats) {
    stats->routed = r.routed;
    stats->kept.resize(static_cast<std::size_t>(E));
    for (int e = 0; e < E; ++e) stats->kept[static_cast<std::size_t>(e)] = r.offsets[static_cast<std::size_t>(e) + 1] - r.offsets[static_cast<std::size_t>(e)];
    stats->capacity = r.capacity;
    stats->dropped = r.dropped;
    stats->rerouted = r.rerouted;
    stats->idle = r.idle_experts();
    stats->aux = r.aux;
    stats->z = r.z;
  }
  std::vector<Variable> inputs{x, router, up, down};
  if (has_noise) inputs.push_back(noise);
  // Agirliklar geri gecis aninda okunur (otograd degerleri ileri ve geri arasinda degismez).
  const Variable rv = router, uv = up, dv = down, nv = noise;
  Variable out = ag::custom_function(
      "lm_moe_ffn", inputs, std::move(y), xs, [cache, s, rv, uv, dv, nv, has_noise, N, d, E, F](const Variable& g) {
        std::vector<float> dx(static_cast<std::size_t>(N) * d), dr(static_cast<std::size_t>(d) * E, 0.0f),
            du(static_cast<std::size_t>(E) * d * 2 * F, 0.0f), dd(static_cast<std::size_t>(E) * F * d, 0.0f),
            dn(has_noise ? static_cast<std::size_t>(d) * E : 0, 0.0f);
        moe::layer_backward(g.values().data(), *cache, rv.values().data(), has_noise ? nv.values().data() : nullptr,
                            uv.values().data(), dv.values().data(), s, 0.0f, dx.data(), dr.data(), has_noise ? dn.data() : nullptr,
                            du.data(), dd.data());
        std::vector<Variable> grads{Variable::make(std::move(dx), Shape{{N, d}}, false),
                                    Variable::make(std::move(dr), rv.shape(), false),
                                    Variable::make(std::move(du), uv.shape(), false),
                                    Variable::make(std::move(dd), dv.shape(), false)};
        if (has_noise) grads.push_back(Variable::make(std::move(dn), nv.shape(), false));
        return grads;
      });
  if (router_loss) {
    *router_loss = ag::custom_function(
        "lm_moe_router_loss", {x, router}, {r.aux + r.z}, Shape{{}}, [cache, s, rv, N, d, E](const Variable& g) {
          const float gl = g.values()[0];
          std::vector<float> dx(static_cast<std::size_t>(N) * d, 0.0f), dr(static_cast<std::size_t>(d) * E, 0.0f);
          const std::vector<float> zero_gate(static_cast<std::size_t>(N) * s.top_k, 0.0f);
          moe::router_backward(cache->routing, cache->x.data(), rv.values().data(), nullptr, zero_gate.data(), s, gl,
                               dx.data(), dr.data(), nullptr);
          return std::vector<Variable>{Variable::make(std::move(dx), Shape{{N, d}}, false),
                                       Variable::make(std::move(dr), rv.shape(), false)};
        });
  }
  return out;
}

}  // namespace lm
}  // namespace nexus
