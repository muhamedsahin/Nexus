/**
 * @file checkpoint.cpp
 * @brief FAZ 7.1: saved-tensor kancasi tabanli (non-reentrant) aktivasyon yeniden hesaplama.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/autograd/autograd.hpp"

#include <stdexcept>
#include <utility>

namespace nexus {
namespace autograd {

namespace {

thread_local std::size_t* g_counter = nullptr;  // etkin SavedBytesCounter (yoksa nullptr)

std::size_t bytes_of(const Variable& v) { return static_cast<std::size_t>(v.numel()) * sizeof(float); }

// Bir checkpoint bolgesinin durumu: ileri geciste kac tensör "kaydedildi" (yalniz sira),
// geri geciste yeniden hesaplanan degerler.
struct Frame {
  std::function<Variable(const std::vector<Variable>&)> fn;
  std::vector<Variable> inputs;
  std::size_t count = 0;
  bool recomputed = false;
  std::vector<Variable> values;
};

struct Handle {
  std::shared_ptr<Frame> frame;
  std::size_t index;
};

void recompute(Frame& f) {
  std::vector<Variable> stash;
  stash.reserve(f.count);
  {
    // Ic ice checkpoint kendi kancasini bunun ustune kurar; burada yalniz bu duzeyin
    // kayitlari (ileri gecisle ayni sira) toplanir.
    SavedTensorsHooksGuard guard(
        [&stash](const Variable& v) -> std::shared_ptr<void> {
          stash.push_back(v);
          return std::make_shared<std::size_t>(stash.size() - 1);
        },
        [](std::shared_ptr<void>&&) -> Variable {
          throw std::logic_error("Nexus checkpoint: yeniden hesaplama grafi geri yurutulmez");
        });
    EnableGradGuard enable;  // motor geri geciste grad'i kapatir; kayitlar ileri gecisteki gibi olmali
    (void)f.fn(f.inputs);
  }
  if (stash.size() != f.count) {
    throw std::runtime_error("Nexus checkpoint: yeniden hesaplama farkli sayida tensör kaydetti (fn belirlenimci degil)");
  }
  f.values = std::move(stash);
  f.recomputed = true;
}

}  // namespace

Variable checkpoint(const std::function<Variable(const std::vector<Variable>&)>& fn, const std::vector<Variable>& inputs,
                    const CheckpointOptions& options) {
  if (!options.preserve_rng) {
    throw std::invalid_argument("Nexus checkpoint: preserve_rng=false desteklenmez (RNG acik seed/offset ile)");
  }
  if (!grad_enabled()) {
    return fn(inputs);  // grafik yok: kaydedilecek bir sey de yok
  }
  auto frame = std::make_shared<Frame>();
  frame->fn = fn;
  frame->inputs = inputs;
  if (g_counter != nullptr) {
    for (const Variable& v : inputs) {
      *g_counter += bytes_of(v);  // bolge girisleri saklanir (tek bellek maliyeti)
    }
  }
  Variable out;
  {
    SavedTensorsHooksGuard guard(
        [frame](const Variable&) -> std::shared_ptr<void> {
          // Kayit yalniz sira numarasi + bolgeye guclu referans (unpack zamanina kadar yasasin).
          return std::make_shared<Handle>(Handle{frame, frame->count++});
        },
        [](std::shared_ptr<void>&& packed) -> Variable {
          auto* h = static_cast<Handle*>(packed.get());
          if (!h->frame->recomputed) {
            recompute(*h->frame);
          }
          return h->frame->values.at(h->index);
        });
    out = fn(inputs);
  }
  return out;
}

SavedBytesCounter::SavedBytesCounter() : bytes_(std::make_shared<std::size_t>(0)) {
  std::size_t* counter = bytes_.get();
  guard_ = std::make_unique<SavedTensorsHooksGuard>(
      [counter](const Variable& v) -> std::shared_ptr<void> {
        *counter += bytes_of(v);
        return std::make_shared<Variable>(v);
      },
      [](std::shared_ptr<void>&& packed) -> Variable { return *static_cast<Variable*>(packed.get()); });
  g_counter = counter;
}

SavedBytesCounter::~SavedBytesCounter() {
  guard_.reset();
  g_counter = nullptr;
}

}  // namespace autograd
}  // namespace nexus
