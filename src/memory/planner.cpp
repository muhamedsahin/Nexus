/**
 * @file planner.cpp
 * @brief FAZ 7.4: boyuta gore acgozlu en iyi uyum ofset atamasi.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/memory/planner.hpp"

#include "nexus/core/error.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace nexus {
namespace memory {

namespace {

std::uint64_t align_up(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

}  // namespace

Plan plan_offsets(const std::vector<Lifetime>& items, std::uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw TensorError("Nexus planner: hizalama 2'nin kuvveti olmali");
  }
  for (const Lifetime& t : items) {
    if (t.end <= t.begin) {
      throw TensorError("Nexus planner: end > begin olmali");
    }
  }
  Plan plan;
  plan.offsets.assign(items.size(), 0);
  std::vector<std::size_t> order(items.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return items[a].bytes > items[b].bytes; });
  std::vector<std::size_t> placed;
  placed.reserve(items.size());
  for (const std::size_t i : order) {
    const Lifetime& t = items[i];
    const std::uint64_t size = align_up(t.bytes, alignment);
    // Zamanda cakisan yerlesmisler, ofsete gore sirali.
    std::vector<std::size_t> live;
    for (const std::size_t j : placed) {
      if (items[j].begin < t.end && t.begin < items[j].end) {
        live.push_back(j);
      }
    }
    std::sort(live.begin(), live.end(), [&](std::size_t a, std::size_t b) { return plan.offsets[a] < plan.offsets[b]; });
    std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t best_gap = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t cursor = 0;
    for (const std::size_t j : live) {
      const std::uint64_t start = plan.offsets[j];
      if (start >= cursor && start - cursor >= size && start - cursor < best_gap) {
        best = cursor;
        best_gap = start - cursor;
      }
      cursor = std::max(cursor, align_up(start + items[j].bytes, alignment));
    }
    plan.offsets[i] = best != std::numeric_limits<std::uint64_t>::max() ? best : cursor;
    plan.arena_bytes = std::max(plan.arena_bytes, plan.offsets[i] + t.bytes);
    placed.push_back(i);
  }
  // Tepe canli bayt (alt sinir): olay taramasi.
  std::vector<std::pair<int, std::int64_t>> events;
  for (const Lifetime& t : items) {
    events.emplace_back(t.begin, static_cast<std::int64_t>(t.bytes));
    events.emplace_back(t.end, -static_cast<std::int64_t>(t.bytes));
  }
  std::sort(events.begin(), events.end());  // ayni adimda once birakma (negatif once)
  std::int64_t live = 0;
  for (const auto& [time, delta] : events) {
    (void)time;
    live += delta;
    plan.peak_live_bytes = std::max(plan.peak_live_bytes, static_cast<std::uint64_t>(std::max<std::int64_t>(live, 0)));
  }
  return plan;
}

void LifetimeRecorder::on_alloc(std::uint64_t id, std::uint64_t bytes) {
  if (open_.count(id) != 0) {
    throw TensorError("Nexus planner: ayni kimlik iki kez ayrildi");
  }
  open_[id] = items_.size();
  items_.push_back({clock_++, 0, bytes});
}

void LifetimeRecorder::on_free(std::uint64_t id) {
  const auto it = open_.find(id);
  if (it == open_.end()) {
    throw TensorError("Nexus planner: bilinmeyen kimlik birakildi");
  }
  items_[it->second].end = ++clock_;
  open_.erase(it);
}

std::vector<Lifetime> LifetimeRecorder::lifetimes() const {
  std::vector<Lifetime> out = items_;
  for (const auto& [id, index] : open_) {
    (void)id;
    out[index].end = clock_ + 1;
  }
  return out;
}

}  // namespace memory
}  // namespace nexus
