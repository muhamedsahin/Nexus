/**
 * @file test_phase8_loader.cpp
 * @brief FAZ 8.F: asenkron GPU token yukleyici — sira/yeniden kullanim dogrulugu, uretici hatasi,
 *        GPT egitiminde veri bekleme olcumu (NXTK manifest'inden).
 *
 *   test_phase8_loader order              : yavas tuketicide bile her yigin dogru ve sirali
 *   test_phase8_loader error              : uretici istisnasi next()'te yeniden firlatilir
 *   test_phase8_loader gpt MANIFEST ADIM  : GPT-2 124M BF16, senkron besleme vs TokenLoader
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/cuda/gpt.hpp"
#include "nexus/cuda/token_loader.hpp"
#include "nexus_test.hpp"
#include "nexusdata/text/token_shard.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nc = nexus::cuda;
namespace nt = nexusdata::text;

namespace {

constexpr int kSkip = 77;

void CUDART_CB slow_host(void*) { std::this_thread::sleep_for(std::chrono::milliseconds(3)); }

int test_order() {
  nexus_test::section("test_loader_order: 2/4 yuva, yavas tuketici, 200 yigin");
  const std::size_t count = 1 << 16;
  for (const int slots : {2, 4}) {
    std::int64_t produced = 0;
    nc::TokenLoader loader(
        [&](std::int32_t* dst, std::size_t n) {
          for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<std::int32_t>(produced * 7919 + static_cast<std::int64_t>(i));
          ++produced;
        },
        count, slots);
    nc::Stream compute = nc::Stream::create(0);
    auto* check = static_cast<std::int32_t*>(nc::pinned_allocator().allocate(count * 4));
    bool ok = true;
    for (int b = 0; b < 200 && ok; ++b) {
      const std::int32_t* d = loader.next(compute);
      // Hesap akisini yapay olarak yavaslat: is parcacigi yuvayi erken yeniden yazarsa veri bozulur.
      NEXUS_CUDA_CHECK(cudaLaunchHostFunc(compute.handle(), slow_host, nullptr));
      NEXUS_CUDA_CHECK(cudaMemcpyAsync(check, d, count * 4, cudaMemcpyDeviceToHost, compute.handle()));
      compute.synchronize();
      for (std::size_t i = 0; i < count && ok; i += 97) {
        ok = check[i] == static_cast<std::int32_t>(static_cast<std::int64_t>(b) * 7919 + static_cast<std::int64_t>(i));
      }
    }
    std::printf("  info  %d yuva: %lld yigin, bekleme %.1f ms\n", slots, static_cast<long long>(loader.batches()),
                loader.wait_ms());
    nexus_test::expect(ok, (std::to_string(slots) + " yuva: her yigin dogru ve sirali").c_str());
    nc::pinned_allocator().deallocate(check);
    compute.destroy();
  }
  return 0;
}

int test_error() {
  nexus_test::section("test_loader_error: uretici istisnasi tuketiciye ulasir");
  nc::Stream compute = nc::Stream::create(0);
  int calls = 0;
  bool thrown = false;
  int got = 0;
  {
    nc::TokenLoader loader(
        [&](std::int32_t* dst, std::size_t n) {
          if (++calls > 5) throw std::runtime_error("parca okunamadi");
          std::memset(dst, 0, n * 4);
        },
        1024, 3);
    try {
      for (int i = 0; i < 10; ++i) {
        (void)loader.next(compute);
        ++got;
      }
    } catch (const std::runtime_error& e) {
      thrown = std::string(e.what()) == "parca okunamadi";
    }
  }
  compute.destroy();
  nexus_test::expect(thrown && got == 5, "5 gecerli yigin, sonra uretici hatasi yeniden firlatilir");
  return 0;
}

int test_gpt(const char* manifest_path, int steps) {
  nexus_test::section("loader_wait_gpt: GPT-2 124M BF16 flash, B=4 T=1024");
  const nt::Manifest m = nt::Manifest::load(manifest_path);
  const int B = 4;
  const int T = 1024;
  nc::GptConfig config;
  config.compute_dtype = nexus::DType::BF16;
  nc::Stream stream = nc::Stream::create(0);
  nc::StreamGuard guard(stream);
  nc::GptModel model(config, B, T, 20261025, stream);
  nc::DeviceAdamWConfig adam;
  adam.lr = 6e-4;
  adam.weight_decay = 0.1;
  adam.max_grad_norm = 1.0;
  nc::GptTrainer trainer(model, adam, 10);
  const std::size_t count = static_cast<std::size_t>(B) * T;
  // Pencere T - 1: ornek T token (girdi + kaydirilmis hedef GptModel icinde).
  nt::TokenStream sync_stream(m, T - 1, 1);
  nt::TokenStream async_stream(m, T - 1, 1);
  double produce_ms = 0.0;
  auto produce = [&](nt::TokenStream& st, std::int32_t* dst) {
    const auto t0 = std::chrono::steady_clock::now();
    nt::PackedSample s;
    for (int b = 0; b < B; ++b) {
      st.next(s);
      std::memcpy(dst + static_cast<std::size_t>(b) * T, s.tokens.data(), static_cast<std::size_t>(T) * 4);
    }
    produce_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  // Isinma
  std::vector<std::int32_t> host(count);
  for (int i = 0; i < 5; ++i) {
    produce(sync_stream, host.data());
    (void)trainer.step(host.data());
  }
  trainer.synchronize();
  produce_ms = 0.0;
  // (1) Senkron: ana is parcacigi yigini uretir, sonra adim.
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < steps; ++i) {
    produce(sync_stream, host.data());
    (void)trainer.step(host.data());
  }
  const float loss_sync = trainer.synchronize();
  const double sync_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / steps;
  const double sync_produce = produce_ms / steps;
  produce_ms = 0.0;
  // (2) TokenLoader: uretim arka planda, H2D kopya akisinda.
  double loader_wait = 0.0;
  double loader_ms = 0.0;
  float loss_async = 0.0f;
  {
    nc::TokenLoader loader([&](std::int32_t* dst, std::size_t) { produce(async_stream, dst); }, count, 4);
    for (int i = 0; i < 3; ++i) (void)trainer.step_device(loader.next(stream));
    trainer.synchronize();
    const double w0 = loader.wait_ms();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i) (void)trainer.step_device(loader.next(stream));
    loss_async = trainer.synchronize();
    loader_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / steps;
    loader_wait = (loader.wait_ms() - w0) / steps;
  }
  std::printf("  info  veri: %zu parca, %.1f M token\n", m.shards.size(), m.total_tokens() / 1e6);
  std::printf("  info  senkron:     %.2f ms/adim (uretim %.3f ms/adim ana is parcaciginda), kayip %.4f\n", sync_ms,
              sync_produce, loss_sync);
  std::printf("  info  TokenLoader: %.2f ms/adim, veri bekleme %.4f ms/adim (%.3f%%), kayip %.4f\n", loader_ms,
              loader_wait, 100.0 * loader_wait / loader_ms, loss_async);
  std::printf("  info  token/s: senkron %.0f, loader %.0f\n", count / (sync_ms / 1e3), count / (loader_ms / 1e3));
  nexus_test::expect(loader_wait < 0.01 * loader_ms, "veri bekleme adim suresinin %1'inden az");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return kSkip;
  }
  const std::string mode = argc > 1 ? argv[1] : "order";
  try {
    if (mode == "order") test_order();
    else if (mode == "error") test_error();
    else if (mode == "gpt" && argc > 3) test_gpt(argv[2], std::stoi(argv[3]));
    else {
      std::printf("kullanim: test_phase8_loader <order|error|gpt MANIFEST ADIM>\n");
      return 2;
    }
  } catch (const std::exception& e) {
    std::printf("  FAIL  istisna: %s\n", e.what());
    return 1;
  }
  return nexus_test::finish("faz8-loader");
}
