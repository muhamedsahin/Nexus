/**
 * @file test_cuda_allocator.cpp
 * @brief FAZ 1.3/1.4 GPU: akis/olay semantigi, onbellekli aygit ayiricisi, pinned havuz.
 *
 * promt.txt FAZ 1 TESTLER:
 *  - test_caching_allocator: 10^6 rastgele boyutlu ayirma/serbest; sizinti yok;
 *    akislar arasi record_stream senaryosu (compute-sanitizer racecheck ile ayrica kosulur).
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/core/memory.hpp"
#include "nexus/cuda/caching_allocator.hpp"
#include "nexus/cuda/kernels.hpp"
#include "nexus/cuda/runtime.hpp"
#include "nexus_test.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace nc = nexus::cuda;

namespace {

int run() {
  if (nc::device_count() == 0) {
    std::printf("  SKIP  CUDA aygiti yok\n");
    return 77;
  }
  auto& pool = nc::caching_device_allocator();
  pool.empty_cache();

  nexus_test::section("akis ve olay semantigi");
  {
    nc::Stream s = nc::Stream::create(0);
    nexus_test::expect(!s.is_default() && s.device() == 0, "yeni akis varsayilan degil");
    {
      nc::StreamGuard guard(s);
      nexus_test::expect(nc::current_stream(0) == s, "StreamGuard gecerli akisi degistirir");
    }
    nexus_test::expect(nc::current_stream(0).is_default(), "StreamGuard cikista eski akisi geri koyar");
    nc::Event start;
    nc::Event stop;
    nc::DeviceBuffer buffer(64 << 20, s);
    start.record(s);
    nc::fill(buffer.as<float>(), 1.0f, (64u << 20) / 4u, s.handle());
    stop.record(s);
    stop.synchronize();
    nexus_test::expect(stop.query() && s.query(), "olay ve akis tamamlandi");
    nexus_test::expect(start.elapsed_ms(stop) > 0.0f, "olaylar arasi sure olculur");
    buffer.reset();
    s.destroy();
  }

  nexus_test::section("blok bolme, birlestirme ve yeniden kullanim");
  {
    const nc::Stream s;  // varsayilan akis
    const auto before = pool.stats(0);
    void* a = pool.allocate(1000, s);
    void* b = pool.allocate(1000, s);
    const auto mid = pool.stats(0);
    nexus_test::expect(mid.num_cuda_mallocs == before.num_cuda_mallocs + 1, "iki kucuk istek tek 2 MiB segmentten");
    nexus_test::expect(static_cast<char*>(b) - static_cast<char*>(a) == 1024, "512'ye yuvarlama + bolme: bitisik bloklar");
    nexus_test::expect(reinterpret_cast<std::uintptr_t>(a) % 512 == 0, "512 bayt hizali");
    pool.deallocate(a);
    pool.deallocate(b);
    const auto merged = pool.stats(0);
    nexus_test::expect(merged.inactive_split_bytes == 0, "serbest komsular birlesti (bolunmus serbest bayt 0)");
    void* c = pool.allocate(4000, s);
    nexus_test::expect(c == a, "birlesik blok ayni adresten yeniden verildi");
    nexus_test::expect(pool.stats(0).cache_hits > merged.cache_hits, "onbellek isabeti sayildi");
    pool.deallocate(c);
    void* big = pool.allocate(64u << 20, s);
    nexus_test::expect(pool.allocation_size(big) == (64u << 20), "buyuk havuz: istenen boyut kayitli");
    pool.deallocate(big);
    bool threw = false;
    try {
      pool.deallocate(big);
    } catch (const std::exception&) {
      threw = true;
    }
    nexus_test::expect(threw, "cift serbest birakma acik hata");
  }

  nexus_test::section("test_caching_allocator: 10^6 rastgele ayirma/serbest");
  {
    pool.empty_cache();
    const nc::Stream s;
    std::mt19937_64 rng(20261024);
    std::uniform_int_distribution<int> small(1, 1 << 20);
    std::uniform_int_distribution<int> large(1 << 20, 24 << 20);
    std::vector<void*> live;
    live.reserve(2048);
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kOps = 1000000;
    for (int op = 0; op < kOps; ++op) {
      if (!live.empty() && (live.size() >= 2048 || (rng() & 1u) == 0u)) {
        const std::size_t pick = static_cast<std::size_t>(rng() % live.size());
        pool.deallocate(live[pick]);
        live[pick] = live.back();
        live.pop_back();
      } else {
        const bool is_large = rng() % 64 == 0;
        live.push_back(pool.allocate(static_cast<std::size_t>(is_large ? large(rng) : small(rng)), s));
      }
    }
    for (void* p : live) {
      pool.deallocate(p);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto st = pool.stats(0);
    nexus_test::info("allocator_ns_per_op", seconds * 1.0e9 / kOps);
    nexus_test::info("allocator_cuda_mallocs", static_cast<double>(st.num_cuda_mallocs));
    nexus_test::info("allocator_peak_reserved_mib", static_cast<double>(st.peak_reserved_bytes) / 1048576.0);
    nexus_test::expect(st.allocated_bytes == 0 && st.requested_bytes == 0, "sizinti yok: ayrilmis bayt 0");
    nexus_test::expect(st.inactive_split_bytes == 0, "tum bolmeler geri birlesti");
    nexus_test::expect(st.cache_hits > st.num_cuda_mallocs * 100, "cudaMalloc cagrisi isabetlerin yaninda nadir");
    pool.empty_cache();
    nexus_test::expect(pool.stats(0).reserved_bytes == 0, "empty_cache tum segmentleri birakir");
  }

  nexus_test::section("record_stream: baska akista kullanim erken yeniden kullanimi engeller");
  {
    pool.empty_cache();
    nc::Stream producer = nc::Stream::create(0);
    nc::Stream consumer = nc::Stream::create(0);
    constexpr std::size_t n = 16u << 20;  // 64 MiB float
    float* data = static_cast<float*>(pool.allocate(n * 4, producer));
    nc::fill(data, 3.0f, n, producer.handle());
    nc::Event ready;
    ready.record(producer);
    ready.wait(consumer);
    nc::DeviceBuffer out(sizeof(float), consumer);
    nc::DeviceBuffer scratch(nc::sum_scratch_bytes(n), consumer);
    // Tuketici akista uzun bir okuma; tampon bu sirada uretici tarafinda serbest birakilir.
    for (int i = 0; i < 20; ++i) {
      nc::sum(out.as<float>(), data, n, scratch.data(), consumer.handle());
    }
    pool.record_stream(data, consumer);
    pool.deallocate(data);
    const auto deferred = pool.stats(0);
    nexus_test::expect(deferred.active_bytes >= n * 4, "serbest birakma tuketici bitene kadar ertelendi (aktif)");
    void* reuse = pool.allocate(n * 4, producer);
    nexus_test::expect(reuse != data || consumer.query(), "ertelenmis blok tuketici bitmeden yeniden verilmedi");
    consumer.synchronize();
    float host = 0.0f;
    NEXUS_CUDA_CHECK(cudaMemcpy(&host, out.data(), sizeof(float), cudaMemcpyDeviceToHost));
    nexus_test::expect(host == 3.0f * static_cast<float>(n), "tuketici dogru veriyi okudu (yaris yok)");
    pool.deallocate(reuse);
    out.reset();
    scratch.reset();
    pool.empty_cache();
    nexus_test::expect(pool.stats(0).reserved_bytes == 0, "olaylar islendi, segmentler birakildi");
    producer.destroy();
    consumer.destroy();
  }

  nexus_test::section("OOM: onbellek bosalt -> tekrar dene -> ayrintili hata");
  {
    pool.empty_cache();
    const nc::Stream s;
    std::size_t free_bytes = 0;
    std::size_t total = 0;
    NEXUS_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total));
    // Onbellege buyuk bir blok birak, sonra karttan fazlasini iste.
    void* cached = pool.allocate(free_bytes / 4, s);
    pool.deallocate(cached);
    const auto before = pool.stats(0);
    std::string message;
    try {
      // Windows WDDM surucusu "sysmem fallback" ile VRAM'i asan istekleri sistem RAM'ine
      // tasiyabilir; bu yuzden ne VRAM'in ne RAM'in karsilayabilecegi bir boyut istenir.
      void* impossible = pool.allocate(total * 512, s);
      pool.deallocate(impossible);
    } catch (const std::exception& error) {
      message = error.what();
    }
    const auto after = pool.stats(0);
    nexus_test::expect(after.num_alloc_retries == before.num_alloc_retries + 1, "OOM'da bir kez onbellek bosaltilip denendi");
    nexus_test::expect(after.num_ooms == before.num_ooms + 1, "OOM sayildi");
    nexus_test::expect(message.find("OOM") != std::string::npos && message.find("Etiket") != std::string::npos,
                       "OOM mesaji bayt ve etiket kirilimi icerir");
    nexus_test::expect(after.reserved_bytes == 0, "deneme oncesi onbellek bosaltildi");
  }

  nexus_test::section("nexus::allocate_device onbellekli ayiricidan gecer");
  {
    const auto before = pool.stats(0);
    void* p = nexus::allocate_device(nexus::DeviceRef::cuda(0), 1 << 20);
    nexus_test::expect(pool.stats(0).num_allocs == before.num_allocs + 1, "kapi ayirmasi havuzdan");
    nexus::release_owned(p);
    nexus_test::expect(pool.stats(0).allocated_bytes == before.allocated_bytes, "release_owned havuza dondu");
  }

  nexus_test::section("pinned host havuzu");
  {
    auto& host = nc::pinned_allocator();
    nc::Stream s = nc::Stream::create(0);
    void* a = host.allocate(10000);
    nc::DeviceBuffer dev(10000, s);
    NEXUS_CUDA_CHECK(cudaMemcpyAsync(dev.data(), a, 10000, cudaMemcpyHostToDevice, s.handle()));
    host.record_stream(a, s);
    host.deallocate(a);
    nexus_test::expect(host.stats().deferred_frees >= 1, "asenkron kopya surerken serbest ertelendi");
    s.synchronize();
    void* b = host.allocate(9000);
    nexus_test::expect(b == a && host.stats().cache_hits >= 1, "kopya bitince ayni pinned blok yeniden verildi");
    host.deallocate(b);
    dev.reset();
    host.empty_cache();
    nexus_test::expect(host.stats().reserved_bytes == 0, "pinned empty_cache");
    s.destroy();
  }

  return nexus_test::finish("faz1-cuda-ayirici");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    return run();
  } catch (const std::exception& error) {
    std::printf("  FAIL  beklenmeyen istisna: %s\n", error.what());
    return 1;
  }
}
