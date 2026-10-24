# FAZ 0 — Ölçüm Altyapısı ve Tahmin Aracı (2026-10-23)

Bu klasör FAZ 0'ın makinayla okunur çıktı köküdür. Çalıştırma çıktıları buraya yazılır:

| Dosya | Üreten | Ne zaman |
|---|---|---|
| `bench_phase0.json` | `bench_phase0` (ctest) | her benchmark koşusunda |
| `bench_phase0_trace.json` | `bench_phase0` | her benchmark koşusunda (Chrome izi) |
| `lm_baseline_result.json` | `lm_baseline --json ...` | taban çizgisi koşusunda |
| `lm_baseline_config.json` | elle (girdi) | sabit |
| `estimate_gpt2_124m.json`, `estimate_llama7b.json` | elle (girdi) | sabit |

## FAZ 0 bileşenleri

1. **Bellek muhasebe defteri** — `include/nexus/prof/memory_ledger.hpp`, `src/prof/prof.cpp`.
   `nexus::allocate_aligned/pinned/device` kapısından geçen her ayırma,
   `AllocationLabelScope` ile seçilen etiketle (parameter, gradient, optimizer_state,
   activation, temporary, comm_buffer, kv_cache) deftere yazılır; serbest bırakmada
   aynı etiketten düşülür. Anlık + doruk bayt, etiket ve adres uzayı kırılımı.
2. **Sayaçlar** — `nexus/prof/counters.hpp`: FLOP/bayt/çağrı, ad bazında, kilitli.
3. **Kapsamlı zamanlayıcı** — `nexus/prof/timer.hpp`: `ScopedTimer`, iç içe aralıklar;
   profiler kapalıyken kayıt yazmaz (yalnızca iki saat okuması).
4. **Chrome izi** — `nexus/prof/trace.hpp`: `dump_chrome_trace(path)`; `chrome://tracing`
   ve Perfetto ile açılır ("X" complete olayları + "C" sayaç olayı).
5. **NVTX** — `nexus/prof/nvtx.hpp`: yalnız `NEXUS_ENABLE_NVTX` tanımlı ve NVTX başlığı
   bulunabilirken gerçek çağrı; aksi halde derleyicinin sildiği boş gövde. Bayrak
   kapalıyken uydurma işaret üretilmez.
6. **nexus_estimate** — `tools/nexus_estimate/`: kapalı biçim parametre/bellek/FLOP
   tahmini; JSON + tablo çıktısı. Python aynası: `benchmarks/runner/estimate.py`
   (birebir aynı formüller; parite `test_estimate_parity` ile doğrulanır).
7. **lm_baseline** — `apps/lm_baseline/`: küçük dil modeli taban çizgisi; gerçek adım
   süreleri, kayıp eğrisi, süreç doruk belleği, tahmini FLOP ve (yalnız aygit zirvesi
   verildiyse) MFU + roofline MFU üst sınırı.

## Formül kaynakları

- FLOP: Kaplan ve ark. arXiv:2001.08361; Hoffmann ve ark. arXiv:2203.15556 (C ≈ 6NT);
  dikkat terimi PaLM arXiv:2204.02311 Ek B (token başına 12·L·d·s).
- Aktivasyon belleği: Korthikanti ve ark. arXiv:2205.05198
  (katman başına s·b·h·(34 + 5·a·s/h) bayt, 2 bayt/eleman; tam yeniden hesaplamada 2·s·b·h).
- ZeRO bölüşümü: Rajbhandari ve ark. arXiv:1910.02054 (evre 1 optimizer, evre 2 gradyan,
  evre 3 parametre bölüşümü).

## Kalibrasyon (test_estimate_parity ile zorunlu)

- GPT-2 124M: formülden **124.439.808** parametre; nominal 124M'e fark ~%0,35 (≤ %0,5).
- LLaMA-7B: formülden **6.738.415.616**; yayımlanan ~6,74B'ye fark ~%0,02 (≤ %0,5).

## Dürüstlük sınırları

- **Çatılma/parçalanma oranı FAZ 0'da "n/a"**: k64 havuz ayırıcısı (CachingAllocator)
  FAZ 1'de gelir; oran `reserved/current` ile o zaman ölçülür. FAZ 0 defteri yalnız
  gerçek ayırma-serbest bırakma akışını sayar, oran uydurmaz.
- **MFU**: aygit zirve FLOP/s verilmeden hesaplanmaz (null). Üst sınır roofline'dan:
  `min(zirve, yoğunluk × bant) / zirve`.
- Defter yalnız nexus kapısından geçen ayrımları sayar; `nexus_model` tensorları kendi
  ayırıcısını kullanır. Birleşik tek ayırıcı FAZ 1'in işidir.

## Çalıştırma

```
cmake --build build --target nexus_estimate bench_phase0 lm_baseline
ctest --test-dir build -R "test_phase0_prof|test_estimate_parity" --output-on-failure
ctest --test-dir build -R bench_phase0 --output-on-failure
./build/apps/lm_baseline/lm_baseline --config docs/benchmarks/2026-10-23-faz0/lm_baseline_config.json ^
    --json docs/benchmarks/2026-10-23-faz0/lm_baseline_result.json ^
    --trace docs/benchmarks/2026-10-23-faz0/lm_baseline_trace.json
./build/tools/nexus_estimate/nexus_estimate --config docs/benchmarks/2026-10-23-faz0/estimate_gpt2_124m.json
python benchmarks/runner/estimate.py --config docs/benchmarks/2026-10-23-faz0/estimate_llama7b.json
```

## 2026-10-24 denetim notu

Bu fazin kodu ve testleri 24 Ekim denetiminde yeniden gozden gecirildi: derleme hatalari,
kosulmamis testler ve eksik spec maddeleri `../2026-10-24-faz0-3-denetim/README.md` icinde
listelenmistir (bulgular, eklenen testler, olcumler ve acik kalan isler).
