# FAZ 1 — k64 Önbellekli Ayırıcı (CachingAllocator) ve Parçalanma Metriği (2026-10-23)

Bu klasör FAZ 1'in makinayla okunur çıktı köküdür. `bench_phase1` koşusu buraya
`bench_phase1.json` yazar.

## FAZ 1 ne ekler

FAZ 0 ölçümü kurmuştu ama gerçek bir havuz ayırıcı yoktu: her `allocate_aligned`
doğrudan OS'a (`_aligned_malloc`/`std::aligned_alloc`) gidiyor, serbest bırakma da
doğrudan OS'a dönüyordu. FAZ 1 bunu tek bir ayırıcıda birleştirir:

1. **`nexus::CachingAllocator`** — `include/nexus/core/allocator.hpp`, `src/core/allocator.cpp`.
   - 64 bayt tabanlı, power-of-two boyut sınıfları (64, 128, 256, ...).
   - Serbest bırakılan bloklar yok edilmez; sınıfına göre önbelleklenir, uygun
     istekte yeniden kullanılır (`cache_hits`).
   - `empty_cache()` önbelleği OS'a geri verir; `reset()` test için sayaçları sıfırlar.
   - Mutex altında, çok iş parçacıklı güvenli.
2. **Bellek kapısına kanca** — `src/core/memory.cpp`: `allocate_aligned` artık havuzdan
   beslenir; `release_owned` (AlignedHost) bloğu OS'a değil havuza geri verir.
   Kayıt tablosu (track/untrack) ve FAZ 0 defteri (ledger) aynen korunur;
   `allocation_count()` mantıksal ayırma sayısını saymaya devam eder.
3. **Gerçek parçalanma metriği** — `nexus::memory_fragmentation_ratio()` =
   `reserved/current`. FAZ 0'daki `"n/a"` artık çalışma zamanında ölçülür.
   - `reserved`: OS'tan tutulan toplam (canlı + önbellekli).
   - `current`: kullanıcıda canlı olan toplam (yuvarlanmış).
   - Oran ≥ 1.0; 1.0 = parçalanma yok. Canlı ayırma yokken `reserved>0` ise −1.0
     (yalnızca önbellek), ikisi de 0 ise 1.0.

## Dürüstlük sınırları

- FAZ 1 yalnızca **AlignedHost** (sayfalanabilir host) yolunu havuzlar; pinned ve
  CUDA cihaz yolları mevcut doğrudan-OS davranışını korur (cihaz havuzu sonraki faz).
- Boyut sınıfları power-of-two'dur; ince taneli sınıf bölümleme sonraki optimizasyon.
  İç parçalanma `reserved/current` ile dürüstçe görünür, gizlenmez.

## Test / Benchmark

```
ctest --test-dir build -R "test_phase1_allocator" --output-on-failure
ctest --test-dir build -R "bench_phase1" --output-on-failure
```

`test_phase1_allocator` şunları doğrular: 64 bayt hizalama, önbellekten yeniden
kullanım (adres eşitliği + `cache_hits`), `reserved ≥ current`, oran hesabı,
`empty_cache`, büyük blok, çok iş parçacıklı denge ve Tensor'un havuzdan geçtiği.

## 2026-10-24 denetim notu

Bu fazin kodu ve testleri 24 Ekim denetiminde yeniden gozden gecirildi: derleme hatalari,
kosulmamis testler ve eksik spec maddeleri `../2026-10-24-faz0-3-denetim/README.md` icinde
listelenmistir (bulgular, eklenen testler, olcumler ve acik kalan isler).
