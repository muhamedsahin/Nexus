# FAZ 13 — Grafik IR, Füzyon Derleyicisi, Kod Üretimi, CUDA Graphs, Otomatik Ayar (2026-10-26)

Kapsam: `promt.txt` FAZ 13 (DUVAR 12). İki yol var:
- **CPU derleyicisi** (`nexus::compile`): `lm::LanguageModel` izlenir → IR → geçişler → kod üretimi (cl.exe JIT) → yürütücü.
- **GPU yolu** (`trainer.compile(true)`): `cuda::gpt_backend` için GEMM boyut dolgusu, CE→BF16 füzyonu, tüm eğitim
  adımı tek CUDA Graph, `max-autotune`'da cuBLASLt algoritma ölçümü.

GPU yolu IR'yi çalıştırmaz. IR'nin GPU kod üretimi (NVRTC) ayrı sınandı (aşağıda).

Makine: RTX 3070 Laptop 8 GB, Ryzen 7 5800H, Windows 11, **fişte** (`Win32_Battery` 2; pilde yapılan bir ön ölçüm atıldı).
Referans: PyTorch 2.14.1+cu132, nanoGPT (3adf61e) `torch.compile`, triton-windows 3.8.

## Kabul ölçütleri

| Ölçüt | Sonuç | Kanıt |
|---|---|---|
| gpt2s tek kartta Nexus compile token/s ≥ PyTorch `torch.compile(max-autotune)` | ✅ **24 048 vs 22 762 tok/s (+%5.6)**. Medyanlar 3 karışık sıralı turdan (her turda Nexus ve PyTorch sırayla, aynı ısıl koşul); 3 turun hepsinde Nexus önde. PyTorch'un en hızlı modu `max-autotune-no-cudagraphs` 23 041: Nexus +%4.4. Nexus `compile` (ayarsız) 23 591. | `interleaved.jsonl` |
| Soğuk derleme süresi ≤ PyTorch | ✅ İlk iki adım: Nexus ısınma 2.4 s + yakalama 1.6 s ≈ **4 s**. PyTorch, boş inductor/Triton önbelleğiyle ilk adım: **182 s** (max-autotune), 49 s (default). **Dikkat:** Nexus GPU yolu yeni çekirdek kodu üretmiyor (dolgu + graph yakalama + GEMM ölçümü). Kod üreten CPU JIT'in soğuk derlemesi 4.7 s, NVRTC 0.3 s. | `torch_cold.jsonl`, `interleaved.jsonl` |

Ayar: GPT-2 124M BF16, T=1024, B=4 × 8 biriktirme (32 768 token/adım), V=50257, selective recompute (Nexus).
nanoGPT'de SDPA flash ve fused AdamW açık. Bu ayar FAZ 9'da PyTorch'un en iyi olduğu ayardı.

## Throughput ve host ek yükü (gpt2s, B=4×8)

| Yol | tok/s (medyan, 3 tur) | Adım başına host süresi | Not |
|---|---|---|---|
| Nexus eager | 21 451 | 1 114–1 134 ms | kuyruk dolu: host GPU'yu bekliyor |
| Nexus eager + GEMM dolgusu (ayrı oturum, 2 tur) | 23 087 / 21 058 | 1 048 ms | dolgunun tek başına etkisi +%6–9 |
| **Nexus compile** | **23 591** | **4.5 ms** | dolgu + CE→BF16 + CUDA Graph (3 639 düğüm) |
| **Nexus compile max-autotune** | **24 048** | 4.3–5.8 ms | + cuBLASLt algoritma ölçümü |
| nanoGPT eager (ayrı oturum) | 19 947 / 20 044 | — | tepe 7.20 GB |
| torch.compile default (ayrı oturum) | 22 409 / 22 412 | — | |
| torch.compile max-autotune | 22 762 | — | `.grad` sabit tamponla (aşağıda) |
| torch.compile max-autotune-no-cudagraphs | 23 041 | — | |

Tepe bellek: Nexus 4.61 GB, PyTorch compile 5.54–5.97 GB.

- **Kazancın kaynağı (profil):**
  - lm_head GEMM'leri: V=50257 8'in katı değil, cuBLASLt `align1` sm75 çekirdeklerine düşüyordu (mikro adım başına 37 + 30 + 24 ms).
    Dolgu (64'ün katı) sonrası `align8` sm80 çekirdekleri: 26 + 24 + 23 ms. Dolgu sütunları CE'de yok sayılıyor, gradyanları 0;
    kayıp ve gerçek parametrelerin matematiği aynı. Bu, inductor'ın `pad_mm` karşılığı.
  - CE çekirdeği tek okumalı çevrimiçi softmax + float4: 20 → 15 ms.
  - CE dlogits'i doğrudan BF16 yazıyor: ayrı `cast` çekirdeği ve FP32 yazımı kalktı.
- **CUDA Graph:** host süresi 1.1 s → ~5 ms (≈ 0). Bu ayarda GPU zaten tam dolu olduğundan token/s'e doğrudan katkısı küçük.
  Kazanç dolgu ve füzyondan geliyor.
- **Gürültü:** Ayrı oturumlar arasında %5–7 ısıl fark görüldü (aynı yapılandırma 23.1k → 21.1k). Bu yüzden kabul karşılaştırması
  karışık sıralı yapıldı.
- **PyTorch max-autotune notu:** Biriktirmeli döngü, PyTorch'un CUDA Graph'larıyla çakışıyor
  ("accessing gradient tensor output of CUDAGraphs that has been overwritten"). Hata iletisinin önerisi uygulandı:
  `.grad` tamponları önceden sıfırla ayrılıp `zero_grad(set_to_none=False)` kullanıldı (`faz9_nanogpt.py`).

## Testler

| Test | Ne denetlenir | Sonuç |
|---|---|---|
| `test_compile_numerics` | **Tüm ön tanımlılar** küçültülerek (2 katman, d 64; norm/MLP/konum/GQA/bias/bağlama yapısı korunur): derlenmiş, yalnız-füzyon (üretilen AVX2 çekirdekleri) ve derleyici kapalı yollar, eager otograda karşı kayıp ve **tüm** gradyanlar | ✅ 10/10: kayıp farkı **0**, gradyan ≤ 5.2e-7. `mistral-7b` kayan pencere yüzünden açık hatayla atlanıyor |
| `test_fusion_kernel_count` | gpt2-124m adımı (ileri + geri). Nexus eager (otograd) çağrı sayısı ile derlenmiş IR çekirdek sayısı | 🟡 468 → **226** (**2.07×**); min-cut kopyalarıyla 252 (1.86×). **Spec hedefi 3× tutmadı.** Nexus eager zaten kaba taneli (füzyonlu LN/CE/linear, flash). Test ≥ 2× gerilemesini denetliyor |
| `test_dynamic_shape_no_recompile` | `dynamic=true`: T 16 → 32 → 24 (B 3) → 64 (B 1) | ✅ yeniden derleme 0, derleyici çağrısı 0, kayıp farkı 0. `dynamic=false` koruyucusu yeni T'yi yeniden izleme olarak sayıyor; kaynak aynı olduğu için önbellekten |
| `test_cache_hit_zero_compile` | ikinci derleme | ✅ 0 derleyici çağrısı, aynı sonuç. Soğuk 4.7 s (vcvars ortamı bir kez yakalanır: 9.1 → 4.7 s), sıcak 6–7 ms |
| `test_compile_min_cut_memory` | min-cut yeniden hesaplama + statik arena | ✅ saklanan 4.35 MB (hepsi saklansa 5.12 MB), sonuç **bit bit aynı**. 🟡 tepe arena yalnız 6461.8 → 6457.8 KB: tepe, gradyan tamponlarının olduğu bölümde. Arena toplam tampon belleğinin 0.41'i |
| `test_compile_comm_reorder` | FSDP benzeri grafik, 2 rank thread: AllGather'lar hesaptan önce başlıyor, asenkron yürütme | ✅ sonuç aynı |
| `test_compile_export_stablehlo` | StableHLO (func @main, dot_general, dinamik `?` boyut, custom_call) + IR dökümü (her geçişten sonra .txt + .dot + kaynak) + HLO metni → MatrixFlash `compile_hlo` → aynı sonuç | ✅ (StableHLO `stablehlo-opt` ile doğrulanmadı: kurulu değil) |
| `test_cuda_graph_replay_equal` | tüm adım tek graph (biriktirme 3, değişken lr): aynı çekirdekli eager ile 12 adım kayıp, norm, parametre | ✅ **bit bit aynı**. Dolgusuz eager ile kayıp farkı 4.3e-5 |
| `test_nvrtc_fusion_kernels` | IR füzyon kümeleri → CUDA kaynağı → NVRTC → cubin (disk önbelleği) → çalıştırma | ✅ CPU JIT ile en büyük göreli fark 1.2e-7; soğuk 298 ms, sıcak 9 ms (0 NVRTC çağrısı) |

Regresyon:
- CPU **115/115** (`ctest_cpu.txt`).
- GPU **38/38** (`ctest_cuda.txt`).
- ASan (FAZ 13 CPU testleri + TP/4B): **9/9** (`asan_ctest.txt`).

**Bulunan ve düzeltilen gerileme:** İlk sürümde hızlı (çevrimiçi softmax) CE çekirdeği eager yolunun da varsayılanıydı. Toplama sırası
değiştiği için küçük modelin kaotik 300–500 adımlık eğri testleri (`test_gpt_flash_vs_math`, `test_bf16_training_converges`,
`test_master_weight_modes`) %1 bandının dışına çıktı (deterministik; çekirdek testleri geçiyordu). Eager artık FAZ 3 toplama
sırasıyla aynı referans çekirdeği kullanıyor; geçmiş eğriler bit bit aynı (flash/math @300: 7.85879 / 7.86535, FAZ 12 ile aynı).
Hızlı çekirdek yalnız `compile` yolunda (`GptConfig::fast_ce`). Nexus eager satırı bu düzeltmeden sonra yeniden ölçüldü:
21 453 / 21 196 tok/s (öncekiyle aynı).

## CPU derleyicisi: ölçüm (eager ile)

| Model (4 katman, d 256, T 128, B 4) | eager | derlenmiş | çekirdek | arena / toplam |
|---|---|---|---|---|
| GPT tipi | 980 ms | 1 199 ms (**0.82×**) | 167 → 92 | 36.9 / 92.0 MB |
| LLaMA tipi | 1 336 ms | 1 390 ms (**0.96×**) | 170 → 151 | 40.0 / 141.4 MB |

**Gerideyiz:** CPU yürütücüsü eager'dan %4–18 yavaş. Aynı GEMM çekirdekleri kullanılıyor; fark norm/CE döngüleri ve kopyalardan
geliyor; profillenmedi. `max-autotune` (AVX2 ve skaler varyant ölçümü) bu modellerde etkisiz: tam desen modunda eleman eleman
küme kalmıyor.

## Tasarım özeti

- **13.1 IR:**
  - SSA, düğüm = op, değer = düğüm kimliği.
  - Çok çıktılı düğümler `Output(i)` ile adlandırılıyor.
  - Sembolik boyutlar (katsayı × sembol çarpımı: N = B·T).
  - Metin ve Graphviz dökümü, yapısal parmak izi.
- **13.2 İzleme:**
  - `trace_lm`: meta tensörlerle (veri yok) `lm::LmConfig` ileri geçişi; parametre adları `lm::LanguageModel` ile aynı.
  - Dikkat ayrıştırılmış izleniyor: SplitHeads/BMMNT/Scale/CausalMask/Softmax/BMM.
  - Python yok, grafik kırılması yok.
  - Autodiff ile birleşik ileri + geri grafik. VJP kuralı olmayan op açık hata.
  - Koruyucu: `dynamic=false` yeni (B, T) için yeniden izliyor.
- **13.3 Geçişler:**
  - Temizlik: DCE, sabit katlama, CSE (yeniden hesaplanan kopyalar ayrı anahtarlanıyor).
  - Desenler: softmax(QKᵀ)V → Attention (paketli QKV: dilimsiz okuma), silu·b → SwiGLU, matmul+bias+gelu → LinearBiasGelu,
    matmul+bias, add+norm → AddNorm, lm_head+CE → FusedLinearCE (logit tamponu yok).
  - Geri epilogları: DGELU, BGRAD, beta=1 birikim, norm geri (dx + dw + db + artık tek çekirdek).
  - Eleman eleman dikey + yatay füzyon (döngü güvenlik kuralıyla).
  - Min-cut yeniden hesaplama: Dinic; pahalı düğümler kaynağa bağlı; `Output` düğümünün girdileri ebeveyninin girdileri.
  - Statik arena (ömür aralıkları, ilk uyan).
  - İletişim öncelikli liste zamanlama.
- **13.4 Kod üretimi:**
  - CPU: AVX2 + MSVC SVML intrinsics, grafik başına tek `cl.exe` çağrısı → DLL, disk önbelleği.
  - CUDA: ızgara adımlı + float4 → NVRTC → cubin → `cuModuleLoadData`, disk önbelleği. Triton bağımlılığı yok.
- **13.5 Otomatik ayar:**
  - cuBLASLt aday algoritmaları ölçülüyor (FAZ 3 altyapısı, `max-autotune`).
  - CPU füzyon kümelerinde vektör / skaler varyant ölçümü.
  - 🟡 cuBLASLt sonuçları yalnız bellekte; diske yazılmıyor.
  - 🟡 Flash blok boyutu ayarı yok.
- **13.6 CUDA Graphs:**
  - Isınma adımı eager. Sonraki adımda tüm adım tek grafik: H2D token kopyaları, M × forward_backward, AdamW, D2H kayıp/norm.
  - Değişen değerler (lr, bias düzeltmesi, adım, mikro adım ağırlıkları) aygıttaki bir yapıdan okunuyor; her yeniden
    oynatmada pinned bellekten bir memcpy düğümü yeniliyor.
  - AdamW tanımlayıcıları kalıcı scratch'te.
  - 🟡 Uzunluk kovaları yok (`GptModel` sabit T).
  - 🟡 DDP, 8-bit AdamW, F16 kayıp ölçeği ve ana kopyasız modda graph açık hata.
- **13.7:** StableHLO MLIR metni (özel çekirdekler `stablehlo.custom_call @nexus.*`) ve HLO metni.
- **13.8 API:**
  - `nexus::compile::compile(model, {mode, dynamic, fullgraph, patterns, fusion, recompute, dump_dir, cache_dir})`.
  - `LanguageTrainerBuilder::compile(true, mode)`. Desteklemeyen arka uçlar (nn::Module CPU, paralel) açık hata veriyor.

## Gerideyiz / yapılmadı

- **GPU uçtan uca IR yürütücüsü yok.** GPU kazancı elle yazılmış `GptModel` yolunda geçişlerin karşılıkları uygulanarak alındı:
  dolgu, CE→BF16 füzyonu, graph, autotune. IR'den üretilen CUDA çekirdekleri yalnız birim testte çalıştırıldı.
- Çekirdek sayısı: 3× hedefi tutmadı (2.07×).
- CPU yürütücüsü eager'dan yavaş (0.82–0.96×).
- Min-cut tepe bellekte belirgin kazanç sağlamadı.
- LLaMA tipi GPU modeli yok: GPU ölçümü yalnız gpt2s; `llama1b` ölçülmedi.
- TF XLA kurulu değil, ölçülmedi.
- cuBLASLt autotune disk önbelleği yok; flash blok ayarı yok; uzunluk kovası yok.
- StableHLO resmi araçla doğrulanmadı.

## Dosyalar

- Kod:
  - `include/nexus/compile/{ir,trace,passes,codegen,compile}.hpp`;
  - `src/compile/{ir,trace,passes,codegen,executor,compile}.cpp`;
  - `include/nexus/cuda/nvrtc_jit.hpp`, `src/cuda/nvrtc_jit.cpp`;
  - GPU: `gpt.{hpp,cpp}` (dolgu, CE→BF16, graph parametreleri), `kernels_train.cu` (CE, AdamW aygıt hiperparametreleri),
    `lm_backend.{hpp,cpp}` (yakalama/yeniden oynatma), `lm_trainer` (`compile`).
- Testler: `tests/parity/test_phase13_compile.cpp`, `tests/cuda/test_phase13_compile_cuda.cpp`.
- Ölçüm: `benchmarks/runner/faz9_nanogpt.py` (`--compile-mode`).
- Ham çıktılar: `interleaved.jsonl`, `gpu_bench.jsonl`, `torch_bench.jsonl`, `torch_cold.jsonl`, `cpu_bench.jsonl`, `ctest_*.txt`.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
