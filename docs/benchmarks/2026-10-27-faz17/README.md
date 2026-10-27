# FAZ 17 — Birlikte Çalışabilirlik: safetensors / HF İçe Aktarma, GGUF / ONNX Dışa Aktarma, DLPack, Python API (2026-10-27)

Kapsam: `promt.txt` FAZ 17 (DUVAR 15).

**İndirmeler (kullanıcı kararı):** yalnız **TinyLlama-1.1B-Chat-v1.0** indirildi (HF, safetensors BF16, 2.2 GB).

**İndirilmeyenler:**
- pip paketleri: transformers, onnx, onnxruntime, safetensors, gguf;
- llama.cpp.

**Bu yüzden doğrulama yolu:**
- Her biçim, depodaki **bağımsız Python referansıyla** sınandı: `benchmarks/runner/faz17_ref.py`, yalnız numpy + torch kullanıyor.
  - kendi safetensors ayrıştırıcısı;
  - HF tanımıyla LLaMA / Mistral / Qwen2 ileri geçişi;
  - GGUF okuyucu ve nicemleme açıcı;
  - genel protobuf çözücü ve ONNX yorumlayıcı.
- llama.cpp ve ONNX Runtime ile doğrulama **yapılamadı** (⛔, aşağıda).

Makine: Ryzen 7 5800H, 32 GB, NVMe, RTX 3070 Laptop (DLPack CUDA testi), Windows 11, Python 3.14.8, torch 2.14.1+cu132.
Ölçümler **fişte** (`Win32_Battery` 2).

## Ne yapıldı

| Alt adım | Durum | Nerede |
|---|---|---|
| 17.A safetensors | ✅ Okuma: mmap (Windows `MapViewOfFile` / POSIX `mmap`), sıfır kopya `bytes()`. Sıkı doğrulama: başlık boyu, dtype, şekil × boyut, taşma, aralıklar bitişik / boşluksuz / üst üste değil. Yazma: atomik `.part` → yeniden adlandırma, başlık 8'e hizalı | `interop::SafeTensors`, `write_safetensors` |
| 17.A ad eşleme | ✅ Tek tablo (`hf.hpp` başlığında): HF `[out, in]` ↔ Nexus `[in, out]`; q/k/v → birleşik qkv; gate/up → `mlp.up`; bağlı gömme | `interop/hf.cpp` |
| 17.B HF içe / dışa aktarma | ✅ Sürümlü şablonlar `llama/v1` (LLaMA-2/3, TinyLlama; rope_scaling none / linear / dynamic / yarn / llama3), `mistral/v1` (kayan pencere), `qwen2/v1` (qkv bias, bağlı gömme). Parçalı checkpoint (`model.safetensors.index.json`). Eksik / fazla / şekli uyuşmayan tensör ve desteklenmeyen alan açık hata. `save_hf`: Nexus → config.json + model.safetensors (F32 / F16 / BF16) | `load_hf`, `save_hf`, `hf_config` |
| 17.C GGUF | ✅ v3 dışa aktarma (llama.cpp `llama` kuralları):<br>• meta veri `general.*`, `llama.*`, `tokenizer.ggml.*`;<br>• tensör adları llama.cpp ile aynı;<br>• Q/K `convert_hf_to_gguf.permute`;<br>• F32 / F16 / Q8_0 / Q4_0, 32 hizalı.<br>Ayrıca C++ okuyucu ve `import_gguf` (gidiş-dönüş) | `export_gguf`, `read_gguf`, `import_gguf` |
| 17.D ONNX | ✅ FAZ 13 IR (`trace_lm`) → ONNX ModelProto (el yazımı protobuf):<br>• opset 17 / 18;<br>• dinamik `batch` / `seq` eksenleri;<br>• dış veri (4096 hizalı);<br>• yalnız standart işlemler (ayrıntı `onnx.hpp`'de) | `export_onnx` |
| 17.E DLPack | ✅ v0.8 ABI; `nexus::Tensor ↔ DLManagedTensor` sıfır kopya. Yaşam süresi `adopt_shared` + deleter ile. Kapsül adları `dltensor` → `used_dltensor`. float32 dışı, bitişik olmayan ya da desteklenmeyen cihaz açık hata | `interop/dlpack.*`, Python `nexus.interop` |
| 17.F Python API | ✅ `_core` Python **3.14** için yeniden derlendi (eski Anaconda 3.11 kaldırılmıştı). pybind11 2.13.6, `PYBIND11_FINDPYTHON`. Yeni alt modüller:<br>• `nexus.interop`: Tensor, `from_dlpack`, safetensors, HF, GGUF, ONNX;<br>• `nexus.lm`: build, logits, KV önbellekli generate;<br>• `nexus.config` | `python/bindings/interop.cpp` |

## Testler

| Test | Sonuç |
|---|---|
| `test_safetensors_roundtrip` | ✅ F32 / F16 (alt normal dahil) / BF16 / I64 / boş tensör / metadata. mmap ve tam kopya aynı. 8 bozuk / kötü niyetli dosya açık hatayla reddediliyor:<br>• başlık boyu; • kesik dosya; • bilinmeyen dtype; • şekil uyuşmazlığı;<br>• negatif boyut; • 2⁶⁴ taşması; • bozuk JSON; • üst üste aralık. |
| `test_hf_templates` | ✅ Rastgele küçük HF dizinleri (HF parametre adlarıyla, Python'da üretilir): Nexus vs bağımsız torch FP64:<br>• `llama/v1` 7.4e-7;<br>• `mistral/v1` (pencere 7) 7.4e-7;<br>• `qwen2/v1` (qkv bias, bağlı) 1.1e-6.<br>`save_hf → load_hf` bit bit aynı. GELU, GPT-NeoX ve şekil uyuşmazlığı açık hata. |
| `test_hf_import_logits` | ✅ **TinyLlama-1.1B** (gerçek checkpoint, 22 katman, GQA 32/4): 16 token, Nexus FP32 vs bağımsız torch FP64 göreli **1.39e-6**, argmax 16/16. |
| `test_gguf_roundtrip` (dosya düzeyi) | ✅ Nexus → GGUF → bağımsız Python okuyucu + ileri geçiş, ile Nexus'un aynı dosyayı `import_gguf` ile yüklemesi karşılaştırıldı (F32 / F16 / Q8_0 / Q4_0). Fark ≤ **1.2e-6**; bu, biçim, permütasyon ve nicemleme yerleşiminin doğruluğunu gösteriyor. TinyLlama Q4_0 dosyasında 6.1e-7. Blok hatası: Q8_0 ≤ amax/254; Q4_0 ≤ amax/8 (ggml'nin asimetrik d = max/−8 tanımı). |
| `test_onnx_roundtrip` | ✅ gpt2 tipi (öğrenilen konum, LayerNorm + bias, GELU, bağlı) ve llama tipi (RoPE, RMSNorm, SwiGLU, GQA 4/2); opset 17 / 18 × gömülü / dış veri. Bağımsız yorumlayıcı (B, T) = (1, 9) ve (2, 16) ile Nexus'a göre ≤ **1.4e-6** (dinamik eksenler). TinyLlama: 2 114 düğüm + 4.4 GB dış veri, **9.6e-7**. |
| `test_dlpack_torch` (Python, ctest'e bağlı) | ✅ 16/16:<br>• torch → nexus → torch aynı işaretçi, CPU ve **CUDA**;<br>• nexus'ta yerinde işlem (CPU döngüsü, CUDA'da `nexus::cuda::add_inplace`) torch'ta görünüyor;<br>• numpy görünümü sıfır kopya;<br>• kaynak silinse de veri geçerli (sahiplik);<br>• float64 / bitişik olmayan / tüketilmiş kapsül açık hata;<br>• Python'dan load_hf (qwen2) 1.2e-6; export_gguf → import_gguf 1.4e-7; export_onnx. |

### Kabul kriterleri

| Kriter | Sonuç |
|---|---|
| Hazır LLaMA-2 (1.3B) ağırlığı yüklenir, logits HF ile eşleşir | ✅ TinyLlama-1.1B (LLaMA-2 mimarisi): 1.39e-6, argmax 16/16. Referans `transformers` değil, HF tanımının bağımsız torch uygulaması (transformers kurulmadı). |
| GGUF çıktısı llama.cpp'de çalışır | ⛔ **Ölçülmedi** (llama.cpp indirilmedi). Dosya düzeyinde bağımsız okuyucuyla doğrulandı (yukarıda). Tokenizer meta verisi tokenizer.json'dan yazılıyor; SentencePiece skorları yok (`tokenizer.model` indirilmedi; skor = −id). |
| ONNX çıktısı ORT'de çalışır | ⛔ **Ölçülmedi** (onnxruntime yok). Bağımsız protobuf çözücü + yorumlayıcıyla 1e-6 düzeyinde eşit. |
| PyTorch ile DLPack alışverişi sıfır kopya | ✅ CPU ve CUDA'da işaretçi eşitliği, yerinde değişiklik görünür. |

## Benchmark'lar

### Ağırlık yükleme (TinyLlama-1.1B, 2.20 GB BF16 → 4.4 GB FP32 lm::LanguageModel)

| Yol | Açma + doğrulama | Toplam | Okunan hız |
|---|---|---|---|
| **mmap** | 1 ms | **3 970 ms** | 0.55 GB/s |
| Tam kopya (dosyayı belleğe okuma) | 3 309 ms | 6 317 ms | 0.35 GB/s |

- **İlk sürüm 47 s sürüyordu. İki düzeltme:**
  - Rastgele başlatma atlanıyor: `LanguageModel(config, zero_init)`; 1.1 milyar gaussian gereksizdi.
  - [out,in] → [in,out] devriği 64×64 karolu ve paralel.

  Sonuç: 47 → 10.3 → 3.97 s.
- Kalan süre BF16 → FP32 açma ve devrik kopyası. Nexus `lm` yolu FP32 olduğundan ağırlık kopyası kaçınılmaz.
  mmap'in "sıfır kopya" kısmı dosya → tampon adımında.

### Dışa aktarma (TinyLlama-1.1B)

| Biçim | Dosya | Süre | Logit sapması (FP32'ye göre) | argmax (8 token) |
|---|---|---|---|---|
| GGUF F16 | 2.20 GB | 11.0 s | — | — |
| GGUF Q8_0 | **1.17 GB** | 10.4 s | 0.046 | 8/8 |
| GGUF Q4_0 | **0.62 GB** | 6.3 s | 0.65 | 7/8 |
| ONNX FP32 (dış veri) | 174 KB + 4.40 GB | 3.8 s | 9.6e-7 | — |

- Logit sapması, bütün logitlerin en büyük mutlak farkının en büyük logite oranı (en kötü durum ölçüsü).
- Q4_0 (blok başına tek ölçek, sıfır noktası yok) bu ölçüde büyük sapma veriyor. Perplexity ölçülmedi: TinyLlama'nın SentencePiece tokenizer'ı C++'ta yok.

## Gerideyiz / yapılmadı

- **llama.cpp ile GGUF ve onnxruntime ile ONNX doğrulaması ⛔:** paketler indirilmedi. Doğrulama bağımsız ama yine bizim yazdığımız okuyucularla.
  Gerçek llama.cpp yüklemesinde tokenizer meta verisi eksik kalabilir: SentencePiece skorları ve `tokenizer.ggml.merges` yok.
- **HF parity transformers'a karşı değil:** HF tanımının bağımsız torch uygulamasına karşı. Mistral / Qwen2 şablonları gerçek checkpoint ile değil, HF adlı rastgele küçük modellerle sınandı.
- GGUF yalnız `llama` mimarisi; K-quant (Q4_K, Q6_K) yok.
- ONNX: MoE, ALiBi, kayan pencere, RoPE ölçekleme, softcap desteklenmiyor (`trace_lm` sınırları; açık hata). KV önbellekli (decode) ONNX grafiği yok.
- **Python:**
  - eski `python/tests/test_faz*.py` (pytest) bu turda 3.14 ile koşulmadı (pytest kurulu değil);
  - `pip install` tekerleği üretilmedi (scikit-build-core yapılandırması var, derleme `build-py` ile);
  - `nexus.autograd` / `nexus.data` için yeni bağlama eklenmedi (mevcut kabuk korunuyor).
- **DLPack:**
  - yalnız float32;
  - CUDA'da yalnız `add_` / `sum`;
  - `__dlpack__(stream)` akışı yok sayılıyor (Nexus işlemleri dönmeden senkronize ediyor).

## Dosyalar

- C++:
  - `include/nexus/interop/{json,safetensors,hf,gguf,onnx,dlpack}.hpp`;
  - `src/interop/*.cpp`;
  - `src/interop/transpose.hpp`;
  - `lm::LanguageModel(config, zero_init)`.
- Python:
  - `python/bindings/interop.cpp`;
  - `python/tests/test_faz17.py`;
  - `python/CMakeLists.txt` (nexus::cuda bağlantısı, 3.14).
- Referans: `benchmarks/runner/faz17_ref.py`. Komutları:
  - `logits`: HF ileri geçişi;
  - `tiny`: rastgele HF dizini;
  - `gguf_logits`: GGUF okuyucu;
  - `onnx_logits`: ONNX yorumlayıcı.
- Testler: `tests/parity/test_phase17_interop.cpp`, `test_dlpack_torch` (ctest).
- Ham çıktılar: `bench.jsonl`, `tests_cpu.log`, `tests_python.log`, `ctest_cpu.txt`, `ctest_cuda.txt`, `asan_ctest.txt`.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
