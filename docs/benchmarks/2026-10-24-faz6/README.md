# FAZ 6 — Modern LLM Bloğu ve Yapılandırma Güdümlü Kurucu (2026-10-24)

Kapsam: `promt.txt` FAZ 6. Ön-normalizasyonlu `DecoderBlock`, RoPE ailesi, SwiGLU/GeGLU, GQA,
bağlı ağırlık, sözlük dolgusu, µP, `nexus::lm::build(config)` + ön tanımlılar, NXS. Mevcut
`CausalTransformerBlock` (tür 33) değişmedi. Makine: RTX 3070 Laptop (sm_86), 16 mantıksal
çekirdek, MSVC 19.44, CUDA 13.4.

## Uygulanan

| Spec | Ne | Dosya |
|---|---|---|
| 6.1 / 6.C | `DecoderBlock` (NXS tür 34): ön-norm (varsayılan), paralel blok (GPT-J/PaLM), son-norm (tür 33 tarifi, karşılaştırma için); layernorm/rmsnorm, bias'lı/bias'sız; gelu/swiglu/geglu; GQA/MQA, head_dim, qkv_bias, qk_norm, pencere, softcap, ALiBi, artık ve çekirdek içi dikkat dropout'u | `include/nexus/lm/model.hpp`, `src/lm/model.cpp` |
| 6.A | SwiGLU/GeGLU: W1 ve W3 tek birleşik GEMM (`mlp.up` [d, 2F]) + tek düğümde ileri/geri. GPU: `gated_forward/backward` çekirdekleri | `src/lm/ops.cpp`, `src/cuda/kernels_lm.cu` |
| 6.2 / 6.B | RoPE (HF "rotate_half" düzeni), θ, kısmi döndürme (rotary_dim < head_dim), ölçekleme: linear, NTK, dinamik NTK, YaRN, LLaMA-3.1. cos/sin önbelleği tembel büyür (CPU ve aygıt). Geri geçiş = transpoz (ters açı). GPU: qkv tamponunda yerinde, Q ve K tek çağrıda | `include/nexus/lm/rope.hpp`, `src/lm/rope.cpp`, `src/cuda/kernels_lm.cu` |
| 6.3 | Konum: none, sinusoidal (tembel tablo, max_len sınırı yok), learned (wpe), rope, alibi | `src/lm/model.cpp` |
| 6.4 / 6.D | Ağırlık bağlama (tek `wte`, gradyan iki yoldan otomatik toplanır, NXS'de tek kopya); sözlük dolgusu (`vocab_pad_multiple`, dolgu logitleri −inf); final norm | aynı |
| 6.5 / 6.E | Başlatma gpt2 / llama / µP, kesik normal seçeneği; tohumdan, platformdan bağımsız RNG (splitmix64 + Box-Muller, `std::normal_distribution` değil) | aynı |
| 6.6 / 6.F | `LmConfig` + JSON (gidiş-dönüş, overrides), `count_parameters` (ayırmadan), 11 ön tanımlı: gpt2-{124m,350m,774m,1.5b}, llama2-7b, llama3-8b, mistral-7b, qwen2-{0.5b,1.5b,7b}, tinyllama-1.1b | `include/nexus/lm/config.hpp`, `src/lm/config.cpp` |
| 6.7 | NXS: kök tür 35 (`LanguageModel`, config JSON gömülü) + tür 34 bloklar + adlı ağırlıklar; `LanguageModel::load_nxs` aynı modeli kurar. `module_from_spec` bu türler için açık hata verir (autograd tabanlı, `nn::Module` değil) | `src/model/nxs.cpp` |
| — | `lm::linear` / `linear_t`: NexusModel AVX2 GEMM'li autograd op (otograd `matmul` naif üçlü döngü); CPU flash için kalıcı iş parçacığı havuzu | `src/lm/ops.cpp`, `src/attention/cpu.cpp` |

Yol: CPU tarafı FAZ 2 autograd motoru üstünde, FP32. GPU tarafında yalnız çekirdekler var
(aşağıda "Açık kalanlar").

## Testler

| Test | Sonuç |
|---|---|
| `test_rope_vs_reference` | 10 frekans vakası (none ×2, linear, ntk, dynamic ×2, yarn ×2, llama3 ×2) HF formüllerine **≤ 5.3e-8**; `apply_rotary_pos_emb` (YaRN, tam ve kısmi döndürme) **≤ 2.4e-8**; transpoz∘ileri = a²·I. Bilinen değerler: llama3 inv_freq[0] değişmez, inv_freq[63] = taban/8; YaRN s=4 → 1.138629 |
| `test_swiglu_gradcheck` | SwiGLU, GeGLU, RoPE gradcheck geçti; GQA flash dikkat, skoru açıkça yazan autograd dikkatine karşı O **7e-8**, dQ/dK/dV **2.4e-7** |
| `test_decoder_block_parity` | 5 yapılandırma, PyTorch FP32'ye karşı **kayıp ve tüm parametre gradyanları**: LLaMA (GQA 4/2, SwiGLU, RMSNorm) 9.2e-7 · Qwen2 (MQA, qkv_bias, qk_norm, kısmi rot, YaRN, bağlı) 1.3e-6 · GPT-2 (LN+bias, wpe, GELU, bağlı, dolgu 97→128) 7.1e-7 · paralel + GeGLU + llama3 ölçekleme 8.7e-7 · son-norm 5.6e-7 (eşik 1e-4) |
| `test_tied_embeddings_single_copy` | bağlı kayıp = bağsız (lm_head = wte), bağlı dwte = dwte(gömme) + dW(lm_head) (hata 0), NXS'de tek kopya; dolgulu kayıp = dolgusuz |
| `test_build_param_counts` | 11 ön tanımlının hepsi resmi sayıyla **birebir** (örn. llama3-8b 8 030 261 248, qwen2-0.5b 494 032 768); analitik = kurulan 5 yapılandırmada; gpt2-124m gerçekten kurulur |
| `test_lm_nxs_roundtrip` | yüklenen model aynı logitleri **bit-bit** verir |
| `test_lm_mup_multipliers` | gizli matris std m=4'te 0.5×; Adam LR çarpanı gizli 1/m, gömme/norm 1 |
| `test_deep_stack_stability` | aşağıda |
| `test_rope_swiglu_cuda` | GPU RoPE (llama3 ölçekli, offset, kısmi) F32 3e-8, BF16 1.1e-3; V sütunları dokunulmaz; SwiGLU/GeGLU ileri/geri F32 5e-8, BF16 1.7e-3. compute-sanitizer 4 araç temiz (`sanitizer/summary.txt`) |

**Parity referansı hakkında:** `transformers` paketi bu makinede kurulu değil (kurmak indirme
izni ister). `tests/parity/reference_lm_torch.py` HF kaynak kodunun ilgili fonksiyonlarının
(`_compute_*_parameters`, `rotate_half`, `apply_rotary_pos_emb`, `repeat_kv`, `LlamaRMSNorm`,
`LlamaMLP`) PyTorch'ta satır satır kopyasıdır. Yani kıyas "HF formülleri"ne karşıdır, HF
paketinin kendisine değil. Spec'in BF16 toleransı yerine FP32'de 1e-4 eşiğiyle geçti.

### Derinlik kararlılık deneyi (6.G)

d=64, H=4, d_ff=172, V=128, B=4, T=32, AdamW (β2=0.95, wd=0.1, kırpma 1.0, ısınma yok),
GPT-2 başlatması. Veri: stokastik Markov zinciri (taban **1.208** nat). Değerler son 20
adımın kayıp ortalaması. Ham çıktı: `deep_stack_stability.txt`.

| lr | L | Yerleşim | 100 | 300 | 500 | NaN | Katman grad normu max/min (ilk → son) |
|---|---:|---|---:|---:|---:|---|---|
| 1e-3 | 32 | ön-norm | 2.638 | 1.553 | 1.476 | yok | 1.22 → 5.98 |
| 1e-3 | 32 | son-norm | 2.447 | 1.445 | 1.413 | yok | 1.01 → 1.26 |
| 1e-3 | 48 | ön-norm | 2.789 | 1.580 | 1.491 | yok | 1.45 → 7.94 |
| 1e-3 | 48 | son-norm | 2.482 | 1.465 | 1.429 | yok | 1.02 → 1.49 |
| 3e-3 | 32 | ön-norm | 4.000 | 1.693 | 1.542 | yok | 1.22 → 41.3 |
| 3e-3 | 32 | son-norm | 1.664 | 1.469 | 1.507 | yok | 1.01 → 4.08 |
| 3e-3 | 48 | ön-norm | 4.602 | 4.069 | 2.800 | yok | 1.45 → 27.6 |
| 3e-3 | 48 | son-norm | 1.709 | 1.497 | 1.534 | yok | 1.02 → 3.97 |

Kabul ölçütü ("48 katmanlı ön-norm yığın kararlı eğitilir") sağlandı: 500 adımın hepsinde
NaN yok, kayıp öğreniyor. **Ama beklenen karşılaştırma tersine çıktı:** bu küçük genişlikte
son-norm hiç kararsızlaşmadı ve her ayarda daha hızlı öğrendi. lr=3e-3, L=48'de ön-norm
belirgin şekilde yavaş (500. adımda 2.80, son-norm 1.53) ve katman grad normları arasındaki
fark 27×'a açılıyor. Literatürdeki son-norm kararsızlığı (Xiong ve ark. 2020) daha geniş
modellerde ve gradyan kırpması olmadan görülür; bu deney o koşulu üretmedi. Daha geniş model,
kırpmasız ve ısınmasız ayarla tekrar edilmeli — açık iş.

## Benchmark

Ham veri: `bench_cuda_phase6.json` (10 ısınma + 10 tekrar, CUDA olayları, medyan).

| Ölçüm | Sonuç |
|---|---|
| SwiGLU MLP N=4096, d=4096, d_ff=11008 BF16, ileri | 36.6 ms, **30.3 TFLOPS** |
| aynı, geri | 71.0 ms, **31.2 TFLOPS** |
| aynı, ileri+geri | 115.7 ms, 28.7 TFLOPS (yalnız 6 GEMM: 103.1 ms, 32.3 TF) |
| SwiGLU ileri+geri çekirdekleri | 2.16 ms, 334 GB/s (toplamın **%1.9**'u) |
| RoPE BF16, 16k token, H=32, Hkv=8, D=128 (Q+K yerinde) | 0.95 ms, **356 GB/s** (kartın 448 GB/s bant genişliğinin %80'i) |
| RoPE F32, aynı | 2.22 ms, 303 GB/s |

FLOP: ileri 6·N·d·F, geri 2 × ileri. İlk koşuda (3 ısınma) ileri 116 ms ölçüldü; laptop GPU
saat yükselmesine denk geldi, 10 ısınmayla tekrarlandı.

- **PyTorch eager/compile karşılaştırması yok:** kurulu PyTorch yalnız CPU.
- **RoPE füzyonu (FlashAttention yükleme aşamasında) yapılmadı**; yalnız ayrı çekirdek ölçüldü.
- **"llama-tarzı 1B token/s" ölçülmedi:** GPU'da LLaMA modeli yok (GPU `GptModel` yalnız GPT-2
  şekli). Ayrıca 1.1B parametreyle AdamW eğitimi bu karta sığmaz: FP32 ana kopya + gradyan +
  2 moment = 16 B/parametre + BF16 gölge 2 B ≈ **19.8 GB > 8 GB**.
- CPU yolu: 48 katman (d=64, B·T=128) adım başına ~0.24 s. Kalıcı havuz ve AVX2 `lm::linear`
  öncesi ~2.4 s'ydi (otograd naif matmul + her çağrıda 16 iş parçacığı açma).

## Açık kalanlar

- GPU LLaMA modeli (`GptModel`'i RMSNorm/SwiGLU/RoPE/GQA/bağsız kafa ile genelleştirmek) ve
  token/s ölçümü; RoPE'nin flash yükleme aşamasına füzyonu ve iki yolun karşılaştırması.
- Python API (`nexus.lm.build("llama3-8b", overrides=...)`) yazılmadı; C++ `build(name, json)` var.
- `transformers` kurulup gerçek HF katmanlarıyla parity (indirme izni gerekir).
- µP için LR aktarım deneyi (küçük modelde ayarla, büyükte doğrula) yapılmadı; yalnız çarpanlar test edildi.
- Derinlik deneyinin son-norm kararsızlığını gösterecek ölçekte tekrarı.
- GELU erf sürümüdür; orijinal GPT-2 tanh yaklaşımını kullanır (fark ~1e-3).

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
