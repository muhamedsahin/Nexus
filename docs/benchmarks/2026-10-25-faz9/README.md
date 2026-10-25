# FAZ 9 — Dil Modeli Eğitmeni v2 (2026-10-25)

Kapsam: `promt.txt` FAZ 9 (DUVAR 11). Builder ile kurulan tek süreç ön eğitim döngüsü:
gradyan biriktirme (token ağırlıklı), LR zamanlayıcıları, global norm kırpma, parametre grupları,
sabit kümede değerlendirme, ölçüm (token/s, MFU, tepe bellek, veri süresi, atlanan adım),
günlükçüler (konsol, CSV, JSONL, TensorBoard, W&B JSONL), NaN/sıçrama teşhisi, eski API yönlendirmesi
ve sınıflandırma `Trainer`'ının ortak altyapıya taşınması.
Makine: RTX 3070 Laptop (8 GB, WDDM, 40 SM, 115 W sınırı), Ryzen 7 5800H, 32 GB RAM, CUDA 13.4,
MSVC 19.44. Referans: karpathy/nanoGPT `3adf61e` (`model.py`), PyTorch 2.14.1+cu132,
triton-windows 3.8 (`.bench-venv-cuda`). Veri: TinyStoriesV2-GPT4 (HF `roneneldan/TinyStories`),
`nexus_tokenize` ile GPT-2 BPE: eğitim 541.5 M token (2 717 495 hikâye, 6 parça), doğrulama 5.47 M.

**Ölçüm koşulu.** Throughput tablosu fişteyken alındı (BatteryStatus 2). Kayıp eğrisi koşularının
bir kısmında laptop yeniden pile geçti (16:50'de %24, deşarj; GPU bellek saati 810 MHz). Bu yalnız
süreleri etkiler, kayıp değerlerini değil; eğri koşularının süreleri hız kanıtı olarak kullanılmadı.
MFU tepesi: BF16 dense Tensor Core, FP32 biriktirme, 40 SM × 512 FLOP/saat × 2100 MHz =
**43.0 TFLOPS** (NVML azami SM saati = `nvidia-smi clocks.max.sm`; FAZ 7 ile aynı tanım).
Ölçüm sırasında kart güç sınırında 1620–1680 MHz'de çalışıyor; o saatle tepe ~34 TFLOPS olurdu.
MFU = (6·N + 12·L·H·d·T) · token/s / tepe, N = wpe hariç parametre (nanoGPT `get_num_params`).

## Kabul ölçütleri

| Ölçüt | Sonuç | Kanıt |
|---|---|---|
| gpt2s tek kartta token/s ≥ nanoGPT eager | ✅ Aynı ayarda (B=4×8 ve B=2×16, T=1024, 32k token/adım) Nexus **%3.8 ve %3.3 hızlı**, tepe bellek **4.61 / 3.46 GB** (nanoGPT 7.20 / 4.83). B=8×4'te nanoGPT 8 GB'a sığmıyor (11.9 GB, WDDM taşması, 2.1k tok/s); Nexus 23.3k tok/s, 6.90 GB. En iyi-en iyi: Nexus 23.3k, eager 21.3k (**+%9**). | `throughput_raw.txt` |
| torch.compile ile fark raporlanır | 🟡 Geride. B=4×8: compile **24.3k** tok/s, Nexus 22.2k (**−%8.7**). B=2×16: 21.2k vs 20.6k (−%2.5). En iyi-en iyi: compile 24.3k (B=4) vs Nexus 23.3k (B=8): **−%3.9**. Bellek: Nexus her ayarda daha az (4.61 vs 5.52 GB, 3.46 vs 4.02 GB). Fark FAZ 13'te (füzyon) kapanacak. | `throughput_raw.txt` |
| MFU raporlanır | ✅ Eğitmen her adımda yazıyor. gpt2s BF16 B=8×4: **%46.4** (43 TFLOPS'a göre). | `throughput_raw.txt`, `curve_nexus_gpt2s.txt` |
| Kayıp eğrisi nanoGPT ile aynı veri/tohumda ±%1 | 🟡 Isınma sonrası ✅, ısınma içinde değil. Aynı başlangıç ağırlıkları (Nexus'tan aktarıldı) ve aynı 300 × 32 768 token. Adım 0 kaybı **11.0185 / 11.0186**. Adım 30–299 (270 adım): hepsi ±%1 içinde, en çok **%0.68**, ortalama %0.17. Adım 10–29 (ısınma, kayıp 8.3 → 5.0): en çok **%3.5** (adım 17), 10 adımlık kayan ortalamada %1.5. Tüm eğride adımların %96.9'u ±%1 içinde. Nexus'un kendi koşudan koşuya farkı (BF16, atomik dQ): en çok %0.44. Son 50 adım ortalaması 3.0299 / 3.0342. Doğrulama kaybı 11.016 → **3.000** (ppl 20.1). | `curve_compare.json`, `curve_*` |

## Throughput (GPT-2 124M BF16, T=1024, global yığın 32 768 token, fişte)

Nexus: flash dikkat + selective recompute + foreach AdamW (aygıtta global norm kırpma 1.0) + v2 eğitmen
(adım başına tek senkron). nanoGPT: `train.py` tek GPU döngüsü (autocast bf16, fused AdamW,
`loss / accum`, `clip_grad_norm_`, adım başına `.item()`), SDPA flash açık.

| Mikro × accum | Nexus v2 | nanoGPT eager | nanoGPT torch.compile |
|---|---|---|---|
| 8 × 4 | **23 316 tok/s**, MFU %46.4, 6.90 GB | sığmıyor: 11.87 GB ayırma, 2 116 tok/s | sığmıyor: 7.78 GB, 1 226 tok/s |
| 4 × 8 | 22 164, MFU %44.1, **4.61 GB** | 21 347, MFU %42.5, 7.20 GB | **24 270**, MFU %48.3, 5.52 GB |
| 2 × 16 | 20 639, MFU %41.0, **3.46 GB** | 19 973, MFU %39.7, 4.83 GB | 21 159, MFU %42.1, 4.02 GB |

Bellek: Nexus = ayırıcının tepe rezervi, PyTorch = `max_memory_reserved`. Masaüstü uygulamaları
ayrıca 0.6–1.6 GB tutuyor; 8 GB'ın üstündeki ayırmalar WDDM'de sistem belleğine taşıyor.
torch.compile ilk adımı 57–65 s (derleme), tablo ısınma sonrası 6 adımın ortalaması. Her hücre tek koşu.

Eğitmen yükü (aynı oturum, B=8): ham `GptTrainer` (FAZ 3–8 yolu, 10 adımda bir senkron) 356.3 ms/mikro
adım = 22 990 tok/s. v2 accum=1 (her adımda senkron + adım kaydı) 358.1 ms = 22 877 tok/s (**−%0.5**).
v2 accum=4: 23 316 tok/s (+%1.4: optimizer adımı 4 mikro adıma bölünüyor).

**Biriktirme ilk sürümü yavaştı, değiştirildi.** İlk uygulamada biriktirme ayrı bir FP32 tamponla
yapılıyordu (kopyala/topla, +0.5 GB). B=8'de bu tepe belleği 6.90 → 7.40 GB'a çıkardı ve WDDM taşmasıyla
accum=4 21 841 tok/s'e düştü (accum=1'den −%4.6). Şimdi `forward_backward(..., accumulate=true)`
gradyana doğrudan ekliyor: ağırlık GEMM'leri beta=1, BGRADB epiloğunun bias gradyanı [4C] ara tampondan
`+=`, LN ve gömme gradyanları zaten `+=`. Ek bellek yok (accum=4 tepesi 6.90 GB).

## 9.H uçtan uca: tiny

`train --model tiny` (L4 H4 C256, 13.3M + gömme; B=16 T=256 accum 4 = 16 384 token/adım, 500 adım,
WarmupCosine 50/500, lr 1e-3, BF16, fişte): doğrulama kaybı 10.88 → **2.770** (ppl 16.0), 78 s,
108k tok/s, MFU %25, tepe 1.79 GB, NaN/sıçrama yok (`train_tiny.txt`).

## Kayıp eğrisi kurulumu

- Nexus koşusu: `test_phase9_trainer_cuda train --model gpt2s --micro 4 --seq 1024 --accum 8 --steps 300
  --warmup 30 --lr 6e-4 --precision bf16 --recompute selective` (WarmupCosine, min %10, AdamW β=(0.9, 0.95),
  wd 0.1 yalnız ≥2B ağırlıklarda, kırpma 1.0). Başlangıç ağırlıkları (`--dump-init`) ve her adımın
  tokenları (`--dump-batches`, 39.3 MB) diske yazıldı. İkinci koşu aynı yığın dosyasını üretti (`cmp` aynı).
- nanoGPT: `faz9_nanogpt.py curve`, aynı ağırlıklar (matrisler transpoze), aynı tokenlar; Nexus satırı
  1024 token → nanoGPT girdi `[:1023]`, hedef `[1:]` (block_size 1023; nedensel maske ile eşdeğer).
  nanoGPT'nin `nn.GELU()` (erf) yerine tanh GELU (Nexus çekirdeği), vocab 50257, bias=True.
- nanoGPT eğrisi B=2×16 ile koşuldu (aynı 32 satır/adım, aynı sıra). B=4×8 denemesi pildeyken
  7.55 GB'ta taşmaya girip 22 dakikada bitmedi, durduruldu.
- Isınmadaki fark için ölçülmüş bir neden yok. Aday: nanoGPT logitleri autocast altında BF16 GEMM çıktısı;
  Nexus logitleri FP32 (FAZ 4.6). Doğrulanmadı.

## Testler

CPU (`tests/parity/test_phase9_trainer.cpp`, Release + ASan ✅ 11/11):

| Test | Ne denetlenir | Sonuç |
|---|---|---|
| `test_old_api_still_works` | `LanguageTrainer(model, 0.02, 0.01).fit` kayıpları FAZ 9 öncesi koddan (7a63578) alınan altın değerlerle | **bit bit aynı** (6 değer), 15 fit epoch'u = 15 optimizer adımı |
| `test_accum_equivalence` | micro 2 × accum 4 ≡ micro 8 × accum 1 (FP32, kırpmalı) | gradyan rel 3.8e-5, adım sonrası ağırlık rel 4.6e-6, kayıp aynı |
| `test_schedule_values` | WarmupCosine = nanoGPT `get_lr` (0..1200), ısınmasız kosinüs = NexusOptim `CosineAnnealingLR`, WSD (lineer, 1−√), lineer, ters karekök, sabit | farklar **0** ve 2.2e-16 |
| `test_param_groups_wd` | sıfır gradyanla tek adım: yalnız ≥2B, gömme dışı ağırlıklar (1 − lr·wd·s) küçülür; `3.` önekine LR ×2 | hata ≤ 3.4e-8 |
| `test_nan_policy` | atla (ağırlık değişmez, eğitim sürer, geçici NaN tekrar etmez), durdur (adım 2), geri dön (bozuk ağırlık → ilk NaN `2.CausalTransformerBlock`, parametre adıyla; adım 4'e dönüş, LR ×0.5) | ✅ |
| `test_loss_spike` | adım 40'ta gürültülü yığın → sıçrama, örnek kimlikleri 40000/40001 | ✅, yanlış alarm yok |
| `test_trainer_resume` | 3 adım + `save` + yeni model (sıfırlanmış) + `resume` + 3 adım ≡ 6 adım | **bit bit aynı** |
| `test_eval_fixed_set` | aynı token kümesi (kaynak ilerlemez), `eval_every`, ppl = exp(kayıp), çoktan seçmeli puan = −Σ token kaybı | ✅ |
| `test_trainer_loggers` | CSV, JSONL (NexusTrain), W&B JSONL, TensorBoard (geri okuma CRC denetimli; crc32c test vektörü) | ✅ |
| `test_classifier_shared_infra` | sınıflandırma `Trainer`: micro 2 × accum 4 ≡ yığın 8 (kırpma + adım zamanlayıcısı), eksik son pencere adım atar | ağırlık rel 3.0e-7 |

GPU (`tests/cuda/test_phase9_trainer_cuda.cpp`):

| Test | Ne denetlenir | Sonuç |
|---|---|---|
| `test_accum_equivalence_cuda` | FP32 (TF32 kapalı), deterministik: micro 2 × 4 ≡ micro 8 × 1 | gradyan rel 2.3e-5 (taban 1e-3·max\|g\|), adım sonrası \|Δp\| ≤ lr'nin %0.1'i |
| `test_param_groups_wd_cuda` | gömme sönümü kapalı, `fc_w` LR ×3 | hata ≤ 6e-9 |
| `test_nan_policy_cuda` | atla: aygıtta parametre bit bit aynı, `skipped_steps` = 1; geri dön: ilk NaN **`h.1`**, bozuk parametre **`h.1.fc_w`** | ✅ |
| `test_eval_no_grad_memory` | değerlendirmede yeni ayırma 0, tepe artışı 0, `cudaMemGetInfo` aynı | ✅ |
| `test_trainer_resume_cuda` | BF16, 3 + kayıt + 3 ≡ 6 | **bit bit aynı** |

Regresyon: CPU paketi 77/77 ✅ (`ctest_cpu_tail.txt`), GPU paketi 30/30 ✅ (`ctest_cuda.txt`; FAZ 3–8 GPU testleri dahil, `lr_scale` ve `grad_scale` değişikliklerinden sonra).
"Ağırlık rel" eşiği bazı yerlerde |Δ|/lr: ilk Adam adımında |g| ≈ 1e-11 olan sıfırdan başlamış bias'larda
g/(|g|+ε) işaret gürültüsü lr·1e-3 mertebesinde fark verir. Bu bir yuvarlama etkisi, hata değil.

## Tasarım özeti

- **9.1 API.** `train::LanguageTrainerBuilder(model | cuda::gpt_backend(config))`
  `.data(TokenStream train, val) .optimizer(AdamW, ParamGroupPolicy) .schedule(WarmupCosine | WSD | LinearLr |
  ConstantLr | InverseSqrt) .global_batch_tokens .micro_batch .seq_len .grad_clip .precision(BF16, Fp32)
  .recompute(Selective) .eval_every(k, tokens) .log_every .execution_policy(cuda(0)) .seed .deterministic
  .nan_policy .spike_detector .histogram_every .log_csv/jsonl/tensorboard/wandb .build()`;
  `train(max_steps)`, `evaluate()`, `score_choices()`, `save()` / `resume()`.
  Model bir `LmBackend` arkasında: CPU `nn::Module` (FP32, NexusOptim AdamW gruplarıyla) ve
  CUDA `GptModel`. Eski `LanguageTrainer::fit(rows, epochs)` bu döngüye yönlenir:
  "epoch" = aynı yığında bir optimizer adımı. Tek davranış farkı: sonlu olmayan gradyanlı adım artık atlanıyor.
- **9.2 Biriktirme.** accum = global / (micro·T) (tam bölünmüyorsa hata). Mikro adım ağırlığı
  w_i = geçerli_i / Σ geçerli: dlogits'e çarpan (GPU CE `grad_scale`), yani token ağırlıklı ortalama.
  GPU'da doğrudan `+=` biriktirme, ek tampon yok. DDP `no_sync` FAZ 10'da.
- **9.3 Gruplar.** Varsayılan nanoGPT: ≥2 boyutlu (katman dilimi) ağırlıklar sönümlü; bias, norm sönümsüz;
  gömme isteğe bağlı. Ad önekli kural LR/wd çarpanı verir (µP). GPU AdamW çekirdeklerine slot başına `lr_scale` eklendi;
  1.0'da sonuç eskiyle bit bit aynı (`__fmul_rn(x, 1) = x`).
- **9.4 Zamanlayıcılar.** Durumsuz `factor(adım)`. Devam yalnız adım sayacını saklar.
- **9.5 Değerlendirme.** Doğrulama kaynağının başlangıç durumu saklanır, her değerlendirmede ona dönülür
  (aynı tokenlar) ve sonra eski durum geri yüklenir. GPU'da yalnız ileri geçiş, önceden ayrılmış tamponlarda.
  Çoktan seçmeli: konum başına CE kaybı (`device_token_losses`), puan = seçenek tokenlarının −Σ kaybı (FAZ 20 kancası).
- **9.6 Ölçüm.** Adım kaydı: kayıp, LR, grad norm, token/s, MFU, adım ms, veri hazırlama ms (aygıt
  çalışırken), tepe bellek, atlanan adım, doğrulama. Sonraki adımın verisi aygıt çalışırken alınır,
  adım sonucu tek senkronla okunur.
- **9.7 Teşhis.** GPU'da kayıp ölçeği olmadan da inf/NaN koruması (`set_nonfinite_guard`): AdamW norm geçişi
  bayrağı yazar, adım aygıtta atlanır, bias düzeltmesi atlananları çıkarır. Anomalide adımın her mikro yığını
  anomali modunda yeniden çalışır: ağırlık taraması, ileri geçiş (her katman çıktısında sonlu-olmayan
  sayacı, ilk katman adıyla), geri geçiş (sonlu olmayan gradyanlar, geri geçiş sırasıyla).
  Politikalar: atla | durdur | geri dön (host anlık görüntüsü, LR × faktör, veri ileri devam eder, kötü yığın
  tekrar görülmez). Sıçrama: kayan medyan + k·1.4826·MAD, adımın tüm örnek kimlikleriyle kaydedilir.
  Katman başına ağırlık/gradyan normu her k adımda.
- **9.8 Sınıflandırma Trainer'ı.** Ortak yardımcılar (`grad_utils.hpp`: global norm, sonluluk, torch kırpma
  katsayısı) ve zamanlayıcılar (`schedule.hpp`). `TrainerBuilder::gradient_accumulation` (örnek ağırlıklı)
  ve `step_schedule` eklendi. Kırpma artık `min(1, max/(norm+1e-6))` (eskiden yalnız norm > max iken max/norm).

## Açık kalanlar / gerideyiz

- **torch.compile'ın gerisindeyiz.** Aynı ayarda (B=4×8) %8.7, en iyi-en iyi %3.9. Füzyon ve CUDA Graphs FAZ 13'te.
- **Kayıp eğrisi ısınmada ±%1 dışında** (adım 10–29, en çok %3.5). Nedeni ölçülmedi.
- **TF/XLA (KerasNLP GPT-2, jit_compile) referansı ölçülmedi.** Windows'ta TensorFlow GPU desteği yok, kurulu da değil.
- **Değişken uzunluklu token ağırlığı GPU'da gerçek maskeyle sınanmadı.** Ağırlık kodu geçerli hedef sayar
  (`-100`), ama GPU modelinde hedef tamponu girdiyle aynı. Kayıp maskesi / `cu_seqlens` bağlantısı hâlâ yok
  (FAZ 8'den açık). Bugün tüm mikro yığınlar eşit ağırlıklı (1/accum).
- **`.checkpoint(policy)` yok.** `save/resume` tek dosya, FAZ 15'e kadar geçici. Rollback anlık görüntüsü host RAM'de.
- **Kapsamı dar özellikler:** ZeroOffload yalnız accum=1 ve NaN korumasız; AdamW8bitFused v2 eğitmende
  desteklenmiyor (hata verir).
- **9.8 eksik:** CUDA ve BF16 sınıflandırma `Trainer`'ında yok (`nn::Sequential` CPU katmanları).
- **TensorBoard:** yalnız skaler (histogram protosu yok); katman normları skaler olarak yazılıyor.
- **Ölçüm sınırları:** Throughput tablosunda her hücre tek koşu; tekrar ve varyans yok. MFU tepesi azami saatle
  (2100 MHz); kart güç sınırında ~1650 MHz'de çalışıyor.
- **CPU arka ucu MFU yazmıyor:** tepe FLOPS bilinmiyor, MFU NaN.

## Dosyalar

| Spec | Dosya |
|---|---|
| 9.1 / 9.A–9.F | `include/nexus/train/lm_trainer.hpp`, `src/train/lm_trainer.cpp` (döngü, CPU arka ucu, değerlendirme, teşhis, kayıt) |
| 9.4 | `include/nexus/train/schedule.hpp` |
| 9.6 | `include/nexus/train/loggers.hpp`, `src/train/loggers.cpp` (TensorBoard, W&B JSONL) |
| 9.2 / 9.3 / 9.7 GPU | `include/nexus/cuda/gpt.hpp`, `src/cuda/gpt.cpp` (`forward_backward(..., accumulate)`, `set_nonfinite_guard`, `set_param_scales`, `save_state/load_state`, `diagnose_nonfinite`, `layer_norms`), `src/cuda/kernels_train.cu` (`lr_scale`, `nonfinite_guard_update`), `src/cuda/kernels_elementwise.cu` (`count_nonfinite`) |
| GPU arka ucu | `include/nexus/cuda/lm_backend.hpp`, `src/cuda/lm_backend.cpp` (`gpt_backend`, NVML azami saatle tepe FLOPS) |
| Eski API | `include/nexus/train/language.hpp` |
| 9.8 | `include/nexus/train/grad_utils.hpp`, `include/nexus/train/trainer.hpp` |
| Testler / araçlar | `tests/parity/test_phase9_trainer.cpp`, `tests/cuda/test_phase9_trainer_cuda.cpp` (+ `train`, `bench` modları), `benchmarks/runner/faz9_nanogpt.py`, `faz9_compare_curves.py`, `faz9_tinystories_jsonl.py` |

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
