# Dalga 2 — dtype'lı, Adımlı, Paylaşımlı Depolamalı Tensor + Dağıtıcı (2026-10-31)

Kapsam: `plan.txt` Dalga 2 (FAZ 1 borcu: "tensör yalnız float*, dtype yok, genel stride / view yok").

**Karar (D3):** API kırılmasına izin sorusu açık kaldığı için planın varsayılanı uygulandı: **uyumlu evrim**.

- Eski `nexus::Tensor` API'sinin anlamı değişmedi.
- Autograd `Variable`'ın `vector<float>` verisine dokunulmadı (34 dosya / 318 satır doğrudan erişiyor).

## Ne yapıldı

| Parça | Durum | Nerede |
|---|---|---|
| `Storage` | ✅ Paylaşılan bayt tamponu. Sahip olan (aligned / pinned / cihaz ayırıcı) ya da sahip olmayan (`wrap` + yaşam tutucu). Sürüm sayacı yerinde yazımlarda artıyor. | `include/nexus/core/tensor.hpp`, `src/core/tensor.cpp` |
| `Tensor` | ✅ Storage + **dtype** (11 tip: F32 / F16 / BF16 / F64 / I32 / I64 / I8 / U8 / Bool / F8E4M3 / F8E5M2) + şekil + eleman cinsinden **int64 adımlar** + **ofset**. Fabrikalar: `empty`, `zeros(…, dtype)`, `from_values`, `alias_raw`, `adopt_shared_raw`, `from_storage` (sınır denetimli). | aynı |
| Kopyasız görünümler | ✅ `view_as`, `reshape` (bitişikse görünüm, değilse kopya), `narrow`, `slice` (adımlı), `transpose`, `permute`, `expand` (adım 0 yayın), `unsqueeze`, `squeeze`, `select`, `shallow`. Yazılar tüm görünümlerde görünüyor. | aynı |
| Dönüşümler | ✅ `contiguous`, `to(dtype)`, `to(alan, cihaz)`, `copy_from` (adımlı + dtype dönüşümlü). Kurallar:<br>• kayan → BF16 / F16: RNE (`half.hpp` ile aynı bit);<br>• kayan → tamsayı: sıfıra doğru kesme, NaN / aralık dışı **açık hata**;<br>• I64 kopyası double'dan geçmiyor;<br>• FP8 doygun. | aynı |
| Dağıtıcı | ✅ (işlem × dtype × cihaz) → çekirdek. Düz dizi, O(1), kilitsiz arama. CPU çekirdekleri `call_once` ile tembel kaydediliyor (statik kütüphane bağlayıcı tuzağına karşı). Kayıtsız birleşim → `"… icin cekirdek yok"` (sessiz kaçış yok). Aynı anahtara ikinci kayıt da hata veriyor. | `include/nexus/core/tensor_ops.hpp`, `src/core/tensor_ops.cpp` |
| CPU çekirdekleri | ✅ Ayrıntı aşağıda. | aynı |
| Geriye uyum | ✅ Aynen kalanlar: `zeros / full / pinned / alias / adopt_shared / data() / view() / operator[] / clone / swap`, derin kopya kurucusu, taşımada kaynağın boşalması. Değişen: `data()` / `operator[]` artık `noexcept` değil. F32 dışında açık hata veriyorlar; adımlı tensörde `operator[]` de hata. | — |

CPU çekirdekleri:

- `add` / `mul`: F32 / F64 / BF16 / F16 / I32 / I64. NumPy yayını; bitişik aynı şekilli girdide düz döngü. BF16 / F16 FP32'de hesaplanıp RNE ile yuvarlanıyor.
- `sum`: kayanlar FP64 biriktiriyor; tamsayılar I64 veriyor.
- `matmul` 2B:
  - F32 / BF16 / F16 → FAZ 18 AVX2 `cpu::gemm`. Devrik görünüm kopyasız `a_trans` / `b_trans` bayrağına çevriliyor.
  - BF16 / F16'da A bir kez FP32'ye çevriliyor; B olduğu gibi paketleniyor.
  - F64 doğrudan.
- `fill_` / `copy_`: her dtype.
- `add_out` / `mul_out`: ayırmasız sıcak yol.

**Kurallar:**

- İkili işlemlerde iki girdi aynı dtype olmalı. Otomatik yükseltme bilerek yok (`Tensor::to` ile açıkça).
- `Shape` rank 0 = 0 eleman (FAZ 1 kuralı): skaler `{1}` şekliyle temsil ediliyor.

## Testler

| Test | Sonuç |
|---|---|
| `test_tensor_views` | ✅ 14/14:<br>• adımlar, transpose / slice (adım 2) / select / expand / permute elemanları elle hesaplanan değerlerle;<br>• görünüme yazı aynı depolamada görünüyor, komşuya dokunmuyor, sürüm artıyor;<br>• bitişik olmayan `view` hata, `reshape` kopya;<br>• `contiguous` mantıksal sırayı koruyor;<br>• geçersiz slice / expand / permute hata;<br>• kopya kurucusu derin. |
| `test_tensor_dtype_cast` | ✅ 12/12:<br>• F32 → BF16 / F16 bit eşit (`half.hpp`), F64 tam;<br>• kesme, 300 → I8 ve NaN → I32 hata;<br>• I64 2^62+1 kopyası tam, I32'ye hata;<br>• Bool, F8E4M3 doygunluğu (1000 → 448);<br>• BF16'da `data()` hata;<br>• devrik görünüm → F64 kopya. |
| `test_tensor_ops_dispatch` | ✅ 14/14:<br>• yayın, adımlı mul, BF16 add RNE, I64 add, sum FP64 / I32 → I64;<br>• matmul 4 şekil × 4 yerleşim (normal / devrik A / devrik B / ikisi) double referansla: F32 en kötü **6.8e-7**, BF16 / F16 **2.9e-3** (çıktı yuvarlaması, ≤ 2^-8);<br>• dtype / yayın uyuşmazlığı, kayıtsız çekirdek (F8 add, I32 matmul) ve çift kayıt açık hata;<br>• `add_out == add`. |
| `test_tensor_autograd_bridge` | ✅ 6/6: `tensor_view` aynı adres; adımlı görünüm; yaşam tutucu Variable'dan uzun yaşıyor; autograd skaleri `{1}`; BF16 devrik → `from_tensor` doğru sıra; `ops::matmul(tensor_view)` ile `autograd::matmul` farkı 4.8e-7. |
| `test_tensor_legacy_api` | ✅ 6/6: alias aynı adres, `adopt_shared` yaşam tutucu, derin kopya, `full + view()`, taşıma kaynağı boşaltıyor, adımlıda `operator[]` hata. |
| Mevcut Tensor kullanıcıları | ✅ 30/30 (değişiklikten hemen sonra): `test_phase1_*`, DLPack-torch (CPU), mirrored, custom ext, RL, PJRT, backend, safetensors. |
| Tam regresyon | aşağıda |

**Tam regresyon:** CPU **167 / 168** (`cpu_ctest2`), GPU Tensor kullanıcıları **44 / 44**, ASan **34 / 34** + köprü **13 / 13**, AddressSanitizer hatası 0.

**Tek başarısız test:** `test_grouped_gemm_vs_loop`. Bu bir **hız** ölçütü ("paralel gruplu GEMM sıralıdan hızlı"); bit eşitliği iki koşuda da ✅.

- Dalga 2'den bağımsız olduğu A/B ile gösterildi. Aynı testin Dalga 2 öncesi kodla (`git stash`) ve yeni kodla 6'şar koşusu: oran 1.01–1.04× vs 1.02–1.05×, süre ~815 ms her ikisinde (`grouped_gemm_ab_old.txt`).
- Ölçüt ~1.0× sınırında oturuyor; gürültüde kalıyor.
- Asıl bulgu: MoE gruplu GEMM yalnız **8–15 GFLOP/s** (AVX2 GEMM 300+). Paralel yol hızlandırmıyor. Plan Dalga 7.3 (MoE iş parçacığı / GEMM) kapsamında, burada düzeltilmedi.

## Eager işlem ek yükü (fişte, 3 koşu, medyan µs / işlem)

| n | Nexus `add` (ayırma + dağıtım + çekirdek) | Nexus `add_out` | Nexus görünüm | PyTorch `torch.add` | PyTorch `out=` | PyTorch `.view` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.63–0.76 | 0.15–0.20 | 0.044 | 2.7–3.7 | 2.4–3.2 | 1.7–2.3 |
| 64 | 0.65–0.80 | 0.15–0.17 | 0.045 | 2.7–3.2 | 2.1–2.7 | 1.7–2.5 |
| 4096 | 1.20–1.25 | 0.71–0.72 | 0.044 | 3.5–4.2 | 3.0–3.1 | 1.8–2.1 |

- **Plan hedefi (≤ 2 µs / işlem) ✅.**
- **Kıyas eşit değil:**
  - PyTorch sayıları Python'dan ölçüldü; yorumlayıcı + bağlama yükünü (~1–2 µs) içeriyor.
  - Nexus doğrudan C++ API.
  - libtorch (C++) kurulu değil; Nexus'un Python bağlaması bu yeni API'yi henüz sunmuyor.
  - Bu yüzden **"PyTorch'tan hızlı" iddiası yok**; yalnız mutlak hedef ölçüldü.

## CUDA çekirdekleri (Dalga 2 devamı, 2026-10-31)

`nexus::cuda::register_tensor_kernels()` (`include/nexus/cuda/tensor_kernels.hpp`, `src/cuda/tensor_kernels.cu`) dağıtıcıya `DeviceKind::Device` çekirdeklerini kaydediyor:

- **Açıkça çağrılıyor ve idempotent.** Statik kütüphanede kendini kaydeden nesneyi bağlayıcı atabilir.
- **Kayıt öncesi davranış:** cihaz işlemi açık hata veriyor ve mesaj bu çağrıyı öneriyor.

| Çekirdek | dtype | Ayrıntı |
|---|---|---|
| copy | F32 F16 BF16 F64 I32 I64 I8 U8 Bool | Adımlı + dönüşümlü, D2D / H2D / D2H. Host tarafı bitişik ara tamponla taşınıyor. Kayan → tamsayı NaN / aralık dışı: cihaz bayrağı → açık hata. |
| fill | aynı | 1 ögelik cihaz skalerinin adım 0 kopyası (adımlı hedef). |
| add / mul | F32 F16 BF16 F64 I32 I64 | Yayın; bitişik aynı şekilde düz yol. BF16 / F16 FP32'de hesaplanıyor, RNE. |
| sum | F32 F16 BF16 F64 → FP64 biriktirme · I32 I64 → I64 | Belirlenimci: sabit ızgarada (≤ 1024 blok) blok kısmi toplamları, ardından tek iş parçacığı sıralı toplam. |
| matmul | F32 F16 BF16 | cuBLASLt (`nexus::cuda::gemm`, FP32 biriktirme; F32'de `float32_matmul_precision` geçerli). Devrik görünüm kopyasız `trans` bayrağına çevriliyor. |

- **FP8 cihaz dönüşümü ve I32 / I64 matmul kayıtlı değil** → açık hata.
- **Akış:** tüm çekirdekler **eski varsayılan akışta** (stream 0) çalışıyor. Böylece Tensor'ün `copy_bytes` hızlı yolu (`cudaMemcpy`) ile doğal sıralı. Nexus akışları `cudaStreamNonBlocking` olduğundan başka akışta çalışan kodla (örn. GptModel) paylaşılan tamponlarda çağıran açık senkronizasyon yapmalı. Bu, PyTorch'un "geçerli akış" modelinden farklı; akış farkındalığı açık iş.

### Testler (`tests/cuda/test_tensor_cuda.cpp`)

| Test | Sonuç |
|---|---|
| `test_tensor_cuda_copy` | ✅ 8/8:<br>• H2D / D2H gidiş-dönüş bit bit;<br>• cihazda `permute().contiguous()` host ile bit bit;<br>• cihazda devrik + dönüşüm (BF16 / F16 / F64 / I32 / I64 / I8 / U8 / Bool) host çekirdekleriyle **bit bit**;<br>• BF16 → F32;<br>• host → cihaz adımlı + dönüşümlü hedef; cihaz → host adımlı hedef;<br>• cihazda NaN → I32 ve 300 → I8 açık hata. |
| `test_tensor_cuda_ops` | ✅ 6/6:<br>• add / mul (yayın [4,1,6] + [3,6], adımlı) F32 / BF16 / F16 / F64 / I64 host ile **bit bit**;<br>• sum 65536 öge host'a ≤ 1e-6 ve iki çağrı bit bit (belirlenimci); sum I32 → I64;<br>• adımlı `fill_` komşulara dokunmuyor;<br>• matmul (3 şekil × 4 yerleşim) F32 (Highest) double referansla **2.0e-7**, BF16 / F16 **3.3e-3**. |
| `test_tensor_cuda_errors` | ✅ 4/4: kayıt öncesi hata + öneri; FP8 / I32 matmul kayıtsız → açık hata; host + cihaz karışık ikili işlem açık hata. |
| compute-sanitizer | ✅ memcheck / initcheck / racecheck / synccheck × 3 mod: **0 hata, 0 yarış**; sanitizer altında testler de geçti (`cuda-sanitizer/`). |
| GPU regresyonu | ✅ 54 / 54 (Tensor kullanıcıları + CUDA çekirdek / GPT / flash / KV / MoE grouped alt kümesi, `gpu_ctest_cuda_kernels.log`). |

Hız ölçülmedi. Bu adım doğruluk ve kapsam içindi; cihaz eager ek yükü ve bant genişliği sonraki dalgada ölçülecek.

## Autograd'ın Tensor'a taşınması (Dalga 2 devamı, 2026-10-31)

**Yaklaşım: uyumlu, iki depolamalı `Variable`.**

- `data_`: FAZ 2 F32 host `vector<float>`. 34 dosya / 318 satır değişmeden çalışıyor.
- `tensor_`: yeni; herhangi bir dtype ve cihaz (CUDA dahil).
- İkisi aynı anda dolu olmuyor.

| Parça | Ne |
|---|---|
| API | `Variable::wrap(Tensor, requires_grad)` (kopyasız) · `tensor()` (iki depolamada da; vektörde kopyasız görünüm) · `dtype()` · `on_device()` · `is_tensor_backed()`. Tensor depolamada `values()` / `mutable_values()` / `at()` **açık hata** veriyor (sessiz FP32 kopya yok). |
| Op'lar (Tensor yolu) | `add`, `mul`, `matmul`, `sum`: girdilerden biri Tensor depolamalıysa `nexus::ops` ile çalışıyor (her kayıtlı dtype, CUDA dahil). Yeni: `to(a, dtype)` ve `to(a, alan, cihaz)`, geri geçiş kaynak dtype / cihaza dönüyor. `matmul` geri geçişinde `dA = g Bᵀ`, `dB = Aᵀ g`, devrik görünümler kopyasız. Tensor yolunda yayın yok (eksen indirgemesi yok) → şekil farkı açık hata. |
| Gradyan biriktirme | `AutogradMeta::tensor_backed`: Tensor yaprak gradyanı **aynı dtype / cihazda** Tensor olarak biriktiriyor (`ops::add_out`). FAZ 2 yaprağına gelen Tensor gradyanı F32 host vektöre çevriliyor; eski optimizer'lar ve `grad().values()` aynen çalışıyor. |
| Karışık grafik | Tensor yolu op'u gradyanı her girdiye **kendi depolama türünde** döndürüyor. FAZ 2 düğümü (relu, softmax, …) vektör gradyanı alıyor; Tensor girdisine giren FAZ 2 op'u açık hata veriyor. |
| Motor | Çok kenarlı birikim ve `retain_grad` Tensor'da `ops::add` kullanıyor. Kök tohumu çıktının dtype / cihazında 1. Anomali modu Tensor gradyanını host'a kopyalayıp NaN / Inf denetliyor. `detach` / `add_` / `mul_` / `fill_` Tensor depolamada çalışıyor. |
| Sürüm denetimi | Tensor depolamalı `Variable::version()` = meta sürümü + depolama sürümü. Tensor işlemleriyle (`ops::fill_`, `copy_from`, `add_out`) yapılan yerinde yazılar kaydedilmiş tensör denetimine takılıyor. |

### Testler

| Test | Sonuç |
|---|---|
| `test_tensor_autograd` (CPU) | ✅ 9/9:<br>• **F64 sonlu fark gradcheck** (matmul / mul / add / sum) en kötü **1.2e-9**;<br>• Tensor yaprağın gradyanı Tensor, aynı dtype;<br>• FAZ 2 vektör yolu ile Tensor yolu (F32) kayıp + gradyanlar **9.4e-8**;<br>• **BF16 autograd** gradyanları BF16, F32'ye **2.5e-3**;<br>• karışık grafik (FAZ 2 relu + Tensor add) doğru türlerde gradyan;<br>• Tensor girdide FAZ 2 op'u açık hata;<br>• kaydedilmiş Tensor yerinde değişince `backward` hata (sürüm);<br>• `to(BF16)` geri geçişi F32, iki backward birikiyor;<br>• dtype uyuşmazlığı hata. |
| `test_tensor_cuda_autograd` (GPU) | ✅ 5/5:<br>• cihaz gradyanları cihazda, F32 ve BF16 dtype'ında;<br>• cihaz autograd host Tensor yoluna göre F32 **3.1e-7**, BF16 **3.6e-3**;<br>• host FAZ 2 yaprak → `to(GPU)` → matmul: gradyan host F32 vektör olarak dönüyor ve doğru. |
| compute-sanitizer (autograd) | ✅ memcheck / initcheck / racecheck / synccheck: **0 hata, 0 yarış**; sanitizer altında 5/5 (`cuda-sanitizer/*-autograd.log`). |
| Regresyon | ✅ CPU **169 / 169** (`cpu_ctest3`) · **tam GPU 222 / 223** (`gpu_ctest4`; tek hata yine kararsız `test_grouped_gemm_vs_loop` hız ölçütü, yukarıdaki A/B ile Dalga 2'den bağımsız, bu turda CPU koşusunda geçti) · ASan **25 / 25** (autograd / Tensor / LM / checkpoint). |

### Kapsam dışı (açık)

- **Tensor yolu yalnız 5 op:** add, mul, matmul, sum, to. Diğer FAZ 2 op'ları (sub, div, relu, gelu, softmax, cross_entropy, layernorm, embedding, …) hâlâ yalnız F32 vektör yolunda. Yani **BF16 / GPU uçtan uca bir model henüz Variable üzerinden eğitilemiyor**. Bu op'ların taşınması sıradaki iş.
- Optimizer'lar (AdamW vb.) Tensor depolamalı parametreleri güncellemiyor.
- Tensor yolunda yayın ve eksenli indirgeme yok.
- Hız ölçülmedi.

## Açık kalanlar

- **Variable köprüsü (artık kısmen taşındı, bkz. yukarı):** FAZ 2 op'ları hâlâ `vector<float>`. Köprü eklendi: `Variable::tensor_view()` (kopyasız F32 görünüm, yaşam tutucu; salt okunur: görünümden yazı Variable sürümünü artırmaz) ve `Variable::from_tensor()` (her dtype / cihaz / adım → F32). `test_tensor_autograd_bridge` 6/6, Tensor matmul ↔ autograd matmul 4.8e-7. Autograd'ın dtype'lı Tensor üzerine taşınması (BF16 / F16 autograd) yapılmadı. Bu, en büyük kalan iş.
- **CUDA:** akış farkındalığı (geçerli akış), FP8 cihaz dönüşümü, cihaz eager ek yükü ölçümü.
- **Diğer:**
  - Otomatik dtype yükseltme yok.
  - Batch matmul (3B+) yok.
  - Diğer işlemler (div, exp, reduce eksenli, …) yok.
  - Python bağlaması yok.

## Dosyalar

- Ham ölçüm: `nexus_eager.jsonl`, `torch_eager_python.jsonl`.
- Betik: `benchmarks/runner/dalga2_eager_overhead.py`.
- Test: `tests/parity/test_tensor_core.cpp`.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
