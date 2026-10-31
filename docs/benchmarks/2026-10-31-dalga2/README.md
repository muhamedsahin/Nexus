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

## Açık kalanlar

- **Variable köprüsü:** autograd `Variable`'ın verisi hâlâ `vector<float>`. Köprü eklendi: `Variable::tensor_view()` (kopyasız F32 görünüm, yaşam tutucu; salt okunur: görünümden yazı Variable sürümünü artırmaz) ve `Variable::from_tensor()` (her dtype / cihaz / adım → F32). `test_tensor_autograd_bridge` 6/6, Tensor matmul ↔ autograd matmul 4.8e-7. Autograd'ın dtype'lı Tensor üzerine taşınması (BF16 / F16 autograd) yapılmadı. Bu, en büyük kalan iş.
- **CUDA çekirdekleri:**
  - Dağıtıcıya kayıtlı cihaz çekirdeği yok. Cihazda yalnız bitişik + aynı dtype kopya (`copy_bytes`) çalışıyor.
  - Cihazda adımlı / dönüşümlü kopya, add / mul / matmul **açık hata veriyor**.
  - nexus_cuda'dan kayıt (cuBLASLt matmul, eleman çekirdekleri) sonraki adım.
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
