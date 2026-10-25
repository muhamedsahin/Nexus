# FAZ 10 — Çok Süreç / Çok Kart: Başlatıcı, Rendezvous, ProcessGroup, DDP (2026-10-25)

Kapsam: `promt.txt` FAZ 10 (DUVAR 9'un veri paralelliği kısmı). Süreç içi "rank tamponu
ortalaması"nın yerine gerçek çok süreçli eğitim geldi:
- `ProcessGroup` ve asenkron `Work`;
- thread ve tcp arka uçları;
- TCPStore ve `nexus-run` başlatıcısı;
- kovalı ve örtüşmeli DDP (CPU `nn::Module` + GPU `GptModel`);
- `no_sync`, BF16 sıkıştırma, hiyerarşik all-reduce, kollektif doğrulama modu;
- dil modeli eğitmeni v2'de dp > 1.

Makine: RTX 3070 Laptop (8 GB, WDDM), Ryzen 7 5800H (8 çekirdek / 16 iş parçacığı), 32 GB, Windows 11,
CUDA 13.4, MSVC 19.44, fişte. Referans: PyTorch 2.14.1+cu132 `torch.distributed` gloo.

**Donanım sınırı (riskler bölümünde öngörülen durum).** Makinede **tek GPU** var ve Windows'ta
**NCCL yok**. WDDM sürücüsünde CUDA IPC ve çok GPU P2P da yok. Bu yüzden:
- Çok GPU ölçekleme ölçülmedi.
- GPU DDP, aynı GPU'yu paylaşan süreçlerle sınandı: gradyan host'a kopyalanıp tcp all-reduce ile toplanıyor.
  Bu doğruluk ve örtüşme kanıtı içindir, hız ölçeklemesi için değil.
- CPU DDP ölçeklemesinde tüm rank'ler aynı 8 çekirdeği paylaşıyor.
- **Gerçek GPU kümesinde ölçülmedi.**

## Kabul ölçütleri

| Ölçüt | Sonuç | Kanıt |
|---|---|---|
| Eşdeğerlik testleri geçer | ✅ CPU: dp=4 × micro 2 ≡ dp=1 × micro 8 (FP32), 100 adım, ağırlık farkı **2.4e-7** (≤1e-6). Thread ve **4 ayrı süreç (tcp)** aynı sonucu veriyor. GPU: aynı GPU'yu paylaşan **2 süreç**, dp=2 × 2 ≡ dp=1 × 4 (FP32, TF32 kapalı), 100 adım, **1.5e-7**. Tüm rank'lerin ağırlıkları bit bit aynı. Bir istisna aşağıda. | `ddp_cpu_equiv.txt`, `ddp_gpu_equiv.txt` |
| Örtüşme kanıtlanır | ✅ CPU (2 rank, 7 kova): ilk kova iletişimi geri geçişin **0.43. ms**'inde başladı (geri geçiş 2.59 ms), 0.23 ms'si geri geçişle örtüştü. GPU (CUDA olay zamanları, 2 süreç, 7 kova): ilk iletişim **1.69 ms**'de başladı (geri geçiş 11.07 ms), 9.38 ms örtüştü. | `ddp_cpu_overlap.txt`, `ddp_gpu_overlap.txt` |
| Çok GPU'da ölçek verimliliği ≥ PyTorch DDP | ⛔ Ölçülemedi: tek GPU. Ek bilgi: bu makinede PyTorch gloo CUDA tensörü all-reduce ederken **çöküyor** (0xC0000005, `torch_gloo_cuda_crash_repro.py`), yani paylaşılan GPU'da bile PyTorch GPU DDP referansı yok. CPU'da (gloo) karşılaştırma aşağıda. | — |

**Eşdeğerlikteki istisna (raporlanıyor, gizlenmiyor).** Dikkatteki anahtar projeksiyonu bias'ının gerçek
gradyanı matematiksel olarak sıfır: softmax, tüm anahtarlara eklenen sabite duyarsız. Gradyanda yalnız
yuvarlama gürültüsü kalıyor (|g_p|/|g| = 1.8e-8). Adam bu gürültüyü ~lr büyüklüğünde adımlara çeviriyor;
dp=1 ve dp=4 farklı toplama sırasıyla farklı rastgele yürüyor. Sonuç: 100 adımda **2.6e-3** fark (CPU; 16 öğe).
GPU'da aynı sınıf 2240 öğe (anahtar bias'ı + 64'lük hizalama dolgusu), fark 7.6e-6.
Test bu öğeleri otomatik ayırıyor: referans koşudaki en büyük |g| < 1e-6·max ölçütü. Kalan her öğede ≤1e-6 arıyor.
Bu, iletişim hatası değil, optimizer'ın gürültüye duyarlılığı.

## all-reduce bus bandwidth (float32 sum, nccl-tests busbw = algbw · 2(n−1)/n, en yavaş rank)

| Bayt | Nexus tcp 2 rank | PyTorch gloo 2 | Nexus tcp 4 rank | PyTorch gloo 4 | Nexus thread 2 | Nexus thread 4 |
|---|---|---|---|---|---|---|
| 8 B | 212 µs | 795 µs | 489 µs | 1 248 µs | — | — |
| 32 KB | 0.17 GB/s | 0.04 | 0.08 | 0.02 | — | — |
| 256 KB | 0.82 | 0.10 | 0.54 | 0.19 | — | — |
| 2 MB | **1.32** | 0.08 | **1.43** | 0.81 | 1.26 | 1.25 |
| 16 MB | 0.45 | 0.21 | 0.77 | 0.59 | 1.80 | 1.14 |
| 128 MB | 0.98 | 0.17 | 0.65 | 0.64 | 1.83 | 1.02 |
| 1 GB | **1.17** | 0.17 | **0.75** | 0.55 | 1.87 | 1.12 |

Ham çıktılar `busbw_*.txt`. Loopback TCP (tek makine), her boyutta 2–50 tekrar, tek koşu.
Nexus tcp her boyutta gloo'dan hızlı: küçük mesajda gecikme 3.7× düşük, 1 GB'ta busbw 2 rank'te 6.9×, 4 rank'te 1.36×.
16 MB'ta 2 rank'te bir çukur var (0.45 GB/s, komşu boyutlarda 1.3 / 0.98); nedeni incelenmedi.
Thread arka ucu kopyalı posta kutusu kullandığı için bellek bant genişliğiyle sınırlı.

## DDP ölçekleme (zayıf ölçek, rank başına sabit iş)

CPU (Nexus: `ddp_bench`, Embedding + 4 × CausalTransformerBlock C=128, rank başına 4 × 64 token, thread arka ucu;
PyTorch: aynı şekilli `nn.MultiheadAttention` modeli, DDP gloo, `bucket_cap_mb=25`, süreç başına 16/n iş parçacığı):

| Rank | Nexus ms/adım | Nexus verim t1/tN | PyTorch ms/adım | PyTorch verim |
|---|---|---|---|---|
| 1 | 17.9 | 1.00 | 21.3 | 1.00 |
| 2 | 23.4 | **0.77** | 46.0 | 0.46 |
| 4 | 32.6 | **0.55** | 44.1 | 0.48 |
| 8 | 57.1 | 0.31 | 69.7 | 0.31 |

İki çerçeve aynı model kodunu kullanmıyor (Nexus nn katmanları / PyTorch modülleri). Bu yüzden mutlak süreler
değil, verim oranları kıyaslanmalı. 8 rank'te ikisi de çekirdek çekişmesiyle sınırlı.
Model 2 MB ve tek kovaya sığıyor, bu yüzden bu ölçümde örtüşme yok (`overlapped_ms` 0). Örtüşme küçük kovalı testte gösterildi.

GPU (aynı GPU'yu paylaşan süreçler; GPT L6 H6 C384, 13.9 M parametre, BF16, rank başına 8 × 256 token):
- Nexus 1 rank: 15.8 ms/adım.
- Nexus 2 rank: 157.4 ms/adım. İletişim kovaları 90 ms meşgul: 55 MB FP32 gradyan D2H → tcp all-reduce → H2D. 7 ms'si örtüşüyor.
- PyTorch 1 rank (nanoGPT, aynı şekil): 30.7 ms.
- PyTorch 2 rank: çöktü (gloo + CUDA, yukarıda).

Bu kurulum hız için değil; gerçek kümede NCCL (ya da P2P) gerekir.

## Testler

| Test | Ne denetlenir | Sonuç |
|---|---|---|
| `test_collectives_thread` | 1..8 rank; all_reduce (Sum/Max/Min/Avg; F32, F64, I32, I64, BF16, F16; 0, 1, 7, 1000, 65537 öğe), reduce_scatter, all_gather, her kökten broadcast, eşit olmayan all_to_all_v, send/recv (iki etiket), barrier, alt grup, hiyerarşik (k=2, 4); doğrulama modu açıkken de | ✅ |
| `test_collectives_tcp_{2,3,4,8}` | aynı gövde, rank = `nexus-run` süreci | ✅ |
| `test_collectives_nccl` | NCCL arka ucu sessiz yedek yerine açık hata | ✅ |
| `test_mismatched_collective_detected` (+ `_tcp`) | rank 0 all_reduce, rank 1 broadcast çağırır → kilitlenme yerine iki tarafı adlandıran hata ("rank 0: all_reduce #1 … / rank 1: broadcast #1 …") | ✅ |
| `test_rank_failure_clean_exit` | 3 süreç; rank 1 kod 3 ile ölür. Diğerleri bağlantı kopmasını görüp **0.1 s**'de kod 4 ile çıkar. Başlatıcı ilk öleni (işletim sisteminin çıkış zamanıyla) seçer ve **3** döndürür. | ✅ (5/5 tekrar) |
| `test_ddp_equivalence` (+ `_tcp`, `_cuda`) | yukarıdaki kabul satırı | ✅ |
| `test_overlap_occurs` (+ `_cuda`) | yukarıdaki kabul satırı | ✅ |
| `test_topology_and_lr_scaling` | düğüm içi/arası ayrımı, algoritma önerisi (1 düğüm: ring, 2 × 4: hierarchical), lineer/karekök LR ölçekleme | ✅ |

Regresyon: CPU paketi **90/90** (`ctest_cpu.txt`), GPU paketi **32/32** (`ctest_cuda.txt`).
ASan: FAZ 9 + 10 CPU testleri **23/23**, tcp çok süreçli testler dahil (`asan_ctest.txt`).

## Tasarım özeti

- **10.1 ProcessGroup** (`include/nexus/dist/process_group.hpp`).
  - İşlemler: all_reduce, reduce_scatter, all_gather, broadcast, all_to_all_v, send/recv, barrier.
    Her biri `Work` döndürür (`wait(timeout)`, `is_completed()`, başlangıç/bitiş zamanı).
  - Grup başına bir iş parçacığı işleri çağrı sırasıyla yürütür.
  - Algoritmalar tek kez yazıldı, bir taşıma katmanı (`Transport`) üzerinde koşar:
    - halka all-reduce (reduce-scatter + all-gather, n parça);
    - binom ağaç broadcast;
    - çift çift all-to-allv.
  - Taşıma çerçeveleri grup kimliği taşır, böylece alt gruplar aynı bağlantıyı paylaşabilir.
  - tcp: çift başına bir soket ve göndermeye ayrılmış bir iş parçacığı, böylece halkada kilitlenme olmaz.
  - Zaman aşımı ya da kopan bağlantı Work'ü hata ile bitirir; grup "bozuk" olur.
  - Avg tamsayıda açık hata verir. BF16/F16 her adımda FP32'de toplanıp yuvarlanır.
- **10.2** `TcpStore` (set/get/add/check/wait/barrier, istemci yeniden dener) ve `init_process_group`
  (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT). `nexus-run`: `--nproc-per-node --nnodes --node-rank
  --master-addr --master-port --grace-s`; ortam değişkenleri RANK / LOCAL_RANK / WORLD_SIZE / LOCAL_WORLD_SIZE / GROUP_RANK;
  hata durumunda kalan süreçleri bir süre bekler, sonra sonlandırır ve rank başına çıkış kodu tablosu yazar.
- **10.3 DDP** (`include/nexus/dist/ddp.hpp`; arka uçlarda).
  - Başlangıçta parametreler rank 0'dan yayılır.
  - Kovalar (varsayılan 25 MB) geri geçiş sırasındaki aşamalardan kurulur:
    - CPU'da her Sequential katmanı bir aşama;
    - GPU'da lnf, katman L−1…0 ve gömme.
  - İlk senkron adımda gerçek sıra ölçülür; tahminden farklıysa kovalar yeniden kurulur.
  - CPU: son mikro adımın geri geçişi katman katman koşar. Kovası tamamlanan all-reduce'u ProcessGroup
    iş parçacığında başlatır; geri geçiş devam eder.
  - GPU: `GptModel::set_backward_hook` aşama olaylarını kaydeder. İletişim iş parçacığı sırayla: ayrı akışta olayı
    bekler → pinned host kovaya D2H → tcp all-reduce → H2D. Hesap akışı bitiş olaylarını bekledikten sonra
    optimizer adımı atılır.
  - `no_sync`: yalnız son mikro adım iletişim yapar.
  - `bf16_compress`: kova BF16'ya sıkıştırılarak indirgenir.
  - `find_unused_parameters`: kovalar geri geçişten sonra indirgenir. Gerçek kullanılmayan parametre tespiti
    yok; tüm parametreler (sıfır gradyanlılar dahil) indirgenir.
- **10.4 Ölçek.** Global yığın = micro · accum · dp; `scale_learning_rate` (lineer / karekök).
  Token ağırlıkları küresel: geçerli token sayısı adım başında all-reduce edilir, gradyan SUM ile indirgenir.
  Kayıp ve değerlendirme rank'ler üzerinden toplanır; günlük yalnız rank 0'da yazılır. save/resume rank başına dosya.
- **10.5 Topoloji.** `discover_cluster`: MatrixFlash `discover_topology` (NUMA, P2P/NVLink matrisi) +
  LOCAL_WORLD_SIZE ile düğüm ayrımı. `all_reduce_hierarchical`: düğüm içi reduce-scatter → düğümler arası
  all-reduce → düğüm içi all-gather. Sayı düğüm başına rank'e bölünmezse halkaya döner ve bunu
  `last_algorithm()` ile söyler; DDP kovası dolgu ekleyerek bölünebilir yapar.
- **10.6 Doğrulama** (`GroupOptions::validate` / `NEXUS_DIST_DEBUG=1`): her kollektiften önce
  (sıra no, işlem, dtype, sayı, kök) parmak izi halka all-gather ile karşılaştırılır. send/recv iki rank'lik
  olduğu için hariç; all_to_all_v'de sayı karşılaştırılmaz (rank başına farklı olabilir).

## Açık kalanlar / gerideyiz

- **Çok GPU ölçekleme ve NCCL:** ölçülmedi, uygulanmadı. Windows'ta NCCL yok, Linux NCCL yolu bu makinede
  derlenip sınanamadığı için yazılmadı; `create_nccl_group` açık hata veriyor. MatrixFlash'ın tek süreç CUDA P2P
  halkası DDP'ye bağlanmadı (çok GPU yok).
- **GPU DDP yavaş:** host üzerinden tcp aktarım nedeniyle (2 süreçte 10× yavaşlama). Gerçek kümede NCCL gerekir.
- **PyTorch GPU DDP referansı yok:** gloo + CUDA bu kurulumda çöküyor.
- **Eksik özellikler:**
  - PowerSGD yok, yalnız BF16 sıkıştırma.
  - `find_unused_parameters` gerçek tespit yapmıyor (yukarıda).
  - Elastik yeniden başlatma yok (FAZ 15).
- **Topoloji:** MatrixFlash `topology.cpp` genişletilmedi, Nexus tarafında kullanıldı. CPU derlemesinde CUDA aygıt
  sayısı 0 görünüyor (MatrixFlash CPU derlemesi).
- **busbw ölçümü:** tek makine loopback, tek koşu; 16 MB'taki çukur incelenmedi.
- **Eşdeğerlik ölçütü:** ≤1e-6, gradyanı yalnız gürültü olan öğeler (anahtar bias'ı) hariç tutulmuş hâliyle sağlanıyor (yukarıda).

## Dosyalar

| Spec | Dosya |
|---|---|
| 10.1 / 10.A / 10.C / 10.F / 10.G | `include/nexus/dist/process_group.hpp`, `src/dist/process_group.cpp`, `src/dist/net.{hpp,cpp}` |
| 10.2 / 10.B | `include/nexus/dist/store.hpp`, `src/dist/store.cpp`, `tools/nexus_run/main.cpp` |
| 10.3 / 10.E | `include/nexus/dist/ddp.hpp`, `src/dist/ddp.cpp`, CPU: `src/train/lm_trainer.cpp`, GPU: `src/cuda/lm_backend.cpp`, `GptModel::set_backward_hook` |
| 10.4 / 10.H | `include/nexus/train/lm_trainer.hpp` (`process_group`), `scale_learning_rate` |
| 10.5 | `include/nexus/dist/topology.hpp`, `src/dist/topology.cpp` |
| 10.D | `create_nccl_group` (açık hata) |
| Testler / araçlar | `tests/parity/test_phase10_dist.cpp`, `tests/cuda/test_phase10_ddp_cuda.cpp`, `benchmarks/runner/faz10_torch_dist.py` |

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi
Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
