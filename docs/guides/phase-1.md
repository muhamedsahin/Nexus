# Faz 1 kapsami

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Faz 1 sunlari yapar:

- Kok `nexus::nexus` hedefi ve `if(NOT TARGET)` korumalari
- 64 bayt hizali host, sayfa kilitli host ve CUDA cihaz bellegi
- `nexus_model::Tensor`, `nexusdata::NDArray`, `nexusdata::Batch` ve `matrix_pro::Tensor` arasinda sifir kopyali gorunum
- `backward_into` ile tahsisatsiz geri yayilim
- `MirroredTensor`: host ve cihaz kopyasini olay sirali akislarda tutan cift tampon
- MatrixFlash-Pro is parcacigi havuzu: her parca tam bir kez kosar, bos cekirdek uyur

Faz 1 sunlari yapmaz:

- AVX GEMM, NCCL, Ring-AllReduce, ROCm/HIP cekirdegi, OpenXLA derlemesi
- Heterojen CPU+GPU is bolucu
- Ozel aktivasyon kaydi, RL ortami, yuksek seviye `TrainerBuilder`

Bu kapilarin CMake secenekleri (`NEXUS_ENABLE_MULTI_GPU`, `NEXUS_ENABLE_TPU_XLA`, `NEXUS_ENABLE_ROCM`) yalnizca derleme tanimidir. Aygit bellegi ayirmak, desteklenmeyen ailede hata verir. Sessiz bos isaretci donulmez.

Dogrulama:

- `tests/test_phase1_bridge.cpp` adres esitligini, 64 bayt hizayi ve `backward_into` sirasinda sifir `operator new` cagrisini kontrol eder
- `tests/test_phase1_loss_families.cpp` her kayip ailesinde sayisal turev, bit esitligi ve sifir ayirma
- `tests/test_phase1_mirrored_tensor.cpp` host/cihaz turu, akis sirasi ve sifir kopyali sarmalama
- `tests/test_phase1_thread_pool.cpp` her parcayi tam bir kez kosturur; cagri donunce tampon serbest birakilir
- `benchmarks/bench_phase1_bridge.cpp` triad bant genisligini, aile basina `backward_into` ns/cagri ve (CUDA acikken) H2D/D2H GB/s olcer

`MirroredTensor` kopyalari kendi engellemeyen akisinda yapar. Cihaz cekirdegi varsayilan akista (veya `set_device_stream` ile verilen akista) calisir. H2D oncesi cihaz akisindaki okuma, D2H oncesi cihaz akisindaki yazma bir CUDA olayi ile beklenir. `pending_` bayragi, DMA hostu okurken `modify_host` yazmasin diye kopyayi bitirir.

Is parcacigi havuzu Chase-Lev kuyrugu degildir. Cagiran ve calisanlar ayni atomik sayactan parca alir. Donus, hem `completed == parca` hem de hicbir calisanin yigindaki is kaydina dokunmamasi (`busy == 0`) saglaninca olur. Bu sozlesme, GEMM cikis tamponunun cagri donunce serbest birakilmasini guvenli kilar.
