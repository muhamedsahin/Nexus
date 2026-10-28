# Faz 19 — arka uç denetimi ve düzeltmeler

28 Ekim yayın kontrolü: kullanıcının seçimiyle mevcut Windows makinesinde doğrulanabilen kapsam
yayına hazırlanmıştır. HIP kaynak uyumluluğu `test_hip_source_compat` olarak ayrı çalışır;
`test_hip_parity` gerçek yürütme uygulanmadığı için 77 (Skipped) döndürür. HIP/Metal motorları
ve gerçek üretici PJRT doğrulaması açık kalır. [Son yayın testleri](audit/publish-release.log).

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

27–28 Ekim 2026 denetimi, önceki rapordaki tamamlanma ve PJRT/XLA iddialarını düzeltti. **Faz 19 bütünüyle kapalı değildir:** HIP ve Metal yürütme motorları eksiktir. Gerçek C++ PJRT C ABI adaptörü kullanıcı onayıyla eklendi; resmî ABI, opaque nesneler, async olaylar ve hata temizliği test edildi. Gerçek üretici PJRT eklentisi üzerinde çalışma bu ortamda doğrulanmadı. StableHLO’nun gerçek XLA CPU tarafından derlenip çalıştırılması ayrı JAX/jaxlib testiyle kanıtlandı.

- [Ayrıntılı denetim raporu, düzeltmeler, kapsam ve tekrar komutları](audit/REPORT.md)
- [Nihai Release: 8 geçti, 1 üretici testi atlandı](audit/release-final-phase19.log)
- [Nihai ASan: 9 geçti, 1 üretici testi atlandı](audit/asan-final.log)
- [Geniş regresyon: ilk koşuda 206/207; kalan test son tekrarda geçti](audit/release-tests.log)
- [Asıl TCP hedefinin son başarılı tekrarı](audit/tcp4-final.log)
- [34 PJRT ABI/cleanup denetimi](audit/pjrt-abi-contract.log)
- [Dört CUDA sanitizer sonucu](audit/sanitizers.json)
- [Yeni 7 tekrarlı GEMM ölçümleri ve kaynak/ikili kimlikleri](audit/backend-bench.json)
- [Gerçek XLA CPU sonucu](audit/xla-cpu.json)

CUDA/DirectML registry sürücülerinin CPU staging işlemleri artık yerel GPU çekirdeği olarak gösterilmez. Desteklenmeyen işlem varsayılan `Error` politikasında hata verir; uyarılı CPU fallback açıkça seçilebilir. Vulkan üzerinde MLP eğitimi, GEMM ve shader parity kontrolleri geçti. Metal etiketiyle sıradan host belleği GPU tamponu gibi sunulmaz.

`bench_faz19.json` önceki raporun tarihsel verisidir. İçindeki eski PJRT girdileri Nexus CPU yorumlayıcısını ölçer; gerçek OpenXLA veya TPU ölçümü değildir. Eski en-iyi-örnek değerleri ile yeni medyanlar farklı protokollerdedir. Bu denetim PyTorch/TensorFlow’u geçme iddiasında bulunmaz.
