# Nexus Faz 1 belgeleri

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.

Faz 1 alti alt kutuphaneyi tek CMake hedefine baglar ve aralarinda sifir kopyali bir tensor koprusu kurar. Faz 2 donanim zamanlayicisini, Faz 3 optimizator ve kayip fuzyonunu, Faz 4 kullanici genisletmelerini, Faz 5 pekistirmeli ogrenmeyi, Faz 6 akici egitmen API'sini ekler.

## Okuma sirasi

1. [Derleme](guides/build.md)
2. [Monorepo hedefleri](architecture/monorepo.md)
3. [Bellek modeli](architecture/memory.md)
4. [Tensör koprusu](architecture/tensor-bridge.md)
5. [Faz 2 donanim](architecture/phase2-hardware.md)
6. [Faz 3 fuzyon](architecture/phase3-fusion.md)
7. [Faz 4 genisletme](architecture/phase4-custom.md)
8. [Faz 5 pekistirme](architecture/phase5-rl.md)
9. [Faz 6 API](architecture/phase6-api.md)
10. [Faz 8 katmanlar](architecture/phase8-layers.md)
11. [Faz 9 bloklar](architecture/phase9-blocks.md)
12. [Faz 10 yapi](architecture/phase10-structure.md)
13. [Faz 11 dizi](architecture/phase11-sequence.md)
14. [Faz 12 katalog](architecture/phase12-catalog.md)
15. [Faz 13 taban](architecture/phase13-baseline.md)
16. [Faz 24 kapanis](architecture/phase24-report.md)
17. [Faz 30 kapanis](architecture/phase30-report.md)
18. [Tensor API](api/tensor.md)
19. [backward_into](api/backward-into.md)
20. [Calisan ornekler](examples/README.md)

## Tek satırda baglama

```cmake
target_link_libraries(my_app PRIVATE nexus::nexus)
```

Bu hedef `nexus_train`, `nexus_model`, `nexus_optim`, `nexusloss`, `nexusdata` ve `matrix_pro` basliklarini, kutuphanelerini ve derleme tanimlarini tasir. CUDA toolkit yoksa `matrix_pro` CPU cekirdekleriyle kalir.
