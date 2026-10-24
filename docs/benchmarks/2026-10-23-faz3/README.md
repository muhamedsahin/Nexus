# FAZ 3 — Füzyonlu Çekirdekler (Fused Kernels) (2026-10-23)

Bu klasör FAZ 3'ün makinayla okunur çıktı köküdür. `bench_phase3_fused` koşusu
buraya `bench_phase3_fused.json` yazar.

## FAZ 3 ne ekler

FAZ 2'nin autograd motorunun üzerine "CustomFunction" deseniyle füzyonlu işlemler:
ileri + geri **tek düğümde**, ara aktivasyon ayrı bir tensör olarak dolaşmadan.

1. **`fused_linear(x, w, b, act)`** — `y = act(x @ w + b)`; matmul epilogunda bias +
   aktivasyon (Identity/ReLU/GELU/SiLU) tek geçişte; geri tek düğümde `dx`, `dw`, `db`.
2. **`fused_layernorm(x, gamma, beta, eps)`** — tek geçiş istatistik + normalize;
   geri tek düğümde `dx`, `dgamma`, `dbeta` (standart layernorm formülü).
3. **`fused_residual_layernorm(x, residual, gamma, beta, eps)`** — `x + residual`
   ayrı tensör olarak dolaşmaz, doğrudan normalize edilir.
4. **`fused_softmax(x)`** — çevrimiçi (online) kararlı softmax (tek geçiş max +
   ölçekli toplam), füzyonlu geri. Tam füzyonlu dikkatin (flash-attention tarzı)
   yapı taşıdır.

## Dürüstlük sınırları

- Füzyon **bellek trafiğini** azaltır, sayısal sonucu değiştirmez; test bunu
  pariteyle doğrular.
- Tam füzyonlu dikkat (dikkat skorunu belleğe yazmadan) sonraki fazdadır.
- Kaydedilen ara veri (örn. `fused_linear`'da pre-aktivasyon) geri geçiş için
  tutulur; bu standarttır (PyTorch'un fused bias+gelu'su da aynısını yapar).

## Test / Benchmark

```
ctest --test-dir build -R "test_phase3_fused" --output-on-failure
ctest --test-dir build -R "bench_phase3_fused" --output-on-failure
```

`test_phase3_fused` şunları doğrular: her füzyonlu opun çıktı ve gradyanı,
füzyonsuz bilesimle birebir (4 aktivasyon için fused_linear), layernorm ve
residual-layernorm referansıyla, softmax autograd::softmax ile; ve sonlu fark
gradyan eşleşmesi.

## 2026-10-24 denetim notu

Bu fazin kodu ve testleri 24 Ekim denetiminde yeniden gozden gecirildi: derleme hatalari,
kosulmamis testler ve eksik spec maddeleri `../2026-10-24-faz0-3-denetim/README.md` icinde
listelenmistir (bulgular, eklenen testler, olcumler ve acik kalan isler).
