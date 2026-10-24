# FAZ 2 — Genel Tensör Autograd Motoru (2026-10-23)

Bu klasör FAZ 2'nin makinayla okunur çıktı köküdür. `bench_phase2` koşusu buraya
`bench_phase2.json` yazar.

## FAZ 2 ne ekler

Elle yazılmış katman backward'larının yanına genel, tensör düzeyinde, ters-mod bir
otograd motoru (PyTorch Node/Function + Edge + AccumulateGrad tasarımının sadeliği):

1. **Düğüm modeli** — `include/nexus/autograd/autograd.hpp`: `Node::apply(grad_outputs)→grad_inputs`,
   `next_edges`, `sequence_nr`, `AutogradMeta` (grad, grad_fn, requires_grad, is_leaf,
   retains_grad, version), yaprak `AccumulateGrad` (`grad +=`).
2. **Motor** — `src/autograd/engine.cpp`: `backward()` bağımlılık sayacı + hazır kuyruğu
   ile topolojik sırada; `retain_graph` grafiği serbest bırakmayı denetler; motor durumu
   çağrı başına yereldir (iç içe backward çağrıları birbirini bozmaz).
3. **SavedVariable + sürüm denetimi** — `save_variable`/`restore_variable`: geri geçişte
   kaydedilmiş tensör yerinde değiştirildiyse açık hata. `saved_tensors_hooks`
   (pack/unpack) FAZ 7 offload/sıkıştırma için uzantı noktasıdır.
4. **Kancalar** — `Variable::register_hook` (grad dönüşümü) ve
   `register_post_accumulate_hook` (DDP/FSDP için FAZ 10/11).
5. **Mod denetimi** — `set_grad_enabled`, `NoGradGuard`, `EnableGradGuard` (thread-local).
6. **Op kümesi** — `src/autograd/ops.cpp`: add/sub/mul/div/neg, add_scalar/mul_scalar,
   add_bias, relu/gelu/silu/sigmoid/tanh/exp/log/pow, sum/mean, reshape/transpose,
   matmul, softmax, cross_entropy, dropout. Her op ileri + geri formülü (sayısal
   gradyan testiyle doğrulanır).

## Dürüstlük sınırları

- **Çift backward** (`create_graph` yüksek mertebeden türev) FAZ 2'de altyapı dışı:
  parametre kabul edilir ama kullanılmaz.
- **İleri-mod AD** kapsam dışı (spece göre).
- **nn::Module birleşmesi / CustomFunction sarmalayıcıları / Python** ayrı bir alt
  adımdır; bu çekirdek motor önce gelir.
- Grad kancaları yaprak gradyanında (ve kök tohumlamasında) ateşlenir; ara (non-leaf)
  tensörlerde gradyan bu basit motorda saklanmaz.

## Test / Benchmark

```
ctest --test-dir build -R "test_phase2_autograd" --output-on-failure
ctest --test-dir build -R "bench_phase2" --output-on-failure
```

`test_phase2_autograd` şunları doğrular: mod guard'ları, `d/dx x²=2x`, 15 op için
sonlu fark gradyan eşleşmesi, yaprak gradyan birikimi (`+=`), sürüm denetimi hatası,
grad kancası, yerinde değişiklik kısıtı ve uçtan uca MLP eğitimi (loss düşer).

## 2026-10-24 denetim notu

Bu fazin kodu ve testleri 24 Ekim denetiminde yeniden gozden gecirildi: derleme hatalari,
kosulmamis testler ve eksik spec maddeleri `../2026-10-24-faz0-3-denetim/README.md` icinde
listelenmistir (bulgular, eklenen testler, olcumler ve acik kalan isler).
