import { CopyBlock } from "../../components/copy-block";

const train = `#include <nexus/nexus.hpp>

auto model = nexus::nn::Sequential()
    .add<nexus::nn::Linear>(16, 32)
    .add(nexus::nn::swish())
    .add<nexus::nn::Linear>(32, 3);

auto trainer = nexus::TrainerBuilder(model)
    .optimizer(nexus::optim::AdamW(model->parameters(), {.lr = 0.05f}))
    .loss(nexus::loss::CrossEntropy(0.0f))
    .execution_policy(nexus::ExecutionPolicy::cpu())
    .gradient_clip_norm(1.0f)
    .metric<nexus::metrics::Accuracy>()
    .callback<nexus::callbacks::EarlyStopping>(5)
    .build();

trainer.fit(train_loader, val_loader, 12);
model.save_nxs("ag.nxs");`;

const gemm = `matrix_pro::gemm_f32(n, n, n, a, n, b, n, c, n, 1.0f, 0.0f);`;

export default function CppPage() {
  return (
    <>
      <p className="kicker">C++</p>
      <h1>Eğitmen ve model dosyası.</h1>
      <p>
        <code>TrainerBuilder</code> çapraz entropi ve düz satırlar üzerinde çalışır. <code>mixed_precision</code> kayıp
        ölçeği uygular, matmul FP32 kalır. <code>execution_policy_hybrid</code> yalnız istenirse Linear çarpımını
        zamanlayıcıya verir. <code>ExecutionPolicy::cuda</code> matmul&apos;u CPU çekirdeğinde bırakır.
      </p>
      <CopyBlock code={train} />
      <h2>.nxs</h2>
      <p>
        <code>save_nxs</code> sihri NXS2 olan tek dosya yazar. NXS1 yalnızca eğitmen durumudur ve model diye açılmaz.
        İsimsiz bir C++ lambdası, kayıtlı gövde yoksa <code>python govdesi bu surecte yok</code> der.
      </p>
      <h2>GEMM</h2>
      <CopyBlock code={gemm} />
      <p>Seçili CPU çekirdeği avx2-4x16. Bu işlemcide AVX-512 seçilmez.</p>
    </>
  );
}
