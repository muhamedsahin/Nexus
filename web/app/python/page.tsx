import { CopyBlock } from "../../components/copy-block";

const fit = `import nexus

model = nexus.nn.Sequential()
model.add(nexus.nn.Linear(2, 8, True))
model.add(nexus.nn.swish())
model.add(nexus.nn.Linear(8, 2, True))
loader = nexus.data.DataLoader(features, labels, 2, 8, False, 1)
trainer = (
    nexus.train.TrainerBuilder(model)
    .adamw(model, lr=0.05)
    .loss()
    .execution_policy_cpu()
    .build()
)
trainer.fit(loader, loader, 8)
model.save_nxs("ag.nxs")
loaded = nexus.load("ag.nxs")`;

const device = `out = nexus.nn.conv2d_device(
    [2.0], [3.0], [0.5],
    1, 1, 1, 1, 1, 1, 1, 1, 0,
)
# out["nchw"][0] == 6.5`;

export default function PythonPage() {
  return (
    <>
      <p className="kicker">Python</p>
      <h1>Aynı adlar, ince kabuk.</h1>
      <p>
        <code>Sequential</code>, <code>TrainerBuilder</code> ve <code>PPOAgent</code> C++ adlarıdır. Swish C++ formülüdür.
        Başka bir adın gövdesi .nxs içine yazılır. Kapalı hücre yalnız float, int veya bool olabilir.
      </p>
      <CopyBlock code={fit} />
      <h2>Aygıt evrişimi</h2>
      <p>
        <code>nn.Conv2d</code> CPU&apos;da kalır. <code>conv2d_device</code> host veriyi aygıta taşır. Varsayılan eğitim
        politikası CPU&apos;dur.
      </p>
      <CopyBlock code={device} />
      <h2>Teyp ve ortam</h2>
      <p>
        <code>nexus.custom.Tape</code> toplama, çarpma, matmul, ReLU ve mevcut <code>Module.backward</code> kaydeder.
        <code>Environment</code> ve <code>Callback</code> Python alt sınıfı kabul eder.
      </p>
    </>
  );
}
