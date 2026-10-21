"""Python kabugunun egitim, nxs, gemm ve RL girisleri."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nexus


def test_siniflandirici_ve_nxs():
    egitim = []
    etiket = []
    for i in range(32):
        x = 1.0 if i % 2 == 0 else -1.0
        egitim.extend((x, 0.0))
        etiket.append(1 if x > 0.0 else 0)
    model = nexus.nn.Sequential()
    model.add(nexus.nn.Linear(2, 8, True))
    model.add(nexus.nn.swish())
    model.add(nexus.nn.Linear(8, 2, True))
    trainer = (
        nexus.train.TrainerBuilder(model)
        .adamw(model, lr=0.08)
        .loss(0.0)
        .execution_policy_cpu()
        .gradient_clip_norm(1.0)
        .accuracy()
        .build()
    )
    loader = nexus.data.DataLoader(egitim, etiket, 2, 8, False, 1)
    trainer.fit(loader, loader, 8)
    assert trainer.history()[-1].train_loss < trainer.history()[0].train_loss
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "ag.nxs")
        model.save_nxs(path)
        yuklu = nexus.load(path)
        girdi = nexus.Tensor([1, 2])
        girdi[0] = 1.0
        girdi[1] = 0.0
        bir = model.forward(girdi).to_list()
        iki = yuklu.forward(girdi).to_list()
        assert max(abs(a - b) for a, b in zip(bir, iki)) < 1.0e-5


def test_isimli_aktivasyon_dosyadan():
    def mine(x):
        return x * 2

    model = nexus.nn.Sequential()
    model.add(nexus.nn.activation("Mine", mine))
    girdi = nexus.Tensor([2])
    girdi.copy_from([1.5, -0.5])
    ilk = model.forward(girdi).to_list()
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "mine.nxs")
        model.save_nxs(path)
        yuklu = nexus.load(path)
        ikinci = yuklu.forward(girdi).to_list()
        assert max(abs(a - b) for a, b in zip(ilk, ikinci)) < 1.0e-5
        kok = str(Path(__file__).resolve().parents[1] / "src")
        ortam = os.environ.copy()
        ortam["PYTHONPATH"] = kok + os.pathsep + ortam.get("PYTHONPATH", "")
        kod = (
            "import nexus\n"
            f"m = nexus.load({path!r})\n"
            "x = nexus.Tensor([2])\n"
            "x.copy_from([1.5, -0.5])\n"
            "print(m.forward(x).to_list())\n"
        )
        proc = subprocess.run([sys.executable, "-c", kod], capture_output=True, text=True, env=ortam)
        assert proc.returncode == 0, proc.stderr
        assert "3.0" in proc.stdout
        assert "-1.0" in proc.stdout

    def dis():
        kutu = [3.0]

        def govde(x):
            return x * kutu[0]

        return govde

    try:
        nexus.nn.activation("Kapali", dis())
    except TypeError as error:
        assert "kapali" in str(error)
    else:
        raise AssertionError("liste hucresi yazildi")


def test_goruntu_yazilir():
    chw = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "kare.png")
        nexus.data.save_image(path, chw, 2, 2, 3)
        geri = nexus.data.load_image(path)
        assert geri["height"] == 2 and geri["width"] == 2 and geri["channels"] == 3
        assert abs(geri["chw"][0] - 1.0) < 1.0e-6
        assert abs(geri["chw"][1] - 0.0) < 1.0e-6


def test_gemm_ve_all_reduce():
    c = nexus.linalg.gemm(2, 2, 2, [1, 0, 0, 1], [2, 0, 0, 3])
    assert abs(c[0] - 2.0) < 1.0e-4
    assert abs(c[3] - 3.0) < 1.0e-4
    assert nexus.linalg.kernel_name()
    ort = nexus.train.all_reduce_mean([[1.0, 3.0], [3.0, 5.0]])
    assert abs(ort[0][0] - 2.0) < 1.0e-5
    assert abs(ort[1][1] - 4.0) < 1.0e-5


def test_ppo_ve_lig():
    env = nexus.rl.LineWorld()
    config = nexus.rl.PPOConfig()
    config.hidden = 8
    config.seed = 2
    config.epochs = 1
    ajan = nexus.rl.PPOAgent(env.observation_dim(), env.action_count(), config)
    ajan.learn(env, 1, 4)
    gozlem = [0.0] * env.observation_dim()
    eylem = ajan.act(gozlem, True)[0]
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "ajan.nxs")
        ajan.save_nxs(path)
        yuklu = nexus.rl.PPOAgent.load_nxs(path)
        assert yuklu.act(gozlem, True)[0] == eylem
    lig = nexus.rl.League()
    oyuncu = lig.add_player(1200.0)
    lig.save(oyuncu, [0.5, 0.25])
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "lig.nxs")
        lig.save_nxs(path)
        geri = nexus.rl.League.load_nxs(path)
        assert abs(geri.elo(0) - 1200.0) < 1.0e-3
        assert geri.weights(0) == [0.5, 0.25]


def test_csv_microtape_mse():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "veri.csv"
        path.write_text("1,0,1\n-1,0,0\n", encoding="utf-8")
        tablo = nexus.data.read_csv(str(path))
        assert tablo.features_per_sample == 2
        assert tablo.labels == [1, 0]
    tape = nexus.custom.MicroTape()
    x = tape.input(3.0)
    y = tape.square(x)
    tape.backward(y)
    assert abs(tape.value(y) - 9.0) < 1.0e-5
    assert abs(tape.grad(x) - 6.0) < 1.0e-4
    kayip, gradyan = nexus.loss.MSE().forward_backward([1.0, 2.0], [1.0, 0.0])
    assert kayip > 0.0
    assert len(gradyan) == 2


def test_vektor_ortam():
    env = nexus.rl.VectorEnv.line_worlds(2)
    gozlem = env.reset(1)
    assert len(gozlem) == 2 * nexus.rl.LineWorld().observation_dim()
    sonuc = env.step([0, 1])
    assert len(sonuc["rewards"]) == 2
