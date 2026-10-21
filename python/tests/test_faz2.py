"""Faz 2: Python ortam, geri cagri ve katman."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nexus


class Sayac(nexus.rl.Environment):
    def __init__(self):
        super().__init__(1, 2, True)
        self.value = 0.0

    def reset(self, seed):
        self.value = 0.0
        return [self.value]

    def step(self, action):
        self.value = 1.0 if int(action) == 1 else 0.0
        return {"observation": [self.value], "reward": float(action), "terminated": False, "truncated": False, "info": "py"}


class Dur(nexus.train.Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []

    def on_epoch_end(self, state):
        self.epochs.append(state["epoch"])

    def stop(self):
        return len(self.epochs) >= 1


def scale_forward(values, params):
    weight = params[0].to_list()[0]
    data = values.to_list()
    out = type(values)(values.shape)
    out.copy_from([item * weight for item in data])
    return out


def scale_backward(grad, params):
    weight = params[0]["values"].to_list()[0]
    data = grad.to_list()
    params[0]["grad"].copy_from([sum(data)])
    out = type(grad)(grad.shape)
    out.copy_from([item * weight for item in data])
    return out


def test_ortam_ve_geri_cagri():
    env = Sayac()
    assert env.reset(1) == [0.0]
    step = env.step(1)
    assert step["observation"] == [1.0]
    assert step["reward"] == 1.0
    assert step["info"] == "py"
    model = nexus.nn.Sequential()
    model.add(nexus.nn.Linear(2, 2, True))
    features = [1.0, 0.0, 0.0, 1.0]
    labels = [0, 1]
    loader = nexus.data.DataLoader(features, labels, 2, 1, False, 1)
    watch = Dur()
    trainer = nexus.train.TrainerBuilder(model).adamw(model, lr=0.05).loss().callback(watch).build()
    trainer.fit(loader, loader, 4)
    assert watch.epochs == [0]
    assert trainer.epochs_ran() == 1


def test_python_katman_dosyadan():
    layer = nexus.nn.PythonModule("Olcek", scale_forward, scale_backward, [[1]])
    layer.parameters()[0]["values"].copy_from([2.0])
    model = nexus.nn.Sequential()
    model.add(layer)
    values = nexus.Tensor([2])
    values.copy_from([1.5, -0.5])
    first = model.forward(values).to_list()
    assert abs(first[0] - 3.0) < 1.0e-5
    grad = nexus.Tensor([2])
    grad.copy_from([1.0, 1.0])
    back = model.backward(grad).to_list()
    assert abs(back[0] - 2.0) < 1.0e-5
    assert abs(model.parameters()[0]["grad"].to_list()[0] - 2.0) < 1.0e-5
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "olcek.nxs")
        model.save_nxs(path)
        loaded = nexus.load(path)
        second = loaded.forward(values).to_list()
        assert max(abs(a - b) for a, b in zip(first, second)) < 1.0e-5
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
