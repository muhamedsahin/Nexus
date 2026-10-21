"""Faz 4: kapali sayi hucresi ve math adi .nxs icinde."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import math

import nexus


def _olcekli(olcek):
    def govde(x):
        return x * olcek

    return govde


def _tanh(x):
    return math.tanh(x)


def test_kapali_sayi_ve_math():
    model = nexus.nn.Sequential()
    model.add(nexus.nn.activation("Carpan", _olcekli(2.0)))
    model.add(nexus.nn.activation("Tanh", _tanh))
    values = nexus.Tensor([1])
    values.copy_from([1.5])
    first = model.forward(values).to_list()
    assert abs(first[0] - math.tanh(3.0)) < 1.0e-5
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "kapali.nxs")
        model.save_nxs(path)
        loaded = nexus.load(path)
        second = loaded.forward(values).to_list()
        assert abs(first[0] - second[0]) < 1.0e-5
        kok = str(Path(__file__).resolve().parents[1] / "src")
        ortam = os.environ.copy()
        ortam["PYTHONPATH"] = kok + os.pathsep + ortam.get("PYTHONPATH", "")
        kod = (
            "import nexus\n"
            f"m = nexus.load({path!r})\n"
            "x = nexus.Tensor([1])\n"
            "x.copy_from([1.5])\n"
            "print(m.forward(x).to_list()[0])\n"
        )
        proc = subprocess.run([sys.executable, "-c", kod], capture_output=True, text=True, env=ortam)
        assert proc.returncode == 0, proc.stderr
        assert abs(float(proc.stdout.strip()) - math.tanh(3.0)) < 1.0e-5

    def liste_tutan(kutu):
        def govde(x):
            return x + kutu[0]

        return govde

    try:
        nexus.nn.activation("Liste", liste_tutan([1.0]))
    except TypeError as error:
        assert "kapali" in str(error)
    else:
        raise AssertionError("liste hucresi yazildi")
