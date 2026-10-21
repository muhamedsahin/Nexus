"""Faz 5: tensör görünümü ve görüntü kanalı."""

import tempfile
from pathlib import Path

import nexus


def test_reshape_ve_indeks():
    tensor = nexus.Tensor([6])
    tensor.copy_from([1, 2, 3, 4, 5, 6])
    view = tensor.reshape([2, 3])
    assert view.shape == [2, 3]
    assert view[1, 2] == 6.0
    view[0, 1] = 9.0
    assert tensor[1] == 9.0
    mirror = memoryview(tensor)
    assert mirror.shape == (6,)
    assert abs(mirror[1] - 9.0) < 1.0e-6


def test_goruntu_kanal_ve_boyut():
    chw = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "kare.png")
        nexus.data.save_image(path, chw, 2, 2, 3)
        geri = nexus.data.load_image(path, channels=3)
        assert geri["height"] == 2 and geri["channels"] == 3
        kucuk = nexus.data.load_image(path, channels=3, height=1, width=1)
        assert kucuk["height"] == 1 and kucuk["width"] == 1
        assert len(kucuk["chw"]) == 3
        gri = nexus.data.load_image(path, channels=1)
        assert gri["channels"] == 1
        assert len(gri["chw"]) == 4
