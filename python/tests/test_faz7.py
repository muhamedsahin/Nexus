"""Faz 7: acik CUDA conv. Varsayilan Conv2d CPU'da kalir."""

import nexus


def test_conv2d_device_tiny():
    out = nexus.nn.conv2d_device([2.0], [3.0], [0.5], 1, 1, 1, 1, 1, 1, 1, 1, 0)
    assert out["height"] == 1 and out["width"] == 1
    assert abs(out["nchw"][0] - 6.5) < 1.0e-4


def test_varsayilan_conv_cpu():
    layer = nexus.nn.Conv2d(1, 1, 1, 1, 0, True)
    image = nexus.Tensor([1, 1, 1, 1])
    image.copy_from([1.0])
    result = layer.forward(image)
    assert result.numel() == 1
