"""FAZ 0.E parity referansi: test_parity_nano'nun kurdugu modeli PyTorch FP32'de kurar.

Girdi klasoru (test_parity_nano yazar):
  manifest.json   yapilandirma yolu, batch/time, tokenlar, parametre adlari ve sekilleri
  weights.bin     parametreler, manifest sirasiyla ardisik float32
Cikti:
  torch_grads.bin gradyanlar ayni sirayla float32
  torch_out.json  {"loss": ...}

PyTorch kurulu degilse 77 ile cikar (ctest SKIPPED).
Iki taraf ayni is parcacigi sayisina sabitlenir (promt.txt FAZ 0 RISKLER); burada
referans tek is parcaciklidir, cunku parity sayisal esitlik olcer, hiz degil.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import json
import math
import os
import sys

SKIP = 77


def main() -> int:
    if len(sys.argv) != 2:
        print("kullanim: reference_torch.py <parity_klasoru>")
        return 2
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as error:  # pragma: no cover - ortam bagimli
        print(f"  SKIP  PyTorch/numpy yok: {error}")
        return SKIP

    torch.set_num_threads(1)
    folder = sys.argv[1]
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(manifest["config"], encoding="utf-8") as handle:
        config = json.load(handle)
    model_cfg = config["model"]
    batch = int(manifest["batch"])
    time = int(manifest["time"])
    dim = int(model_cfg["dim"])
    heads = int(model_cfg["heads"])
    layers = int(model_cfg["layers"])
    mask_value = float(manifest["mask_value"])
    eps = float(manifest["ln_eps"])

    flat = np.fromfile(os.path.join(folder, "weights.bin"), dtype=np.float32)
    params = {}
    order = []
    offset = 0
    for entry in manifest["params"]:
        shape = tuple(entry["shape"])
        count = int(np.prod(shape))
        tensor = torch.from_numpy(flat[offset : offset + count].copy()).reshape(shape)
        params[entry["name"]] = tensor.requires_grad_(True)
        order.append(entry["name"])
        offset += count
    if offset != flat.size:
        print("  FAIL  weights.bin boyutu manifest ile uyusmuyor")
        return 1

    tokens = torch.tensor(manifest["tokens"], dtype=torch.long).reshape(batch, time)
    dh = dim // heads
    mask = torch.triu(torch.full((time, time), mask_value), diagonal=1)

    x = params["wte"][tokens.reshape(-1)] + params["wpe"][:time].repeat(batch, 1)
    for layer in range(layers):
        p = lambda name: params[f"h{layer}.{name}"]  # noqa: E731
        q = x @ p("wq") + p("bq")
        k = x @ p("wk") + p("bk")
        v = x @ p("wv") + p("bv")
        q = q.reshape(batch, time, heads, dh).transpose(1, 2)
        k = k.reshape(batch, time, heads, dh).transpose(1, 2)
        v = v.reshape(batch, time, heads, dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(dh)) + mask
        attn = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(batch * time, dim)
        attn = attn @ p("wo") + p("bo")
        x = F.layer_norm(attn + x, (dim,), p("ln1_g"), p("ln1_b"), eps)
        hidden = F.gelu(x @ p("w1") + p("b1"))
        mlp = hidden @ p("w2") + p("b2")
        x = F.layer_norm(mlp + x, (dim,), p("ln2_g"), p("ln2_b"), eps)
    logits = (x @ params["head_w"] + params["head_b"]).reshape(batch, time, -1)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
    loss.backward()

    with open(os.path.join(folder, "torch_grads.bin"), "wb") as handle:
        for name in order:
            handle.write(params[name].grad.detach().numpy().astype(np.float32).tobytes())
    with open(os.path.join(folder, "torch_out.json"), "w", encoding="utf-8") as handle:
        json.dump({"loss": float(loss.item()), "torch": torch.__version__}, handle)
    print(f"  info  torch_loss={loss.item():.8f} (torch {torch.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
