"""FAZ 3.I parity referansi: test_cuda_gpt'nin GPU'da egittigi GPT-2'yi PyTorch FP32'de kurar.

Ayni baslangic agirligi (weights.bin, Nexus duzeni [giris, cikis]), ayni token akisi,
ayni AdamW (2B matrislere sonum, digerlerine yok) ve ayni global norm kirpma.
Karsilastirilanlar:
  - her adimin kaybi: |nexus - torch FP64| <= 1e-4 (promt.txt test_gpt2s_parity_fp32; referans
    olarak ayni egitimin FP64 kosusu kullanilir, PyTorch FP32 farki bilgi olarak raporlanir)
  - ilk adimin tum gradyanlari: norm-bagil max_rel <= 1e-4 (analitik ~0 tensörler
    gurultu tabaninda, compare.py ile ayni kural)
PyTorch yoksa 77 (SKIPPED).

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
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as error:  # pragma: no cover
        print(f"  SKIP  PyTorch/numpy yok: {error}")
        return SKIP

    folder = sys.argv[1]
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as handle:
        m = json.load(handle)
    flat = np.fromfile(os.path.join(folder, "weights.bin"), dtype=np.float32)
    tokens_all = np.fromfile(os.path.join(folder, "tokens.bin"), dtype=np.int32)
    nexus_grads = np.fromfile(os.path.join(folder, "nexus_grads0.bin"), dtype=np.float32)
    B, T, C, H, L = m["batch"], m["time"], m["dim"], m["heads"], m["layers"]
    D = C // H
    eps = m["ln_eps"]
    steps = m["steps"]

    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    adam = m["adam"]

    def make_params(dtype):
        out = {}
        for p in m["params"]:
            values = flat[p["offset"] : p["offset"] + p["numel"]].copy().reshape(p["shape"])
            out[p["name"]] = torch.from_numpy(values).to(dtype).requires_grad_(True)
        return out

    def loss_fn(P, tokens):
        x = P["wte"][tokens] + P["wpe"][:T].unsqueeze(0)
        for l in range(L):
            h = F.layer_norm(x, (C,), P["ln1_w"][l], P["ln1_b"][l], eps)
            qkv = h @ P["qkv_w"][l] + P["qkv_b"][l]
            q, k, v = qkv.split(C, dim=-1)
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, H, D).transpose(1, 2)
            v = v.view(B, T, H, D).transpose(1, 2)
            att = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(D))
            att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
            y = (att @ v).transpose(1, 2).reshape(B, T, C)
            x = x + y @ P["attproj_w"][l] + P["attproj_b"][l]
            h2 = F.layer_norm(x, (C,), P["ln2_w"][l], P["ln2_b"][l], eps)
            x = x + F.gelu(h2 @ P["fc_w"][l] + P["fc_b"][l], approximate="tanh") @ P["fcproj_w"][l] + P["fcproj_b"][l]
        x = F.layer_norm(x, (C,), P["lnf_w"], P["lnf_b"], eps)
        logits = x @ P["wte"].t()
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))

    def train(dtype, check_grads):
        params = make_params(dtype)
        decay = [params[p["name"]] for p in m["params"] if p["decay"]]
        no_decay = [params[p["name"]] for p in m["params"] if not p["decay"]]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": adam["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
            lr=adam["lr"],
            betas=(adam["beta1"], adam["beta2"]),
            eps=adam["eps"],
            foreach=False,
        )
        all_params = list(params.values())
        losses = []
        grad_report = None
        for s in range(steps):
            tokens = torch.from_numpy(tokens_all[s * B * T : (s + 1) * B * T].astype(np.int64)).view(B, T)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(params, tokens)
            loss.backward()
            losses.append(float(loss.item()))
            if s == 0 and check_grads:
                grad_report = compare_grads(params)
            torch.nn.utils.clip_grad_norm_(all_params, adam["max_grad_norm"])
            opt.step()
        return losses, grad_report

    def compare_grads(params):
        torch_grads = {k: v.grad.detach().numpy().copy() for k, v in params.items()}
        global_scale = max(float(np.max(np.abs(g))) for g in torch_grads.values())
        worst, worst_name, floor = 0.0, "", []
        for p in m["params"]:
            ours = nexus_grads[p["offset"] : p["offset"] + p["numel"]].reshape(p["shape"])
            ref = torch_grads[p["name"]]
            scale = float(np.max(np.abs(ref)))
            err = float(np.max(np.abs(ours - ref)))
            if scale < 1e-6 * global_scale:
                floor.append(p["name"])
                rel = err / global_scale
            else:
                rel = err / scale
            if rel > worst:
                worst, worst_name = rel, p["name"]
        return (worst, worst_name, floor)

    losses, grad_report = train(torch.float32, True)
    # FP64 referansi: "dogru" egri. PyTorch FP32'nin kendisi de FP64'ten sapar; AdamW, gradyani
    # yuvarlama gurultusu duzeyinde olan agirliklari (baslangicta Q/K) +-lr adimina normalize
    # ettigi icin bu sapma adim adim buyur. Nexus FP32, PyTorch FP32'den kotu olmamali.
    losses64, _ = train(torch.float64, False)
    nexus_losses = m["nexus_losses"]
    diffs = [abs(a - b) for a, b in zip(nexus_losses, losses)]
    worst_loss = max(diffs)
    nexus64 = [abs(a - c) for a, c in zip(nexus_losses, losses64)]
    torch64 = [abs(b - c) for b, c in zip(losses, losses64)]
    # Gecis (spec esigi 1e-4, dogru referansa karsi): her adimda |Nexus - FP64| <= 1e-4.
    # Bilgi: |Nexus - PyTorch FP32| ayrica raporlanir; 24 Ekim olcumunde PyTorch CPU FP32
    # 10 adimda FP64'ten 1.25e-3 saptı, Nexus GPU 7.9e-6 (sapma PyTorch tarafinda).
    within = all(n <= 2.0 * t + 1e-6 for n, t in zip(nexus64, torch64))
    strict = all(n <= 1e-4 for n in nexus64)
    ok = strict and grad_report[0] <= 1e-4
    print(f"  info  torch {torch.__version__} kayip: {losses[0]:.6f} -> {losses[-1]:.6f}")
    for s, (a, b, c) in enumerate(zip(nexus_losses, losses, losses64)):
        print(
            f"  info  adim {s}: nexus={a:.7f} torch32={b:.7f} torch64={c:.7f} "
            f"|nexus-torch32|={abs(a - b):.2e} |nexus-fp64|={abs(a - c):.2e} |torch32-fp64|={abs(b - c):.2e}"
        )
    print(f"  info  en buyuk |nexus-torch32|={worst_loss:.3e} (spec esigi 1e-4; bilgi)")
    print(
        f"  info  en buyuk |nexus-fp64|={max(nexus64):.3e}, en buyuk |torch32-fp64|={max(torch64):.3e} "
        f"(gecis |nexus-fp64|<=1e-4 her adimda: {strict}; nexus <= 2*torch32+1e-6: {within})"
    )
    print(
        f"  info  ilk adim gradyan max_rel={grad_report[0]:.3e} ({grad_report[1]}), "
        f"gurultu tabani: {', '.join(grad_report[2]) or 'yok'} (esik 1e-4)"
    )
    with open(os.path.join(folder, "parity_report.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "torch_version": torch.__version__,
                "nexus_losses": nexus_losses,
                "torch_losses": losses,
                "torch_fp64_losses": losses64,
                "nexus_vs_fp64": nexus64,
                "torch32_vs_fp64": torch64,
                "max_loss_diff": worst_loss,
                "grad_max_rel": grad_report[0],
                "grad_worst_param": grad_report[1],
                "noise_floor": grad_report[2],
                "passed": ok,
            },
            handle,
            indent=2,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
