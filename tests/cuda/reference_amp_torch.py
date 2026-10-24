"""FAZ 4.G: Nexus BF16/FP32 kayip egrisini PyTorch BF16 autocast/FP32 ile karsilastirir.

test_phase4_cuda_training dump <klasor> <adim> ayni baslangic agirligini (Nexus duzeni
[giris, cikis]), ayni Markov batch'lerini ve Nexus FP32/BF16 egrilerini yazar. Bu betik ayni
GPT-2'yi PyTorch'ta (CPU) FP32 ve torch.autocast(bfloat16) ile ayni AdamW + kirpma ayariyla
egitir ve 50 adimlik pencere ortalamalarini yan yana koyar.

Soru: Nexus BF16'nin FP32'ye gore gecikmesi, PyTorch BF16 autocast'in kendi FP32'sine gore
gecikmesiyle ayni buyuklukte mi? (Ayni hassasiyet seciminin dogal maliyeti mi, yoksa Nexus'a
ozgu bir kayip mi.)

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import json
import math
import os
import sys
import time

SKIP = 77


def main() -> int:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as error:  # pragma: no cover
        print(f"  SKIP  PyTorch yok: {error}")
        return SKIP
    folder = sys.argv[1]
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as handle:
        m = json.load(handle)
    flat = np.fromfile(os.path.join(folder, "weights.bin"), dtype=np.float32)
    tokens_all = np.fromfile(os.path.join(folder, "tokens.bin"), dtype=np.int32)
    B, T, C, H, L, steps = m["batch"], m["time"], m["dim"], m["heads"], m["layers"], m["steps"]
    D = C // H
    eps = m["ln_eps"]
    adam = m["adam"]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

    def train(autocast_bf16: bool):
        params = {}
        for p in m["params"]:
            values = flat[p["offset"] : p["offset"] + p["numel"]].copy().reshape(p["shape"])
            params[p["name"]] = torch.from_numpy(values).requires_grad_(True)
        decay = [params[p["name"]] for p in m["params"] if p["decay"]]
        rest = [params[p["name"]] for p in m["params"] if not p["decay"]]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": adam["weight_decay"]}, {"params": rest, "weight_decay": 0.0}],
            lr=adam["lr"], betas=(adam["beta1"], adam["beta2"]), eps=adam["eps"], foreach=False)
        all_params = list(params.values())
        P = params
        losses = []
        for s in range(steps):
            tok = torch.from_numpy(tokens_all[s * B * T : (s + 1) * B * T].astype(np.int64)).view(B, T)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_bf16):
                x = P["wte"][tok] + P["wpe"][:T].unsqueeze(0)
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
            # Nexus gibi: kafa ciktisi FP32'ye yukseltilerek CE'ye girer (4.6).
            loss = F.cross_entropy(logits.float()[:, :-1].reshape(-1, logits.shape[-1]), tok[:, 1:].reshape(-1))
            loss.backward()
            losses.append(float(loss.item()))
            torch.nn.utils.clip_grad_norm_(all_params, adam["max_grad_norm"])
            opt.step()
        return losses

    t0 = time.time()
    torch32 = train(False)
    torch16 = train(True)
    print(f"  info  torch {torch.__version__} CPU, {steps} adim x 2, {time.time() - t0:.0f} s")
    n32, n16 = m["nexus_fp32"], m["nexus_bf16"]

    def win(v, at, w=50):
        return sum(v[at - w : at]) / w

    rows = []
    for at in range(100, steps + 1, 100):
        a, b, c, d = win(n32, at), win(n16, at), win(torch32, at), win(torch16, at)
        rows.append({"step": at, "nexus_fp32": a, "nexus_bf16": b, "torch_fp32": c, "torch_bf16": d,
                     "nexus_gap": (b - a) / a, "torch_gap": (d - c) / c})
        print(f"  info  {at:5d}: nexus fp32={a:.5f} bf16={b:.5f} ({(b - a) / a:+.3%}) | "
              f"torch fp32={c:.5f} bf16={d:.5f} ({(d - c) / c:+.3%}) | nexus32-torch32={(a - c) / c:+.3%}")
    with open(os.path.join(folder, "amp_report.json"), "w", encoding="utf-8") as handle:
        json.dump({"torch_version": torch.__version__, "rows": rows}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
