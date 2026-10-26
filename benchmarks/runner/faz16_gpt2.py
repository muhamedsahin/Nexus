"""FAZ 16: GPT-2 (HF onbellegi) -> Nexus cikarim agirlik dosyasi + sozluk + PyTorch referansi / olcumu.

Kullanim:
  python faz16_gpt2.py export <cikti_dizini>            # gpt2.nxg (FP32), gpt2.tiktoken
  python faz16_gpt2.py logits <cikti_dizini> <tok.bin>  # FP32 tam ileri gecis logitleri (referans)
  python faz16_gpt2.py bench <cikti_dizini> <prompt> <gen> [batch]   # fp16, KV onbellekli kendi generate (eager)

gpt2.nxg: "NXG2" | u32 katman, bas, d, sozluk, n_ctx | FP32 tensorler sirayla:
  wte [V,d], wpe [ctx,d], katman basina: ln1 w,b; attn w [d,3d], b; proj w [d,d], b; ln2 w,b; fc w [d,4d], b;
  fc2 w [4d,d], b; son ln_f w,b. (HF Conv1D zaten [giris, cikis] duzeninde: satir-ana C = A W.)

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import base64
import glob
import json
import math
import os
import struct
import sys
import time

import torch
import torch.nn.functional as F

HF = os.path.expanduser("~/.cache/huggingface/hub/models--gpt2/snapshots")


def snapshot():
    snaps = glob.glob(os.path.join(HF, "*"))
    if not snaps:
        raise SystemExit("gpt2 HF onbellegi yok: " + HF)
    return snaps[0]


def load_state():
    sd = torch.load(os.path.join(snapshot(), "pytorch_model.bin"), map_location="cpu", weights_only=True)
    return {k.replace("transformer.", ""): v.float() for k, v in sd.items()}


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def export(out):
    os.makedirs(out, exist_ok=True)
    sd = load_state()
    cfg = json.load(open(os.path.join(snapshot(), "config.json")))
    L, H, d, V, ctx = cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["vocab_size"], cfg["n_positions"]
    order = ["wte.weight", "wpe.weight"]
    for l in range(L):
        p = f"h.{l}."
        order += [p + "ln_1.weight", p + "ln_1.bias", p + "attn.c_attn.weight", p + "attn.c_attn.bias",
                  p + "attn.c_proj.weight", p + "attn.c_proj.bias", p + "ln_2.weight", p + "ln_2.bias",
                  p + "mlp.c_fc.weight", p + "mlp.c_fc.bias", p + "mlp.c_proj.weight", p + "mlp.c_proj.bias"]
    order += ["ln_f.weight", "ln_f.bias"]
    with open(os.path.join(out, "gpt2.nxg"), "wb") as f:
        f.write(b"NXG2")
        f.write(struct.pack("<5I", L, H, d, V, ctx))
        for k in order:
            f.write(sd[k].contiguous().numpy().astype("<f4").tobytes())
    dec = bytes_to_unicode()
    vocab = json.load(open(os.path.join(snapshot(), "vocab.json"), encoding="utf-8"))
    with open(os.path.join(out, "gpt2.tiktoken"), "w") as f:
        for tok, rank in sorted(vocab.items(), key=lambda kv: kv[1]):
            if tok == "<|endoftext|>":
                continue
            raw = bytes(dec[c] for c in tok)
            f.write(base64.b64encode(raw).decode() + " " + str(rank) + "\n")
    print(json.dumps({"layers": L, "heads": H, "d": d, "vocab": V, "ctx": ctx, "params": sum(sd[k].numel() for k in order)}))


class Gpt2:
    def __init__(self, sd, L, H, dtype, device):
        self.sd = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
        self.L, self.H = L, H

    def block(self, l, x, cache, pos):
        sd, p = self.sd, f"h.{l}."
        B, T, d = x.shape
        h = F.layer_norm(x, (d,), sd[p + "ln_1.weight"], sd[p + "ln_1.bias"], 1e-5)
        qkv = h @ sd[p + "attn.c_attn.weight"] + sd[p + "attn.c_attn.bias"]
        q, k, v = qkv.split(d, dim=2)
        q, k, v = (t.view(B, T, self.H, d // self.H).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            ck, cv = cache[l]
            ck[:, :, pos:pos + T] = k
            cv[:, :, pos:pos + T] = v
            k, v = ck[:, :, :pos + T], cv[:, :, :pos + T]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1), ) if cache is None or pos == 0 else \
            F.scaled_dot_product_attention(q, k, v, attn_mask=None if T == 1 else
                                           torch.ones(T, pos + T, dtype=torch.bool, device=x.device).tril(pos))
        y = y.transpose(1, 2).reshape(B, T, d)
        x = x + y @ sd[p + "attn.c_proj.weight"] + sd[p + "attn.c_proj.bias"]
        h = F.layer_norm(x, (d,), sd[p + "ln_2.weight"], sd[p + "ln_2.bias"], 1e-5)
        h = F.gelu(h @ sd[p + "mlp.c_fc.weight"] + sd[p + "mlp.c_fc.bias"], approximate="tanh")
        return x + h @ sd[p + "mlp.c_proj.weight"] + sd[p + "mlp.c_proj.bias"]

    def forward(self, tok, cache=None, pos=0):
        sd = self.sd
        T = tok.shape[1]
        x = sd["wte.weight"][tok] + sd["wpe.weight"][pos:pos + T]
        for l in range(self.L):
            x = self.block(l, x, cache, pos)
        x = F.layer_norm(x, (x.shape[-1],), sd["ln_f.weight"], sd["ln_f.bias"], 1e-5)
        return x @ sd["wte.weight"].t()


def logits(out, tokfile):
    raw = open(tokfile, "rb").read()
    toks = torch.tensor(struct.unpack("<%di" % (len(raw) // 4), raw), dtype=torch.long)[None]
    sd = load_state()
    m = Gpt2(sd, 12, 12, torch.float64, "cpu")
    with torch.no_grad():
        lg = m.forward(toks)[0].float()
    open(os.path.join(out, "ref_logits.bin"), "wb").write(lg.numpy().astype("<f4").tobytes())
    print(json.dumps({"tokens": toks.shape[1], "argmax_last": int(lg[-1].argmax())}))


def bench(out, prompt, gen, batch=1):
    dev = "cuda"
    sd = load_state()
    m = Gpt2(sd, 12, 12, torch.float16, dev)
    d, H = 768, 12
    res = []
    with torch.no_grad():
        for rep in range(4):
            tok = torch.randint(0, 50257, (batch, prompt), device=dev)
            cache = [(torch.zeros(batch, H, prompt + gen, d // H, device=dev, dtype=torch.float16),
                      torch.zeros(batch, H, prompt + gen, d // H, device=dev, dtype=torch.float16)) for _ in range(12)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            lg = m.forward(tok, cache, 0)
            nxt = lg[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            ttft = (time.perf_counter() - t0) * 1e3
            t1 = time.perf_counter()
            for i in range(gen - 1):
                lg = m.forward(nxt, cache, prompt + i)
                nxt = lg[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            tpot = (time.perf_counter() - t1) * 1e3 / max(1, gen - 1)
            if rep > 0:
                res.append((ttft, tpot))
    ttft = sorted(r[0] for r in res)[len(res) // 2]
    tpot = sorted(r[1] for r in res)[len(res) // 2]
    print(json.dumps({"mode": "torch_gpt2_generate", "torch": torch.__version__, "dtype": "fp16", "batch": batch,
                      "prompt": prompt, "gen": gen, "ttft_ms": round(ttft, 3), "tpot_ms": round(tpot, 3),
                      "tokens_per_s": round(batch * 1000.0 / tpot, 1)}))


def bench_compiled(out, prompt, gen):
    """Statik KV onbellegi + torch.compile(mode='reduce-overhead') decode adimi (CUDA Graph). batch 1."""
    dev, d, H, L = "cuda", 768, 12, 12
    T = prompt + gen

    class Step(torch.nn.Module):
        # gpt-fast deseni: onbellek ve agirliklar modul tamponu (statik girdi) -> CUDA Graph mutasyona izin verir
        def __init__(self):
            super().__init__()
            for k, v in load_state().items():
                self.register_buffer(k.replace(".", "_"), v.to(device=dev, dtype=torch.float16))
            self.register_buffer("kc", torch.zeros(L, 1, H, T, d // H, device=dev, dtype=torch.float16))
            self.register_buffer("vc", torch.zeros(L, 1, H, T, d // H, device=dev, dtype=torch.float16))
            self.register_buffer("ar", torch.arange(T, device=dev))

        def w(self, name):
            return getattr(self, name.replace(".", "_"))

        def forward(self, tok, pos):  # tok [1, 1], pos [1] (aygit)
            x = self.w("wte.weight")[tok] + self.w("wpe.weight")[pos][None]
            mask = (self.ar <= pos)[None, None, None, :]
            for l in range(L):
                p = f"h.{l}."
                h = F.layer_norm(x, (d,), self.w(p + "ln_1.weight"), self.w(p + "ln_1.bias"), 1e-5)
                qkv = h @ self.w(p + "attn.c_attn.weight") + self.w(p + "attn.c_attn.bias")
                q, k, v = qkv.split(d, dim=2)
                q, k, v = (t.view(1, 1, H, d // H).transpose(1, 2) for t in (q, k, v))
                self.kc[l].index_copy_(2, pos, k)
                self.vc[l].index_copy_(2, pos, v)
                y = F.scaled_dot_product_attention(q, self.kc[l], self.vc[l], attn_mask=mask)
                x = x + y.transpose(1, 2).reshape(1, 1, d) @ self.w(p + "attn.c_proj.weight") + self.w(p + "attn.c_proj.bias")
                h = F.layer_norm(x, (d,), self.w(p + "ln_2.weight"), self.w(p + "ln_2.bias"), 1e-5)
                h = F.gelu(h @ self.w(p + "mlp.c_fc.weight") + self.w(p + "mlp.c_fc.bias"), approximate="tanh")
                x = x + h @ self.w(p + "mlp.c_proj.weight") + self.w(p + "mlp.c_proj.bias")
            x = F.layer_norm(x, (d,), self.w("ln_f.weight"), self.w("ln_f.bias"), 1e-5)
            return (x @ self.w("wte.weight").t())[:, -1].argmax(-1, keepdim=True)

    mod = Step()
    kc, vc = mod.kc, mod.vc
    step = mod

    cstep = torch.compile(step, mode="reduce-overhead", fullgraph=True)
    m = Gpt2(load_state(), L, H, torch.float16, dev)
    res = []
    with torch.no_grad():
        t_compile = time.perf_counter()
        for rep in range(4):
            tok = torch.randint(0, 50257, (1, prompt), device=dev)
            cache = [(kc[l], vc[l]) for l in range(L)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nxt = m.forward(tok, cache, 0)[:, -1].argmax(-1, keepdim=True)  # prefill: eager (statik onbellege yazar)
            torch.cuda.synchronize()
            ttft = (time.perf_counter() - t0) * 1e3
            t1 = time.perf_counter()
            for i in range(gen - 1):
                nxt = cstep(nxt.clone(), torch.tensor([prompt + i], device=dev))
            torch.cuda.synchronize()
            tpot = (time.perf_counter() - t1) * 1e3 / (gen - 1)
            if rep == 0:
                first = (time.perf_counter() - t_compile)
            else:
                res.append((ttft, tpot))
    ttft = sorted(r[0] for r in res)[len(res) // 2]
    tpot = sorted(r[1] for r in res)[len(res) // 2]
    print(json.dumps({"mode": "torch_gpt2_compiled_decode", "torch": torch.__version__, "compile": "reduce-overhead",
                      "prompt": prompt, "gen": gen, "ttft_ms_eager_prefill": round(ttft, 3), "tpot_ms": round(tpot, 3),
                      "tokens_per_s": round(1000.0 / tpot, 1), "first_run_s": round(first, 1)}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "export":
        export(sys.argv[2])
    elif cmd == "logits":
        logits(sys.argv[2], sys.argv[3])
    elif cmd == "bench_compiled":
        bench_compiled(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "bench":
        bench(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 1)
    else:
        raise SystemExit("bilinmeyen komut " + cmd)
