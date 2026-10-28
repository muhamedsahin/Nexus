"""FAZ 9 PyTorch referansi: nanoGPT (karpathy/nanoGPT model.py) + train.py egitim dongusu.

Dongu train.py'nin tek-GPU yolunun kopyasidir: torch.amp.autocast(bf16), fused AdamW
(configure_optimizers: >=2B parametreler sonumlu), mikro adim basina loss / accum, clip_grad_norm_,
zero_grad(set_to_none=True), get_lr (warmup + cosine), istege bagli torch.compile.
Farklar (Nexus ile ayni modeli kurmak icin, raporda yazili): vocab 50257 (nanoGPT 50304'e
yuvarlar), bias=True (GPT-2), GELU tanh yaklasimi (Nexus cekirdegi; nanoGPT nn.GELU() erf kullanir),
kayip olarak tum mikro adimlarin ortalamasi (train.py yalniz sonuncuyu yazar).

Modlar:
  bench  --nanogpt DIR --micro B --seq T --accum A --steps N [--compile 0|1] [--vocab 50257]
         Sentetik tokenlarla adim suresi, token/s, MFU (6N + 12 L H Q T, tepe --peak-tflops), tepe bellek.
  curve  --nanogpt DIR --init NEXUS_INIT.bin --batches BATCHES.bin --micro B --seq T --accum A --steps N
         --lr LR --warmup W --min-lr-ratio R [--compile 0|1]
         Nexus'un baslangic agirliklari ve ayni token yiginlari ile egitim; adim kayiplarini JSON yazar.
         Nexus satiri T token: girdi [:T-1], hedef [1:T]; konum tablosunun tam boyutu korunur.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn


def load_nanogpt(directory):
    sys.path.insert(0, os.path.abspath(directory))
    import model as nanogpt_model  # karpathy/nanoGPT model.py

    return nanogpt_model


def build(nanogpt, block, vocab, device):
    cfg = nanogpt.GPTConfig(block_size=block, vocab_size=vocab, n_layer=12, n_head=12, n_embd=768,
                            dropout=0.0, bias=True)
    model = nanogpt.GPT(cfg)
    for blk in model.transformer.h:
        blk.mlp.gelu = nn.GELU(approximate="tanh")  # Nexus GELU (tanh) ile ayni aktivasyon
    return model.to(device)


def load_nexus_init(model, path, block):
    meta = json.load(open(path + ".json"))
    flat = np.fromfile(path, dtype=np.float32)
    L, C = meta["layers"], meta["dim"]
    p = {e["name"]: e for e in meta["params"]}

    def get(name):
        e = p[name]
        n = int(np.prod(e["shape"]))
        return flat[e["offset"]:e["offset"] + n].reshape(e["shape"])

    sd = model.state_dict()
    with torch.no_grad():
        def put(key, arr):
            t = torch.from_numpy(np.ascontiguousarray(arr))
            assert sd[key].shape == t.shape, (key, sd[key].shape, t.shape)
            sd[key].copy_(t)

        put("transformer.wte.weight", get("wte"))
        put("transformer.wpe.weight", get("wpe")[:block])
        for l in range(L):
            h = f"transformer.h.{l}."
            put(h + "ln_1.weight", get("ln1_w")[l]); put(h + "ln_1.bias", get("ln1_b")[l])
            put(h + "attn.c_attn.weight", get("qkv_w")[l].T); put(h + "attn.c_attn.bias", get("qkv_b")[l])
            put(h + "attn.c_proj.weight", get("attproj_w")[l].T); put(h + "attn.c_proj.bias", get("attproj_b")[l])
            put(h + "ln_2.weight", get("ln2_w")[l]); put(h + "ln_2.bias", get("ln2_b")[l])
            put(h + "mlp.c_fc.weight", get("fc_w")[l].T); put(h + "mlp.c_fc.bias", get("fc_b")[l])
            put(h + "mlp.c_proj.weight", get("fcproj_w")[l].T); put(h + "mlp.c_proj.bias", get("fcproj_b")[l])
        put("transformer.ln_f.weight", get("lnf_w")); put("transformer.ln_f.bias", get("lnf_b"))
        assert torch.equal(model.lm_head.weight, model.transformer.wte.weight)  # bagli kafa


def get_lr(it, lr, warmup, decay_iters, min_lr):  # train.py get_lr
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > decay_iters:
        return min_lr
    decay_ratio = (it - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def train(args, model, batches, device, steps):
    optimizer = model.configure_optimizers(args.wd, args.lr, (0.9, 0.95), "cuda")
    # FAZ 13: --compile-mode max-autotune (Triton GEMM/cekirdek ayari); varsayilan torch.compile "default".
    run = torch.compile(model, mode=args.compile_mode) if args.compile else model
    # max-autotune / reduce-overhead CUDA Graph kullanir: biriktirmede .grad tamponlari yakalamadan once
    # sabit ayrilmali (PyTorch hata iletisinin onerisi), her adimda yerinde sifirlanir.
    stable_grads = args.compile and args.compile_mode in ("max-autotune", "reduce-overhead")
    if stable_grads:
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" else contextlib.nullcontext())
    losses, times = [], []
    peak_alloc = peak_reserved = 0
    torch.cuda.reset_peak_memory_stats()
    for it in range(steps):
        t0 = time.perf_counter()
        lr = get_lr(it, args.lr, args.warmup, steps, args.lr * args.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr
        total = torch.zeros((), device=device)
        for micro in range(args.accum):
            X, Y = batches(it, micro)
            with ctx:
                _, loss = run(X, Y)
                loss = loss / args.accum
            total += loss.detach()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=not stable_grads)
        lossf = total.item()  # train.py gibi adim basina senkron (log_interval = 1)
        times.append((time.perf_counter() - t0) * 1e3)
        losses.append(lossf)
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return losses, times, peak_alloc, peak_reserved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "curve"])
    ap.add_argument("--nanogpt", required=True)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--precision", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--peak-tflops", type=float, default=43.0)
    ap.add_argument("--init")
    ap.add_argument("--batches")
    ap.add_argument("--out")
    args = ap.parse_args()

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = args.precision != "f32"
    torch.backends.cudnn.allow_tf32 = args.precision != "f32"
    if args.precision == "f32":
        torch.set_float32_matmul_precision("highest")
    device = "cuda"
    nanogpt = load_nanogpt(args.nanogpt)
    B, T, A = args.micro, args.seq, args.accum

    if args.mode == "bench":
        model = build(nanogpt, T, args.vocab, device)
        gen = torch.Generator(device="cpu").manual_seed(3)
        data = torch.randint(0, args.vocab, (B, T + 1), generator=gen).to(device)
        x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()

        def batches(it, micro):
            return x, y

        n = args.warmup_steps + args.steps
        losses, times, pa, pr = train(args, model, batches, device, n)
        ms = sum(times[args.warmup_steps:]) / args.steps
        tps = B * T * A / (ms / 1e3)
        N = model.get_num_params()  # wpe haric (nanoGPT)
        fpt = 6 * N + 12 * 12 * 12 * 64 * T
        out = {"framework": "nanoGPT/PyTorch " + torch.__version__, "compile": bool(args.compile), "compile_mode": args.compile_mode, "B": B, "T": T,
               "accum": A, "vocab": args.vocab, "ms_per_step": ms, "tokens_per_s": tps,
               "mfu": fpt * tps / (args.peak_tflops * 1e12), "peak_allocated_gb": pa / 1e9,
               "peak_reserved_gb": pr / 1e9, "first_step_ms": times[0], "flops_per_token": fpt,
               "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(), "last_loss": losses[-1]}
        print(json.dumps(out))
        return 0

    # curve
    # Preserve all learned position parameters, including the unused final position.
    # Slicing wpe to T-1 changes parameter count/optimizer state in strict scale parity.
    block = json.load(open(args.init + ".json"))["max_seq"]
    model = build(nanogpt, block, args.vocab, device)
    load_nexus_init(model, args.init, block)
    flat = np.memmap(args.batches, dtype=np.int32, mode="r")
    per_step = A * B * T
    if flat.size < per_step * args.steps:
        raise SystemExit(f"yigin dosyasi kisa: {flat.size} < {per_step * args.steps}")

    def batches(it, micro):
        off = (it * A + micro) * B * T
        rows = torch.from_numpy(np.array(flat[off:off + B * T]).reshape(B, T).astype(np.int64)).to(device)
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    losses, times, pa, pr = train(args, model, batches, device, args.steps)
    out = {"framework": "nanoGPT/PyTorch " + torch.__version__, "compile": bool(args.compile), "precision": args.precision, "losses": losses,
           "ms": times, "peak_allocated_gb": pa / 1e9, "peak_reserved_gb": pr / 1e9,
           "parameter_count": sum(p.numel() for p in model.parameters()),
           "lr": args.lr, "warmup": args.warmup, "target_tokens_per_step": B * (T - 1) * A,
           "gelu": "tanh (explicit Nexus compatibility adaptation)",
           "nanogpt_model_sha256": hashlib.sha256(open(nanogpt.__file__, "rb").read()).hexdigest()}
    if args.out:
        json.dump(out, open(args.out, "w"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("losses", "ms")} |
                     {"first": losses[:3], "last": losses[-3:], "mean_ms": sum(times[5:]) / max(1, len(times) - 5)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
