"""Dalga 1.3: 1.3B icin PyTorch ZeRO-Offload esdegeri (Nexus `--optimizer offload` ile ayni algoritma).

Windows'ta DeepSpeed yok; ayni tarif elle:
  - GPU'da model: --precision bf16 -> parametreler saf BF16 (8 GB'ta FP32 parametre + autocast sigmaz);
    --precision f32 -> FP32 parametre, TF32 kapali (FP32 "gercek" egrisi).
  - Gradyan: register_post_accumulate_grad_hook ile olusur olusmaz pinned CPU tamponuna (BF16 modda BF16, Nexus
    offload_grad_dtype=BF16 gibi), GPU gradyani hemen birakilir (GPU'da tam gradyan tamponu yok).
  - CPU: FP32 ana kopya + torch.optim.AdamW (nanoGPT gruplari: 2B decay 0.1, digerleri 0), global norm kirpma 1.0
    (clip_grad_norm_), nanoGPT get_lr (warmup + cosine, min oran 0.1).
  - Agirlik geri: FP32 -> BF16 pinned ara tampon -> H2D (Nexus BF16 golge gibi).
  - Blok basina activation checkpointing (Nexus Recompute::Full). Kayip FP32 logit uzerinden (F.cross_entropy).
Cikti faz9_nanogpt.py `curve` bicimiyle ayni (losses, ms, peak_*).

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif
ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

sys.path.insert(0, str(Path(__file__).parent))
from faz9_nanogpt import configure_determinism, get_lr, load_nanogpt, load_nexus_init, perturb_one_ulp  # noqa: E402


def build(nanogpt, meta, device, dtype):
    cfg = nanogpt.GPTConfig(block_size=meta["max_seq"], vocab_size=meta["vocab"], n_layer=meta["layers"],
                            n_head=meta["heads"], n_embd=meta["dim"], dropout=0.0, bias=True)
    with torch.device(device):
        model = nanogpt.GPT(cfg)
    for blk in model.transformer.h:
        blk.mlp.gelu = nn.GELU(approximate="tanh")  # Nexus GELU (tanh)
    return model.to(dtype)


def forward_loss(model, idx, targets, checkpoint):
    t = idx.shape[1]
    tr = model.transformer
    x = tr.wte(idx) + tr.wpe(torch.arange(t, device=idx.device))
    for blk in tr.h:
        x = ckpt.checkpoint(blk, x, use_reentrant=False) if checkpoint else blk(x)
    logits = model.lm_head(tr.ln_f(x)).float()  # CE FP32 (Nexus CE FP32 biriktirir)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def clip_fp64_norm(params, max_norm):
    """clip_grad_norm_ ile ayni katsayi ifadesi (FP32: max_norm / (norm + 1e-6), en fazla 1), ama norm FP64'te.
    Dalga 1.3 olcumu: CPU'da torch.nn.utils.clip_grad_norm_ 124M'de CUDA'daki ayni cagridan 1.2e-4 kayip farki
    uretti (GPU kirpma + CPU AdamW: fark 0) -> CPU FP32 norm indirgemesi buyuk tensorlerde yeterince dogru degil.
    Nexus offload normu FP64'te hesaplar."""
    grads = [p.grad for p in params]
    total = torch.sqrt(sum(torch.linalg.vector_norm(g, dtype=torch.float64) ** 2 for g in grads)).to(torch.float32)
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for g in grads:
        g.mul_(coef)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanogpt", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--batches", required=True)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--precision", choices=["bf16", "f32"], default="bf16")
    ap.add_argument("--deterministic", type=int, choices=[0, 1], default=0)
    ap.add_argument("--checkpoint", type=int, choices=[0, 1], default=1)
    ap.add_argument("--perturb-ulp", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="CPU AdamW is parcacigi (0: torch varsayilani)")
    ap.add_argument("--adamw-impl", choices=["fused", "foreach", "single"], default="fused")
    ap.add_argument("--out")
    args = ap.parse_args()
    numerics = configure_determinism(bool(args.deterministic))
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.threads:
        torch.set_num_threads(args.threads)
    meta = json.load(open(args.init + ".json"))
    nanogpt = load_nanogpt(args.nanogpt)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    # CPU FP32 ana kopya (= baslangic agirliklari), GPU calisma modeli
    master_model = build(nanogpt, meta, "cpu", torch.float32)
    load_nexus_init(master_model, args.init, meta["max_seq"])
    if args.perturb_ulp:
        perturb_one_ulp(master_model, args.perturb_ulp)
    master = list(master_model.parameters())  # bagli wte/lm_head bir kez
    gpu_model = build(nanogpt, meta, "cuda", dtype)
    gpu = list(gpu_model.parameters())
    assert len(gpu) == len(master) and all(g.shape == m.shape for g, m in zip(gpu, master))
    # Host bellegi (1.3B'de 32 GB makinede sinirda): pinned gradyan tamponu BF16 modda gradyanlar m.grad'a
    # alindiktan sonra H2D ara tamponu olarak yeniden kullanilir (Nexus pinned_w = pinned_grad gibi); FP32 modda
    # dogrudan m.grad'dir (kopya yok) ve agirlik dogrudan (sayfalanabilir) kopyalanir.
    grad_pinned = [torch.empty(m.shape, dtype=dtype, pin_memory=True) for m in master]
    staging = grad_pinned if dtype == torch.bfloat16 else None
    with torch.no_grad():
        for i, (g, m) in enumerate(zip(gpu, master)):
            if staging is not None:
                staging[i].copy_(m)
                g.copy_(staging[i])
            else:
                g.copy_(m)
    for m, gp in zip(master, grad_pinned):
        m.grad = gp if dtype == torch.float32 else torch.zeros_like(m)
    copy_stream = torch.cuda.Stream()

    def offload_hook(i):
        def hook(p):
            ev = torch.cuda.Event()
            ev.record()
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(ev)
                grad_pinned[i].copy_(p.grad, non_blocking=True)
                p.grad.record_stream(copy_stream)
            p.grad = None  # GPU'da gradyan tutulmaz
        return hook

    for i, p in enumerate(gpu):
        p.register_post_accumulate_grad_hook(offload_hook(i))
    decay = [m for m in master if m.dim() >= 2]
    nodecay = [m for m in master if m.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.wd}, {"params": nodecay, "weight_decay": 0.0}]
    impl = dict(fused=dict(fused=True), foreach=dict(foreach=True), single=dict(foreach=False))[args.adamw_impl]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, **impl)  # destek yoksa acik hata
    optimizer_impl = args.adamw_impl + " (CPU)"

    flat = np.memmap(args.batches, dtype=np.int32, mode="r")
    T = args.seq
    if flat.size < T * args.steps:
        raise SystemExit(f"yigin dosyasi kisa: {flat.size} < {T * args.steps}")
    losses, times, cpu_ms = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for it in range(args.steps):
        t0 = time.perf_counter()
        lr = get_lr(it, args.lr, args.warmup, args.steps, args.lr * args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        rows = torch.from_numpy(np.array(flat[it * T:(it + 1) * T]).reshape(1, T).astype(np.int64)).cuda()
        loss = forward_loss(gpu_model, rows[:, :-1], rows[:, 1:], bool(args.checkpoint))
        loss.backward()
        copy_stream.synchronize()
        torch.cuda.synchronize()
        c0 = time.perf_counter()
        if staging is not None:
            with torch.no_grad():
                for m, gp in zip(master, grad_pinned):
                    m.grad.copy_(gp)  # BF16 -> FP32
        clip_fp64_norm(master, args.clip)
        opt.step()
        with torch.no_grad():
            for i, (g, m) in enumerate(zip(gpu, master)):
                if staging is not None:
                    staging[i].copy_(m)
                    g.copy_(staging[i], non_blocking=True)
                else:
                    g.copy_(m)
        torch.cuda.synchronize()
        cpu_ms.append((time.perf_counter() - c0) * 1e3)
        lossf = loss.item()
        times.append((time.perf_counter() - t0) * 1e3)
        losses.append(lossf)
        print(f"step {it} loss {lossf:.6f} ms {times[-1]:.1f} cpu_update_ms {cpu_ms[-1]:.1f}", flush=True)
    out = {"framework": "PyTorch " + torch.__version__ + " manual ZeRO-Offload (post-accumulate-grad hooks, CPU AdamW)",
           "precision": args.precision, "losses": losses, "ms": times, "cpu_update_ms": cpu_ms,
           "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
           "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
           "parameter_count": sum(p.numel() for p in master),
           "cuda_build": int(torch.version.cuda.split('.')[0]) * 1000 + int(torch.version.cuda.split('.')[1]) * 10,
           "lr": args.lr, "warmup": args.warmup, "checkpoint": bool(args.checkpoint), "optimizer_impl": optimizer_impl,
           "cpu_threads": torch.get_num_threads(), "perturb_ulp_seed": args.perturb_ulp,
           "target_tokens_per_step": T - 1, "gelu": "tanh (explicit Nexus compatibility adaptation)",
           "nanogpt_model_sha256": hashlib.sha256(open(nanogpt.__file__, "rb").read()).hexdigest()}
    out.update(numerics)
    if args.out:
        json.dump(out, open(args.out, "w"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("losses", "ms", "cpu_update_ms")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
