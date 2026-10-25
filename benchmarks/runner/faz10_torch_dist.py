"""FAZ 10 PyTorch referansi: torch.distributed gloo (Windows'ta NCCL yok).

Modlar (torch.multiprocessing.spawn, init_method tcp://127.0.0.1:PORT):
  busbw  --ranks N [--max-bytes B]   : float32 all_reduce, 8 B ... 1 GB (x8), nccl-tests busbw tanimi.
  ddp    --ranks N --steps S --device cpu|cuda
         CPU: Nexus test_phase10_dist ddp_bench ile ayni sekil (Embedding + 4 x TransformerBlock C=128, 2 bas,
         FFN 256, T=64, rank basina 4 satir, vocab 11); GPU: nanoGPT tarzi GPT (L6 H6 C384, vocab 8192, T=256,
         rank basina 8 satir, bf16 autocast), DDP bucket_cap_mb=25, AdamW fused (GPU) / foreach (CPU),
         clip 1.0. Tum rank'ler ayni GPU'yu paylasir (Nexus testleriyle ayni kosul).

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


def busbw(rank, n, args):
    sizes = []
    b = 8
    while b <= args.max_bytes:
        sizes.append(b)
        b *= 8
    for bytes_ in sizes:
        x = torch.ones(max(1, bytes_ // 4), dtype=torch.float32)
        iters = 50 if bytes_ <= (1 << 20) else (8 if bytes_ <= (64 << 20) else (3 if bytes_ <= (256 << 20) else 2))
        dist.all_reduce(x)
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        s = (time.perf_counter() - t0) / iters
        w = torch.tensor([s], dtype=torch.float64)
        dist.all_reduce(w, op=dist.ReduceOp.MAX)
        if rank == 0:
            alg = bytes_ / w.item() / 1e9
            print(json.dumps({"bytes": bytes_, "us": w.item() * 1e6, "algbw": alg, "busbw": alg * 2 * (n - 1) / n}))


class Block(nn.Module):
    def __init__(self, c, h, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, h, batch_first=True)
        self.ln2 = nn.LayerNorm(c)
        self.ff = nn.Sequential(nn.Linear(c, ffn), nn.GELU(), nn.Linear(ffn, c))

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class CpuLM(nn.Module):
    def __init__(self, vocab=11, c=128, blocks=4, h=2, ffn=256, t=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, c)
        self.pos = nn.Parameter(torch.zeros(t, c))
        self.blocks = nn.ModuleList([Block(c, h, ffn) for _ in range(blocks)])
        self.head = nn.Linear(c, vocab)
        self.register_buffer("mask", torch.triu(torch.full((t, t), float("-inf")), 1))

    def forward(self, idx):
        x = self.emb(idx) + self.pos[: idx.size(1)]
        for b in self.blocks:
            x = b(x, self.mask[: idx.size(1), : idx.size(1)])
        return self.head(x)


def ddp(rank, n, args):
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(1337)
    if args.device == "cpu":
        torch.set_num_threads(max(1, args.threads // n))
        model = CpuLM()
        rows, T, vocab = 4, 64, 11
        dev = "cpu"
        ctx = None
    else:
        sys.path.insert(0, args.nanogpt)
        import model as nanogpt
        cfg = nanogpt.GPTConfig(block_size=256, vocab_size=8192, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=True)
        model = nanogpt.GPT(cfg).cuda()
        rows, T, vocab = 8, 256, 8192
        dev = "cuda"
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    params = sum(p.numel() for p in model.parameters())
    net = DDP(model, bucket_cap_mb=25) if n > 1 else model
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1,
                            fused=(dev == "cuda"), foreach=(dev == "cpu"))
    g = torch.Generator().manual_seed(rank + 7)
    data = torch.randint(0, vocab, (rows, T + 1), generator=g).to(dev)
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    times = []
    for it in range(args.steps + 3):
        t0 = time.perf_counter()
        if ctx is not None:
            with ctx:
                _, loss = net(x, y)
        else:
            logits = net(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss.item()
        times.append((time.perf_counter() - t0) * 1e3)
    ms = sum(times[3:]) / args.steps
    if rank == 0:
        print(json.dumps({"framework": "pytorch-" + torch.__version__, "device": dev, "ranks": n, "params": params,
                          "rows_per_rank": rows, "T": T, "ms_per_step": ms, "tokens_per_s": rows * T * n / (ms / 1e3)}))


def worker(rank, n, args):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{args.port}", rank=rank, world_size=n)
    try:
        if args.mode == "busbw":
            busbw(rank, n, args)
        else:
            ddp(rank, n, args)
    finally:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["busbw", "ddp"])
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--max-bytes", type=int, default=1 << 30)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--nanogpt", default="")
    ap.add_argument("--port", type=int, default=29700)
    args = ap.parse_args()
    if args.ranks == 1 and args.mode == "ddp":
        worker(0, 1, args)
    else:
        mp.spawn(worker, args=(args.ranks, args), nprocs=args.ranks, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
