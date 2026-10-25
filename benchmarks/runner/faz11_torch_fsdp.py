"""FAZ 11 PyTorch referansi: FSDP2 (torch.distributed.fsdp.fully_shard) CPU + gloo.

Model: Nexus test_phase11_fsdp bench_mem ile ayni sekil (Embedding(11, 256) + 6 x [LN, MHA(2 bas), LN, FFN 512]
+ Linear), rank basina 4 x 64 token, AdamW, clip 1.0. Her blok ayri fully_shard birimi (per-block wrap).
Rank basina kalici bellek: yerel parca parametre + gradyan + AdamW durumu baytlari (DTensor yerel tensorleri
sayilir; Python/kutuphane ek yuku dahil degil — Nexus'un persistent_memory() tanimiyla ayni kapsam).

Kullanim: python faz11_torch_fsdp.py --ranks N [--steps S] [--shard 1|0]  (shard 0: DDP benzeri, yalniz replika)

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, c, h, ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, h, batch_first=True)
        self.ln2 = nn.LayerNorm(c)
        self.ff = nn.Sequential(nn.Linear(c, ffn), nn.GELU(), nn.Linear(ffn, c))

    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.full((t, t), float("-inf")), 1)
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class LM(nn.Module):
    def __init__(self, vocab=11, c=256, blocks=6, h=2, ffn=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, c)
        self.blocks = nn.ModuleList([Block(c, h, ffn) for _ in range(blocks)])
        self.head = nn.Linear(c, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def local_bytes(t):
    if t is None:
        return 0
    if hasattr(t, "to_local"):
        t = t.to_local()
    return t.numel() * t.element_size()


def worker(rank, n, args):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{args.port}", rank=rank, world_size=n)
    torch.manual_seed(1337)
    torch.set_num_threads(max(1, args.threads // n))
    model = LM()
    params = sum(p.numel() for p in model.parameters())
    if args.shard:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        mesh = init_device_mesh("cpu", (n,))
        for b in model.blocks:
            fully_shard(b, mesh=mesh)
        fully_shard(model, mesh=mesh)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, foreach=True)
    g = torch.Generator().manual_seed(rank + 7)
    data = torch.randint(0, 11, (4, 65), generator=g)
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    times = []
    for it in range(args.steps):
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x).reshape(-1, 11), y.reshape(-1))
        loss.backward()
        if not args.shard and n > 1:
            for p in model.parameters():
                dist.all_reduce(p.grad)
                p.grad /= n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=False)
        loss.item()
        times.append((time.perf_counter() - t0) * 1e3)
    pb = sum(local_bytes(p) for p in model.parameters())
    gb = sum(local_bytes(p.grad) for p in model.parameters())
    ob = sum(local_bytes(v) for st in opt.state.values() for k, v in st.items() if torch.is_tensor(v) and k != "step")
    if rank == 0:
        print(json.dumps({"framework": "pytorch-fsdp2-" + torch.__version__ if args.shard else "pytorch-replica",
                          "dp": n, "params": params, "persistent_mb": (pb + gb + ob) / 1e6, "param_mb": pb / 1e6,
                          "grad_mb": gb / 1e6, "opt_mb": ob / 1e6, "mean_step_ms": sum(times[2:]) / max(1, len(times) - 2)}))
    dist.destroy_process_group()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # FSDP2 CPU aygit agi (gloo); GPU'ya tasimasin
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--port", type=int, default=29780)
    args = ap.parse_args()
    mp.spawn(worker, args=(args.ranks, args), nprocs=args.ranks, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
