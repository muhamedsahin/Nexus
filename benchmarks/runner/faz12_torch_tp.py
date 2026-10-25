"""FAZ 12 PyTorch referansi: tensor paralelligi (torch.distributed.tensor.parallel, TorchTitan'in kullandigi
parallelize_module + ColwiseParallel / RowwiseParallel / SequenceParallel + loss_parallel), CPU + gloo.

Model Nexus `test_phase12_parallel bench_tp` ile ayni sekil: LLaMA benzeri (RMSNorm, SwiGLU, RoPE, ayrik kafa),
sozluk 2048, d 256, 8 bas, d_ff 704, 4 katman, T 256, B 2 (tek mikro yigin), AdamW, clip 1.0.
TorchTitan TP plani (models/llama/parallelize_llama.py) izlenir: SP'de norm SequenceParallel, attention/ffn
girisi PrepareModuleInput(Shard(1) -> Replicate), wo/w2 RowwiseParallel(output_layouts=Shard(1)),
cikti ColwiseParallel(Shard(-1)) + loss_parallel (sozluk-parcali CE).

Kullanim: python faz12_torch_tp.py --tp N [--sp 0|1] [--steps S] [--threads K]

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

V, D, H, FF, L, T, B = 2048, 256, 8, 704, 4, 256, 2
HD = D // H


def rope_tables(t, hd, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.outer(torch.arange(t, dtype=torch.float32), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x [B, T, h, hd], yarim dondurme (Nexus/GPT-NeoX duzeni)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, D, bias=False)
        self.wv = nn.Linear(D, D, bias=False)
        self.wo = nn.Linear(D, D, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, -1, HD)  # -1: yerel bas sayisi (TP)
        k = self.wk(x).view(b, t, -1, HD)
        v = self.wv(x).view(b, t, -1, HD)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(b, t, -1))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(D, FF, bias=False)
        self.w2 = nn.Linear(FF, D, bias=False)
        self.w3 = nn.Linear(D, FF, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.RMSNorm(D, eps=1e-5)
        self.attention = Attention()
        self.ffn_norm = nn.RMSNorm(D, eps=1e-5)
        self.feed_forward = FeedForward()

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(V, D)
        self.layers = nn.ModuleList([Block() for _ in range(L)])
        self.norm = nn.RMSNorm(D, eps=1e-5)
        self.output = nn.Linear(D, V, bias=False)
        cos, sin = rope_tables(T, HD)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx):
        x = self.tok_embeddings(idx)
        for blk in self.layers:
            x = blk(x, self.cos, self.sin)
        return self.output(self.norm(x))


def parallelize(model, mesh, sp):
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (ColwiseParallel, PrepareModuleInput, RowwiseParallel,
                                                   SequenceParallel, parallelize_module)
    top = {
        "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1) if sp else Replicate()),
        "output": ColwiseParallel(input_layouts=Shard(1) if sp else Replicate(), output_layouts=Shard(-1),
                                  use_local_output=False),
    }
    if sp:
        top["norm"] = SequenceParallel()
    parallelize_module(model, mesh, top)
    for blk in model.layers:
        plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)) if sp else RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)) if sp else RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }
        if sp:
            plan.update({
                "attention_norm": SequenceParallel(),
                "attention": PrepareModuleInput(input_layouts=(Shard(1), None, None),
                                                desired_input_layouts=(Replicate(), None, None)),
                "ffn_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
            })
        parallelize_module(blk, mesh, plan)
    if not sp:
        # SP'siz: normlar her rank'te tam; kirpma DTensor ile karismasin diye cogaltilmis DTensor (Replicate).
        from torch.distributed.tensor import DTensor, distribute_module

        def in_fn(mod, inputs, device_mesh):
            return tuple(DTensor.from_local(t, device_mesh, [Replicate()]) if isinstance(t, torch.Tensor)
                         and not isinstance(t, DTensor) else t for t in inputs)

        def out_fn(mod, outputs, device_mesh):
            return outputs.to_local()

        for m in [model.norm] + [b.attention_norm for b in model.layers] + [b.ffn_norm for b in model.layers]:
            distribute_module(m, mesh, input_fn=in_fn, output_fn=out_fn)


def worker(rank, n, args):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{args.port}", rank=rank, world_size=n)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    model = LM()
    if n > 1:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh("cpu", (n,))
        parallelize(model, mesh, bool(args.sp))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, foreach=False)
    g = torch.Generator().manual_seed(61)
    data = torch.randint(0, V, (B, T + 1), generator=g)
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    from torch.distributed.tensor.parallel import loss_parallel
    times = []
    for _ in range(args.steps):
        dist.barrier()
        t0 = time.perf_counter()
        logits = model(x)
        if n > 1:
            with loss_parallel():
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                loss.backward()
        else:
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=False)
        (loss.full_tensor() if hasattr(loss, "full_tensor") else loss).item()
        ms = torch.tensor([(time.perf_counter() - t0) * 1e3], dtype=torch.float64)
        dist.all_reduce(ms, op=dist.ReduceOp.MAX)
        times.append(float(ms))
    if rank == 0:
        rest = sorted(times[1:])
        med = rest[len(rest) // 2]
        print(json.dumps({"framework": "pytorch-tp-" + torch.__version__, "tp": n, "sp": args.sp, "threads": args.threads,
                          "median_step_ms": med, "tokens_per_s": B * T / (med * 1e-3)}))
    dist.destroy_process_group()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--sp", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--port", type=int, default=29790)
    args = ap.parse_args()
    mp.spawn(worker, args=(args.tp, args), nprocs=args.tp, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
