"""Dalga 3.1: dalga3_gemm.cpp --epilogue 0 ile ayni GEMM sekilleri, PyTorch (ayni GPU, BF16 -> BF16).

Kullanim: python dalga3_gemm_torch.py --bt 1024 --out sonuc.json [--fused 1]
Satir-ana sozlesme Nexus ile ayni: C[m,n] = op(A)[m,k] op(B)[k,n]; trans_a ise A bellekte [k,m]
(torch'ta a.t() gorunumu, kopyasiz). Olcum: 10 isinma, 7 tekrar x 50 cagri, CUDA event, medyan.
--fused 1: Nexus epilog modunun PyTorch eager karsiligi (fc: linear + gelu(tanh), iki cekirdek).
--graph 1: 50 cagri tek CUDA grafinda yakalanip oynatilir (Python / dagitici ek yuku olcumden cikar;
           yalniz GPU cekirdek suresi kalir).

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""
import argparse
import json

import torch
import torch.nn.functional as F

C, V = 768, 50304


def shapes(bt):
    return [
        ("fwd.qkv", False, False, bt, 3 * C, C),
        ("fwd.attn_proj", False, False, bt, C, C),
        ("fwd.fc", False, False, bt, 4 * C, C),
        ("fwd.fc_proj", False, False, bt, C, 4 * C),
        ("fwd.lm_head", False, True, bt, V, C),
        ("bwd.lm_head.dx", False, False, bt, C, V),
        ("bwd.lm_head.dw", True, False, V, C, bt),
        ("bwd.fc_proj.dx", False, True, bt, 4 * C, C),
        ("bwd.fc_proj.dw", True, False, 4 * C, C, bt),
        ("bwd.fc.dx", False, True, bt, C, 4 * C),
        ("bwd.fc.dw", True, False, C, 4 * C, bt),
        ("bwd.attn_proj.dx", False, True, bt, C, C),
        ("bwd.attn_proj.dw", True, False, C, C, bt),
        ("bwd.qkv.dx", False, True, bt, C, 3 * C),
        ("bwd.qkv.dw", True, False, C, 3 * C, bt),
    ]


def time_ms(fn, graph=False):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(50):
                fn()
        g.replay()
        torch.cuda.synchronize()
        replay = g.replay
    else:
        def replay():
            for _ in range(50):
                fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    v = []
    for _ in range(7):
        a.record()
        replay()
        b.record()
        b.synchronize()
        v.append(a.elapsed_time(b) / 50.0)
    v.sort()
    return v[len(v) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt", type=int, default=1024)
    ap.add_argument("--out")
    ap.add_argument("--fused", type=int, default=0)
    ap.add_argument("--graph", type=int, default=0)
    args = ap.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    print(f"{'sekil':18s} {'m':>6s} {'n':>6s} {'k':>6s} {'torch_ms':>10s} {'TFLOP/s':>8s}")
    for name, ta, tb, m, n, k in shapes(args.bt):
        A = (torch.randn((k, m) if ta else (m, k), device="cuda", generator=g) * 0.02).bfloat16()
        B = (torch.randn((n, k) if tb else (k, n), device="cuda", generator=g) * 0.02).bfloat16()
        a = A.t() if ta else A
        b = B.t() if tb else B
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        if args.fused and name == "fwd.fc":
            bias = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
            fn = lambda: F.gelu(torch.addmm(bias, a, b), approximate="tanh")
        else:
            fn = lambda: torch.matmul(a, b, out=out)
        ms = time_ms(fn, bool(args.graph))
        print(f"{name:18s} {m:6d} {n:6d} {k:6d} {ms:10.4f} {2.0 * m * n * k / ms / 1e9:8.1f}")
        rows.append(dict(name=name, ta=ta, tb=tb, m=m, n=n, k=k, torch_ms=ms))
    res = dict(bt=args.bt, fused=bool(args.fused), graph=bool(args.graph), torch=torch.__version__, cuda=torch.version.cuda,
               cublas_bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
               device=torch.cuda.get_device_name(), shapes=rows)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
