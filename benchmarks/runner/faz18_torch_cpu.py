"""FAZ 18 karsilastirmasi: PyTorch CPU (oneDNN / MKL) GEMM + GEMV, test_phase18_cpu bench ile ayni sekiller.

Kullanim: python faz18_torch_cpu.py [is_parcacigi=8]
Cikti: JSON satirlari (stdout). Olcum: 1 isinma + en iyi (n <= 1024: 10, degilse 3 tekrar) — Nexus ile ayni protokol.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""
import json
import sys
import time

import torch


def best_ms(fn, reps):
    fn()
    b = 1e30
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, (time.perf_counter() - t0) * 1e3)
    return b


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    torch.set_num_threads(threads)
    g = torch.Generator().manual_seed(1)
    for n in (256, 512, 1024, 2048, 4096):
        a = torch.randn(n, n, generator=g)
        b = torch.randn(n, n, generator=g)
        c = torch.empty(n, n)
        reps = 10 if n <= 1024 else 3
        f32 = best_ms(lambda: torch.mm(a, b, out=c), reps)
        row = {"mode": "torch_cpu_gemm", "torch": torch.__version__, "threads": threads, "n": n,
               "f32_gflops": round(2 * n ** 3 / f32 / 1e6, 1)}
        if n <= 1024:  # BF16 AVX2'de patolojik yavas (FAZ 18 ilk olcum 0.2-0.5 GFLOPS); buyuk n saatler surer
            ab, bb = a.bfloat16(), b.bfloat16()
            bf = best_ms(lambda: torch.mm(ab, bb), 1)
            row["bf16_gflops"] = round(2 * n ** 3 / bf / 1e6, 2)
        print(json.dumps(row), flush=True)
    for k in (2048, 4096):
        n = 2 * k
        x = torch.randn(1, k, generator=g)
        w = torch.randn(n, k, generator=g)
        f = best_ms(lambda: torch.nn.functional.linear(x, w), 20)
        wb, xb = w.bfloat16(), x.bfloat16()
        bf = best_ms(lambda: torch.nn.functional.linear(xb, wb), 20)
        print(json.dumps({"mode": "torch_cpu_gemv", "threads": threads, "N": n, "K": k, "f32_ms": round(f, 3),
                          "f32_gb_s": round(n * k * 4 / (f * 1e6), 1), "bf16_ms": round(bf, 3),
                          "bf16_gb_s": round(n * k * 2 / (bf * 1e6), 1)}), flush=True)


if __name__ == "__main__":
    main()
