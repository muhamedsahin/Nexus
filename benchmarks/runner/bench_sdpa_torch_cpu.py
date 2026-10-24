"""FAZ 5.G karsilastirmasi: PyTorch CPU scaled_dot_product_attention (FP32), bench_attention_cpu ile ayni sekiller.

Kullanim: python bench_sdpa_torch_cpu.py <cikti.json>
"""
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F

B, H, D = 1, 8, 64


def median_ms(fn, warmup=1, reps=5):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main():
    out = {"phase": 5, "backend": f"torch {torch.__version__} sdpa cpu fp32", "threads": torch.get_num_threads(),
           "shape": {"batch": B, "heads": H, "head_dim": D}, "cases": []}
    g = torch.Generator().manual_seed(3)
    for T in (512, 1024, 2048, 4096):
        for causal in (False, True):
            q, k, v, do = (torch.randn(B, H, T, D, generator=g) for _ in range(4))
            qg, kg, vg = (x.clone().requires_grad_(True) for x in (q, k, v))

            def fwd():
                with torch.no_grad():
                    F.scaled_dot_product_attention(q, k, v, is_causal=causal)

            def fwd_bwd():
                o = F.scaled_dot_product_attention(qg, kg, vg, is_causal=causal)
                o.backward(do)
                qg.grad = kg.grad = vg.grad = None

            tf, tfb = median_ms(fwd), median_ms(fwd_bwd)
            print(f"  info  T={T:5d} {'causal' if causal else 'full':6s} fwd {tf:9.3f} ms  fwd+bwd {tfb:9.3f} ms")
            out["cases"].append({"T": T, "causal": causal, "fwd_ms": tf, "fwd_bwd_ms": tfb})
    out["protocol"] = {"warmup": 1, "repeats": 5, "stat": "median", "timing": "perf_counter"}
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
