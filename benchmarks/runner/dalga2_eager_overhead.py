"""Dalga 2: kucuk tensörde eager islem ek yuku, PyTorch CPU (test_tensor_core bench ile ayni sekiller / protokol).

1000 isinma, 7 x 10000 cagri, medyan us / islem. add: yeni cikti ayirir; add_out: torch.add(out=); view: .view.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""
import json
import statistics
import time

import torch


def per_call_us(fn):
    for _ in range(1000):
        fn()
    samples = []
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(10000):
            fn()
        samples.append((time.perf_counter() - t0) * 1e6 / 10000)
    return statistics.median(samples)


def main():
    torch.set_num_threads(8)
    g = torch.Generator().manual_seed(1)
    for n in (1, 64, 4096):
        a = torch.randn(n, generator=g)
        b = a.clone()
        out = torch.empty(n)
        with torch.inference_mode():
            add = per_call_us(lambda: torch.add(a, b))
            add_out = per_call_us(lambda: torch.add(a, b, out=out))
            view = per_call_us(lambda: a.view(1, n))
        print(json.dumps({"mode": "torch_eager_add", "torch": torch.__version__, "n": n, "add_us": round(add, 3),
                          "add_out_us": round(add_out, 3), "view_us": round(view, 3)}), flush=True)


if __name__ == "__main__":
    main()
