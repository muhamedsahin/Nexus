"""FAZ 9 kabul: Nexus ve nanoGPT kayip egrilerini adim adim karsilastirir (ayni veri, ayni baslangic).

Kullanim: python faz9_compare_curves.py NEXUS_history.csv NANOGPT_curve.json [NEXUS2_history.csv] [--skip 10]
Cikti: adim basina goreli fark |a - b| / b istatistikleri (ham ve 10 adimlik kayan ortalama),
son 50 adimin ortalama kayiplari. Ucuncu dosya verilirse Nexus'un kendi kosudan kosuya farki da
(BF16 + atomik toplamalar: ayni kod iki kez) taban gurultu olarak yazilir.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import csv
import json
import sys


def nexus_losses(path):
    with open(path) as f:
        return [float(r["loss"]) for r in csv.DictReader(f)]


def smooth(v, w=10):
    out = []
    for i in range(len(v)):
        lo = max(0, i - w + 1)
        out.append(sum(v[lo:i + 1]) / (i + 1 - lo))
    return out


def stats(a, b, skip):
    n = min(len(a), len(b))
    rel = [abs(a[i] - b[i]) / b[i] for i in range(skip, n)]
    sa, sb = smooth(a[:n]), smooth(b[:n])
    srel = [abs(sa[i] - sb[i]) / sb[i] for i in range(skip, n)]
    tail = min(50, n)
    return {
        "steps": n,
        "skip": skip,
        "rel_mean": sum(rel) / len(rel),
        "rel_max": max(rel),
        "rel_max_step": skip + rel.index(max(rel)),
        "smooth10_rel_max": max(srel),
        "smooth10_rel_mean": sum(srel) / len(srel),
        "within_1pct_raw": sum(1 for r in rel if r <= 0.01) / len(rel),
        "tail50_mean_a": sum(a[n - tail:n]) / tail,
        "tail50_mean_b": sum(b[n - tail:n]) / tail,
        "first": [a[0], b[0]],
        "last": [a[n - 1], b[n - 1]],
    }


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    skip = 10
    if "--skip" in sys.argv:
        skip = int(sys.argv[sys.argv.index("--skip") + 1])
        args = [x for x in args if x != str(skip)]
    nexus = nexus_losses(args[0])
    ref = json.load(open(args[1]))["losses"]
    out = {"nexus_vs_nanogpt": stats(nexus, ref, skip)}
    if len(args) > 2:
        out["nexus_run_to_run"] = stats(nexus_losses(args[2]), nexus, skip)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
