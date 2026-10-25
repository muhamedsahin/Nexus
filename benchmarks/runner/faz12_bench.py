"""FAZ 12 olcum kosucusu: Nexus TP/SP/async (thread + tcp), PyTorch TP (CPU gloo), PP kabarcik, CP baglam taramasi.

Her yapilandirma `--repeats` kez kosar; JSONL satirlari docs/benchmarks/2026-10-25-faz12/ altina yazilir.
Kullanim: python faz12_bench.py --build build-phase2-cpu [--repeats 3] [--only tp,torch,pp,cp]

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "benchmarks", "2026-10-25-faz12")


def run_json(cmd, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    rows = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            rows.append(json.loads(line))
    if p.returncode != 0 or not rows:
        print("HATA:", " ".join(cmd), "\n", p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
    return rows


def write(name, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} satir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build-phase2-cpu")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="tp,tcp,torch,pp,cp")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    exe = os.path.join(ROOT, args.build, "tests", "parity", "test_phase12_parallel.exe")
    run = os.path.join(ROOT, args.build, "tools", "nexus_run", "nexus-run.exe")
    py = sys.executable
    only = set(args.only.split(","))

    if "tp" in only:
        rows = []
        for tp, sp, asy in [(1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1), (4, 0, 0), (4, 1, 0), (4, 1, 1)]:
            for rep in range(args.repeats):
                for r in run_json([exe, "bench_tp", str(tp), str(sp), str(asy), "8"]):
                    r["repeat"] = rep
                    rows.append(r)
        write("nexus_tp_thread.jsonl", rows)
    if "tcp" in only:
        rows = []
        port = 29700
        for tp, sp, asy in [(2, 1, 0), (2, 1, 1), (4, 1, 0), (4, 1, 1)]:
            for rep in range(args.repeats):
                port += 1
                for r in run_json([run, "--nproc-per-node", str(tp), "--master-port", str(port), "--", exe, "bench_tp", str(tp),
                                   str(sp), str(asy), "8"]):
                    r["repeat"] = rep
                    rows.append(r)
        write("nexus_tp_tcp.jsonl", rows)
    if "torch" in only:
        rows = []
        port = 29800
        for tp, sp, thr in [(1, 0, 1), (2, 0, 1), (2, 1, 1), (4, 0, 1), (4, 1, 1), (1, 0, 8), (2, 1, 4), (4, 1, 2)]:
            for rep in range(args.repeats):
                port += 1
                for r in run_json([py, os.path.join(ROOT, "benchmarks", "runner", "faz12_torch_tp.py"), "--tp", str(tp), "--sp",
                                   str(sp), "--threads", str(thr), "--port", str(port)]):
                    r["repeat"] = rep
                    rows.append(r)
        write("torch_tp.jsonl", rows)
    if "pp" in only:
        rows = []
        for sched, v in [("gpipe", 1), ("1f1b", 1), ("interleaved_1f1b", 2), ("zb_h1", 1)]:
            for m in (4, 8, 16):
                for rep in range(args.repeats):
                    for r in run_json([exe, "bench_pp", sched, str(v), str(m)]):
                        r["repeat"] = rep
                        rows.append(r)
        write("nexus_pp.jsonl", rows)
    if "cp" in only:
        rows = []
        for T in (8192, 16384, 32768, 65536, 131072):
            rows += run_json([exe, "bench_cp", str(T), "8", "ring", "zigzag"], timeout=7200)
        rows += run_json([exe, "bench_cp", "32768", "8", "ring", "contiguous"])
        for T in (8192, 16384, 32768):
            rows += run_json([exe, "bench_cp", str(T), "1", "ring", "zigzag"])
        write("nexus_cp.jsonl", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
