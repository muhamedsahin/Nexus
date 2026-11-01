"""Dalga 3.1: Nexus GEMM vs PyTorch, donusumlu tekrarlar (GPU saat kaymasi iki tarafa esit dagilsin).

Kullanim:
  python dalga3_gemm_compare.py --nexus build-phase1/benchmarks/runner/dalga3_gemm.exe \
      --torch-python .bench-venv-cu134/Scripts/python.exe --bt 1024 --repeats 5 --out DIR
Her tekrar: Nexus (--epilogue 0, sezgi + autotune) ve PyTorch (torch.matmul eager + CUDA graf) ayri surec, sira tekrar
basina yer degistirir. Sekil basina tekrar medyanlari alinir; oran = torch_ms / nexus_ms (>1 Nexus hizli).
Ayrica fwd.fc epilog karsilastirmasi: Nexus --epilogue 1 (BiasGelu + aux, tek cagri) vs PyTorch eager
addmm + gelu(tanh) (--fused 1). Sonuc DIR/compare-bt{bt}.json ve ekrana Markdown tablo.
Guc: her surecten once ve her tekrar sonunda AC kontrolu (BatteryStatus == 2); degilse acik hata, sonuc yazilmaz.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def power_status():
    out = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                         capture_output=True, text=True).stdout.strip()
    return out


def run_nexus(exe, bt, epilogue, path):
    subprocess.run([exe, "--bt", str(bt), "--epilogue", str(epilogue), "--out", str(path)], check=True,
                   capture_output=True)
    return json.loads(path.read_text())


def run_torch(py, bt, fused, path, graph=0):
    subprocess.run([py, "-I", str(HERE / "dalga3_gemm_torch.py"), "--bt", str(bt), "--fused", str(fused), "--graph",
                    str(graph), "--out", str(path)], check=True, capture_output=True)
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nexus", required=True)
    ap.add_argument("--torch-python", required=True)
    ap.add_argument("--bt", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if power_status() != "2":
        sys.exit("dalga3_gemm_compare: dizustu AC'de degil (BatteryStatus != 2); olcum yapilmadi")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    nexus_runs, torch_runs, torch_graph, nexus_epi, torch_fused = [], [], [], [], []
    torch_meta = None
    power_log = []
    for r in range(args.repeats):
        order = ["nexus", "torch"] if r % 2 == 0 else ["torch", "nexus"]
        for who in order:
            # Her olcumden once ve sonra AC kontrolu: pil gucunde GPU saati dusup sonucu bozar.
            if power_status() != "2":
                sys.exit(f"dalga3_gemm_compare: tekrar {r}, {who} oncesi AC kesildi; sonuc yazilmadi")
            if who == "nexus":
                nexus_runs.append(run_nexus(args.nexus, args.bt, 0, raw / f"nexus-bt{args.bt}-{r}.json"))
                nexus_epi.append(run_nexus(args.nexus, args.bt, 1, raw / f"nexus-epi-bt{args.bt}-{r}.json"))
            else:
                t = run_torch(args.torch_python, args.bt, 0, raw / f"torch-bt{args.bt}-{r}.json")
                torch_runs.append(t)
                torch_meta = {k: v for k, v in t.items() if k != "shapes"}
                torch_fused.append(run_torch(args.torch_python, args.bt, 1, raw / f"torch-fused-bt{args.bt}-{r}.json"))
                torch_graph.append(run_torch(args.torch_python, args.bt, 0, raw / f"torch-graph-bt{args.bt}-{r}.json", 1))
        if power_status() != "2":
            sys.exit(f"dalga3_gemm_compare: tekrar {r} sirasinda AC kesildi; sonuc yazilmadi")
        power_log.append("AC")
        print(f"tekrar {r + 1}/{args.repeats} tamam", file=sys.stderr)

    def med(runs, name, field):
        return statistics.median(next(s[field] for s in run["shapes"] if s["name"] == name) for run in runs)

    def spread(runs, name, field):
        v = [next(s[field] for s in run["shapes"] if s["name"] == name) for run in runs]
        return (max(v) - min(v)) / statistics.median(v)

    rows = []
    for s in nexus_runs[0]["shapes"]:
        n = s["name"]
        h, a, t = med(nexus_runs, n, "heuristic_ms"), med(nexus_runs, n, "autotune_ms"), med(torch_runs, n, "torch_ms")
        tg = med(torch_graph, n, "torch_ms")
        rows.append(dict(name=n, m=s["m"], n=s["n"], k=s["k"], nexus_heuristic_ms=h, nexus_autotune_ms=a, torch_ms=t,
                         torch_graph_ms=tg, ratio_heuristic=t / h, ratio_autotune=t / a, ratio_graph=tg / h,
                         flop=2.0 * s["m"] * s["n"] * s["k"],
                         spread_nexus_heuristic=spread(nexus_runs, n, "heuristic_ms"),
                         spread_torch=spread(torch_runs, n, "torch_ms")))
    fc_epi = med(nexus_epi, "fwd.fc", "heuristic_ms")
    fc_epi_tuned = med(nexus_epi, "fwd.fc", "autotune_ms")
    fc_split = med(nexus_epi, "fwd.fc", "split_ms")
    fc_torch = med(torch_fused, "fwd.fc", "torch_ms")
    tot = lambda key: sum(r[key] for r in rows)
    summary = dict(sum_nexus_heuristic_ms=tot("nexus_heuristic_ms"), sum_nexus_autotune_ms=tot("nexus_autotune_ms"),
                   sum_torch_ms=tot("torch_ms"), sum_torch_graph_ms=tot("torch_graph_ms"),
                   shapes_nexus_faster_than_torch_graph=sum(r["ratio_graph"] >= 1.0 for r in rows),
                   shapes_nexus_heuristic_faster=sum(r["ratio_heuristic"] >= 1.0 for r in rows),
                   shapes_nexus_autotune_faster=sum(r["ratio_autotune"] >= 1.0 for r in rows),
                   fused_fc=dict(nexus_bias_gelu_aux_ms=fc_epi, nexus_bias_gelu_aux_autotune_ms=fc_epi_tuned,
                                 nexus_bias_then_gelu_kernel_ms=fc_split, torch_addmm_gelu_ms=fc_torch,
                                 ratio=fc_torch / min(fc_epi, fc_epi_tuned), ratio_split=fc_torch / fc_split))
    res = dict(bt=args.bt, repeats=args.repeats, power_each_repeat=power_log, torch=torch_meta, rows=rows, summary=summary,
               note="oran = torch_ms / nexus_ms; >1 Nexus hizli. spread = (max-min)/medyan tekrarlar arasi.")
    (out / f"compare-bt{args.bt}.json").write_text(json.dumps(res, indent=1))
    print("| sekil | m | n | k | Nexus sezgi ms | Nexus autotune ms | PyTorch eager ms | PyTorch graf ms | "
          "oran eager (sezgi) | oran graf (sezgi) | oran eager (autotune) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['name']} | {r['m']} | {r['n']} | {r['k']} | {r['nexus_heuristic_ms']:.4f} | "
              f"{r['nexus_autotune_ms']:.4f} | {r['torch_ms']:.4f} | {r['torch_graph_ms']:.4f} | "
              f"{r['ratio_heuristic']:.3f} | {r['ratio_graph']:.3f} | {r['ratio_autotune']:.3f} |")
    sh = summary['sum_nexus_heuristic_ms']
    print(f"| **toplam** | | | | {sh:.3f} | {summary['sum_nexus_autotune_ms']:.3f} | {summary['sum_torch_ms']:.3f} | "
          f"{summary['sum_torch_graph_ms']:.3f} | {summary['sum_torch_ms'] / sh:.3f} | "
          f"{summary['sum_torch_graph_ms'] / sh:.3f} | {summary['sum_torch_ms'] / summary['sum_nexus_autotune_ms']:.3f} |")
    f = summary["fused_fc"]
    print(f"\nfwd.fc epilog: Nexus BiasGelu+aux {f['nexus_bias_gelu_aux_ms']:.4f} ms (autotune "
          f"{f['nexus_bias_gelu_aux_autotune_ms']:.4f}) -> {f['ratio']:.3f}x | Nexus Bias GEMM + GELU cekirdegi "
          f"{f['nexus_bias_then_gelu_kernel_ms']:.4f} ms -> {f['ratio_split']:.3f}x | PyTorch addmm+gelu "
          f"{f['torch_addmm_gelu_ms']:.4f} ms")


if __name__ == "__main__":
    main()
