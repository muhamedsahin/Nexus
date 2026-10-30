"""Dalga 1: FAZ 21 BF16 parity farki (0.0021, LR=1e-4) sistematik mi, hangi taraf FP32'ye yakin?

Ayni baslangic agirliklari ve tokenlarla (FAZ 21 tarifi: 15 adim, LR=1e-4, warmup 2):
  nexus-f32, nexus-bf16 (FAZ 21 gibi --compile 1), torch-f32, torch-bf16 (autocast; bozmasiz + N bozmali
  topluluk, deterministik) ve torch-bf16-compile (max-autotune, FAZ 21 referansi).
Olcu: her BF16 egrisinin ayni uygulamanin FP32 egrisine ve torch-f32'ye uzakligi; torch-bf16 toplulugunun
kendi yayilmasi. Tolerans degistirilmez; yalniz olculur.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi 1. sinif
ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli olarak gelistirilmistir.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from faz20_validate import digest, read_run, write_json  # noqa: E402


def run(command, log, env, product):
    if Path(product).is_file():
        return
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=3600)
    if p.returncode:
        raise RuntimeError(f"basarisiz ({p.returncode}): {log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--nanogpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--perturbations", type=int, default=4)
    ap.add_argument("--torch-compile", type=int, choices=[0, 1], default=1)
    ap.add_argument("--init-from", help="baslangic agirligi silinmisse (yer) ayni SHA'li kopya; SHA ciktiya yazilir")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    exe, tokens = str(Path(args.exe).resolve()), str(Path(args.tokens).resolve())
    init = out / "initial.bin"
    common = ["--steps", str(args.steps), "--warmup", str(args.warmup), "--seq", "1024", "--lr", str(args.lr)]
    nexus = {}
    for prec, extra in (("f32", ["--dump-init", str(init)]), ("bf16", ["--compile", "1"])):
        path = out / f"nexus-{prec}.jsonl"
        run([exe, "--model", "124m", "--tokens", tokens, "--out", str(path), "--precision", prec,
             "--optimizer", "adamw"] + common + extra, out / f"nexus-{prec}.log", env, path)
        nexus[prec] = [r["loss"] for r in read_run(path)[1]]
    if not init.is_file():
        if not args.init_from:
            raise SystemExit("initial.bin yok: --init-from ile ayni SHA'li baslangic verin")
        import shutil
        shutil.copyfile(args.init_from, init)
    ref_script = str(Path(__file__).with_name("faz9_nanogpt.py").resolve())
    base = [str(Path(args.python).resolve()), ref_script, "curve", "--nanogpt", str(Path(args.nanogpt).resolve()),
            "--init", str(init), "--batches", tokens, "--micro", "1", "--accum", "1"] + common

    def torch_curve(name, extra):
        path = out / f"{name}.json"
        run(base + extra + ["--out", str(path)], out / f"{name}.log", env, path)
        return json.loads(path.read_text(encoding="utf-8"))["losses"]

    tf32 = torch_curve("torch-f32", ["--precision", "f32", "--deterministic", "1"])
    tbf = [torch_curve(f"torch-bf16-ulp{k}", ["--precision", "bf16", "--deterministic", "1", "--perturb-ulp", str(k)])
           for k in range(args.perturbations + 1)]
    curves = dict(nexus_f32=nexus["f32"], nexus_bf16=nexus["bf16"], torch_f32=tf32, torch_bf16=tbf[0])
    if args.torch_compile:
        curves["torch_bf16_compile"] = torch_curve("torch-bf16-compile", ["--precision", "bf16", "--compile", "1",
                                                                          "--compile-mode", "max-autotune"])

    def dist(a, b):
        e = [abs(x - y) for x, y in zip(a, b, strict=True)]
        return dict(max_abs=max(e), step0=e[0], errors=e)

    result = dict(lr=args.lr, steps=args.steps, perturbations=args.perturbations,
                  initial_sha256=digest(init), tokens_sha256=digest(Path(tokens)), executable_sha256=digest(Path(exe)),
                  reference_script_sha256=digest(Path(ref_script)),
                  fp32_parity=dist(nexus["f32"], tf32),
                  nexus_bf16_vs_own_f32=dist(nexus["bf16"], nexus["f32"]),
                  nexus_bf16_vs_torch_f32=dist(nexus["bf16"], tf32),
                  torch_bf16_vs_torch_f32=dist(tbf[0], tf32),
                  nexus_bf16_vs_torch_bf16=dist(nexus["bf16"], tbf[0]),
                  torch_bf16_ensemble_max=max(max(abs(x - y) for x, y in zip(c, tbf[0])) for c in tbf[1:]))
    if "torch_bf16_compile" in curves:
        result["torch_bf16_compile_vs_torch_f32"] = dist(curves["torch_bf16_compile"], tf32)
        result["nexus_bf16_vs_torch_bf16_compile"] = dist(nexus["bf16"], curves["torch_bf16_compile"])
        result["torch_bf16_compile_vs_eager"] = dist(curves["torch_bf16_compile"], tbf[0])
    result["curves"] = curves
    write_json(out / "bf16-accuracy.json", result)
    init.unlink()
    print(json.dumps({k: (v["max_abs"] if isinstance(v, dict) and "max_abs" in v else v)
                      for k, v in result.items() if k != "curves"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
