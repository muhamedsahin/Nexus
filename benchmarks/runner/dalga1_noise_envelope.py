"""Dalga 1: 124M FP32 parity'de referansin kendi yuvarlama gurultu zarfi.

Soru: LR=6e-4'teki Nexus-PyTorch kayip farki (0.0437) bir hata mi, yoksa kaotik egitimin
1 ULP'lik herhangi bir farki buyutmesi mi? Deney:
  1. Nexus bir kez (baslangic agirliklari --dump-init ile yazilir).
  2. Deterministik PyTorch ayni agirliklarla (bozmasiz) ve N kez her parametresi rastgele
     isaretli 1 ULP oynatilmis agirliklarla.
  3. Adim adim: zarf(t) = max_k |torch_k(t) - torch_0(t)|, nexus(t) = |nexus(t) - torch_0(t)|.
Her LR icin ayri dizin; cikti envelope.json. Tolerans degistirilmez; yalniz olculur.

Kullanim (depo koku):
  python benchmarks/runner/dalga1_noise_envelope.py --exe build-phase1/benchmarks/runner/bench_cuda_phase20.exe
      --tokens build-phase1/faz20/tokens.bin --python .bench-venv-cu134/Scripts/python.exe
      --nanogpt ~/nexus-data/nanogpt --out docs/benchmarks/<yeni>/ --lr 6e-4 1e-4 --perturbations 4

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


def run(command, log, env, product=None):
    if product is not None and Path(product).is_file():  # yarim kalan kosudan devam: tamamlanmis cikti yeniden kullanilir
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
    ap.add_argument("--lr", type=float, nargs="+", default=[6e-4, 1e-4])
    ap.add_argument("--perturbations", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    ref_script = Path(__file__).with_name("faz9_nanogpt.py").resolve()
    summary = {}
    for lr in args.lr:
        out = root / f"lr{lr:g}"
        out.mkdir(exist_ok=True)
        init, native = out / "initial.bin", out / "nexus.jsonl"
        def native_cmd(path):
            return [str(Path(args.exe).resolve()), "--model", "124m", "--tokens", str(Path(args.tokens).resolve()), "--out", str(path),
                    "--steps", str(args.steps), "--warmup", "10", "--seq", "1024", "--precision", "f32", "--lr", str(lr),
                    "--dump-init", str(init)]
        if not native.is_file():
            run(native_cmd(native), out / "native.log", env)
        elif not init.is_file():
            # baslangic dosyasi silinmis (yer): Nexus yeni yola tekrar kosar; egri bit bit ayni olmali
            k = 1
            while (out / f"nexus-regen{k}.jsonl").exists():
                k += 1
            regen = out / f"nexus-regen{k}.jsonl"
            run(native_cmd(regen), out / f"native-regen{k}.log", env)
            if [r["loss"] for r in read_run(regen)[1]] != [r["loss"] for r in read_run(native)[1]]:
                raise RuntimeError("Nexus tekrari bit bit ayni degil: " + str(regen))
        curves = []
        for seed in range(args.perturbations + 1):
            ref = out / f"torch-ulp{seed}.json"
            run([str(Path(args.python).resolve()), str(ref_script), "curve", "--nanogpt", str(Path(args.nanogpt).resolve()),
                 "--init", str(init), "--batches", str(Path(args.tokens).resolve()), "--micro", "1", "--seq", "1024",
                 "--accum", "1", "--steps", str(args.steps), "--warmup", "10", "--lr", str(lr), "--precision", "f32",
                 "--deterministic", "1", "--perturb-ulp", str(seed), "--out", str(ref)], out / f"torch-ulp{seed}.log", env, ref)
            curves.append(json.loads(ref.read_text(encoding="utf-8"))["losses"])
        _, steps = read_run(native)
        nexus = [r["loss"] for r in steps]
        base = curves[0]
        envelope = [max(abs(c[t] - base[t]) for c in curves[1:]) for t in range(args.steps)]
        nexus_err = [abs(nexus[t] - base[t]) for t in range(args.steps)]
        # Nexus, bozulmus torch kosularinin her birine gore de: referansin kendi dagilimiyla ayni boyda mi?
        nexus_vs_each = [max(abs(nexus[t] - c[t]) for c in curves) for t in range(args.steps)]
        ratio = [n / e if e > 0 else (0.0 if n == 0 else None) for n, e in zip(nexus_err, envelope)]  # None: zarf 0
        torch_meta = json.loads((out / "torch-ulp0.json").read_text(encoding="utf-8"))
        result = dict(lr=lr, steps=args.steps, perturbations=args.perturbations,
                      torch_version=torch_meta["framework"], deterministic=torch_meta.get("deterministic_algorithms"),
                      initial_sha256=digest(init), tokens_sha256=digest(Path(args.tokens)),
                      executable_sha256=digest(Path(args.exe)), reference_script_sha256=digest(ref_script),
                      envelope_max=max(envelope), envelope_worst_step=envelope.index(max(envelope)),
                      nexus_max=max(nexus_err), nexus_worst_step=nexus_err.index(max(nexus_err)),
                      nexus_first10_max=max(nexus_err[:10]), envelope_first10_max=max(envelope[:10]),
                      steps_nexus_exceeds_envelope=sum(n > e for n, e in zip(nexus_err, envelope)),
                      envelope=envelope, nexus_error=nexus_err, nexus_vs_each_max=nexus_vs_each, ratio=ratio)
        write_json(out / "envelope.json", result)
        (init).unlink()  # ~498 MB; SHA kayitli
        summary[f"{lr:g}"] = {k: v for k, v in result.items() if not isinstance(v, list)}
        print(json.dumps(summary[f"{lr:g}"]), flush=True)
    write_json(root / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
