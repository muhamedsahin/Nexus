"""Reproduce the full 124M / 100-step Phase 20 parity gate on identical real tokens.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import argparse
import math
import os
from pathlib import Path
import subprocess
from faz20_validate import compare, digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--python", required=True, help="Python containing CUDA PyTorch and numpy")
    parser.add_argument("--nanogpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lr",type=float,default=.0001)
    parser.add_argument("--deterministic",type=int,choices=[0,1],default=1)
    args = parser.parse_args()
    if not math.isfinite(args.lr) or not 0 < args.lr <= 1:
        parser.error('--lr must be finite and in (0, 1]')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe, tokens = Path(args.exe).resolve(), Path(args.tokens).resolve()
    init = out / "initial.bin"
    native = out / "nexus.jsonl"
    reference = out / "nanogpt.json"
    commands = [
        [str(exe), "--model", "124m", "--tokens", str(tokens), "--out", str(native),
         "--steps", "100", "--warmup", "10", "--seq", "1024", "--precision", "f32",
         "--lr", str(args.lr), "--dump-init", str(init)],
        [str(Path(args.python).resolve()), str(Path(__file__).with_name("faz9_nanogpt.py").resolve()),
         "curve", "--nanogpt", str(Path(args.nanogpt).resolve()), "--init", str(init), "--batches", str(tokens),
         "--micro", "1", "--seq", "1024", "--accum", "1", "--steps", "100", "--warmup", "10",
         "--lr", str(args.lr), "--precision", "f32", "--deterministic",str(args.deterministic),"--out", str(reference)]]
    env = dict(os.environ)
    if args.deterministic:
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    metadata = {"executable_sha256": digest(exe), "tokens_sha256": digest(tokens), "commands": commands,
                "reference_script_sha256":digest(Path(__file__).with_name('faz9_nanogpt.py')),
                "deterministic":bool(args.deterministic),
                "cublas_workspace_config":env.get('CUBLAS_WORKSPACE_CONFIG')}
    write_json(out / "metadata.json", metadata)
    for name, command in zip(("native", "reference"), commands):
        with (out / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600,env=env)
        if process.returncode:
            raise RuntimeError(f"{name} failed: {process.returncode}; see log")
    result = compare(native, reference)
    result["initial_weights_sha256"] = digest(init)
    write_json(out / "parity.json", result)
    print(result["passed"], "max_abs", result["max_abs"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
