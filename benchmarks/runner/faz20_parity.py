"""Reproduce the full 124M / 100-step Phase 20 parity gate on identical real tokens.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import argparse
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
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe, tokens = Path(args.exe).resolve(), Path(args.tokens).resolve()
    init = out / "initial.bin"
    native = out / "nexus.jsonl"
    reference = out / "nanogpt.json"
    commands = [
        [str(exe), "--model", "124m", "--tokens", str(tokens), "--out", str(native),
         "--steps", "100", "--warmup", "10", "--seq", "1024", "--precision", "f32",
         "--lr", "0.0001", "--dump-init", str(init)],
        [str(Path(args.python).resolve()), str(Path(__file__).with_name("faz9_nanogpt.py").resolve()),
         "curve", "--nanogpt", str(Path(args.nanogpt).resolve()), "--init", str(init), "--batches", str(tokens),
         "--micro", "1", "--seq", "1024", "--accum", "1", "--steps", "100", "--warmup", "10",
         "--lr", "0.0001", "--precision", "f32", "--out", str(reference)]]
    metadata = {"executable_sha256": digest(exe), "tokens_sha256": digest(tokens), "commands": commands}
    write_json(out / "metadata.json", metadata)
    for name, command in zip(("native", "reference"), commands):
        with (out / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        if process.returncode:
            raise RuntimeError(f"{name} failed: {process.returncode}; see log")
    result = compare(native, reference)
    result["initial_weights_sha256"] = digest(init)
    write_json(out / "parity.json", result)
    print(result["passed"], "max_abs", result["max_abs"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
