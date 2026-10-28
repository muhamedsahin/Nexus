"""Separate-process CUDA full-state checkpoint proof (short run, never a 24h claim).
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    Path(args.work).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase20-", dir=args.work) as scratch:
        root = Path(scratch)
        tokens = root / "tokens.bin"
        tokens.write_bytes(struct.pack("<512i", *[(i * 17 + 3) % 128 for i in range(512)]))
        base = [str(Path(args.exe).resolve()), "--model", "smoke", "--tokens", str(tokens),
                "--steps", "8", "--warmup", "2", "--seq", "32", "--precision", "bf16"]

        def run(name, extra=()):
            result = subprocess.run(base + ["--out", str(root / (name + ".jsonl"))] + list(extra),
                                    capture_output=True, text=True, timeout=180)
            print(result.stdout, result.stderr, end="")
            if result.returncode == 77:
                return None
            if result.returncode:
                raise RuntimeError("training failed: " + name)
            return [json.loads(line) for line in (root / (name + ".jsonl")).read_text().splitlines()][1:]

        continuous = run("continuous")
        if continuous is None:
            return 77
        first = run("first", ["--stop-after", "4", "--save", str(root / "checkpoint")])
        resumed = run("resumed", ["--resume", str(root / "checkpoint")])
        for actual, expected in zip(first + resumed, continuous, strict=True):
            if any(actual[k] != expected[k] for k in ("step", "loss", "lr")):
                raise AssertionError((actual, expected))
        changed = subprocess.run(base + ["--out", str(root / "changed.jsonl"), "--lr", "0.0001",
                                 "--resume", str(root / "checkpoint")], capture_output=True, text=True)
        if changed.returncode != 1 or "recipe mismatch" not in changed.stderr:
            raise AssertionError("changed recipe accepted")
        values = bytearray(tokens.read_bytes())
        values[0] ^= 1
        tokens.write_bytes(values)
        changed = subprocess.run(base + ["--out", str(root / "changed-data.jsonl"),
                                 "--resume", str(root / "checkpoint")], capture_output=True, text=True)
        if changed.returncode != 1 or "token SHA256 mismatch" not in changed.stderr:
            raise AssertionError("changed token stream accepted")
        for extra in (["--seq", "1"], ["--steps", "x"], ["--bogus", "1"]):
            result = subprocess.run(base + ["--out", str(root / "invalid.jsonl")] + extra, capture_output=True)
            if result.returncode != 1:
                raise AssertionError("invalid option accepted")
        print("PASS: BF16 8 steps == 4 + full checkpoint + fresh process + 4 (exact losses/LR); invalid options rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
