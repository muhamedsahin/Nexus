"""Run the Phase 20 single-GPU scale matrix sequentially, with independent repetitions.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import argparse
import datetime
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
from faz20_validate import digest, summarize, write_json


def capture(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 10 or args.repeats < 5:
        parser.error("at least 10 steps (5 warmup) and 5 independent repetitions required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    exe, tokens = Path(args.exe).resolve(), Path(args.tokens).resolve()
    metadata = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "platform": platform.platform(), "executable_sha256": digest(exe),
                "tokens_sha256": digest(tokens), "steps": args.steps, "repeats": args.repeats,
                "hardware": capture(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,power.limit", "--format=csv"]),
                "git": capture(["git", "rev-parse", "HEAD"]),
                "source_sha256": {str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name("bench_cuda_phase20.cpp"),
                                                             Path(__file__).with_name("faz20_validate.py")]}}
    write_json(out / "metadata.json", metadata)
    results = []
    for model, optimizer in [("124m", "adamw"), ("350m", "adamw"), ("1.3b", "fused8")]:
        runs = []
        for repeat in range(args.repeats):
            tag = f"{model}-{repeat}"
            raw = out / (tag + ".jsonl")
            command = [str(exe), "--model", model, "--tokens", str(tokens), "--out", str(raw),
                       "--steps", str(args.steps), "--warmup", "5", "--optimizer", optimizer,
                       "--precision", "bf16", "--seq", "1024", "--micro", "1"]
            start = time.monotonic()
            with (out / (tag + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
            run = {"command": command, "returncode": process.returncode, "elapsed_s": time.monotonic() - start,
                   "telemetry": capture(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used", "--format=csv"])}
            if process.returncode == 0:
                run["summary"] = summarize(raw)
            runs.append(run)
            write_json(out / (tag + ".json"), run)
            print(tag, "exit", process.returncode, flush=True)
            if process.returncode:
                break  # Do not retry OOM five times or turn failure into a fast timing.
        entry = {"model": model, "optimizer": optimizer, "runs": runs,
                 "completed": len(runs) == args.repeats and all(r["returncode"] == 0 for r in runs)}
        if entry["completed"]:
            medians = [r["summary"]["median_ms"] for r in runs]
            entry["run_medians_ms"] = medians
            entry["median_ms"] = statistics.median(medians)
            entry["min_ms"] = min(medians)
            entry["max_ms"] = max(medians)
            entry["std_ms"] = statistics.pstdev(medians)
            entry["target_tokens_per_s"] = 1023 * 1000 / entry["median_ms"]
        results.append(entry)
        write_json(out / "matrix.json", results)
    return 0 if all(r["completed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
