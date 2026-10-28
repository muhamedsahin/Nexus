#!/usr/bin/env python3
"""Reproduce the Phase 19 backend GEMM measurements without overwriting historical data.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    exe = args.executable.resolve()
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["NEXUS_CPU_THREADS"] = str(args.threads)

    def run(command):
        try:
            result = subprocess.run(command, cwd=root, env=env, text=True, encoding="utf-8",
                                    errors="replace", capture_output=True, timeout=300)
        except FileNotFoundError as error:
            return {"command": [str(x) for x in command], "returncode": 127,
                    "stdout": "", "stderr": str(error)}
        return {"command": [str(x) for x in command], "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}

    measured = run([str(exe), "bench"])
    if measured["returncode"]:
        raise RuntimeError(measured)
    rows = [json.loads(line) for line in measured["stdout"].splitlines() if line.startswith('{')]
    if not rows:
        raise RuntimeError("No backend measurement rows were produced")
    for row in rows:
        samples = row["samples_ms"]
        if len(samples) < 5 or not all(s > 0 for s in samples):
            raise RuntimeError("At least five positive measurements are required")
        if abs(statistics.median(samples) - row["median_ms"]) > 1e-6:
            raise RuntimeError("Median differs from raw samples")
        if abs(statistics.pstdev(samples) - row["std_ms"]) > 1e-6:
            raise RuntimeError("Standard deviation differs from raw samples")
    report = {
        "schema_version": 1, "phase": 19,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "python": platform.python_version(),
        "cpu_threads": args.threads, "synchronization": "driver.synchronize after each GEMM",
        "statistic": "median (population standard deviation)",
        "inputs": {"generator": "((i * seed + 13) % 97 - 48) * scale",
                   "seed_a": 7, "seed_b": 13, "scale": 0.02},
        "executable": str(exe), "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("src/backend/registry.cpp", "src/backend/pjrt.cpp", "src/backend/pjrt_plugin.cpp", "src/backend/vulkan.cpp",
                         "src/backend/directml.cpp", "src/backend/hip.cpp", "src/backend/metal.cpp",
                         "tests/parity/test_phase19_backends.cpp", "benchmarks/runner/faz19_audit.py")
        },
        "git": run(["git", "rev-parse", "HEAD"]),
        "working_tree": run(["git", "status", "--short"]),
        "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "measurements": rows, "raw_run": measured,
        "limitations": ["CPU reference interpreter is not XLA or a TPU measurement",
                        "GEMM timings do not establish language-model token/s or competitor superiority"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "measured_backends": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
