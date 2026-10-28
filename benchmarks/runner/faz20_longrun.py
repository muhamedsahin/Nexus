"""Measured endurance with periodic NXS3 checkpoints and fresh-process continuation probes.

Run in the foreground. A short test cannot satisfy the >24h acceptance gate.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import argparse
import datetime
from pathlib import Path
import subprocess
import time
from faz20_validate import digest, read_run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", choices=["smoke", "124m", "350m"], default="124m")
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--min-seconds", type=float, default=86401)
    args = parser.parse_args()
    if not (2 <= args.seq <= 1024 and args.checkpoint_every >= 4 and args.min_seconds > 0
            and args.max_steps > args.checkpoint_every + 2):
        parser.error("invalid duration, sequence or checkpoint interval")
    exe, tokens = Path(args.exe).resolve(), Path(args.tokens).resolve()
    if tokens.stat().st_size < args.max_steps * args.seq * 4:
        parser.error("not enough real tokens for max-steps; prepare a larger corpus slice")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = out / "checkpoint"
    report = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "executable_sha256": digest(exe), "tokens_sha256": digest(tokens),
              "requested_seconds": args.min_seconds, "active_training_seconds": 0,
              "resume_probes": [], "passed_24h": False, "completed_requested_duration": False,
              "status": "running"}
    start = time.monotonic()
    tail = None
    try:
        for boundary in range(args.checkpoint_every, args.max_steps - 1, args.checkpoint_every):
            stop = boundary + 2
            raw = out / f"until-{stop}.jsonl"
            command = [str(exe), "--model", args.model, "--tokens", str(tokens), "--out", str(raw),
                       "--seq", str(args.seq), "--steps", str(args.max_steps), "--stop-after", str(stop),
                       "--warmup", "2", "--lr", "0.0001", "--precision", "bf16",
                       "--checkpoint-dir", str(checkpoint), "--checkpoint-every", str(args.checkpoint_every)]
            if tail is not None:
                command += ["--resume", str(checkpoint)]
            with (out / f"until-{stop}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"training failed with code {result.returncode}; inspect {raw.with_suffix('.log')}")
            _, rows = read_run(raw)
            if rows[-1]["step"] != stop - 1:
                raise RuntimeError("incomplete native training run")
            if tail is not None:
                if any((a["step"], a["loss"], a["lr"]) != (b["step"], b["loss"], b["lr"])
                       for a, b in zip(rows[:2], tail, strict=True)):
                    raise RuntimeError("fresh-process resume differs from uninterrupted checkpoint continuation")
                report["resume_probes"].append({"step": rows[0]["step"], "exact": True,
                                                 "raw_sha256": digest(raw)})
            # Count useful unique training time; exclude the two replayed probe steps.
            unique_rows = rows[2:] if tail is not None else rows
            report["active_training_seconds"] += sum(r["ms"] for r in unique_rows) / 1000
            tail = rows[-2:]
            report["wall_seconds"] = time.monotonic() - start
            report["last_step"] = stop
            report["completed_requested_duration"] = (report["active_training_seconds"] >= args.min_seconds
                                                       and bool(report["resume_probes"]))
            report["passed_24h"] = report["completed_requested_duration"] and report["active_training_seconds"] > 86400
            write_json(out / "endurance.json", report)
            print(f"step={stop} active_s={report['active_training_seconds']:.1f} exact_resume_probes={len(report['resume_probes'])}", flush=True)
            if report["completed_requested_duration"]:
                report["status"] = "completed"
                break
        else:
            raise RuntimeError("max-steps exhausted before requested duration")
    except BaseException as exc:
        report["status"] = "failed_or_interrupted"
        report["error"] = str(exc)
        write_json(out / "endurance.json", report)
        raise
    write_json(out / "endurance.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
