"""Phase 20 evidence: bounded real-token extraction, finite strict parity, scale metrics.

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract(manifest_path, out, count, offset=0):
    """Contiguous GPT-2 tokens with EOS boundaries; never re-tokenize or invent data."""
    if count < 1 or offset < 0:
        raise ValueError("count must be positive and offset nonnegative")
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["version"] != 1 or manifest["tokenizer"]["name"] != "gpt2":
        raise ValueError("expected NXTK v1 / GPT-2 manifest")
    remaining, skip, sources = count, offset, []
    output = Path(out)
    if output.exists():
        raise ValueError("token output already exists")
    with output.open("xb") as target:
        for shard in manifest["shards"]:
            if skip >= shard["tokens"]:
                skip -= shard["tokens"]
                continue
            source = (path.parent / shard["path"]).resolve()
            if source == output.resolve():
                raise ValueError("source and output collide")
            actual = digest(source)
            if actual != shard["sha256"]:
                raise ValueError("shard SHA256 mismatch: " + str(source))
            with source.open("rb") as stream:
                header = stream.read(64)
                magic, version, width, reserved, tokens, docs, tokenizer = struct.unpack("<4sIIIQQ32s", header)
                if (magic != b"NXTK" or version != 1 or width not in (2, 4) or reserved
                        or tokens != shard["tokens"] or docs != shard["docs"]
                        or tokenizer.hex() != manifest["tokenizer"]["digest"]):
                    raise ValueError("invalid shard header")
                take = min(remaining, tokens - skip)
                stream.seek(64 + skip * width)
                left = take
                while left:
                    chunk = min(left, 262144)
                    raw = stream.read(chunk * width)
                    if len(raw) != chunk * width:
                        raise ValueError("truncated shard")
                    values = array.array("H" if width == 2 else "I", raw)
                    if sys.byteorder != "little":
                        values.byteswap()
                    if max(values) >= manifest["tokenizer"]["vocab_size"]:
                        raise ValueError("token out of vocabulary")
                    encoded = array.array("i", values)
                    if sys.byteorder != "little":
                        encoded.byteswap()
                    target.write(encoded.tobytes())
                    left -= chunk
                sources.append({"path": str(source), "sha256": actual, "offset": skip, "tokens": take})
                remaining -= take
                skip = 0
            if not remaining:
                break
        if remaining:
            raise ValueError("not enough source tokens")
    result = {"tokenizer": manifest["tokenizer"], "manifest_sha256": digest(path),
              "sources": sources, "tokens": count, "sha256": digest(output),
              "format": "little-endian int32; consecutive EOS-separated corpus tokens"}
    write_json(str(output) + ".json", result)
    return result


def read_run(path):
    entries = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not entries or entries[0].get("type") != "config":
        raise ValueError("missing config")
    steps = entries[1:]
    if not steps:
        raise ValueError("empty run")
    for i, row in enumerate(steps):
        if row.get("type") != "step" or (i and row["step"] != steps[i-1]["step"] + 1):
            raise ValueError("non-contiguous step sequence")
        if any(not math.isfinite(row[k]) for k in ("loss", "ms", "lr")) or row["ms"] <= 0:
            raise ValueError("nonfinite run or nonpositive duration")
        if row["target_tokens"] <= 0:
            raise ValueError("invalid token count")
    return entries[0], steps


def compare(nexus_path, reference_path, count=100, tolerance=1e-3):
    if count < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid parity requirements")
    config, steps = read_run(nexus_path)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    expected = reference["losses"]
    if len(steps) != count or len(expected) != count or steps[0]["step"] != 0:
        raise ValueError("parity requires exactly the requested steps from zero")
    if any(not math.isfinite(v) for v in expected):
        raise ValueError("nonfinite reference")
    errors = [abs(row["loss"] - ref) for row, ref in zip(steps, expected)]
    return {"passed": max(errors) <= tolerance, "steps": count, "tolerance": tolerance,
            "max_abs": max(errors), "worst_step": errors.index(max(errors)),
            "mean_abs": statistics.mean(errors), "errors": errors,
            "nexus_sha256": digest(nexus_path), "reference_sha256": digest(reference_path),
            "model": config["model"], "precision": config["precision"]}


def summarize(path, warmup=5):
    config, steps = read_run(path)
    if warmup < 0 or len(steps) - warmup < 5:
        raise ValueError("need at least five measured steps after warmup")
    samples = steps[warmup:]
    times = [r["ms"] for r in samples]
    median = statistics.median(times)
    tokens = config["micro"] * (config["seq"] - 1)
    # Work is done for T input positions, but only T-1 are training targets.
    model_flops = config["micro"] * config["seq"] * (
        6 * (config["parameters"] - 1024 * config["dim"])
        + 12 * config["layers"] * config["dim"] * config["seq"])
    peak = config["peak_flops"]
    result = {"config": config, "warmup": warmup, "samples_ms": times,
              "median_ms": median, "min_ms": min(times), "max_ms": max(times),
              "std_ms": statistics.pstdev(times), "target_tokens_per_s": tokens * 1000 / median,
              "mfu_estimate": model_flops * 1000 / median / peak if peak else None,
              "hfu_full_recompute_estimate": model_flops * (4 / 3) * 1000 / median / peak if peak else None,
              "flops_method": "6*N_no_position + 12*L*d*T per input position; full recompute HFU = 4/3 MFU; excludes optimizer",
              "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in steps),
              "peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in steps),
              "max_inactive_split_fraction": max(r["inactive_split_bytes"] / r["reserved_bytes"] if r["reserved_bytes"] else 0 for r in steps),
              "first_loss": steps[0]["loss"], "last_loss": steps[-1]["loss"],
              "raw_sha256": digest(path)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--count", type=int, required=True)
    prepare.add_argument("--offset", type=int, default=0)
    prepare.add_argument("--out", required=True)
    parity = modes.add_parser("compare")
    parity.add_argument("--nexus", required=True)
    parity.add_argument("--reference", required=True)
    parity.add_argument("--steps", type=int, default=100)
    parity.add_argument("--tolerance", type=float, default=1e-3)
    parity.add_argument("--out", required=True)
    report = modes.add_parser("summarize")
    report.add_argument("--run", required=True)
    report.add_argument("--warmup", type=int, default=5)
    report.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        extract(args.manifest, args.out, args.count, args.offset)
        return 0
    result = (compare(args.nexus, args.reference, args.steps, args.tolerance) if args.mode == "compare"
              else summarize(args.run, args.warmup))
    write_json(args.out, result)
    return 1 if result.get("passed") is False else 0


if __name__ == "__main__":
    sys.exit(main())
