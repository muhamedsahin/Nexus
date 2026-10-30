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


def f32_ulp(x):
    """FP32 kayip degerinin bir ULP'si (kayip FP32 olarak yazilir; daha kucuk fark temsil edilemez)."""
    x = abs(x)
    return 2.0 ** (math.frexp(x)[1] - 24) if x > 0 else 2.0 ** -149


def ensemble_compare(nexus_path, reference_paths, count=100, early_steps=10, early_tolerance=1e-5, z_limit=3.0,
                     tail=20, ulp_floor=2):
    """Dalga 1: kaotik (yuksek LR) egitimde noktasal <=1e-3 referansin kendine karsi bile saglanamaz
    (124M FP32 LR=6e-4: 1 ULP bozulmus 13 PyTorch kosusunun 78 ciftinin hicbiri gecmez). Bu kapi
    referansin KENDI yuvarlama toplulugunu olcut alir; noktasal compare() yerine gecmez, yanina eklenir.

    reference_paths[0] bozmasiz referans, digerleri ayni tarifin 1 ULP bozulmus tekrarlari (>= 4).
    Nexus topluluga bir uye daha gibi sinanir. Gecme sartlarinin HEPSI:
      1. ilk early_steps adim: |nexus - ref0| <= early_tolerance (kaos baslamadan once yuvarlama duzeyi);
      2. uye skoru: max_t max_k |nexus - ref_k| <= referans uyelerinin en buyuk skoru
         (uye j: max_t max_{k!=j} |ref_j - ref_k|), taban ulp_floor x FP32 ULP;
      3. son tail adimin ortalama kaybi: |z| <= z_limit; sd tabani ulp_floor x FP32 ULP.
    v1 tasarimi (adim adim zarf, tabansiz) reddedildi: LR=1e-4'te 1 ULP'lik kayip farkini ve 14. uyenin
    dogal asimini hata sayiyordu (docs/benchmarks/2026-10-31-dalga1). steps_outside_envelope bilgi icindir.
    """
    if len(reference_paths) < 5:
        raise ValueError("ensemble parity needs the unperturbed reference and at least four perturbed repeats")
    if count < early_steps or count < tail or tail < 2 or early_steps < 1 or ulp_floor < 0:
        raise ValueError("invalid ensemble parity requirements")
    config, steps = read_run(nexus_path)
    if len(steps) != count or steps[0]["step"] != 0:
        raise ValueError("ensemble parity requires exactly the requested steps from zero")
    nexus = [row["loss"] for row in steps]
    curves = []
    for path in reference_paths:
        losses = json.loads(Path(path).read_text(encoding="utf-8"))["losses"]
        if len(losses) != count or any(not math.isfinite(v) for v in losses):
            raise ValueError(f"reference curve invalid: {path}")
        curves.append(losses)
    if any(not math.isfinite(v) for v in nexus):
        raise ValueError("nonfinite Nexus loss")
    early = max(abs(nexus[t] - curves[0][t]) for t in range(early_steps))
    floor = ulp_floor * max(f32_ulp(v) for c in curves for v in c)
    pairs = [(a, b) for i, a in enumerate(curves) for b in curves[i + 1:]]
    envelope = [max(abs(a[t] - b[t]) for a, b in pairs) for t in range(count)]
    nexus_spread = [max(abs(nexus[t] - c[t]) for c in curves) for t in range(count)]
    outside = [t for t in range(count) if nexus_spread[t] > max(envelope[t], floor)]
    member_scores = [max(abs(curves[j][t] - curves[k][t]) for t in range(count) for k in range(len(curves)) if k != j)
                     for j in range(len(curves))]
    nexus_score = max(nexus_spread)
    score_limit = max(max(member_scores), floor)
    tails = [statistics.mean(c[-tail:]) for c in curves]
    mean, sd = statistics.mean(tails), statistics.stdev(tails)
    nexus_tail = statistics.mean(nexus[-tail:])
    sd_eff = max(sd, ulp_floor * f32_ulp(mean))
    z = (nexus_tail - mean) / sd_eff
    pointwise_pairs_passing = sum(max(abs(a[t] - b[t]) for t in range(count)) <= 1e-3 for a, b in pairs)
    checks = dict(early=early <= early_tolerance, member_score=nexus_score <= score_limit, tail=abs(z) <= z_limit)
    return {"passed": all(checks.values()), "design": "v2", "checks": checks, "steps": count, "references": len(curves),
            "early_steps": early_steps, "early_tolerance": early_tolerance, "early_max_abs": early,
            "ulp_floor": ulp_floor, "floor_abs": floor,
            "nexus_score": nexus_score, "reference_member_scores": member_scores, "score_limit": score_limit,
            "nexus_score_rank": sum(s < nexus_score for s in member_scores),
            "envelope_max": max(envelope), "steps_outside_envelope": outside,
            "tail_steps": tail, "tail_nexus": nexus_tail, "tail_reference_mean": mean, "tail_reference_sd": sd,
            "tail_sd_effective": sd_eff, "tail_z": z, "z_limit": z_limit,
            "reference_pairs": len(pairs), "reference_pairs_passing_pointwise_1e-3": pointwise_pairs_passing,
            "envelope": envelope, "nexus_spread": nexus_spread,
            "nexus_sha256": digest(nexus_path), "reference_sha256": [digest(p) for p in reference_paths],
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
