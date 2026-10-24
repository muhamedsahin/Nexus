"""FAZ 0.E parity karsilastirmasi: Nexus ve PyTorch kayip/gradyanlari.

Metrikler (promt.txt 0.5): max_abs, max_rel, kosinus benzerligi, ULP histogrami.
  max_rel (tensor)  = max|a - b| / max|b|   (norm-tabanli; sifira yakin tek elemanin
                                             bolumu sismesin diye)
  gurultu tabani    = max|b| < 1e-6 * (tum gradyanlarin max|b|) olan tensörler analitik
                      olarak ~0'dir (or. dikkat key biasi: softmax satir kaymasina
                      degismez). Bunlarda bolum anlamsizdir; max|a-b| / global olcek
                      kullanilir ve raporda "noise_floor": true ile isaretlenir.
  ulp               = |a - b| / ulp(b)       float32 ULP cinsinden, kovalara ayrilir
Rapor <klasor>/parity_report.json'a yazilir. Esikler gecerse 0, aksi halde 1.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import argparse
import json
import os
import sys

import numpy as np

ULP_BUCKETS = [0, 1, 4, 16, 64, 256, 1024]


def ulp_histogram(a: np.ndarray, b: np.ndarray) -> dict:
    spacing = np.spacing(np.abs(b).astype(np.float32)).astype(np.float64)
    spacing[spacing == 0] = np.finfo(np.float32).tiny
    ulps = np.abs(a.astype(np.float64) - b.astype(np.float64)) / spacing
    hist = {}
    previous = -1.0
    for edge in ULP_BUCKETS:
        hist[f"<= {edge}"] = int(np.count_nonzero((ulps > previous) & (ulps <= edge)))
        previous = edge
    hist[f"> {ULP_BUCKETS[-1]}"] = int(np.count_nonzero(ulps > ULP_BUCKETS[-1]))
    return hist


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--loss-rtol", type=float, default=1e-5)
    parser.add_argument("--grad-rtol", type=float, default=1e-4)
    args = parser.parse_args()

    with open(os.path.join(args.folder, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(os.path.join(args.folder, "torch_out.json"), encoding="utf-8") as handle:
        torch_out = json.load(handle)
    nexus_grads = np.fromfile(os.path.join(args.folder, "nexus_grads.bin"), dtype=np.float32)
    torch_grads = np.fromfile(os.path.join(args.folder, "torch_grads.bin"), dtype=np.float32)
    if nexus_grads.size != torch_grads.size:
        print("  FAIL  gradyan dosya boyutlari farkli")
        return 1

    nexus_loss = float(manifest["nexus_loss"])
    torch_loss = float(torch_out["loss"])
    loss_rel = abs(nexus_loss - torch_loss) / max(abs(torch_loss), 1e-12)

    report = {
        "loss": {"nexus": nexus_loss, "torch": torch_loss, "max_rel": loss_rel, "rtol": args.loss_rtol},
        "grad_rtol": args.grad_rtol,
        "torch_version": torch_out.get("torch"),
        "params": [],
    }
    ok = loss_rel < args.loss_rtol
    worst_name, worst_rel = "", 0.0
    global_scale = float(np.max(np.abs(torch_grads))) if torch_grads.size else 0.0
    noise_floor_count = 0
    offset = 0
    for entry in manifest["params"]:
        count = int(np.prod(entry["shape"]))
        a = nexus_grads[offset : offset + count]
        b = torch_grads[offset : offset + count]
        offset += count
        scale = float(np.max(np.abs(b))) if count else 0.0
        max_abs = float(np.max(np.abs(a - b))) if count else 0.0
        noise_floor = scale < 1e-6 * global_scale
        if noise_floor:
            noise_floor_count += 1
            max_rel = max_abs / global_scale if global_scale > 0 else max_abs
        else:
            max_rel = max_abs / scale if scale > 0 else max_abs
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = float(np.dot(a.astype(np.float64), b.astype(np.float64)) / denom) if denom > 0 else 1.0
        report["params"].append(
            {
                "name": entry["name"],
                "shape": entry["shape"],
                "max_abs": max_abs,
                "max_rel": max_rel,
                "cosine": cosine,
                "noise_floor": noise_floor,
                "ulp_histogram": ulp_histogram(a, b),
            }
        )
        if max_rel > worst_rel:
            worst_name, worst_rel = entry["name"], max_rel
        ok = ok and max_rel < args.grad_rtol
    report["worst"] = {"name": worst_name, "max_rel": worst_rel}
    report["global_grad_scale"] = global_scale
    report["noise_floor_tensors"] = [p["name"] for p in report["params"] if p["noise_floor"]]
    report["passed"] = ok

    with open(os.path.join(args.folder, "parity_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    min_cos = min((p["cosine"] for p in report["params"] if not p["noise_floor"]), default=1.0)
    print(
        f"  info  loss nexus={nexus_loss:.8f} torch={torch_loss:.8f} max_rel={loss_rel:.3e} (esik {args.loss_rtol:g})"
    )
    print(
        f"  info  gradyan en kotu max_rel={worst_rel:.3e} ({worst_name}), min kosinus={min_cos:.9f} (esik {args.grad_rtol:g})"
    )
    if noise_floor_count:
        print(
            f"  info  {noise_floor_count} tensör gurultu tabaninda (analitik ~0): "
            + ", ".join(report["noise_floor_tensors"])
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
