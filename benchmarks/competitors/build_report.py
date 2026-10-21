"""Faz JSON dosyalarindan RAPOR.md uretir. Sayi yalniz dosyadan gelir.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LIBRARIES = ("nexus", "numpy", "torch", "tensorflow")
LABELS = {
    "gemm_n256": "GEMM N=256",
    "gemm_n512": "GEMM N=512",
    "gemm_n1024": "GEMM N=1024",
    "gemm_n2048": "GEMM N=2048",
    "conv2d": "Conv2d",
    "depthwise": "Depthwise",
    "attention": "Attention",
    "lstm": "LSTM",
    "gru": "GRU",
    "mse": "MSE",
    "adam": "Adam",
}
ORDER = list(LABELS)
MISSING = "koşulmadı"
NO_GPU_WHEEL = "GPU tekeri yok, koşulmadı"
NO_NEXUS_GPU = "Nexus GPU yolu yok"
NO_SECOND = "ikinci ekosistem koşulmadı"
NEXUS_GPU_NAMES = {"gemm_n256", "gemm_n512", "gemm_n1024", "gemm_n2048"}


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def walk(doc, acc) -> None:
    if not isinstance(doc, dict):
        return
    status = doc.get("status")
    if status:
        acc["statuses"].append({"library": doc.get("library"), "device": doc.get("device"), "status": status})
    if "runs" in doc or "parts" in doc:
        for child in list(doc.get("runs", [])) + list(doc.get("parts", [])):
            walk(child, acc)
        return
    if isinstance(doc.get("rows"), list):
        acc["docs"].append(doc)


def index_rows(docs) -> dict:
    found = {}
    for doc in docs:
        for item in doc.get("rows", []):
            if not isinstance(item, dict) or "name" not in item or "seconds" not in item:
                continue
            key = (item["library"], item["device"], str(item["threads"]), item["name"])
            found[key] = item
    return found


def fmt(value) -> str:
    return format(value, ".8g")


def cell_text(item, phrase: str) -> str:
    if item is None:
        return phrase
    return f"{fmt(item['seconds'])} s · {fmt(item['value'])} {item['unit']}"


def ratio_text(nexus, other) -> str:
    if nexus is None or other is None:
        return MISSING
    if nexus["device"] != other["device"] or str(nexus["threads"]) != str(other["threads"]):
        return MISSING
    return fmt(other["seconds"] / nexus["seconds"])


def lookup(found, library, device, threads, name):
    return found.get((library, device, threads, name))


def table(found, device: str, threads: str, statuses) -> list[str]:
    lines = [
        "| iş | Nexus | NumPy | PyTorch | TensorFlow | NumPy/Nexus | PyTorch/Nexus | TensorFlow/Nexus |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ORDER:
        nexus = lookup(found, "nexus", device, threads, name)
        others = {library: lookup(found, library, device, threads, name) for library in LIBRARIES if library != "nexus"}
        nexus_phrase = MISSING
        if device == "gpu" and nexus is None and name not in NEXUS_GPU_NAMES:
            nexus_phrase = NO_NEXUS_GPU
        elif device == "gpu" and nexus is None:
            nexus_phrase = MISSING
        phrases = {}
        for library in ("numpy", "torch", "tensorflow"):
            phrases[library] = MISSING
            if others[library] is not None:
                continue
            if device == "gpu" and library in ("torch", "tensorflow") and any(
                item["status"] == NO_GPU_WHEEL and item.get("library") in (library, None) for item in statuses
            ):
                phrases[library] = NO_GPU_WHEEL
        lines.append(
            "| {name} | {nexus} | {numpy} | {torch} | {tf} | {rn} | {rt} | {rf} |".format(
                name=LABELS[name],
                nexus=cell_text(nexus, nexus_phrase),
                numpy=cell_text(others["numpy"], phrases["numpy"]),
                torch=cell_text(others["torch"], phrases["torch"]),
                tf=cell_text(others["tensorflow"], phrases["tensorflow"]),
                rn=ratio_text(nexus, others["numpy"]),
                rt=ratio_text(nexus, others["torch"]),
                rf=ratio_text(nexus, others["tensorflow"]),
            )
        )
    return lines


def losses(found) -> list[str]:
    lines = [
        "| iş | cihaz | iş parçacığı | rakip | Nexus saniye | rakip saniye | Nexus değer | rakip değer | oran |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    count = 0
    keys = sorted(found, key=lambda item: (item[1], item[2], ORDER.index(item[3]) if item[3] in ORDER else 99, item[0]))
    for library, device, threads, name in keys:
        if library == "nexus":
            continue
        other = found[(library, device, threads, name)]
        nexus = lookup(found, "nexus", device, threads, name)
        if nexus is None:
            continue
        if other["seconds"] > nexus["seconds"]:
            continue
        count += 1
        lines.append(
            "| {name} | {device} | {threads} | {library} | {ns} | {os} | {nv} {nu} | {ov} {ou} | {ratio} |".format(
                name=LABELS.get(name, name),
                device=device,
                threads=threads,
                library=library,
                ns=fmt(nexus["seconds"]),
                os=fmt(other["seconds"]),
                nv=fmt(nexus["value"]),
                nu=nexus["unit"],
                ov=fmt(other["value"]),
                ou=other["unit"],
                ratio=fmt(other["seconds"] / nexus["seconds"]),
            )
        )
    if count == 0:
        return ["Bu dosyalarda, aynı cihaz ve aynı iş parçacığı kolonunda Nexus'tan kısa süren rakip satırı yok.", ""]
    lines.append("")
    return lines


def blas_summary(text: str) -> str:
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines()[0] if text else ""
    blas = cfg.get("Build Dependencies", {}).get("blas", {})
    name = blas.get("name", "")
    version = blas.get("version", "")
    detail = blas.get("openblas configuration", "")
    return " ".join(part for part in (name, version, detail) if part)


def meta_lines(docs) -> list[str]:
    lines = []
    seen = set()

    def add(line: str) -> None:
        if line not in seen:
            seen.add(line)
            lines.append(line)

    for doc in docs:
        library = doc.get("library")
        threads = doc.get("threads")
        if doc.get("kernel"):
            add(f"- Nexus çekirdek: `{doc['kernel']}`")
        if "pool_workers" in doc:
            add(f"- Nexus havuz iş parçacığı: `{doc['pool_workers']}`")
        if doc.get("compiler"):
            add(f"- Derleyici: `{doc['compiler']}`")
        if doc.get("gpu_name"):
            add(f"- GPU adı: `{doc['gpu_name']}`")
        if "mkldnn" in doc:
            add(f"- PyTorch mkldnn ({doc.get('device')} / {threads}): `{doc['mkldnn']}`")
        if "observed_threads" in doc:
            add(f"- PyTorch get_num_threads ({threads}): `{doc['observed_threads']}`")
        if doc.get("blas_config"):
            add(f"- NumPy BLAS ({threads}): `{blas_summary(doc['blas_config'])}`")
        if doc.get("trace_note"):
            add(f"- TensorFlow: {doc['trace_note']}")
        if "intra_op" in doc:
            add(f"- TensorFlow intra_op ({threads}): `{doc['intra_op']}`")
        if doc.get("conv_format"):
            add(f"- TensorFlow conv düzeni ({threads}): `{doc['conv_format']}`")
        if doc.get("note") and library:
            add(f"- {library}: {doc['note']}")
        if doc.get("jax_version"):
            add(f"- JAX: `{doc['jax_version']}`")
        if doc.get("jax_devices"):
            add(f"- JAX cihazları: `{doc['jax_devices']}`")
    return lines


def env_lines(env) -> list[str]:
    if not env:
        return ["- Ortam dosyası yok.", ""]
    lines = [
        f"- Python: `{env.get('python')}`",
        f"- Yorumlayıcı: `{env.get('executable')}`",
        f"- Platform: `{env.get('platform')}`",
        f"- İşlemci: `{env.get('cpu_name') or env.get('processor')}`",
        f"- numpy: `{env.get('numpy')}`",
        f"- torch: `{env.get('torch')}`",
        f"- tensorflow: `{env.get('tensorflow')}`",
        f"- torch.cuda.is_available(): `{env.get('torch_cuda')}`",
        f"- tf GPU: `{env.get('tf_gpus')}`",
        "",
    ]
    if env.get("note"):
        lines.insert(-1, f"- Ortam: {env['note']}")
    return lines


def jax_table(found) -> list[str]:
    lines = [
        "| iş | iş parçacığı | JAX | Nexus | JAX/Nexus |",
        "| --- | --- | --- | --- | --- |",
    ]
    any_row = False
    for threads in ("8", "default", "device"):
        for name in ("gemm_n256", "gemm_n512", "gemm_n1024", "gemm_n2048", "conv2d", "adam"):
            other = lookup(found, "jax", "cpu" if threads != "device" else "gpu", threads, name)
            if other is None and threads == "device":
                other = lookup(found, "jax", "gpu", threads, name)
            if other is None:
                continue
            any_row = True
            device = other["device"]
            nexus = lookup(found, "nexus", device, "8" if device == "cpu" else "device", name)
            if device == "cpu" and threads != "8":
                nexus = None
            lines.append(
                f"| {LABELS.get(name, name)} | {threads} | {cell_text(other, MISSING)} | {cell_text(nexus, MISSING)} | {ratio_text(nexus, other)} |"
            )
    if not any_row:
        return [NO_SECOND, ""]
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    docs = Path(args.docs)
    files = {
        "env": docs / "phase-1-env.json",
        "nexus": docs / "phase-2-nexus.json",
        "numpy": docs / "phase-3-numpy.json",
        "torch": docs / "phase-4-pytorch-cpu.json",
        "tensorflow": docs / "phase-5-tensorflow-cpu.json",
        "gpu": docs / "phase-6-gpu.json",
        "extra": docs / "phase-7-extra.json",
    }
    protocol = json.loads((Path(__file__).resolve().parent / "protocol.json").read_text(encoding="utf-8"))
    acc = {"docs": [], "statuses": []}
    for path in files.values():
        walk(load(path), acc)
    found = index_rows(acc["docs"])
    env = load(files["env"]) or {}
    lines = [
        "# Rakip benchmark raporu",
        "",
        "Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.",
        "",
        "Sayılar bu klasördeki JSON dosyalarından üretildi.",
        "",
        "## Makine",
        "",
        *env_lines(env),
        *meta_lines(acc["docs"]),
        "",
        "## Sözleşme",
        "",
        f"- FP32. GEMM beta `{protocol['beta']}`. FLOP `2 N³`.",
        f"- GEMM N = `{protocol['gemm']['n']}`. Tekrar `{protocol['repeats']['gemm']}`, ısınma `{protocol['warmup']}`.",
        "- Conv2d NCHW 8×3×32×32, çekirdek 3, stride 1, padding 1, 16 çıkış.",
        "- Depthwise 8×32×16×16, çekirdek 3, padding 1.",
        "- Attention B4 T32 E64, 4 baş, dropout 0.",
        "- LSTM ve GRU: B4 T16, giriş 32, gizli 64, çıktı dizisi.",
        "- MSE 1048576 eleman, ortalama, gradyan dahil.",
        "- Adam 1048576 parametre, bir adım, momentler dahil.",
        f"- Katman ve optimizasyon tekrarı `{protocol['repeats']['layer']}`.",
        "- İş parçacığı kolonları `8` ve `default`. GPU kolonu `device`. Kolonlar karışmaz.",
        "- GPU'da ağırlık ve girdi ölçümden önce cihazda. TF32 kapalı. CPU sayısı GPU hücresine yazılmaz.",
        "",
        "## CPU, 8 iş parçacığı",
        "",
        "Oran = rakip süre / Nexus süre. 1'in üstü, Nexus'un daha kısa sürdüğüdür. Nexus havuz sayısı makine satırındadır; kolon etiketi protokoldeki 8'dir.",
        "",
        *table(found, "cpu", "8", acc["statuses"]),
        "",
        "## CPU, kütüphane varsayılanı",
        "",
        "Nexus'un ayrı bir varsayılan iş parçacığı koşusu yoktur. O hücre `koşulmadı` kalır. Oran, Nexus bu kolonda yoksa hesaplanmaz.",
        "",
        *table(found, "cpu", "default", acc["statuses"]),
        "",
        "## GPU",
        "",
        *table(found, "gpu", "device", acc["statuses"]),
        "",
    ]
    if acc["statuses"]:
        lines.append("Durum notları:")
        lines.append("")
        for item in acc["statuses"]:
            if not item.get("library"):
                continue
            lines.append(f"- `{item.get('library')}` / `{item.get('device')}`: {item['status']}")
        lines.append("")
    lines.extend(
        [
            "## İkinci ekosistem",
            "",
            *jax_table(found),
            "## Nexus'un kazanmadığı satırlar",
            "",
            "Aynı cihaz ve aynı iş parçacığı kolonu. Rakip süresi Nexus süresinden kısa ya da eşit.",
            "",
            *losses(found),
            "## Ham JSON",
            "",
        ]
    )
    for name, path in files.items():
        lines.append(f"- `{name}`: `{path.as_posix()}`")
    lines.append(f"- sözleşme: `{(Path(__file__).resolve().parent / 'protocol.json').as_posix()}`")
    lines.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
