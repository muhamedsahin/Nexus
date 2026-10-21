"""Rakip protokol kosucusu. Stdout'a yazmaz; JSON dosyaya gider.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)


def load_protocol_repeats(name: str, device: str) -> int:
    if device == "gpu":
        return int(PROTOCOL["gpu_repeats"])
    if name.startswith("gemm"):
        return int(PROTOCOL["repeats"]["gemm"])
    return int(PROTOCOL["repeats"]["layer"])


def measure(fn, warmup: int, repeats: int, sync=None):
    out = None
    for _ in range(warmup):
        out = fn()
        if sync is not None:
            sync(out)
    start = time.perf_counter()
    for _ in range(repeats):
        out = fn()
        if sync is not None:
            sync(out)
    seconds = (time.perf_counter() - start) / float(repeats)
    return seconds, out


def row(library: str, device: str, threads: str, name: str, seconds: float, value: float, unit: str):
    return {
        "library": library,
        "device": device,
        "threads": threads,
        "name": name,
        "seconds": seconds,
        "value": value,
        "unit": unit,
    }


def accept(rows, item, probe) -> None:
    if probe is None or not math.isfinite(probe):
        print(f"dusen satir {item['name']}", file=sys.stderr)
        return
    if not math.isfinite(item["seconds"]) or item["seconds"] <= 0.0 or not math.isfinite(item["value"]):
        print(f"dusen satir {item['name']}", file=sys.stderr)
        return
    rows.append(item)


def blas_config() -> str:
    import numpy as np

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cfg = np.show_config()
    if isinstance(cfg, dict):
        return json.dumps(cfg, ensure_ascii=False, default=str, indent=2)
    if cfg is not None:
        return str(cfg)
    return buffer.getvalue()


def numpy_run(device: str, threads: str) -> dict:
    if device != "cpu":
        return {
            "library": "numpy",
            "device": device,
            "threads": threads,
            "status": "GPU tekeri yok, koşulmadı",
            "rows": [],
        }
    import numpy as np

    rows = []
    for n in PROTOCOL["gemm"]["n"]:
        matrix_a = np.full((n, n), np.float32(0.01), dtype=np.float32)
        matrix_b = np.full((n, n), np.float32(0.02), dtype=np.float32)
        repeats = load_protocol_repeats("gemm", device)

        def step(left=matrix_a, right=matrix_b):
            return np.matmul(left, right)

        seconds, out = measure(step, int(PROTOCOL["warmup"]), repeats)
        flops = 2.0 * n * n * n
        probe = float(np.reshape(out, -1)[0])
        accept(rows, row("numpy", device, threads, f"gemm_n{n}", seconds, flops / seconds / 1.0e9, "GFLOPS"), probe)
    return {
        "library": "numpy",
        "device": device,
        "threads": threads,
        "blas_config": blas_config(),
        "rows": rows,
    }


def torch_sync(device: str):
    if device != "gpu":
        return None
    import torch

    def sync(_out):
        torch.cuda.synchronize()

    return sync


def torch_run(device: str, threads: str) -> dict:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cuda = bool(torch.cuda.is_available())
    if device == "gpu" and not cuda:
        return {
            "library": "torch",
            "device": device,
            "threads": "device",
            "status": "GPU tekeri yok, koşulmadı",
            "torch_cuda": False,
            "rows": [],
        }
    if device == "cpu" and threads == "8":
        torch.set_num_threads(8)
    meta_threads = "device" if device == "gpu" else threads

    def place(tensor):
        if device == "gpu":
            return tensor.cuda()
        return tensor

    def place_module(module):
        if device == "gpu":
            return module.cuda()
        return module

    sync = torch_sync(device)
    rows = []
    mkldnn = bool(getattr(torch.backends.mkldnn, "enabled", False))

    def add_call(name: str, fn, unit: str, value_of):
        repeats = load_protocol_repeats(name, device)
        try:
            seconds, out = measure(fn, int(PROTOCOL["warmup"]), repeats, sync)
            probe = float(out.detach().reshape(-1)[0].cpu())
            accept(rows, row("torch", device, meta_threads, name, seconds, value_of(seconds), unit), probe)
        except Exception as exc:
            print(f"dusen satir {name}: {exc}", file=sys.stderr)

    for n in PROTOCOL["gemm"]["n"]:
        left = place(torch.full((n, n), 0.01))
        right = place(torch.full((n, n), 0.02))

        def step(a=left, b=right):
            return torch.mm(a, b)

        flops = 2.0 * n * n * n
        add_call(f"gemm_n{n}", step, "GFLOPS", lambda seconds, flops=flops: flops / seconds / 1.0e9)

    conv = place_module(torch.nn.Conv2d(3, 16, 3, stride=1, padding=1))
    image = place(torch.full((8, 3, 32, 32), 0.02))
    conv_flops = 2.0 * 8 * 16 * 32 * 32 * 3 * 3 * 3

    def conv_step(layer=conv, data=image):
        return layer(data)

    add_call("conv2d", conv_step, "GFLOPS", lambda seconds: conv_flops / seconds / 1.0e9)

    depth = place_module(torch.nn.Conv2d(32, 32, 3, padding=1, groups=32))
    field = place(torch.full((8, 32, 16, 16), 0.02))
    depth_flops = 2.0 * 8 * 32 * 16 * 16 * 9

    def depth_step(layer=depth, data=field):
        return layer(data)

    add_call("depthwise", depth_step, "GFLOPS", lambda seconds: depth_flops / seconds / 1.0e9)

    attention = place_module(torch.nn.MultiheadAttention(64, 4, dropout=0.0, batch_first=True))
    tokens = place(torch.full((4, 32, 64), 0.01))

    def attn_step(layer=attention, data=tokens):
        output, _weights = layer(data, data, data)
        return output

    add_call("attention", attn_step, "cagri/s", lambda seconds: 1.0 / seconds)

    lstm = place_module(torch.nn.LSTM(32, 64, batch_first=True))
    gru = place_module(torch.nn.GRU(32, 64, batch_first=True))
    sequence = place(torch.full((4, 16, 32), 0.01))

    def lstm_step(layer=lstm, data=sequence):
        output, _state = layer(data)
        return output

    def gru_step(layer=gru, data=sequence):
        output, _state = layer(data)
        return output

    add_call("lstm", lstm_step, "cagri/s", lambda seconds: 1.0 / seconds)
    add_call("gru", gru_step, "cagri/s", lambda seconds: 1.0 / seconds)

    pred = place(torch.full((1048576,), 0.2, requires_grad=True))
    target = place(torch.full((1048576,), 0.1))
    mse_bytes = 1048576.0 * 3.0 * 4.0

    def mse_step(predicted=pred, truth=target):
        if predicted.grad is not None:
            predicted.grad = None
        loss = torch.nn.functional.mse_loss(predicted, truth, reduction="mean")
        loss.backward()
        return predicted.grad

    add_call("mse", mse_step, "GB/s", lambda seconds: mse_bytes / seconds / 1.0e9)

    param = torch.nn.Parameter(place(torch.full((1048576,), 0.1)))
    param.grad = place(torch.full((1048576,), 0.01))
    opt = torch.optim.Adam([param], lr=1.0e-3)
    adam_bytes = 1048576.0 * 4.0 * 4.0

    def adam_step(optimizer=opt, weights=param):
        optimizer.step()
        return weights

    add_call("adam", adam_step, "GB/s", lambda seconds: adam_bytes / seconds / 1.0e9)

    return {
        "library": "torch",
        "device": device,
        "threads": meta_threads,
        "observed_threads": int(torch.get_num_threads()),
        "mkldnn": mkldnn,
        "allow_tf32": False,
        "torch_cuda": cuda,
        "note": "Ileri cagri varsayilan eager moddadir. mkldnn kapatilmadi. TF32 kapali.",
        "rows": rows,
    }


def tf_sync(device: str):
    if device != "gpu":
        return None

    def sync(out):
        import tensorflow as tf

        flat = tf.reshape(out, [-1])
        _ = float(flat[0].numpy())

    return sync


def tensorflow_run(device: str, threads: str) -> dict:
    import tensorflow as tf

    try:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    except Exception as exc:
        print(f"tf32: {exc}", file=sys.stderr)
    gpus = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception as exc:
            print(f"cpu gorunurluk: {exc}", file=sys.stderr)
    elif not gpus:
        return {
            "library": "tensorflow",
            "device": device,
            "threads": "device",
            "status": "GPU tekeri yok, koşulmadı",
            "tf_gpus": [],
            "rows": [],
        }
    else:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"memory growth: {exc}", file=sys.stderr)
    if device == "cpu" and threads == "8":
        try:
            tf.config.threading.set_intra_op_parallelism_threads(8)
        except Exception as exc:
            print(f"intra op: {exc}", file=sys.stderr)
    meta_threads = "device" if device == "gpu" else threads
    import numpy as np

    rows = []
    formats = []
    sync = tf_sync(device)

    def add_call(name: str, fn, unit: str, value_of):
        repeats = load_protocol_repeats(name, device)
        try:
            seconds, out = measure(fn, int(PROTOCOL["warmup"]), repeats, sync)
            probe = float(tf.reshape(out, [-1])[0].numpy())
            accept(
                rows,
                row("tensorflow", device, meta_threads, name, seconds, value_of(seconds), unit),
                probe,
            )
        except Exception as exc:
            print(f"dusen satir {name}: {exc}", file=sys.stderr)

    for n in PROTOCOL["gemm"]["n"]:
        left = tf.constant(np.full((n, n), 0.01, np.float32))
        right = tf.constant(np.full((n, n), 0.02, np.float32))

        @tf.function
        def step(a=left, b=right):
            return tf.linalg.matmul(a, b)

        flops = 2.0 * n * n * n
        add_call(f"gemm_n{n}", step, "GFLOPS", lambda seconds, flops=flops: flops / seconds / 1.0e9)

    def conv_pair(channels_in: int, channels_out: int, shape, groups: int, name: str, flops: float):
        source = tf.constant(np.full(shape, 0.02, np.float32))
        data = source
        used = "channels_first"
        try:
            layer = tf.keras.layers.Conv2D(
                channels_out,
                3,
                strides=1,
                padding="same",
                data_format="channels_first",
                groups=groups,
                dtype="float32",
            )
            _ = layer(source)
        except Exception:
            data = tf.transpose(source, [0, 2, 3, 1])
            layer = tf.keras.layers.Conv2D(
                channels_out,
                3,
                strides=1,
                padding="same",
                data_format="channels_last",
                groups=groups,
                dtype="float32",
            )
            _ = layer(data)
            used = "channels_last"
        formats.append(f"{name}:{used}")

        @tf.function
        def step():
            return layer(data)

        add_call(name, step, "GFLOPS", lambda seconds, flops=flops: flops / seconds / 1.0e9)

    conv_pair(3, 16, (8, 3, 32, 32), 1, "conv2d", 2.0 * 8 * 16 * 32 * 32 * 3 * 3 * 3)
    conv_pair(32, 32, (8, 32, 16, 16), 32, "depthwise", 2.0 * 8 * 32 * 16 * 16 * 9)

    tokens = tf.constant(np.full((4, 32, 64), 0.01, np.float32))
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.0, dtype="float32")

    @tf.function
    def attn_step():
        return attention(tokens, tokens)

    add_call("attention", attn_step, "cagri/s", lambda seconds: 1.0 / seconds)

    sequence = tf.constant(np.full((4, 16, 32), 0.01, np.float32))
    lstm = tf.keras.layers.LSTM(64, return_sequences=True, dtype="float32")
    gru = tf.keras.layers.GRU(64, return_sequences=True, dtype="float32")

    @tf.function
    def lstm_step():
        return lstm(sequence)

    @tf.function
    def gru_step():
        return gru(sequence)

    add_call("lstm", lstm_step, "cagri/s", lambda seconds: 1.0 / seconds)
    add_call("gru", gru_step, "cagri/s", lambda seconds: 1.0 / seconds)

    pred = tf.Variable(tf.constant(np.full((1048576,), 0.2, np.float32)))
    target = tf.constant(np.full((1048576,), 0.1, np.float32))
    mse_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def mse_step():
        with tf.GradientTape() as tape:
            loss = mse_fn(target, pred)
        return tape.gradient(loss, pred)

    mse_bytes = 1048576.0 * 3.0 * 4.0
    add_call("mse", mse_step, "GB/s", lambda seconds: mse_bytes / seconds / 1.0e9)

    weights = tf.Variable(tf.constant(np.full((1048576,), 0.1, np.float32)))
    grad = tf.constant(np.full((1048576,), 0.01, np.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)

    @tf.function
    def adam_step():
        optimizer.apply_gradients([(grad, weights)])
        return weights

    adam_bytes = 1048576.0 * 4.0 * 4.0
    add_call("adam", adam_step, "GB/s", lambda seconds: adam_bytes / seconds / 1.0e9)

    try:
        intra = int(tf.config.threading.get_intra_op_parallelism_threads())
    except Exception:
        intra = None
    return {
        "library": "tensorflow",
        "device": device,
        "threads": meta_threads,
        "intra_op": intra,
        "tf_gpus": [gpu.name for gpu in gpus],
        "conv_format": formats,
        "tf32": False,
        "trace_note": "tf.function izi isinma cagrisindadir, tekrar ortalamasina girmez. Nexus her cagrida iz cikarmaz. oneDNN varsayilan birakildi.",
        "rows": rows,
    }


def jax_run(device: str, threads: str) -> dict:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
    except Exception as exc:
        return {
            "library": "jax",
            "device": device,
            "threads": threads,
            "status": "ikinci ekosistem koşulmadı",
            "error": str(exc),
            "rows": [],
        }
    devices = [str(item) for item in jax.devices()]
    gpu_devices = [item for item in devices if "gpu" in item.lower() or "cuda" in item.lower()]
    if device == "gpu" and not gpu_devices:
        return {
            "library": "jax",
            "device": device,
            "threads": "device",
            "status": "GPU tekeri yok, koşulmadı",
            "jax_devices": devices,
            "rows": [],
        }
    rows = []

    def add_call(name: str, fn, unit: str, value_of):
        repeats = load_protocol_repeats(name, device)
        try:
            seconds, out = measure(lambda: fn(), 1, repeats, lambda value: value.block_until_ready())
            probe = float(out.reshape(-1)[0])
            accept(rows, row("jax", "cpu" if device != "gpu" else "gpu", threads if device != "gpu" else "device", name, seconds, value_of(seconds), unit), probe)
        except Exception as exc:
            print(f"dusen satir {name}: {exc}", file=sys.stderr)

    for n in PROTOCOL["gemm"]["n"]:
        left = jnp.full((n, n), 0.01, dtype=jnp.float32)
        right = jnp.full((n, n), 0.02, dtype=jnp.float32)

        @jit
        def mm(a, b):
            return a @ b

        flops = 2.0 * n * n * n
        add_call(f"gemm_n{n}", lambda op=mm, a=left, b=right: op(a, b), "GFLOPS", lambda seconds, flops=flops: flops / seconds / 1.0e9)

    image = jnp.full((8, 3, 32, 32), 0.02, dtype=jnp.float32)
    kernel = jnp.full((16, 3, 3, 3), 0.01, dtype=jnp.float32)

    @jit
    def conv(x, w):
        return jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )

    conv_flops = 2.0 * 8 * 16 * 32 * 32 * 3 * 3 * 3
    add_call("conv2d", lambda op=conv, x=image, w=kernel: op(x, w), "GFLOPS", lambda seconds: conv_flops / seconds / 1.0e9)

    params = jnp.full((1048576,), 0.1, dtype=jnp.float32)
    moment = jnp.zeros((1048576,), dtype=jnp.float32)
    second = jnp.zeros((1048576,), dtype=jnp.float32)
    grad = jnp.full((1048576,), 0.01, dtype=jnp.float32)

    @jit
    def adam(p, m, v, g):
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * (g * g)
        mhat = m / 0.1
        vhat = v / 0.001
        p = p - 1.0e-3 * mhat / (jnp.sqrt(vhat) + 1.0e-8)
        return p, m, v

    state = (params, moment, second)
    adam_bytes = 1048576.0 * 4.0 * 4.0

    def adam_once(slot={"state": state}, op=adam, g=grad):
        slot["state"] = op(*slot["state"], g)
        return slot["state"][0]

    add_call("adam", adam_once, "GB/s", lambda seconds: adam_bytes / seconds / 1.0e9)
    return {
        "library": "jax",
        "device": "cpu" if device != "gpu" else "gpu",
        "threads": threads if device != "gpu" else "device",
        "jax_version": importlib.metadata.version("jax"),
        "jax_devices": devices,
        "note": "GEMM, Conv2d ve Adam. Bias duzeltmesi t=1 sabit, momentler adimlar arasinda tasinir.",
        "rows": rows,
    }


RUNNERS = {
    "numpy": numpy_run,
    "torch": torch_run,
    "tensorflow": tensorflow_run,
    "jax": jax_run,
}


def missing(library: str, device: str, threads: str, message: str) -> dict:
    return {
        "library": library,
        "device": device,
        "threads": threads,
        "status": message,
        "rows": [],
    }


def run_selected(library: str, device: str, threads: str) -> dict:
    try:
        return RUNNERS[library](device, threads)
    except Exception as exc:
        print(f"{library}: {exc}", file=sys.stderr)
        status = "ikinci ekosistem koşulmadı" if library == "jax" else "koşulmadı"
        return missing(library, device, threads, status)


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_docs(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    library = runs[0].get("library") if runs else ""
    device = runs[0].get("device") if runs else ""
    return {"library": library, "device": device, "runs": runs}


def numpy_markdown(doc: dict, path: Path) -> None:
    lines = [
        "# Faz 3 — NumPy",
        "",
        "Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.",
        "",
        "Yalnız GEMM ölçüldü. Conv, dikkat, LSTM, GRU, MSE ve Adam bu fazda yok.",
        "",
    ]
    runs = doc.get("runs", [doc])
    for run in runs:
        lines.append(f"## İş parçacığı: {run.get('threads')}")
        lines.append("")
        if run.get("status"):
            lines.append(run["status"])
            lines.append("")
            continue
        lines.append("BLAS (`numpy.show_config`):")
        lines.append("")
        lines.append("```")
        lines.append(run.get("blas_config", ""))
        lines.append("```")
        lines.append("")
        lines.append("| iş | saniye | değer | birim |")
        lines.append("| --- | --- | --- | --- |")
        for item in run.get("rows", []):
            lines.append(
                f"| {item['name']} | {item['seconds']:.9g} | {item['value']:.9g} | {item['unit']} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def probe(out: Path, markdown: Path | None) -> None:
    doc = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "note": "py -3.12 yok. Sanal ortam Anaconda Python 3.11. Anaconda python.exe yanındaki VCRUNTIME140 resmi CPU tekerlerinin DLL başlatmasını düşürdü. .bench-venv/host aynı yorumlayıcıyı sistem CRT ile çalıştırır.",
    }
    for package in ("numpy", "torch", "tensorflow"):
        try:
            doc[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            doc[package] = None
    try:
        import torch

        doc["torch_cuda"] = bool(torch.cuda.is_available())
        doc["torch_runtime"] = torch.__version__
    except Exception as exc:
        doc["torch_cuda"] = None
        doc["torch_error"] = str(exc)
    try:
        import tensorflow as tf

        doc["tf_runtime"] = tf.__version__
        doc["tf_gpus"] = [item.name for item in tf.config.list_physical_devices("GPU")]
    except Exception as exc:
        doc["tf_gpus"] = None
        doc["tf_error"] = str(exc)
    write_json(out, doc)
    if markdown is None:
        return
    def cell(value):
        return "kurulamadı" if value is None else str(value)

    lines = [
        "# Faz 1 — Ortam",
        "",
        "Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.",
        "",
        "ESP-IDF Python kullanılmadı. `py -3.12` bu makinede yok. Sanal ortam Anaconda Python 3.11 ile açıldı.",
        "Anaconda `python.exe` yanındaki VCRUNTIME140, resmi CPU tekerlerinin DLL başlatmasını düşürdü. `.bench-venv/host` aynı yorumlayıcıyı sistem CRT ile çalıştırır ve `pyvenv.cfg` bu dizini gösterir.",
        "",
        f"- Python: `{doc['python']}`",
        f"- Yorumlayıcı: `{doc['executable']}`",
        f"- Platform: `{doc['platform']}`",
        f"- İşlemci: `{doc['processor']}`",
        f"- numpy: `{cell(doc.get('numpy'))}`",
        f"- torch: `{cell(doc.get('torch'))}`",
        f"- tensorflow: `{cell(doc.get('tensorflow'))}`",
        f"- torch.cuda.is_available(): `{doc.get('torch_cuda')}`",
        f"- tf.config.list_physical_devices('GPU'): `{doc.get('tf_gpus')}`",
        "",
    ]
    if doc.get("torch_error"):
        lines.append(f"- torch içe aktarma: `{doc['torch_error']}`")
    if doc.get("tf_error"):
        lines.append(f"- tensorflow içe aktarma: `{doc['tf_error']}`")
    lines.append("")
    lines.append("Sürümler bu sanal ortamdan okundu.")
    lines.append("")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines), encoding="utf-8")


def relaunch_with_threads(threads: str, library: str) -> int:
    env = os.environ.copy()
    for key in THREAD_KEYS:
        env.pop(key, None)
    env.pop("XLA_FLAGS", None)
    if threads == "8":
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = "8"
        if library == "tensorflow":
            env["TF_NUM_INTRAOP_THREADS"] = "8"
        if library == "jax":
            env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
    env["NEXUS_BENCH_READY"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["PYTHONWARNINGS"] = "ignore"
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    return subprocess.call(command, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", choices=sorted(RUNNERS))
    parser.add_argument("--threads", default="default")
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--out")
    parser.add_argument("--merge", nargs="*")
    parser.add_argument("--md")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--status-out")
    parser.add_argument("--status-text")
    args = parser.parse_args()
    if args.probe:
        probe(Path(args.out), Path(args.md) if args.md else None)
        return 0
    if args.status_out:
        write_json(
            Path(args.status_out),
            {"status": args.status_text or "koşulmadı", "rows": []},
        )
        return 0
    if args.merge:
        doc = merge_docs([Path(item) for item in args.merge])
        if args.out:
            write_json(Path(args.out), doc)
        if args.md:
            numpy_markdown(doc, Path(args.md))
        return 0
    if os.environ.get("NEXUS_BENCH_READY") != "1":
        return relaunch_with_threads(args.threads, args.lib or "")
    doc = run_selected(args.lib, args.device, args.threads)
    if args.out:
        write_json(Path(args.out), doc)
    else:
        json.dump(doc, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
