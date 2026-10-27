"""FAZ 17 Python testleri: DLPack sifir kopya (torch), nexus.lm, nexus.interop (safetensors / HF / GGUF / ONNX).

Calistirma (pytest gerekmez):  set PYTHONPATH=python\\src  &&  .bench-venv-cuda\\Scripts\\python.exe python\\tests\\test_faz17.py

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import gc
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

try:
    import nexus
    from nexus import _core
    _core.interop  # noqa: B018  FAZ 17 bagi var mi
except (ImportError, AttributeError) as e:
    print("  SKIP  nexus._core (FAZ 17 bagi) yuklenemedi:", e)
    sys.exit(77)

io, lmm = _core.interop, _core.lm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "benchmarks", "runner"))
import faz17_ref  # noqa: E402  bagimsiz referans (numpy + torch)

passed, failed = 0, 0


def check(cond, name):
    global passed, failed
    print(("  PASS  " if cond else "  FAIL  ") + name)
    passed += bool(cond)
    failed += not cond


def raises(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return str(e)
    return None


def test_dlpack_cpu():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    n = io.from_dlpack(t)
    check(n.data_ptr == t.data_ptr() and n.shape == [3, 4] and n.device == "cpu", "test_dlpack_torch (CPU): torch -> nexus ayni isaretci")
    n.add_(io.from_dlpack(torch.ones(12)))
    n.scale_(2.0)
    check(torch.equal(t, (torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1) * 2), "nexus'ta yerinde islem torch'ta gorunur (sifir kopya)")
    back = torch.from_dlpack(n)
    check(back.data_ptr() == t.data_ptr() and back.shape == (3, 4), "nexus -> torch ayni isaretci")
    v = n.numpy()
    check(v.ctypes.data == t.data_ptr() and v[2, 3] == 24.0, "numpy goruntusu sifir kopya")
    # yasam suresi: uretici silinse de nexus tensoru gecerli (kapsul deleter'i son sahipte)
    keep_ptr = t.data_ptr()
    del t, back, v
    gc.collect()
    check(n.data_ptr == keep_ptr and abs(n.sum() - sum((i + 1) * 2 for i in range(12))) < 1e-6, "kaynak silindikten sonra veri gecerli (sahiplik dogru)")
    e1 = raises(lambda: io.from_dlpack(torch.zeros(4, dtype=torch.float64)))
    e2 = raises(lambda: io.from_dlpack(torch.zeros(4, 5).t()))
    print("  info  float64 ->", e1)
    print("  info  bitisik olmayan ->", e2)
    check(e1 is not None and "float32" in e1 and e2 is not None and "bitisik" in e2, "uyumsuz dtype / adim acik hata (kopyaya dusme yok)")
    cap = torch.ones(3).__dlpack__()
    io.from_dlpack(cap)
    check(raises(lambda: io.from_dlpack(cap)) is not None, "tuketilmis kapsul ikinci kez kullanilamaz")


def test_dlpack_cuda():
    if not torch.cuda.is_available():
        print("  info  CUDA yok: atlandi")
        return
    t = torch.arange(1 << 20, dtype=torch.float32, device="cuda")
    n = io.from_dlpack(t)
    check(n.data_ptr == t.data_ptr() and n.device == "cuda:0", "test_dlpack_torch (CUDA): torch -> nexus ayni aygit isaretcisi")
    err = raises(lambda: n.add_(io.from_dlpack(torch.ones(1 << 20, device="cuda"))))
    if err and "CUDA'siz" in err:
        print("  info  modul CUDA cekirdeksiz:", err)
    else:
        torch.cuda.synchronize()
        check(torch.equal(t[:5].cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])), "nexus CUDA cekirdegi (add_) torch tensorunu yerinde degistirdi")
    back = torch.from_dlpack(n)
    check(back.data_ptr() == t.data_ptr() and back.device.type == "cuda", "nexus -> torch (CUDA) ayni isaretci")


def test_lm_and_interop():
    m = lmm.build("llama2-7b", json.dumps({"n_layer": 2, "d_model": 64, "n_head": 4, "n_kv_head": 2, "d_ff": 128,
                                          "vocab_size": 101, "max_position": 64}))
    lg = m.logits(np.arange(10) % 101)
    out = m.generate([1, 2, 3], 6)
    check(lg.shape == (10, 101) and len(out) == 6 and m.num_parameters() > 0, "nexus.lm: build / logits / KV onbellekli generate")
    cfg = json.loads(_core.config.preset("gpt2-124m"))
    check(cfg["n_layer"] == 12 and _core.config.count_parameters(_core.config.preset("gpt2-124m")) > 100_000_000, "nexus.config preset / sayim")
    d = tempfile.mkdtemp(prefix="nexus_faz17_py_")
    arrs = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(4, dtype=np.float32) * -2}
    io.save_safetensors(os.path.join(d, "x.safetensors"), arrs)
    back = io.load_safetensors(os.path.join(d, "x.safetensors"))
    st_ok = all(np.array_equal(back[k], v) for k, v in arrs.items())
    ref = faz17_ref.read_safetensors(os.path.join(d, "x.safetensors"))  # bagimsiz okuyucu
    check(st_ok and np.array_equal(ref["a"].numpy(), arrs["a"]), "safetensors yaz (nexus) -> oku (nexus + bagimsiz)")
    faz17_ref.tiny(os.path.join(d, "qwen"), "qwen2", 3)
    model, info = io.load_hf(os.path.join(d, "qwen"))
    toks = [5, 9, 100, 3, 7, 7, 1, 0, 42]
    cfg_h, w = faz17_ref.load_dir(os.path.join(d, "qwen"))
    with torch.no_grad():
        r = faz17_ref.forward(cfg_h, w, toks, torch.float64).numpy()
    e = np.abs(model.logits(toks) - r).max() / np.abs(r).max()
    check(info["template"] == "qwen2/v1" and e <= 1e-4, f"python: load_hf (qwen2/v1) logitleri referansla {e:.1e}")
    small = lmm.build("llama2-7b", json.dumps({"n_layer": 1, "d_model": 64, "n_head": 4, "n_kv_head": 4, "d_ff": 128,
                                               "vocab_size": 64, "max_position": 32}))
    g = io.export_gguf(small, os.path.join(d, "m.gguf"), "Q8_0")
    back_m = io.import_gguf(os.path.join(d, "m.gguf"))
    kv, tens = faz17_ref.read_gguf(os.path.join(d, "m.gguf"))
    cfg_g, w_g = faz17_ref.gguf_to_hf(kv, tens)
    with torch.no_grad():
        rg = faz17_ref.forward(cfg_g, w_g, [t % 64 for t in toks], torch.float64).numpy()
    eg = np.abs(back_m.logits([t % 64 for t in toks]) - rg).max() / np.abs(rg).max()
    check(eg <= 1e-4, f"python: export_gguf (Q8_0) -> import_gguf == bagimsiz GGUF okuyucu ({eg:.1e})")
    o = io.export_onnx(m, os.path.join(d, "m.onnx"), 18, True)
    check(g["bytes"] > 0 and o["nodes"] > 0 and o["data_bytes"] > 0, "python: export_gguf / export_onnx")


if __name__ == "__main__":
    t0 = time.time()
    print(f"nexus {nexus.__file__}, python {sys.version.split()[0]}, torch {torch.__version__}")
    test_dlpack_cpu()
    test_dlpack_cuda()
    test_lm_and_interop()
    print(f"test_faz17.py: {passed} gecti, {failed} basarisiz ({time.time() - t0:.1f} s)")
    sys.exit(1 if failed else 0)
