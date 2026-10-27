"""FAZ 17: bagimsiz referanslar (transformers / onnxruntime / gguf paketleri YOK; yalniz numpy + torch).

  python faz17_ref.py logits <hf_dizini> <tok.bin> <cikti.bin> [float32|float64]
      safetensors'u kendi ayristiricisiyla okur, LLaMA / Mistral / Qwen2 ileri gecisini HF tanimiyla (rotate_half
      RoPE, GQA repeat_kv, RMSNorm, SwiGLU, Mistral kayan pencere, Qwen2 qkv bias) torch'ta hesaplar.
  python faz17_ref.py tiny <cikti_dizini> <llama|mistral|qwen2> <tohum>
      rastgele kucuk HF dizini (config.json + model.safetensors F32, HF parametre adlari).

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import json
import os
import struct
import sys

import numpy as np
import torch

DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32}


def read_safetensors(path):
    with open(path, "rb") as f:
        raw = f.read()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    data = memoryview(raw)[8 + n:]
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        b, e = v["data_offsets"]
        buf = data[b:e]
        if v["dtype"] == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        else:
            arr = np.frombuffer(buf, dtype=DT[v["dtype"]])
        out[k] = torch.from_numpy(arr.astype(np.float64).reshape(v["shape"]))
    return out


def load_dir(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    idx = os.path.join(d, "model.safetensors.index.json")
    w = {}
    if os.path.exists(idx):
        for f in sorted(set(json.load(open(idx))["weight_map"].values())):
            w.update(read_safetensors(os.path.join(d, f)))
    else:
        w = read_safetensors(os.path.join(d, "model.safetensors"))
    return cfg, w


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def forward(cfg, w, toks, dtype):
    arch = cfg["architectures"][0]
    d = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    Hk = cfg.get("num_key_value_heads", H)
    D = cfg.get("head_dim") or d // H
    eps = cfg.get("rms_norm_eps", 1e-6)
    theta = cfg.get("rope_theta", 10000.0)
    if cfg.get("rope_scaling"):
        raise SystemExit("referans: rope_scaling desteklenmez")
    win = cfg.get("sliding_window") if arch == "MistralForCausalLM" else None
    W = {k: v.to(dtype) for k, v in w.items()}
    T = len(toks)
    x = W["model.embed_tokens.weight"][torch.tensor(toks)]
    pos = torch.arange(T, dtype=dtype)
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=dtype) / D))
    ang = pos[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    i = torch.arange(T)
    mask = i[None, :] <= i[:, None]
    if win:
        mask = mask & (i[:, None] - i[None, :] < win)
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        h = rms(x, W[p + "input_layernorm.weight"], eps)

        def proj(n, out):
            y = h @ W[p + f"self_attn.{n}_proj.weight"].t()
            if p + f"self_attn.{n}_proj.bias" in W:
                y = y + W[p + f"self_attn.{n}_proj.bias"]
            return y.view(T, out, D).transpose(0, 1)

        q, k, v = proj("q", H), proj("k", Hk), proj("v", Hk)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        k = k.repeat_interleave(H // Hk, dim=0)
        v = v.repeat_interleave(H // Hk, dim=0)
        s = (q @ k.transpose(1, 2)) / (D ** 0.5)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, dim=-1) @ v
        x = x + a.transpose(0, 1).reshape(T, H * D) @ W[p + "self_attn.o_proj.weight"].t()
        h = rms(x, W[p + "post_attention_layernorm.weight"], eps)
        g = torch.nn.functional.silu(h @ W[p + "mlp.gate_proj.weight"].t()) * (h @ W[p + "mlp.up_proj.weight"].t())
        x = x + g @ W[p + "mlp.down_proj.weight"].t()
    x = rms(x, W["model.norm.weight"], eps)
    head = W["lm_head.weight"] if "lm_head.weight" in W else W["model.embed_tokens.weight"]
    return x @ head.t()


def write_safetensors(path, tensors):
    header, off, blobs = {}, 0, []
    for k, arr in tensors.items():
        b = np.ascontiguousarray(arr.astype(np.float32)).tobytes()
        header[k] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def tiny(out, kind, seed):
    rng = np.random.default_rng(seed)
    d, H, Hk, F, L, V = 64, 4, 2, 128, 2, 211
    arch = {"llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM", "qwen2": "Qwen2ForCausalLM"}[kind]
    cfg = {"architectures": [arch], "model_type": kind, "hidden_size": d, "num_attention_heads": H,
           "num_key_value_heads": Hk, "intermediate_size": F, "num_hidden_layers": L, "vocab_size": V,
           "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0 if kind != "qwen2" else 1000000.0,
           "hidden_act": "silu", "tie_word_embeddings": kind == "qwen2", "rope_scaling": None}
    if kind == "mistral":
        cfg["sliding_window"] = 7
    if kind == "qwen2":
        cfg["use_sliding_window"] = False
    D = d // H
    t = {"model.embed_tokens.weight": rng.normal(0, 0.5, (V, d))}
    for l in range(L):
        p = f"model.layers.{l}."
        t[p + "input_layernorm.weight"] = 1 + 0.1 * rng.normal(size=d)
        t[p + "post_attention_layernorm.weight"] = 1 + 0.1 * rng.normal(size=d)
        t[p + "self_attn.q_proj.weight"] = rng.normal(0, 0.2, (H * D, d))
        t[p + "self_attn.k_proj.weight"] = rng.normal(0, 0.2, (Hk * D, d))
        t[p + "self_attn.v_proj.weight"] = rng.normal(0, 0.2, (Hk * D, d))
        if kind == "qwen2":
            t[p + "self_attn.q_proj.bias"] = rng.normal(0, 0.2, H * D)
            t[p + "self_attn.k_proj.bias"] = rng.normal(0, 0.2, Hk * D)
            t[p + "self_attn.v_proj.bias"] = rng.normal(0, 0.2, Hk * D)
        t[p + "self_attn.o_proj.weight"] = rng.normal(0, 0.2, (d, H * D))
        t[p + "mlp.gate_proj.weight"] = rng.normal(0, 0.2, (F, d))
        t[p + "mlp.up_proj.weight"] = rng.normal(0, 0.2, (F, d))
        t[p + "mlp.down_proj.weight"] = rng.normal(0, 0.2, (d, F))
    t["model.norm.weight"] = 1 + 0.1 * rng.normal(size=d)
    if kind != "qwen2":
        t["lm_head.weight"] = rng.normal(0, 0.5, (V, d))
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    write_safetensors(os.path.join(out, "model.safetensors"), t)
    print(json.dumps({"tiny": kind, "dir": out}))


def read_gguf(path):
    """GGUF v3 (ggml belgesindeki bicim) -> (kv, {ad: (ne, tip, ndarray FP64)})."""
    raw = open(path, "rb").read()
    at = [0]

    def take(fmt):
        v = struct.unpack_from("<" + fmt, raw, at[0])
        at[0] += struct.calcsize("<" + fmt)
        return v[0]

    def string():
        n = take("Q")
        s = raw[at[0]:at[0] + n].decode("utf-8", errors="replace")
        at[0] += n
        return s

    SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = take("I"), take("Q")
            return [value(et) for _ in range(n)]
        return take(SC[t])

    assert raw[:4] == b"GGUF", "sihir"
    at[0] = 4
    ver, nt, nkv = take("I"), take("Q"), take("Q")
    assert ver == 3
    kv = {}
    for _ in range(nkv):
        k = string()
        kv[k] = value(take("I"))
    infos = []
    for _ in range(nt):
        name = string()
        nd = take("I")
        ne = [take("Q") for _ in range(nd)]
        infos.append((name, ne, take("I"), take("Q")))
    align = kv.get("general.alignment", 32)
    base = (at[0] + align - 1) // align * align
    out = {}
    for name, ne, t, off in infos:
        n = int(np.prod(ne))
        p = base + off
        if t == 0:
            a = np.frombuffer(raw, np.float32, n, p).astype(np.float32)
        elif t == 1:
            a = np.frombuffer(raw, np.float16, n, p).astype(np.float32)
        elif t == 8:  # Q8_0: f16 d + 32 x int8
            blk = np.frombuffer(raw, np.uint8, n // 32 * 34, p).reshape(-1, 34)
            d = blk[:, :2].copy().view(np.float16).astype(np.float32)
            a = (d * blk[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)
        elif t == 2:  # Q4_0: f16 d + 16 bayt; dusuk yarim j, yuksek yarim j + 16
            blk = np.frombuffer(raw, np.uint8, n // 32 * 18, p).reshape(-1, 18)
            d = blk[:, :2].copy().view(np.float16).astype(np.float32)
            qs = blk[:, 2:]
            q = np.concatenate([(qs & 15), (qs >> 4)], axis=1).astype(np.float32) - 8.0
            a = (d * q).reshape(-1)
        else:
            raise SystemExit(f"tip {t} desteklenmez")
        out[name] = (ne, t, a.reshape(list(reversed(ne))))
    return kv, out


def gguf_to_hf(kv, t):
    """llama.cpp adlari -> HF adlari; Q/K ters permutasyon (convert_hf_to_gguf.permute'un tersi)."""
    H, Hk = kv["llama.attention.head_count"], kv["llama.attention.head_count_kv"]
    d, L = kv["llama.embedding_length"], kv["llama.block_count"]

    def unpermute(w, heads):
        rows, cols = w.shape
        return w.reshape(heads, rows // heads // 2, 2, cols).swapaxes(1, 2).reshape(rows, cols)

    W = {"model.embed_tokens.weight": t["token_embd.weight"][2], "model.norm.weight": t["output_norm.weight"][2],
         "lm_head.weight": t["output.weight"][2]}
    for l in range(L):
        g, p = f"blk.{l}.", f"model.layers.{l}."
        W[p + "input_layernorm.weight"] = t[g + "attn_norm.weight"][2]
        W[p + "post_attention_layernorm.weight"] = t[g + "ffn_norm.weight"][2]
        W[p + "self_attn.q_proj.weight"] = unpermute(t[g + "attn_q.weight"][2], H)
        W[p + "self_attn.k_proj.weight"] = unpermute(t[g + "attn_k.weight"][2], Hk)
        W[p + "self_attn.v_proj.weight"] = t[g + "attn_v.weight"][2]
        W[p + "self_attn.o_proj.weight"] = t[g + "attn_output.weight"][2]
        W[p + "mlp.gate_proj.weight"] = t[g + "ffn_gate.weight"][2]
        W[p + "mlp.up_proj.weight"] = t[g + "ffn_up.weight"][2]
        W[p + "mlp.down_proj.weight"] = t[g + "ffn_down.weight"][2]
    cfg = {"architectures": ["LlamaForCausalLM"], "hidden_size": d, "num_attention_heads": H, "num_key_value_heads": Hk,
           "num_hidden_layers": L, "rms_norm_eps": kv["llama.attention.layer_norm_rms_epsilon"],
           "rope_theta": kv["llama.rope.freq_base"], "rope_scaling": None}
    return cfg, {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in W.items()}


def pb_decode(buf):
    """Genel protobuf cozucu: {alan: [(tel_tipi, deger)]}; uzunluk sinirli alanlar bytes olarak kalir."""
    out, i, n = {}, 0, len(buf)

    def varint():
        nonlocal i
        shift, v = 0, 0
        while True:
            c = buf[i]
            i += 1
            v |= (c & 0x7F) << shift
            if c < 0x80:
                return v
            shift += 7

    while i < n:
        key = varint()
        f, w = key >> 3, key & 7
        if w == 0:
            v = varint()
        elif w == 1:
            v = buf[i:i + 8]
            i += 8
        elif w == 2:
            ln = varint()
            v = bytes(buf[i:i + ln])
            i += ln
        elif w == 5:
            v = buf[i:i + 4]
            i += 4
        else:
            raise SystemExit(f"tel tipi {w}")
        out.setdefault(f, []).append((w, v))
    return out


def s64(v):
    return v - (1 << 64) if v >= (1 << 63) else v


def onnx_load(path):
    model = pb_decode(open(path, "rb").read())
    opset = pb_decode(model[8][0][1])[2][0][1]
    g = pb_decode(model[7][0][1])
    base = os.path.dirname(path)
    inits = {}
    for _, tb in g.get(5, []):
        t = pb_decode(tb)
        dims = [s64(v) for _, v in t.get(1, [])]
        dt = t[2][0][1]
        name = t[8][0][1].decode()
        if 9 in t:
            raw = t[9][0][1]
        else:
            kv = {pb_decode(e)[1][0][1].decode(): pb_decode(e)[2][0][1].decode() for _, e in t[13]}
            with open(os.path.join(base, kv["location"]), "rb") as f:
                f.seek(int(kv["offset"]))
                raw = f.read(int(kv["length"]))
        npdt = {1: np.float32, 7: np.int64, 6: np.int32, 9: np.bool_}[dt]
        a = np.frombuffer(raw, dtype=npdt).reshape(dims)
        inits[name] = torch.from_numpy(a.copy())
    nodes = []
    for _, nb in g.get(1, []):
        nd = pb_decode(nb)
        attrs = {}
        for _, ab in nd.get(5, []):
            a = pb_decode(ab)
            an = a[1][0][1].decode()
            at = a[20][0][1]
            if at == 1:
                attrs[an] = struct.unpack("<f", a[2][0][1])[0]
            elif at == 2:
                attrs[an] = s64(a[3][0][1])
            elif at == 7:
                attrs[an] = [s64(v) for _, v in a.get(8, [])]
            else:
                raise SystemExit(f"oznitelik tipi {at}")
        nodes.append((nd[4][0][1].decode(), [x.decode() for _, x in nd.get(1, [])], [x.decode() for _, x in nd[2]], attrs))
    return opset, inits, nodes


def onnx_run(path, toks, dtype, batch=1):
    opset, env, nodes = onnx_load(path)
    env = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in env.items()}
    env["tokens"] = torch.tensor(toks, dtype=torch.int64).reshape(batch, -1)
    used = set()
    for op, ins, outs, at in nodes:
        used.add(op)
        x = [env[i] for i in ins]
        if op == "Shape":
            y = torch.tensor(list(x[0].shape), dtype=torch.int64)
        elif op == "Slice":
            y = x[0]
            for s, e, ax in zip(x[1].tolist(), x[2].tolist(), x[3].tolist() if len(x) > 3 else range(len(x[1]))):
                n = y.shape[ax]
                s, e = max(0, s + n if s < 0 else s), min(n, e + n if e < 0 else e)
                y = y.narrow(ax, s, max(0, e - s))
        elif op == "Reshape":
            shape = x[1].tolist()
            shape = [y if y != 0 else x[0].shape[i] for i, y in enumerate(shape)]
            y = x[0].reshape(shape)
        elif op == "Squeeze":
            y = x[0].squeeze(x[1].tolist()[0])
        elif op == "Unsqueeze":
            y = x[0]
            for a in sorted(x[1].tolist()):
                y = y.unsqueeze(a)
        elif op == "Concat":
            y = torch.cat(x, dim=at["axis"])
        elif op in ("Add", "Sub", "Mul", "Div", "Mod", "Greater"):
            a, b = x
            y = {"Add": torch.add, "Sub": torch.sub, "Mul": torch.mul, "Div": torch.div, "Mod": torch.remainder,
                 "Greater": torch.gt}[op](a, b)
        elif op == "Range":
            y = torch.arange(int(x[0]), int(x[1]), int(x[2]), dtype=torch.int64)
        elif op == "Gather":
            y = torch.index_select(x[0], at.get("axis", 0), x[1].reshape(-1)).reshape(list(x[1].shape) + list(x[0].shape[1:]))
        elif op == "MatMul":
            y = torch.matmul(x[0], x[1])
        elif op == "Gemm":
            b = x[1].t() if at.get("transB", 0) else x[1]
            a = x[0].t() if at.get("transA", 0) else x[0]
            y = a @ b
        elif op == "Transpose":
            y = x[0].permute(at["perm"])
        elif op == "Sqrt":
            y = torch.sqrt(x[0])
        elif op == "ReduceMean":
            axes = at["axes"] if opset < 18 else x[1].tolist()
            y = x[0].mean(dim=axes, keepdim=bool(at.get("keepdims", 1)))
        elif op == "Erf":
            y = torch.special.erf(x[0])
        elif op == "Tanh":
            y = torch.tanh(x[0])
        elif op == "Sigmoid":
            y = torch.sigmoid(x[0])
        elif op == "Softmax":
            y = torch.softmax(x[0], dim=at.get("axis", -1))
        elif op == "Where":
            y = torch.where(x[0], x[1], x[2])
        elif op == "Expand":
            y = x[0].expand(x[1].tolist())
        elif op == "LayerNormalization":
            y = torch.nn.functional.layer_norm(x[0], (x[0].shape[-1],), x[1], x[2], at.get("epsilon", 1e-5))
        else:
            raise SystemExit("ONNX islemi yorumlanamadi: " + op)
        env[outs[0]] = y
    return env["logits"], opset, sorted(used), len(nodes)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "onnx_logits":
        raw = open(sys.argv[3], "rb").read()
        toks = list(struct.unpack("<%di" % (len(raw) // 4), raw))
        dtype = torch.float32 if (len(sys.argv) > 5 and sys.argv[5] == "float32") else torch.float64
        batch = int(sys.argv[6]) if len(sys.argv) > 6 else 1
        with torch.no_grad():
            lg, opset, ops, nn = onnx_run(sys.argv[2], toks, dtype, batch)
        lg = lg.reshape(-1, lg.shape[-1]).to(torch.float32).numpy()
        open(sys.argv[4], "wb").write(lg.astype("<f4").tobytes())
        print(json.dumps({"onnx": os.path.basename(sys.argv[2]), "opset": opset, "nodes": nn, "ops": ops}))
        sys.exit(0)
    if cmd == "gguf_logits":
        kv, t = read_gguf(sys.argv[2])
        cfg, w = gguf_to_hf(kv, t)
        raw = open(sys.argv[3], "rb").read()
        toks = list(struct.unpack("<%di" % (len(raw) // 4), raw))
        dtype = torch.float64 if (len(sys.argv) > 5 and sys.argv[5] == "float64") else torch.float32
        with torch.no_grad():
            lg = forward(cfg, w, toks, dtype).to(torch.float32).numpy()
        open(sys.argv[4], "wb").write(lg.astype("<f4").tobytes())
        types = sorted({v[1] for v in t.values()})
        print(json.dumps({"gguf": os.path.basename(sys.argv[2]), "kv": len(kv), "tensors": len(t), "types": types,
                          "tokens_meta": len(kv.get("tokenizer.ggml.tokens", [])), "arch": kv["general.architecture"]}))
        sys.exit(0)
    if cmd == "logits":
        cfg, w = load_dir(sys.argv[2])
        raw = open(sys.argv[3], "rb").read()
        toks = list(struct.unpack("<%di" % (len(raw) // 4), raw))
        dtype = torch.float64 if (len(sys.argv) > 5 and sys.argv[5] == "float64") else torch.float32
        with torch.no_grad():
            lg = forward(cfg, w, toks, dtype).to(torch.float32).numpy()
        open(sys.argv[4], "wb").write(lg.astype("<f4").tobytes())
        print(json.dumps({"tokens": len(toks), "vocab": int(lg.shape[1]), "argmax_last": int(lg[-1].argmax())}))
    elif cmd == "tiny":
        tiny(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit("bilinmeyen komut " + cmd)
