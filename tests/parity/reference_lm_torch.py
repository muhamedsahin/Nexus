"""FAZ 6 parity referansi: HF transformers LLaMA/Qwen2/GPT-2 matematiginin PyTorch kopyasi.

transformers paketi bu makinede kurulu degil; asagidaki fonksiyonlar HF kaynak kodunun
(`modeling_rope_utils._compute_*_parameters`, `modeling_llama.rotate_half`,
`apply_rotary_pos_emb`, `repeat_kv`, `LlamaRMSNorm`, `LlamaMLP`) satir satir kopyasidir.

Kullanim:
  python reference_lm_torch.py rope  <dir>   # rope_cases.json -> rope_ref.bin
  python reference_lm_torch.py model <dir>   # manifest.json + weights.bin + tokens.bin -> ref_loss.bin, ref_grads.bin
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F


# ---- HF modeling_rope_utils ---------------------------------------------------------------
def default_inv_freq(base, dim):
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def compute_inv_freq(case):
    t = case["type"]
    base = case["theta"]
    dim = case["dim"]
    factor = case.get("factor", 1.0)
    max_pos = case["max_position"]
    orig = case.get("original_max_position", 0) or max_pos
    attention_factor = 1.0
    if t == "none":
        inv = default_inv_freq(base, dim)
    elif t == "linear":
        inv = default_inv_freq(base, dim) / factor
    elif t == "ntk":
        inv = default_inv_freq(base * factor ** (dim / (dim - 2)), dim)
    elif t == "dynamic":
        seq_len = max(case["seq_len"], orig)
        b = base * ((factor * seq_len / orig) - (factor - 1)) ** (dim / (dim - 2))
        inv = default_inv_freq(b, dim)
    elif t == "yarn":
        beta_fast = case.get("beta_fast", 32.0)
        beta_slow = case.get("beta_slow", 1.0)
        attention_factor = case.get("attention_factor", 0.0) or (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)

        def find_correction_dim(num_rotations):
            return (dim * math.log(orig / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        low = max(math.floor(find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
        extrap = 1.0 / pos_freqs
        interp = 1.0 / (factor * pos_freqs)
        extrap_factor = 1 - ramp
        inv = interp * (1 - extrap_factor) + extrap * extrap_factor
    elif t == "llama3":
        inv = default_inv_freq(base, dim)
        low_freq_factor = case.get("low_freq_factor", 1.0)
        high_freq_factor = case.get("high_freq_factor", 4.0)
        low_freq_wavelen = orig / low_freq_factor
        high_freq_wavelen = orig / high_freq_factor
        wavelen = 2 * math.pi / inv
        inv_llama = torch.where(wavelen > low_freq_wavelen, inv / factor, inv)
        smooth = (orig / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed = (1 - smooth) * inv_llama / factor + smooth * inv_llama
        is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv = torch.where(is_medium, smoothed, inv_llama)
    else:
        raise ValueError(t)
    return inv.float(), attention_factor


# ---- HF modeling_llama --------------------------------------------------------------------
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def rope_cos_sin(inv_freq, attention_factor, positions):
    # HF LlamaRotaryEmbedding.forward (FP32)
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * attention_factor, emb.sin() * attention_factor


def apply_partial(x, cos, sin, rot):
    # GPT-NeoX kismi dondurme: ilk rot boyut doner, kalani gecer. cos/sin x[..., :rot]'a yayinlanir.
    xr, xp = x[..., :rot], x[..., rot:]
    xr = (xr * cos) + (rotate_half(xr) * sin)
    return torch.cat((xr, xp), dim=-1)


def repeat_kv(hidden, n_rep):
    b, kvh, t, d = hidden.shape
    if n_rep == 1:
        return hidden
    hidden = hidden[:, :, None, :, :].expand(b, kvh, n_rep, t, d)
    return hidden.reshape(b, kvh * n_rep, t, d)


def rms_norm(x, w, eps):
    # LlamaRMSNorm (FP32 yolu)
    var = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(var + eps))


def read_f32(path, n=None):
    t = torch.from_file(path, dtype=torch.float32, size=n if n is not None else os.path.getsize(path) // 4)
    return t.clone()


def run_rope(d):
    cases = json.load(open(os.path.join(d, "rope_cases.json")))
    out = []
    for case in cases["freq"]:
        inv, af = compute_inv_freq(case)
        out.append(inv)
        out.append(torch.tensor([af], dtype=torch.float32))
    a = cases["apply"]
    x = read_f32(os.path.join(d, "rope_x.bin"))
    pos = torch.tensor(a["positions"], dtype=torch.int64)
    n, h, dd, rot = len(a["positions"]), a["heads"], a["head_dim"], a["rot_dim"]
    x = x.view(n, h, dd)
    inv, af = compute_inv_freq(a["freq"])
    cos, sin = rope_cos_sin(inv, af, pos)  # [n, rot]
    xr = x.permute(1, 0, 2)[None]  # [1, H, N, D]
    if rot == dd:
        y, _ = apply_rotary_pos_emb(xr, xr, cos[None], sin[None])
    else:
        y = apply_partial(xr[0].permute(1, 0, 2), cos[:, None], sin[:, None], rot).permute(1, 0, 2)[None]
    y = y[0].permute(1, 0, 2).reshape(-1)
    out.append(y)
    torch.cat([t.reshape(-1) for t in out]).numpy().tofile(os.path.join(d, "rope_ref.bin"))


def run_model(d):
    m = json.load(open(os.path.join(d, "manifest.json")))
    c = m["config"]
    B, T = m["batch"], m["time"]
    flat = read_f32(os.path.join(d, "weights.bin"))
    params, off = {}, 0
    for p in m["params"]:
        n = 1
        for s in p["shape"]:
            n *= s
        params[p["name"]] = flat[off : off + n].view(*p["shape"]).clone().requires_grad_(True)
        off += n
    tokens = torch.from_file(os.path.join(d, "tokens.bin"), dtype=torch.int32, size=B * T).long()
    L, dm, H = c["n_layer"], c["d_model"], c["n_head"]
    Hk = c["n_kv_head"] or H
    D = c["head_dim"] or dm // H
    Fh = c["d_ff"]
    eps = c["norm_eps"]
    rot = c["rotary_dim"] or D
    V, Vp = c["vocab_size"], params["wte"].shape[0]

    def norm(x, prefix):
        if c["norm"] == "rmsnorm":
            return rms_norm(x, params[prefix + ".weight"], eps)
        b = params.get(prefix + ".bias")
        return F.layer_norm(x, (dm,), params[prefix + ".weight"], b, eps)

    def lin(x, prefix):
        y = x @ params[prefix + ".weight"]
        b = params.get(prefix + ".bias")
        return y + b if b is not None else y

    x = params["wte"][tokens.view(-1)]
    if c["pos"] == "learned":
        x = x + params["wpe"][torch.arange(T).repeat(B)]
    pos = torch.arange(T)
    if c["pos"] == "rope":
        rs = dict(c["rope_scaling"])
        case = {"type": rs["type"], "theta": c["rope_theta"], "dim": rot, "factor": rs["factor"],
                "max_position": c["max_position"], "original_max_position": rs["original_max_position"],
                "low_freq_factor": rs["low_freq_factor"], "high_freq_factor": rs["high_freq_factor"],
                "beta_fast": rs["beta_fast"], "beta_slow": rs["beta_slow"],
                "attention_factor": rs["attention_factor"], "seq_len": T}
        inv, af = compute_inv_freq(case)
        cos, sin = rope_cos_sin(inv, af, pos)
    for l in range(L):
        p = f"h{l}."
        h = norm(x, p + "norm1") if c["prenorm"] else x
        qkv = lin(h, p + "attn.qkv")
        q = qkv[:, : H * D].view(B, T, H, D).transpose(1, 2)
        k = qkv[:, H * D : (H + Hk) * D].view(B, T, Hk, D).transpose(1, 2)
        v = qkv[:, (H + Hk) * D :].view(B, T, Hk, D).transpose(1, 2)
        if c["qk_norm"]:
            q = rms_norm(q, params[p + "attn.q_norm.weight"], eps)
            k = rms_norm(k, params[p + "attn.k_norm.weight"], eps)
        if c["pos"] == "rope":
            if rot == D:
                q, k = apply_rotary_pos_emb(q, k, cos[None], sin[None])
            else:
                q = apply_partial(q, cos[None, None], sin[None, None], rot)
                k = apply_partial(k, cos[None, None], sin[None, None], rot)
        k = repeat_kv(k, H // Hk)
        v = repeat_kv(v, H // Hk)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = lin(a.transpose(1, 2).reshape(B * T, H * D), p + "attn.o")

        def mlp(z):
            up = lin(z, p + "mlp.up")
            if c["mlp"] == "gelu":
                act = F.gelu(up)
            elif c["mlp"] == "swiglu":
                act = F.silu(up[:, :Fh]) * up[:, Fh:]  # LlamaMLP: down(act(gate) * up)
            else:
                act = F.gelu(up[:, :Fh]) * up[:, Fh:]
            return lin(act, p + "mlp.down")

        if not c["prenorm"]:
            hh = norm(x + a, p + "norm1")
            x = norm(hh + mlp(hh), p + "norm2")
        elif c["parallel_block"]:
            x = x + a + mlp(h)
        else:
            r = x + a
            x = r + mlp(norm(r, p + "norm2"))
    x = norm(x, "norm_f")
    head = params["wte"] if c["tie_embeddings"] else params["lm_head.weight"]
    logits = x @ head.t()
    if Vp > V:
        logits[:, V:] = float("-inf")
    lg = logits.view(B, T, Vp)[:, :-1].reshape(-1, Vp)
    tg = tokens.view(B, T)[:, 1:].reshape(-1)
    loss = F.cross_entropy(lg, tg)
    loss.backward()
    torch.tensor([loss.item()], dtype=torch.float32).numpy().tofile(os.path.join(d, "ref_loss.bin"))
    torch.cat([params[p["name"]].grad.reshape(-1) for p in m["params"]]).numpy().tofile(os.path.join(d, "ref_grads.bin"))


if __name__ == "__main__":
    torch.manual_seed(0)
    {"rope": run_rope, "model": run_model}[sys.argv[1]](sys.argv[2])
