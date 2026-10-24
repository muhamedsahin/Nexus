#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nexus_estimate Python aynasi (FAZ 0).

tools/nexus_estimate/estimate.cpp ile birebir ayni formuller; test_estimate_parity
C++ araci ile bu betigin JSON ciktilarini karsilastirir. Formullerin kaynaklari
estimate.hpp basliginda belgelenmistir (Kaplan/Chinchilla 6N, PaLM dikkat terimi,
Korthikanti aktivasyon bellegi, Rajbhandari ZeRO bolusumu).

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import argparse
import json
import sys

GIB = 1 << 30


def preset_shape(name):
    if name == "gpt2-124m":
        return dict(n_layer=12, d_model=768, n_head=12, n_kv_head=12, d_ff=3072, vocab=50257,
                    seq_len=1024, micro_batch=1, learned_pos=1024, gradient_accum=1,
                    tie_head=True, ffn_swiglu=False, bias=True, rmsnorm=False)
    if name == "llama-7b":
        return dict(n_layer=32, d_model=4096, n_head=32, n_kv_head=32, d_ff=11008, vocab=32000,
                    seq_len=2048, micro_batch=1, learned_pos=0, gradient_accum=1,
                    tie_head=False, ffn_swiglu=True, bias=False, rmsnorm=True)
    raise ValueError("bilinmeyen preset: " + name)


def validate(shape):
    for key in ("n_layer", "d_model", "n_head", "n_kv_head", "d_ff", "vocab", "seq_len",
                "micro_batch", "gradient_accum"):
        if shape[key] <= 0:
            raise ValueError("nexus estimate: mimari alanlari pozitif olmali")
    if shape["d_model"] % shape["n_head"] != 0:
        raise ValueError("nexus estimate: d_model % n_head == 0 olmali")
    if shape["n_head"] % shape["n_kv_head"] != 0:
        raise ValueError("nexus estimate: n_head % n_kv_head == 0 olmali (GQA gruplama)")


def count_parameters(shape):
    validate(shape)
    d = shape["d_model"]
    d_head = d // shape["n_head"]
    d_kv = d_head * shape["n_kv_head"]

    token_embed = shape["vocab"] * d
    pos_embed = shape["learned_pos"] * d

    attn_weights = d * d + d * d + 2 * d * d_kv
    attn_bias = (2 * d + 2 * d_kv) if shape["bias"] else 0
    attention = shape["n_layer"] * (attn_weights + attn_bias)

    ffn_weights = 3 * d * shape["d_ff"] if shape["ffn_swiglu"] else 2 * d * shape["d_ff"]
    if shape["bias"]:
        ffn_bias = (2 * shape["d_ff"] + d) if shape["ffn_swiglu"] else (shape["d_ff"] + d)
    else:
        ffn_bias = 0
    ffn = shape["n_layer"] * (ffn_weights + ffn_bias)

    norm_unit = d if shape["rmsnorm"] else 2 * d
    norms = shape["n_layer"] * 2 * norm_unit + norm_unit

    head = 0 if shape["tie_head"] else shape["vocab"] * d

    total = token_embed + pos_embed + attention + ffn + norms + head
    return dict(token_embed=token_embed, pos_embed=pos_embed, attention=attention, ffn=ffn,
                norms=norms, head=head, total=total)


def estimate_memory(shape, dtype, plan):
    validate(shape)
    if plan["dp"] <= 0 or plan["tp"] <= 0 or plan["pp"] <= 0 or not (0 <= plan["zero_stage"] <= 3):
        raise ValueError("nexus estimate: gecersiz paralellik plani")
    params = count_parameters(shape)
    total = params["total"]

    param_bytes = total * dtype["param_bytes"]
    grad_bytes = total * dtype["grad_bytes"]
    optim_bytes = total * dtype["optim_bytes"]
    master_bytes = total * dtype["master_bytes"]

    model_parallel = plan["tp"] * plan["pp"]
    optim_div = model_parallel * (plan["dp"] if plan["zero_stage"] >= 1 else 1)
    grad_div = model_parallel * (plan["dp"] if plan["zero_stage"] >= 2 else 1)
    param_div = model_parallel * (plan["dp"] if plan["zero_stage"] >= 3 else 1)
    static_sharded = (param_bytes // param_div + grad_bytes // grad_div +
                      optim_bytes // optim_div + master_bytes // param_div)

    s = float(shape["seq_len"])
    b = float(shape["micro_batch"])
    h = float(shape["d_model"])
    a = float(shape["n_head"])
    scale = float(dtype["activation_elem_bytes"]) / 2.0
    if plan["full_recompute"]:
        per_layer = 2.0 * s * b * h * scale
    else:
        per_layer = s * b * h * (34.0 + 5.0 * a * s / h) * scale
    activation_bytes = int(per_layer * float(shape["n_layer"] // plan["pp"]))

    logits_bytes = 2 * shape["micro_batch"] * shape["seq_len"] * shape["vocab"] * dtype["activation_elem_bytes"]

    budget_bytes = 8 * GIB
    per_gpu_bytes = static_sharded + activation_bytes + logits_bytes
    return dict(param_bytes=param_bytes, grad_bytes=grad_bytes, optim_bytes=optim_bytes,
                master_bytes=master_bytes, static_sharded=static_sharded,
                activation_bytes=activation_bytes, logits_bytes=logits_bytes,
                per_gpu_bytes=per_gpu_bytes, budget_bytes=budget_bytes,
                fits_budget=per_gpu_bytes <= budget_bytes)



def estimate_flops(shape, plan):
    validate(shape)
    if plan["dp"] <= 0:
        raise ValueError("nexus estimate: dp pozitif olmali")
    params = count_parameters(shape)
    matmul_params = params["total"] - params["pos_embed"] - (0 if shape["tie_head"] else params["token_embed"])
    tokens_per_step = shape["micro_batch"] * shape["seq_len"] * plan["dp"] * shape["gradient_accum"]
    per_token = (6.0 * float(matmul_params) +
                 12.0 * float(shape["n_layer"]) * float(shape["d_model"]) * float(shape["seq_len"]))
    return dict(matmul_params=matmul_params, tokens_per_step=tokens_per_step,
                flops_per_token=per_token, flops_per_step=per_token * float(tokens_per_step))


def roofline_bound(flops_per_step, bytes_per_step, peak_flops_per_s, mem_bytes_per_s):
    if bytes_per_step <= 0.0 or peak_flops_per_s <= 0.0 or mem_bytes_per_s <= 0.0:
        raise ValueError("nexus estimate: roofline girdileri pozitif olmali")
    intensity = flops_per_step / bytes_per_step
    ceiling = min(peak_flops_per_s, intensity * mem_bytes_per_s)
    return dict(intensity=intensity, ceiling_flops_per_s=ceiling)


def build_output(shape, dtype, plan, peak_tflops, mem_gbps):
    params = count_parameters(shape)
    memory = estimate_memory(shape, dtype, plan)
    flops = estimate_flops(shape, plan)
    output = {
        "tool": "nexus_estimate",
        "phase": 0,
        "input": {"model": dict(shape), "dtype": dict(dtype), "plan": dict(plan)},
        "params": params,
        "memory": memory,
        "flops": flops,
    }
    if peak_tflops > 0.0 and mem_gbps > 0.0:
        bytes_per_step = (float(memory["activation_bytes"] + memory["logits_bytes"]) +
                          3.0 * float(flops["matmul_params"]) * float(dtype["param_bytes"]))
        roof = roofline_bound(flops["flops_per_step"], bytes_per_step, peak_tflops * 1e12, mem_gbps * 1e9)
        roof["bytes_per_step"] = bytes_per_step
        output["roofline"] = roof
    else:
        output["roofline"] = None
    output["fragmentation"] = {"ratio": "n/a", "note": "FAZ 1 k64 havuzu reserved/current ile olcecek"}
    return output



def main(argv=None):
    parser = argparse.ArgumentParser(prog="estimate.py", description="nexus_estimate Python aynasi (FAZ 0)")
    parser.add_argument("--preset", choices=["gpt2-124m", "llama-7b"])
    parser.add_argument("--config")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--seq", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--learned-pos", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--tie", dest="tie", action="store_true", default=None)
    parser.add_argument("--no-tie", dest="tie", action="store_false")
    parser.add_argument("--swiglu", dest="swiglu", action="store_true", default=None)
    parser.add_argument("--no-swiglu", dest="swiglu", action="store_false")
    parser.add_argument("--bias", dest="bias", action="store_true", default=None)
    parser.add_argument("--no-bias", dest="bias", action="store_false")
    parser.add_argument("--rmsnorm", dest="rmsnorm", action="store_true", default=None)
    parser.add_argument("--no-rmsnorm", dest="rmsnorm", action="store_false")
    parser.add_argument("--dp", type=int)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--pp", type=int)
    parser.add_argument("--zero", type=int)
    parser.add_argument("--recompute", action="store_true", default=None)
    parser.add_argument("--param-bytes", type=int)
    parser.add_argument("--grad-bytes", type=int)
    parser.add_argument("--optim-bytes", type=int)
    parser.add_argument("--master-bytes", type=int)
    parser.add_argument("--act-bytes", type=int)
    parser.add_argument("--peak-tflops", type=float, default=0.0)
    parser.add_argument("--mem-gbps", type=float, default=0.0)
    args = parser.parse_args(argv)

    shape = preset_shape(args.preset) if args.preset else preset_shape("gpt2-124m")
    dtype = dict(param_bytes=4, grad_bytes=4, optim_bytes=8, master_bytes=0, activation_elem_bytes=4)
    plan = dict(dp=1, tp=1, pp=1, zero_stage=0, full_recompute=False)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        for key, value in config.get("model", {}).items():
            shape[key] = value
        for key, value in config.get("dtype", {}).items():
            dtype[key] = value
        for key, value in config.get("plan", {}).items():
            plan[key] = value
        device = config.get("device", {})
        args.peak_tflops = device.get("peak_tflops", args.peak_tflops)
        args.mem_gbps = device.get("mem_gbps", args.mem_gbps)

    overrides = {
        "n_layer": args.layers, "d_model": args.d_model, "n_head": args.heads,
        "n_kv_head": args.kv_heads, "d_ff": args.d_ff, "vocab": args.vocab,
        "seq_len": args.seq, "micro_batch": args.batch, "learned_pos": args.learned_pos,
        "gradient_accum": args.grad_accum, "tie_head": args.tie, "ffn_swiglu": args.swiglu,
        "bias": args.bias, "rmsnorm": args.rmsnorm,
    }
    for key, value in overrides.items():
        if value is not None:
            shape[key] = value
    for key, value in (("dp", args.dp), ("tp", args.tp), ("pp", args.pp), ("zero_stage", args.zero)):
        if value is not None:
            plan[key] = value
    if args.recompute is not None:
        plan["full_recompute"] = args.recompute
    for key, value in (("param_bytes", args.param_bytes), ("grad_bytes", args.grad_bytes),
                       ("optim_bytes", args.optim_bytes), ("master_bytes", args.master_bytes),
                       ("activation_elem_bytes", args.act_bytes)):
        if value is not None:
            dtype[key] = value

    output = build_output(shape, dtype, plan, args.peak_tflops, args.mem_gbps)
    text = json.dumps(output, indent=2, sort_keys=False)

    params = output["params"]
    memory = output["memory"]
    flops = output["flops"]
    print("== nexus_estimate (FAZ 0, python) ==")
    print("Parametreler: toplam {total} (embed {te} + konum {pe} + dikkat {at} + ffn {ff} + norm {no} + head {he})"
          .format(total=params["total"], te=params["token_embed"], pe=params["pos_embed"],
                  at=params["attention"], ff=params["ffn"], no=params["norms"], he=params["head"]))
    print("Bellek/aygit: statik {st} + aktivasyon {ac} + logit {lg} = {tot} bayt (butce {bu}, sigar: {fit})"
          .format(st=memory["static_sharded"], ac=memory["activation_bytes"], lg=memory["logits_bytes"],
                  tot=memory["per_gpu_bytes"], bu=memory["budget_bytes"],
                  fit="evet" if memory["fits_budget"] else "hayir"))
    print("FLOP/adim: {fs:.3e} (token/adim {tp}, token basina {ft:.3e})"
          .format(fs=flops["flops_per_step"], tp=flops["tokens_per_step"], ft=flops["flops_per_token"]))
    print("Catilma orani: n/a (FAZ 1 k64 havuzu reserved/current ile olcecek)")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print("JSON yazildi: " + args.json_path)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
