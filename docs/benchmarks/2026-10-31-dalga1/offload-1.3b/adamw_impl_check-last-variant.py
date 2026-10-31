import json, os, sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, r'C:\Users\muham\Desktop\Nexus\benchmarks\runner')
import numpy as np, torch
from faz9_nanogpt import configure_determinism, get_lr, load_nanogpt, load_nexus_init
from torch_offload_ref import build, forward_loss
configure_determinism(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
init = r'C:\Users\muham\Desktop\Nexus\build-phase1\faz20\init.bin'
toks = np.memmap(r'C:\Users\muham\Desktop\Nexus\build-phase1\faz20\tokens.bin', dtype=np.int32, mode='r')
meta = json.load(open(init + '.json'))
ng = load_nanogpt(r'C:\Users\muham\nexus-data\nanogpt')
ref = json.load(open(r'C:\Users\muham\Desktop\Nexus\docs\benchmarks\2026-10-31-dalga1\bf16-accuracy\torch-f32.json'))['losses']
out = {}
for impl in ('gpuclip-cpuadam',):
    torch.manual_seed(1337)
    m = build(ng, meta, 'cuda', torch.float32)
    load_nexus_init(m, init, meta['max_seq'])
    ps = list(m.parameters())
    groups = [{'params': [p for p in ps if p.dim() >= 2], 'weight_decay': 0.1}, {'params': [p for p in ps if p.dim() < 2], 'weight_decay': 0.0}]
    master = [p.detach().cpu().clone() for p in ps]
    for mp in master: mp.grad = torch.zeros_like(mp)
    mg = [{'params': [mp for mp in master if mp.dim() >= 2], 'weight_decay': 0.1}, {'params': [mp for mp in master if mp.dim() < 2], 'weight_decay': 0.0}]
    o = torch.optim.AdamW(mg, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    losses = []
    for it in range(15):
        for g in o.param_groups: g['lr'] = get_lr(it, 1e-4, 2, 15, 1e-5)
        r = torch.from_numpy(np.array(toks[it * 1024:(it + 1) * 1024]).reshape(1, 1024).astype(np.int64)).cuda()
        loss = forward_loss(m, r[:, :-1], r[:, 1:], False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        with torch.no_grad():
            for mp, p in zip(master, ps): mp.grad.copy_(p.grad)
        o.step()
        with torch.no_grad():
            for mp, p in zip(master, ps): p.copy_(mp); p.grad = None
        losses.append(loss.item())
    out[impl] = max(abs(a - b) for a, b in zip(losses, ref))
    print('GPU', impl, 'vs nanoGPT(fused CUDA) max %.3e' % out[impl], flush=True)
    del m, o
    torch.cuda.empty_cache()
json.dump(out, open(r'C:\Users\muham\nexus-data\dalga1-13b\adamw-impl-check-gpuclip.json', 'w'))
