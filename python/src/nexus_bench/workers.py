"""Phase 21: versioned evidence, isolated sequential runs, strict comparisons and reports.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import importlib.metadata
import json
import math
import time

def worker(framework):
    if framework != 'torch':
        try:
            version = importlib.metadata.version(framework if framework != 'jax' else 'jaxlib')
        except importlib.metadata.PackageNotFoundError:
            print(json.dumps(dict(status='unavailable', reason=framework+' is not installed')))
            return 77
        samples = []
        if framework == 'jax':
            import jax
            import jax.numpy as jnp
            cpu = jax.devices('cpu')[0]
            a = jax.device_put(jnp.full((2048,2048), .03125, dtype=jnp.float32), cpu)
            b = jax.device_put(jnp.full((2048,2048), .015625, dtype=jnp.float32), cpu)
            op = jax.jit(lambda x,y: x@y)
            def fn(): return op(a,b).block_until_ready()
            def error(x): return float(jnp.max(jnp.abs(x-1)))
        else:
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(8)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            with tf.device('/CPU:0'):
                a = tf.fill((2048,2048), .03125)
                b = tf.fill((2048,2048), .015625)
            @tf.function(jit_compile=True)
            def op(x,y):
                with tf.device('/CPU:0'): return tf.matmul(x,y)
            def fn(): return op(a,b).numpy()
            def error(x): return float(abs(x-1).max())
        for _ in range(2): value = fn()
        maximum = error(value)
        for _ in range(7):
            t0 = time.perf_counter(); fn(); samples.append((time.perf_counter()-t0)*1000)
        print(json.dumps(dict(version=version, rows=[dict(case='gemm-2048',device='cpu',dtype='f32',
              shape=[2048]*3,samples_ms=samples,max_abs=maximum,warmup=2,seed=1337,timing='host_synchronized')])))
        return 0
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(8)
    torch.manual_seed(1337)
    rows = []
    def measure(name, device, dtype, shape, fn, check):
        sync = torch.cuda.synchronize if device == 'cuda' else lambda: None
        for _ in range(2): fn()
        sync()
        error = float(check())
        if not math.isfinite(error) or error > .02:
            raise ValueError('PyTorch correctness failed')
        samples = []
        if device=='cuda': torch.cuda.reset_peak_memory_stats()
        for _ in range(7):
            t0 = time.perf_counter()
            fn(); sync()
            samples.append((time.perf_counter()-t0)*1000)
        rows.append(dict(case=name, device=device, dtype=dtype, shape=shape, samples_ms=samples,
                         max_abs=error, warmup=2, seed=1337, timing='host_synchronized',cpu_threads=torch.get_num_threads() if device=='cpu' else None))
        if device=='cuda':
            free,total=torch.cuda.mem_get_info()
            rows[-1].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),driver_free_bytes=free,driver_total_bytes=total,
                            cuda_build=int(torch.version.cuda.split('.')[0])*1000+int(torch.version.cuda.split('.')[1])*10)
    for device, dtype in [('cpu', torch.float32)] + ([('cuda', torch.bfloat16)] if torch.cuda.is_available() else []):
        a = torch.full((2048,2048), .03125, device=device, dtype=dtype)
        b = torch.full_like(a, .015625)
        c = torch.empty_like(a)
        measure('gemm-2048', device, 'f32' if dtype == torch.float32 else 'bf16', [2048]*3,
                lambda: torch.mm(a, b, out=c), lambda: (c.float()-1).abs().max().item())
    unavailable = []
    if torch.cuda.is_available() and not torch.backends.cuda.is_flash_attention_available():
        unavailable.append(dict(case='flash-fwd-bwd',framework='torch',reason='installed Windows PyTorch wheel has no FlashAttention kernel'))
    if torch.cuda.is_available() and torch.backends.cuda.is_flash_attention_available():
        from torch.nn.attention import sdpa_kernel, SDPBackend
        q = torch.zeros((1,2048,8,64), device='cuda', dtype=torch.bfloat16).transpose(1,2).detach().requires_grad_()
        k = torch.zeros_like(q).requires_grad_()
        v = torch.full_like(q, .25).requires_grad_()
        dout = torch.full_like(q, .03125)
        output = None
        def attention():
            nonlocal output
            q.grad = k.grad = v.grad = None
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = F.scaled_dot_product_attention(q,k,v,is_causal=True)
                output.backward(dout)
        def check_attention():
            harmonic = torch.arange(1,2049,device='cuda',dtype=torch.float64).reciprocal().flip(0).cumsum(0).flip(0)/32
            return max((output.float()-.25).abs().max().item(), q.grad.abs().max().item(), k.grad.abs().max().item(),
                       (v.grad.double()-harmonic[None,None,:,None]).abs().max().item())
        measure('flash-fwd-bwd', 'cuda', 'bf16', [1,8,2048,64], attention, check_attention)
    print(json.dumps(dict(rows=rows, unavailable=unavailable, version=torch.__version__, cuda=torch.version.cuda)))
    return 0


