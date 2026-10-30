"""Controlled numerical investigation; reuse the canonical trainer, no speed claims."""
import json,os,sys
from pathlib import Path
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'python/src'))
from nexus_bench.cli import capture
from nexus_bench.evidence import sha,write

def reference_run(layout,initial,out):
    import types,numpy as np,torch
    sys.path.insert(0,str(ROOT/'benchmarks/runner'))
    import faz9_nanogpt as reference
    source=Path.home()/'nexus-data/nanogpt'
    expected='7c01703240dbec5d554527dc666e35b3df8391d0b117fddc07afcf325a21d11c'
    assert sha(source/'model.py')==expected,'nanoGPT source differs from the pinned reference'
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model=reference.build(reference.load_nanogpt(str(source)),1024,50257,'cuda')
    reference.load_nexus_init(model,str(initial),1024)
    tokens=np.memmap(ROOT/'build-phase1/faz20/tokens.bin',dtype=np.int32,mode='r')
    def batches(step,micro):
        rows=torch.from_numpy(np.array(tokens[step*1024:(step+1)*1024]).reshape(1,1024).astype(np.int64)).to('cuda')
        if layout=='prefix': return rows[:,:-1].contiguous(),rows[:,1:].contiguous()
        targets=torch.full_like(rows,-1); targets[:,:-1]=rows[:,1:]
        return rows,targets
    args=types.SimpleNamespace(wd=.1,lr=.0006,compile=0,compile_mode='default',precision='f32',
                               accum=1,warmup=10,min_lr_ratio=.1,clip=1)
    losses,times,allocated,reserved=reference.train(args,model,batches,'cuda',100)
    write(out,dict(layout=layout,losses=losses,ms=times,peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,torch_version=torch.__version__,cuda_build=torch.version.cuda,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        parameter_count=sum(p.numel() for p in model.parameters()),
        valid_targets_per_step=1023,input_positions=1023 if layout=='prefix' else 1024,
        nanogpt_sha256=expected,precision='f32',seed=1337,lr=.0006))
    print(layout,'completed',len(losses),'steps')

if len(sys.argv)>1 and sys.argv[1]=='reference':
    reference_run(sys.argv[2],Path(sys.argv[3]),Path(sys.argv[4])); sys.exit(0)

release=ROOT/'docs/benchmarks/2026-10-29-plan/02-unsandboxed-baseline/release-execution.json'
race=ROOT/'docs/benchmarks/2026-10-29-plan/07-racecheck-retry/verification.json'
assert release.exists() and json.loads(release.read_text())['returncode']==0,'Release gate not finished/green'
assert race.exists() and json.loads(race.read_text())['passed'],'Racecheck not finished/green'
scratch=ROOT/'build-phase1/plan-parity-layout'; scratch.mkdir(exist_ok=False)
out=ROOT/'docs/benchmarks/2026-10-29-plan/08-parity-layout'; out.mkdir(exist_ok=False)
binary=ROOT/'build-phase1/benchmarks/runner/bench_cuda_phase20.exe'
initial=scratch/'initial.bin'; native=out/'nexus.jsonl'
command=[str(binary),'--model','124m','--tokens',str(ROOT/'build-phase1/faz20/tokens.bin'),
    '--out',str(native),'--steps','100','--warmup','10','--seq','1024','--precision','f32',
    '--lr','0.0006','--dump-init',str(initial)]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
result=capture(command,env,timeout=1800); write(out/'nexus-execution.json',result)
assert result['returncode']==0,'Native diagnostic failed'
write(out/'metadata.json',dict(purpose='numerical_correctness_investigation_not_speed_benchmark',
    native_sha256=sha(binary),tokens_sha256=sha(ROOT/'build-phase1/faz20/tokens.bin'),
    initial_sha256=sha(initial),reference_script_sha256=sha(ROOT/'benchmarks/runner/faz9_nanogpt.py'),
    experiment_script_sha256=sha(__file__),steps=100,precision='f32',lr=.0006,
    variables='Only reference input geometry changes; valid target tokens stay identical.'))
curves={}
for name,layout in [('prefix-0','prefix'),('prefix-1','prefix'),('full-ignore-last','full')]:
    target=out/(name+'.json')
    result=capture([str(ROOT/'.bench-venv-cuda/Scripts/python.exe'),str(Path(__file__).resolve()),
                    'reference',layout,str(initial),str(target)],env,timeout=1800)
    write(out/(name+'-execution.json'),result)
    assert result['returncode']==0,'Reference diagnostic failed: '+name
    curves[name]=json.loads(target.read_text())['losses']
    print(name,'completed',flush=True)
native_losses=[json.loads(line)['loss'] for line in native.read_text().splitlines()[1:]]
def difference(a,b):
    values=[abs(x-y) for x,y in zip(a,b,strict=True)]
    assert len(values)==100
    return dict(max_abs=max(values),worst_step=values.index(max(values)),passed=max(values)<=.001,errors=values)
summary={name:difference(native_losses,curve) for name,curve in curves.items()}
summary['reference_self_repeat']=difference(curves['prefix-0'],curves['prefix-1'])
write(out/'comparison.json',summary)
print(json.dumps({name:{k:v for k,v in row.items() if k!='errors'} for name,row in summary.items()},indent=2))
