"""Phase 21: versioned evidence, isolated sequential runs, strict comparisons and reports.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

if __package__:
    from .evidence import SCHEMA, aggregate, claims, key, regression, report, sha, stats, validate, validate_document, write
    from .workers import worker
else:
    from evidence import SCHEMA, aggregate, claims, key, regression, report, sha, stats, validate, validate_document, write
    from workers import worker

def capture(command, env=None, timeout=30):
    close_job = None
    p = None
    try:
        p = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding='utf-8', errors='replace', start_new_session=os.name!='nt',
                             creationflags=4 if os.name=='nt' else 0)
        if os.name=='nt':
            if __package__:
                from .windows_job import adopt_and_resume
            else:
                from windows_job import adopt_and_resume
            close_job=adopt_and_resume(p)
        try:
            stdout,stderr = p.communicate(timeout=timeout)
            code = p.returncode
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                close_job(); close_job=None
            else:
                os.killpg(p.pid,signal.SIGKILL)
            stdout,stderr = p.communicate(timeout=30)
            code = 124
            stderr += f'\nBenchmark timeout after {timeout}s; owned process tree terminated.'
        return dict(command=[str(x) for x in command], returncode=code, stdout=stdout, stderr=stderr)
    except (OSError,RuntimeError) as exc:
        if p is not None:
            p.kill(); p.communicate(timeout=30)
        return dict(command=[str(x) for x in command], returncode=-1, stdout='', stderr=str(exc))
    except BaseException:
        if p is not None and p.poll() is None:
            if os.name != 'nt': os.killpg(p.pid,signal.SIGKILL)
            elif close_job is None: p.kill()
        raise
    finally:
        if close_job is not None: close_job()


def training(args, out, env):
    repo = Path(args.repo).resolve()
    native = repo/'build-phase1/benchmarks/runner'/('bench_cuda_phase20.exe' if os.name == 'nt' else 'bench_cuda_phase20')
    if not args.tokens or not args.nanogpt or not native.is_file():
        return dict(status='no_data', reason='provide --tokens, --nanogpt, --repo and build bench_cuda_phase20')
    sys.path.insert(0,str(repo/'benchmarks/runner'))
    from faz20_validate import summarize, read_run
    directory = out/'training'; directory.mkdir()
    token_path = str(Path(args.tokens).resolve())
    scheduler_warmup_steps, timing_discard_steps = 2, 5
    result = dict(tokens_sha256=sha(token_path), native_sha256=sha(native), runs=[], status='measured',
                  scheduler_warmup_steps=scheduler_warmup_steps,timing_discard_steps=timing_discard_steps)
    for model in ['124m','1.3b']:
        for i in range(args.repeats):
            raw = directory/f'{model}-nexus-{i}.jsonl'
            command = [str(native),'--model',model,'--tokens',token_path,'--out',str(raw),
                       '--steps','15','--warmup',str(scheduler_warmup_steps),'--lr','0.0001','--seq','1024','--precision','bf16',
                       '--optimizer','adamw' if model == '124m' else 'fused8']
            init = directory/'initial.bin'
            if model == '124m' and i == 0: command += ['--dump-init',str(init)]
            if model == '124m': command += ['--compile','1']
            t0 = time.perf_counter()
            execution = capture(command,env,timeout=1800)
            write(directory/f'{model}-nexus-{i}-execution.json',execution)
            if execution['returncode']:
                result['status'] = 'failed'; result['reason'] = f'{model} native training failed'; return result
            summary = summarize(raw,warmup=timing_discard_steps)
            result['runs'].append(dict(model=model, framework='nexus',repeat=i, summary=summary,
                                       wall_s=time.perf_counter()-t0, first_step_ms=read_run(raw)[1][0]['ms']))
            if model != '124m': continue
            ref = directory/f'124m-torch-{i}.json'
            command = [str(Path(args.reference_python).resolve()), str(repo/'benchmarks/runner/faz9_nanogpt.py'),
                       'curve','--nanogpt',str(Path(args.nanogpt).resolve()),'--init',str(init),'--batches',token_path,
                       '--micro','1','--seq','1024','--accum','1','--steps','15','--warmup',str(scheduler_warmup_steps),'--lr','0.0001',
                       '--precision','bf16','--compile','1','--compile-mode','max-autotune','--out',str(ref)]
            t0 = time.perf_counter(); execution = capture(command,env,timeout=args.compile_timeout)
            write(directory/f'124m-torch-{i}-execution.json',execution)
            if execution['returncode']:
                result['status']='failed'; result['reason']='torch.compile reference failed; see raw execution log'; return result
            reference = json.loads(ref.read_text())
            _, native_steps = read_run(raw)
            if len(reference['losses']) != 15 or any(not math.isfinite(x) for x in reference['losses']):
                raise ValueError('invalid reference loss trajectory')
            maximum = max(abs(a['loss']-b) for a,b in zip(native_steps,reference['losses'],strict=True))
            # Dalga 1 (docs/benchmarks/2026-10-31-dalga1): BF16'yi BF16'ya karsi olcmek PyTorch'un kendi BF16 hatasini
            # Nexus'a yukluyordu (torch.compile BF16 FP32'den 1.8e-3, Nexus BF16 4.4e-4). Kapi: her iki BF16 egrisi
            # ayni agirlikli deterministik FP32 PyTorch egrisine karsi; Nexus BF16 <= 1e-3 gecer.
            if i == 0:
                truth_path = directory/'124m-torch-f32-truth.json'
                command = [str(Path(args.reference_python).resolve()), str(repo/'benchmarks/runner/faz9_nanogpt.py'),
                           'curve','--nanogpt',str(Path(args.nanogpt).resolve()),'--init',str(init),'--batches',token_path,
                           '--micro','1','--seq','1024','--accum','1','--steps','15','--warmup',str(scheduler_warmup_steps),
                           '--lr','0.0001','--precision','f32','--deterministic','1','--out',str(truth_path)]
                execution = capture(command, dict(env, CUBLAS_WORKSPACE_CONFIG=':4096:8'), timeout=args.compile_timeout)
                write(directory/'124m-torch-f32-truth-execution.json',execution)
                if execution['returncode']:
                    result['status']='failed'; result['reason']='FP32 truth reference failed; see raw execution log'; return result
                truth = json.loads(truth_path.read_text())['losses']
                if len(truth) != 15 or any(not math.isfinite(x) for x in truth):
                    raise ValueError('invalid FP32 truth trajectory')
            nexus_vs_truth = max(abs(a['loss']-b) for a,b in zip(native_steps,truth,strict=True))
            torch_vs_truth = max(abs(a-b) for a,b in zip(reference['losses'],truth,strict=True))
            ref_summary=stats(reference['ms'][timing_discard_steps:])
            ref_summary['target_tokens_per_s']=1023*1000/ref_summary['median_ms']
            peak=summary['config']['peak_flops']
            flops=1023*(6*(reference['parameter_count']-1024*768)+12*12*768*1023)
            ref_summary['mfu_estimate']=flops*1000/ref_summary['median_ms']/peak if peak else None
            result['runs'].append(dict(model=model,framework='torch.compile-max-autotune',repeat=i,
                summary=ref_summary,first_step_ms=reference['ms'][0],wall_s=time.perf_counter()-t0,
                peak_reserved_gb=reference['peak_reserved_gb'],max_loss_abs=maximum,
                nexus_bf16_vs_fp32_truth=nexus_vs_truth,torch_bf16_vs_fp32_truth=torch_vs_truth,
                parity_reference='deterministic PyTorch FP32, same initial weights and tokens',
                cuda_build=reference.get('cuda_build'),
                parity_passed=nexus_vs_truth <= 1e-3))
            print('training',model,i,'measured',flush=True)
    torch_runs = [r for r in result['runs'] if r['framework'].startswith('torch')]
    native_runs = [r for r in result['runs'] if r['framework']=='nexus' and r['model']=='124m']
    native_ms = statistics.median(r['summary']['median_ms'] for r in native_runs)
    ref_ms = statistics.median(r['summary']['median_ms'] for r in torch_runs)
    result['124m_claim'] = dict(status='passed' if all(r['parity_passed'] for r in torch_runs) and native_ms<=ref_ms else 'failed',
                               nexus_median_ms=native_ms,torch_median_ms=ref_ms,reason='speed AND Nexus BF16 within 1e-3 of deterministic PyTorch FP32 truth')
    memory_passed=max(r['summary']['peak_reserved_bytes'] for r in native_runs)<=max(r['peak_reserved_gb'] for r in torch_runs)*1e9
    result['124m_claim']['memory_passed']=memory_passed
    if not memory_passed: result['124m_claim']['status']='failed'
    if any(r['summary']['config'].get('cuda_build') != t.get('cuda_build') for r,t in zip(native_runs,torch_runs,strict=True)):
        result['124m_claim'].update(status='no_data',reason='CUDA build differs; D.2 exact-version comparison condition unmet; inspect separate parity flags')
    result['1.3b_claim'] = dict(status='no_data',reason='fused8 delayed clipping has no matching PyTorch optimizer reference')
    return result


def inference(args, out, env):
    if not args.inference_weights:
        return dict(status='no_data',reason='provide --inference-weights GPT2.nxg for real 124M TTFT/TPOT')
    result = dict(status='measured',weights_sha256=sha(args.inference_weights),runs=[])
    expected = None
    for i in range(args.repeats):
        execution = capture([str(Path(args.native).resolve()),'--infer',str(Path(args.inference_weights).resolve())],env,timeout=300)
        write(out/'raw'/f'inference-{i}.json',execution)
        if execution['returncode']:
            return dict(status='failed',reason='native inference failed; see raw log')
        row = json.loads(execution['stdout'])
        stats(row['ttft_ms']); stats(row['tpot_ms'])
        if expected is None: expected = row['tokens']
        if row['tokens'] != expected: raise ValueError('inference tokens differ between independent processes')
        result['runs'].append(row)
    result['ttft'] = stats([statistics.median(r['ttft_ms']) for r in result['runs']])
    result['tpot'] = stats([statistics.median(r['tpot_ms']) for r in result['runs']])
    return result


def legacy(args,out,env):
    exe = Path(args.repo).resolve()/'build-phase1/benchmarks'/('bench_competitor_protocol.exe' if os.name=='nt' else 'bench_competitor_protocol')
    if not exe.is_file(): return dict(status='no_data',reason='build bench_competitor_protocol for historical CPU matrix')
    measurements = {}
    workers = set()
    for i in range(args.repeats):
        execution = capture([str(exe)],env,timeout=180)
        write(out/'raw'/f'legacy-cpu-{i}.json',execution)
        if execution['returncode']: return dict(status='failed',reason='historical CPU benchmark failed')
        doc = json.loads(execution['stdout'])
        workers.add(doc['pool_workers'])
        for row in doc['rows']:
            if not math.isfinite(row['seconds']) or row['seconds']<=0: raise ValueError('invalid historical timing')
            measurements.setdefault(row['name'],[]).append(row['seconds']*1000)
    return dict(status='measured',actual_pool_workers=sorted(workers),
                protocol='Five fresh processes; each observation is the existing warmup+3/4-iteration average, not a single invocation.',
                comparison='No speed claim: historical label says 8 threads but pool count is reported separately.',
                rows={name:dict(samples_ms=values,summary=stats(values)) for name,values in measurements.items()})


def run(args):
    if args.repeats < 5:
        raise ValueError('prompt protocol requires at least five independent repetitions')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'raw').mkdir()
    env = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               NEXUS_CPU_THREADS='8', PYTHONHASHSEED='1337',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8')
    temp = out / 'scratch'
    temp.mkdir()
    env.update(TEMP=str(temp), TMP=str(temp))
    metadata = dict(schema=SCHEMA, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    platform=platform.platform(), python=sys.version, seed=1337, threads_requested=8, repeats=args.repeats,
                    machine=dict(host=platform.node(),processor=platform.processor(),logical_cpus=os.cpu_count()),
                    git=capture(['git','rev-parse','HEAD']), dirty=capture(['git','status','--porcelain']),
                    hardware=capture(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm','--format=csv']),
                    power=power_status(),
                    source_sha256=sha(__file__), native_sha256=sha(args.native) if Path(args.native).is_file() else None)
    write(out/'metadata.json', metadata)
    source_root=Path(args.repo).resolve()
    sources=['python/src/nexus_bench/cli.py','python/src/nexus_bench/evidence.py','python/src/nexus_bench/workers.py','python/src/nexus_bench/windows_job.py',
             'benchmarks/runner/bench_phase21.cpp','benchmarks/runner/bench_cuda_phase20.cpp',
             'benchmarks/runner/faz9_nanogpt.py','benchmarks/competitors/protocol.json']
    write(out/'source-manifest.json',{p:sha(source_root/p) for p in sources if (source_root/p).is_file()})
    runs, unavailable, failures = [], [], []
    for framework in ['nexus','torch','tensorflow','jax']:
        for repeat in range(args.repeats):
            command = ([str(Path(args.native).resolve())] if framework == 'nexus' else
                       [str(Path(framework_python(args, framework)).resolve()), str(Path(__file__).resolve()), '_worker', framework])
            result = capture(command, env, timeout=args.timeout)
            write(out/'raw'/f'{framework}-{repeat}.json', result)
            if result['returncode'] == 77:
                unavailable.append(dict(framework=framework, reason=result['stdout'].strip()))
                break
            if result['returncode']:
                failures.append(dict(framework=framework, repeat=repeat, returncode=result['returncode']))
                break
            try:
                if framework == 'nexus':
                    payload = [json.loads(line) for line in result['stdout'].splitlines() if line.startswith('{')]
                    rows = [r for r in payload if 'samples_ms' in r]
                    unavailable.extend(r for r in payload if r.get('status') == 'unavailable')
                else:
                    payload = json.loads(result['stdout'])
                    rows = payload['rows']
                    for row in rows: row['framework_version']=payload.get('version')
                    unavailable.extend(payload.get('unavailable',[]))
                for row in rows: validate(row)
            except (ValueError,TypeError,KeyError) as exc:
                failures.append(dict(framework=framework,repeat=repeat,reason='invalid worker evidence: '+str(exc)))
                break
            if not rows:
                failures.append(dict(framework=framework, repeat=repeat, reason='no measurements'))
                break
            runs.append(dict(framework=framework, repeat=repeat, rows=rows))
            print(framework, repeat, 'measured', flush=True)
    # Failed frameworks remain explicit; their partial repetitions cannot produce speed claims.
    failed_frameworks = {f['framework'] for f in failures}
    completed = [r for r in runs if r['framework'] not in failed_frameworks]
    unavailable=list({json.dumps(r,sort_keys=True):r for r in unavailable}.values())
    document = dict(schema=SCHEMA, repeats=args.repeats, rows=aggregate(completed,args.repeats), runs=runs,
                    unavailable=unavailable, failures=failures)
    if args.baseline:
        baseline = Path(args.baseline)
        previous = json.loads((baseline/'results.json').read_text(encoding='utf-8'))
        validate_document(previous)
        previous_meta = json.loads((baseline/'metadata.json').read_text(encoding='utf-8'))
        document['regression'] = regression(document['rows'],previous['rows'],
            [metadata['machine'],metadata['hardware']['stdout']],[previous_meta['machine'],previous_meta['hardware']['stdout']])
    if args.matrix == 'all':
        document['legacy_cpu'] = legacy(args,out,env)
        document['training'] = training(args, out, env)
        document['inference'] = inference(args,out,env)
    write(out/'results.json', document)
    report(out, document)
    freeze=capture([args.reference_python,'-m','pip','freeze'],env)
    write(out/'environment-lock.json',freeze)
    if freeze['returncode']==0: (out/'reference-requirements.lock').write_text(freeze['stdout'],encoding='utf-8')
    for name, extra in (('jax', args.jax_python), ('tensorflow', args.tensorflow_python)):
        if not extra: continue
        freeze=capture([extra,'-m','pip','freeze'],env)
        write(out/f'{name}-environment-lock.json',freeze)
        if freeze['returncode']==0: (out/f'{name}-requirements.lock').write_text(freeze['stdout'],encoding='utf-8')
    write(out/'telemetry-end.json', capture(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used','--format=csv']))
    print('Report:',out/'REPORT.md',flush=True)
    return 1 if failures or any(document.get(k,{}).get('status')=='failed' for k in ['legacy_cpu','training','inference']) or any(r['status']=='failed' for r in document.get('regression',[])) else 0


def framework_python(args, framework):
    """Her referans kendi yorumlayicisinda: TensorFlow Windows'ta Python 3.14 tekerlegi sunmuyor (3.12 venv)."""
    if framework == 'jax' and args.jax_python: return args.jax_python
    if framework == 'tensorflow' and args.tensorflow_python: return args.tensorflow_python
    return args.reference_python


def power_status():
    """Zaman olcumu fiste yapilmali (KURAL K5): Win32_Battery.BatteryStatus 2 = AC, 1 = pil; pilsiz makine None."""
    if os.name != 'nt': return dict(status=None, reason='non-Windows: not recorded')
    result = capture(['powershell','-NoProfile','-c','(Get-CimInstance Win32_Battery).BatteryStatus'])
    value = result['stdout'].strip()
    return dict(status=int(value) if value.isdigit() else None, on_ac=value == '2', raw=result)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '_worker':
        return worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    bench = commands.add_parser('bench')
    bench.add_argument('--matrix',choices=['all','kernels'],default='all')
    bench.add_argument('--native',default='build-phase1/benchmarks/runner/bench_phase21.exe' if os.name == 'nt' else 'build-phase1/benchmarks/runner/bench_phase21')
    # Nexus ile ayni CUDA toolkit'li torch (Dalga 0: 2.15.0+cu134) varsa o; yoksa eski cu132 venv.
    venv_python=lambda name: Path.cwd()/name/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    ref_python=next((p for p in (venv_python('.bench-venv-cu134'), venv_python('.bench-venv-cuda')) if p.is_file()), Path(sys.executable))
    bench.add_argument('--reference-python',default=str(ref_python) if ref_python.is_file() else sys.executable)
    bench.add_argument('--jax-python')
    bench.add_argument('--tensorflow-python')
    bench.add_argument('--repo',default=str(Path.cwd()))
    bench.add_argument('--tokens')
    bench.add_argument('--nanogpt')
    bench.add_argument('--inference-weights')
    bench.add_argument('--out',default='docs/benchmarks/faz21/'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    bench.add_argument('--repeats',type=int,default=5)
    bench.add_argument('--timeout',type=int,default=180)
    bench.add_argument('--compile-timeout',type=int,default=600)
    bench.add_argument('--baseline')
    regenerate = commands.add_parser('report')
    regenerate.add_argument('--out',required=True)
    args = parser.parse_args(argv)
    if args.command == 'report':
        report(args.out,json.loads((Path(args.out)/'results.json').read_text(encoding='utf-8')))
        return 0
    if args.timeout <= 0 or args.compile_timeout <= 0: parser.error('timeouts must be positive')
    repo=Path(args.repo).resolve()
    if not args.tokens and (repo/'build-phase1/faz20/tokens.bin').is_file(): args.tokens=str(repo/'build-phase1/faz20/tokens.bin')
    if not args.nanogpt and (Path.home()/'nexus-data/nanogpt/model.py').is_file(): args.nanogpt=str(Path.home()/'nexus-data/nanogpt')
    if not args.inference_weights and (Path.home()/'nexus-data/gpt2/gpt2.nxg').is_file(): args.inference_weights=str(Path.home()/'nexus-data/gpt2/gpt2.nxg')
    jax_python=repo/'build-phase1/faz19-xla-venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not args.jax_python and jax_python.is_file(): args.jax_python=str(jax_python)
    for attr, venv in (('jax_python', '.bench-venv-jax'), ('tensorflow_python', '.bench-venv-tf')):
        candidate=repo/venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if not getattr(args, attr) and candidate.is_file(): setattr(args, attr, str(candidate))
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
