"""Phase 21: versioned evidence, isolated sequential runs, strict comparisons and reports.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

ATTRIBUTION = ('Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf '
               'öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.')
SCHEMA = 1


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stats(samples):
    if len(samples) < 5 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError('at least five positive finite timing samples required')
    mean = statistics.mean(samples)
    return dict(median_ms=statistics.median(samples), mean_ms=mean, min_ms=min(samples),
                max_ms=max(samples), std_ms=statistics.pstdev(samples), variance_ms2=statistics.pvariance(samples),
                cv=statistics.pstdev(samples)/mean)


def key(row):
    return row['case'], row['device'], row['dtype'], tuple(row['shape'])


def validate(row):
    stats(row['samples_ms'])
    if row['warmup'] < 1 or not math.isfinite(row['max_abs']) or row['max_abs'] < 0 or row['max_abs'] > .02:
        raise ValueError('missing warmup or failed correctness gate')
    if row['timing'] != 'host_synchronized':
        raise ValueError('incompatible timing method')
    if row['seed'] != 1337 or not row['shape'] or any(type(x) is not int or x <= 0 for x in row['shape']):
        raise ValueError('invalid seed/shape')
    return row


def aggregate(runs, repeats):
    groups = {}
    for run in runs:
        seen = set()
        for row in run['rows']:
            validate(row)
            identity = key(row)
            if identity in seen:
                raise ValueError('duplicate case in process')
            seen.add(identity)
            groups.setdefault((run['framework'], identity), []).append(row)
    rows = []
    for (framework, identity), values in sorted(groups.items()):
        if len(values) != repeats:
            raise ValueError('incomplete independent repetitions')
        for name in ['cuda_build','cpu_threads','framework_version']:
            if len({v.get(name) for v in values}) != 1: raise ValueError(name+' changed during repetitions')
        medians = [stats(v['samples_ms'])['median_ms'] for v in values]
        rows.append(dict(framework=framework, case=identity[0], device=identity[1], dtype=identity[2],
                         shape=list(identity[3]), process_medians_ms=medians, summary=stats(medians),
                         max_abs=max(v['max_abs'] for v in values), timing='host_synchronized',cuda_build=values[0].get('cuda_build'),
                         cpu_threads=values[0].get('cpu_threads'),framework_version=values[0].get('framework_version')))
        rows[-1]['memory']={name:max((v[name] for v in values if name in v),default=None)
                           for name in ['peak_allocated_bytes','peak_reserved_bytes']}
    return rows


def claims(rows):
    indexed = {(r['framework'], key(r)): r for r in rows}
    result = []
    for row in rows:
        if row['framework'] != 'nexus':
            continue
        reference = indexed.get(('torch', key(row)))
        ratio = reference['summary']['median_ms'] / row['summary']['median_ms'] if reference else None
        noisy = reference is not None and max(row['summary']['cv'],reference['summary']['cv']) > .1
        incompatible = reference is not None and row['device']=='cuda' and row.get('cuda_build') != reference.get('cuda_build')
        wrong_threads = reference is not None and row['device']=='cpu' and row.get('cpu_threads') != reference.get('cpu_threads')
        result.append(dict(case=row['case'], device=row['device'], dtype=row['dtype'],
                           status='no_data' if noisy or incompatible or wrong_threads else ('passed' if ratio >= 1 else 'failed') if ratio is not None else 'no_data',
                           reason='matched PyTorch reference unavailable' if reference is None else 'CPU thread counts differ' if wrong_threads else 'CUDA build differs; D.2 exact-version condition unmet' if incompatible else 'CV >10%; insufficient stability' if noisy else 'same shape/dtype/timing',
                           torch_over_nexus=ratio))
    for case in ['train-124m', 'train-1.3b', 'time-to-validation-loss', 'inference-7b-int4', 'cluster-7b-mfu']:
        result.append(dict(case=case, status='no_data', reason='requires matched measured reference and correctness evidence'))
    return result


def regression(current, previous, current_machine, previous_machine, limit=.03):
    if not math.isfinite(limit) or limit < 0: raise ValueError('invalid regression threshold')
    if current_machine != previous_machine:
        raise ValueError('baseline hardware/driver differs; refuse performance comparison')
    old = {(r['framework'],key(r)):r for r in previous}
    rows = []
    for row in current:
        before = old.get((row['framework'],key(row)))
        if before is None: continue
        for item in [row,before]:
            median=item['summary']['median_ms']; cv=item['summary']['cv']
            if isinstance(median,bool) or not math.isfinite(median) or median<=0 or not math.isfinite(cv) or cv<0:
                raise ValueError('nonfinite or invalid regression statistics')
        if any(row.get(name)!=before.get(name) for name in ['cuda_build','cpu_threads','framework_version']):
            raise ValueError('baseline dependency/thread/CUDA version differs')
        ratio = row['summary']['median_ms']/before['summary']['median_ms']-1
        noisy = max(row['summary']['cv'],before['summary']['cv']) > .1
        rows.append(dict(framework=row['framework'],case=row['case'],change=ratio,
                         status='no_data' if noisy else 'failed' if ratio>limit else 'passed',
                         reason='CV >10%; insufficient stability' if noisy else 'same hardware/driver/shape/dtype'))
    return rows


def report(directory, document):
    if document['schema'] != SCHEMA:
        raise ValueError('unsupported evidence schema')
    if 'runs' in document:
        failed = {r['framework'] for r in document.get('failures',[])}
        rebuilt = aggregate([r for r in document['runs'] if r['framework'] not in failed],document['repeats'])
        if rebuilt != document['rows']: raise ValueError('report rows disagree with raw measurements')
    for row in document['rows']:
        # Recompute instead of trusting edited/precomputed aggregates.
        if row['summary'] != stats(row['process_medians_ms']):
            raise ValueError('aggregate disagrees with raw process medians')
    directory = Path(directory)
    claim_rows = claims(document['rows'])
    if document.get('training',{}).get('124m_claim'):
        for row in claim_rows:
            if row['case']=='train-124m': row.update(document['training']['124m_claim'])
    write(directory / 'claims.json', claim_rows)
    lines = ['# Faz 21 benchmark raporu', '', ATTRIBUTION, '',
             'Çekirdek tablosu aynı oturumda, sıralı bağımsız süreçlerde ölçüldü. Her süreçte 2 ısınma ve 7 örnek;',
             'tablo bağımsız süreç medyanlarının istatistiğidir. Ham sonuçlar raw/ altında saklanır.', '',
             '| İş | Framework | Cihaz | Tip | Medyan ms | Min | Max | Std | CV |',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for row in document['rows']:
        s = row['summary']
        lines.append(f"| {row['case']} | {row['framework']} | {row['device']} | {row['dtype']} | "
                     f"{s['median_ms']:.6f} | {s['min_ms']:.6f} | {s['max_ms']:.6f} | {s['std_ms']:.6f} | {s['cv']:.3f} |")
    lines += ['', '## Hedefler', '', '| Hedef | Cihaz | Durum | PyTorch/Nexus süre oranı |', '|---|---|---|---:|']
    for claim in claim_rows:
        ratio = claim.get('torch_over_nexus')
        lines.append(f"| {claim['case']} | {claim.get('device', '')} | {claim['status']} | {ratio if ratio is not None else 'veri yok'} |")
    lines += ['', '## Sınırlar', '',
              'Kernel girdileri deterministik sabit değerlerdir; GEMM ve causal attention ileri/geri çıktıları analitik referansla sınanır.',
              'Bu mikro ölçümler uçtan uca eğitim veya dünyanın en hızlısı iddiası değildir.',
              'Resmî FlashAttention-2 paketinin %95 hedefi PyTorch SDPA ölçümüyle kapatılmaz.',
              'Faz 20 uzun eğitim, litgpt ve küme ön koşulları açık kalır. Ölçülmeyen hedefler no_data olarak yazılır.',
              'Saatler/güç kilitlenmez; sürücü ve GPU telemetrisi metadata.json içindedir.',
              'Farklı optimizer/dtype/shape sonuçları karşılaştırılmaz. MFU/HFU, bellek ve TTFT/TPOT eğitim/çıkarım kanıtlarında ayrıca raporlanır.', '',
              '![Süre grafiği](timings.svg)', '']
    if 'training' in document:
        training_result = document['training']
        lines += ['## Uçtan uca eğitim', '',
                  'Aynı gerçek token dosyası, başlangıç ağırlıkları, B=1, T=1024, AdamW beta=(0.9,0.95), LR=1e-4.',
                  'Her bağımsız süreç 15 adım; ilk 5 adım ısınma. PyTorch referansı torch.compile(max-autotune) kullanır.',
                  'İlk adım ve süreç duvar süresi ayrıca kaydedilir; derleme yükü sıcak süreye gizlenmez.', '',
                  'Tam ham sonuçlar results.json / training altında saklanır.', '',
                  f"Durum: {training_result['status']}; {training_result.get('reason','')}", '',
                  '| Model | Framework | Tekrar | Medyan ms | İlk adım ms | Süre s |', '|---|---|---:|---:|---:|---:|']
        for r in training_result.get('runs',[]):
            lines.append(f"| {r['model']} | {r['framework']} | {r['repeat']} | {r['summary']['median_ms']:.3f} | {r['first_step_ms']:.3f} | {r['wall_s']:.1f} |")
        lines += ['',str(training_result.get('124m_claim','')),str(training_result.get('1.3b_claim','')),'']
    if 'inference' in document:
        inference_result = document['inference']
        lines += ['## GPT-2 124M çıkarım (F16)', '',f"Durum: {inference_result['status']}",
                  '32 token istem + 32 token greedy üretim; tüm süreçlerde token dizisi birebir denetlenir.',
                  f"TTFT: {inference_result.get('ttft',{})}",f"TPOT: {inference_result.get('tpot',{})}", '']
    if 'legacy_cpu' in document:
        old = document['legacy_cpu']
        lines += ['## Tarihsel CPU matrisi', '',f"Durum: {old['status']}",old.get('protocol',''),old.get('comparison',''),
                  f"Gerçek pool worker sayısı: {old.get('actual_pool_workers',[])}",'',
                  '| İş | Medyan ms | Min | Max | Std |','|---|---:|---:|---:|---:|']
        for name,row in old.get('rows',{}).items():
            s = row['summary']; lines.append(f"| {name} | {s['median_ms']:.6f} | {s['min_ms']:.6f} | {s['max_ms']:.6f} | {s['std_ms']:.6f} |")
    if document.get('unavailable') or document.get('failures'):
        lines += ['', '## Kullanılamayan veya başarısız yollar', '', '```json',
                  json.dumps(dict(unavailable=document.get('unavailable',[]),failures=document.get('failures',[])),ensure_ascii=False,indent=2),'```','']
    (directory / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    # Separate per-case scales prevent a large CPU bar hiding a small CUDA measurement.
    groups = sorted({key(r) for r in document['rows']})
    height = max(100, len(document['rows'])*42 + len(groups)*28 + 40)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" role="img" aria-label="Median milliseconds per operation">',
           '<rect width="100%" height="100%" fill="white"/>']
    y = 22
    for identity in groups:
        rows = [r for r in document['rows'] if key(r) == identity]
        maximum = max(r['summary']['median_ms'] for r in rows)
        svg.append(f'<text x="12" y="{y}" font-size="14">{html.escape(str(identity))} — independent scale</text>')
        y += 20
        for row in rows:
            value = row['summary']['median_ms']
            svg += [f'<text x="12" y="{y+16}">{html.escape(row["framework"])}</text>',
                    f'<rect x="130" y="{y}" width="{600*value/maximum:.3f}" height="24" fill="#{"2563eb" if row["framework"] == "nexus" else "64748b"}"/>',
                    f'<text x="{145+600*value/maximum:.3f}" y="{y+16}">{value:.6f} ms</text>']
            y += 42
        y += 8
    svg.append('</svg>')
    (directory / 'timings.svg').write_text('\n'.join(svg), encoding='utf-8')
