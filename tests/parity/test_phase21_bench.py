"""Phase 21 regression tests for evidence integrity and report generation.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python/src'))
from nexus_bench.cli import aggregate, claims, report, stats, validate, regression, capture
from nexus_bench.evidence import SCHEMA, distribution, median_interval, validate_document


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(case='gemm-2048',device='cpu',dtype='f32',shape=[2048]*3,
                        samples_ms=[4]*7,warmup=2,seed=1337,max_abs=0,timing='host_synchronized')
        self.runs = [dict(framework='nexus',repeat=i,rows=[copy.deepcopy(self.row)]) for i in range(5)]

    def test_bench_deterministic(self):
        first = aggregate(self.runs,5)
        self.assertEqual(first,aggregate(copy.deepcopy(self.runs),5))
        self.assertEqual(first[0]['summary']['median_ms'],4)
        self.assertEqual(first[0]['summary']['variance_ms2'],0)
        with self.assertRaises(ValueError): aggregate(self.runs[:-1],5)

    def test_finite_and_complete(self):
        for samples in [[1]*4,[1,2,3,4,0],[1,2,3,4,float('nan')],[1,2,3,4,float('inf')]]:
            with self.assertRaises(ValueError): stats(samples)
        for name,value in [('max_abs',float('nan')),('max_abs',False),('warmup',True),
                           ('warmup',1.5),('seed',1337.0)]:
            row=copy.deepcopy(self.row); row[name]=value
            with self.assertRaises(ValueError): validate(row)

    def test_no_cross_shape_claim(self):
        a = aggregate(self.runs,5)
        b = copy.deepcopy(a[0]); b['framework']='torch'; b['shape']=[512]*3
        self.assertEqual(claims(a+[b])[0]['status'],'no_data')
        b['shape']=[2048]*3; b['summary']['median_ms']=2
        self.assertEqual(claims(a+[b])[0]['status'],'failed')

    def test_duplicate_rejected(self):
        self.runs[0]['rows'].append(copy.deepcopy(self.row))
        with self.assertRaises(ValueError): aggregate(self.runs,5)

    def test_repeat_identity(self):
        for invalid in [0,-1,5,True,1.0]:
            runs=copy.deepcopy(self.runs); runs[1]['repeat']=invalid
            with self.assertRaises(ValueError): aggregate(runs,5)

    def test_noisy_iterations_cannot_hide_behind_stable_medians(self):
        runs=copy.deepcopy(self.runs)
        for run in runs: run['rows'][0]['samples_ms']=[1]*6+[100]
        noisy=aggregate(runs,5)
        reference=copy.deepcopy(self.runs)
        for run in reference:
            run['framework']='torch'; run['rows'][0]['samples_ms']=[2]*7
        rows=aggregate(runs+reference,5)
        self.assertEqual(noisy[0]['summary']['cv'],0)
        self.assertGreater(noisy[0]['dispersion']['max_within_process_cv'],2)
        self.assertEqual(claims(rows)[0]['status'],'no_data')
        stable=aggregate(self.runs,5)
        self.assertEqual(regression(noisy,stable,'same','same')[0]['status'],'no_data')

    def test_distribution_and_process_bootstrap(self):
        self.assertEqual(distribution([1,2,3,4,5,6,7])['iqr_ms'],3)
        fixed=median_interval([4]*5)
        self.assertEqual((fixed['low_ms'],fixed['high_ms']),(4,4))
        samples=[1,2,3,4,5]
        spread=median_interval(samples)
        self.assertEqual((spread['low_ms'],spread['high_ms']),(1,5))
        self.assertEqual(spread,median_interval(list(reversed(samples))))
        self.assertLessEqual(spread['low_ms'],3)
        self.assertGreaterEqual(spread['high_ms'],3)

    def test_legacy_rows_do_not_supply_missing_dispersion(self):
        current=aggregate(self.runs,5)
        legacy=aggregate(self.runs,5,schema=1)
        self.assertNotIn('dispersion',legacy[0])
        self.assertEqual(regression(current,legacy,'same','same')[0]['status'],'no_data')
        reference=copy.deepcopy(legacy[0]); reference['framework']='torch'
        self.assertEqual(claims(legacy+[reference])[0]['status'],'no_data')
        with tempfile.TemporaryDirectory() as temp:
            report(temp,dict(schema=1,repeats=5,runs=self.runs,rows=legacy))

    def test_report_rejects_edited_dispersion(self):
        rows=aggregate(self.runs,5)
        rows[0]['dispersion']['max_within_process_cv']=.01
        with self.assertRaises(ValueError):
            validate_document(dict(schema=SCHEMA,repeats=5,runs=self.runs,rows=rows))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                report(temp,dict(schema=SCHEMA,repeats=5,runs=self.runs,rows=rows))

    def test_version_one_archive_replays_exactly(self):
        source=Path(__file__).resolve().parents[2]/'docs/benchmarks/2026-10-28-faz21/run-03'
        original=json.loads((source/'results.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temp:
            report(temp,original)
            for name in ['REPORT.md','claims.json','timings.svg']:
                self.assertEqual((source/name).read_bytes(),(Path(temp)/name).read_bytes())

    def test_regression_machine_and_threshold(self):
        old = aggregate(self.runs,5)
        new = copy.deepcopy(old); new[0]['summary']['median_ms'] *= 1.04
        self.assertEqual(regression(new,old,'same GPU driver','same GPU driver')[0]['status'],'failed')
        with self.assertRaises(ValueError): regression(new,old,'machine A','machine B')
        new[0]['summary']['cv']=.5
        self.assertEqual(regression(new,old,'same','same')[0]['status'],'no_data')

    def test_timeout_kills_owned_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'late.txt'
            code = 'import time,pathlib;time.sleep(1);pathlib.Path('+repr(str(target))+').write_text("late")'
            result = capture([sys.executable,'-c',code],timeout=.1)
            self.assertEqual(result['returncode'],124)
            time.sleep(1.1)
            self.assertFalse(target.exists())

    def test_regression_rejects_nonfinite_baseline(self):
        current=aggregate(self.runs,5)
        for value in [float('nan'),float('inf'),0,-1]:
            old=copy.deepcopy(current);old[0]['summary']['median_ms']=value
            with self.assertRaises(ValueError): regression(current,old,'same','same')
        for value in [float('nan'),float('inf'),-1,True,None,'0']:
            old=copy.deepcopy(current); old[0]['dispersion']['max_within_process_cv']=value
            with self.assertRaises(ValueError): regression(current,old,'same','same')

    def test_report_generation(self):
        document = dict(schema=SCHEMA,repeats=5,runs=self.runs,rows=aggregate(self.runs,5))
        document['training'] = dict(status='measured',scheduler_warmup_steps=2,timing_discard_steps=5,
            runs=[dict(model='124m',framework='nexus',repeat=0,
            summary=dict(median_ms=10,target_tokens_per_s=102300,peak_reserved_bytes=2000000000,
                         mfu_estimate=.25,hfu_full_recompute_estimate=1/3),first_step_ms=100,wall_s=2)])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            report(path,document)
            before = (path/'REPORT.md').read_bytes()
            text = before.decode('utf-8')
            for metric in ['Hedef token/s','Tepe ayrılmış GB','MFU tahmini','HFU tahmini','Parity',
                           '102300.000000','2.000000','Doğrulama kaybına erişme süresi ölçülmedi',
                           'En yüksek süreç içi CV','IQR ms','%95 aralığı ms',
                           'LR scheduler ısınması: 2 adım','zamanlamadan çıkarılan ilk adımlar: 5']:
                self.assertIn(metric,text)
            report(path,document)
            self.assertEqual(before,(path/'REPORT.md').read_bytes())
            ET.parse(path/'timings.svg')
            self.assertTrue(json.loads((path/'claims.json').read_text())[0]['status']=='no_data')
            document['rows'][0]['summary']['median_ms']=1
            with self.assertRaises(ValueError): report(path,document)


if __name__=='__main__': unittest.main()
