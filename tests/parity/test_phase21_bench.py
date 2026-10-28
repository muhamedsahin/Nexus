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


class BenchTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(case='gemm-2048',device='cpu',dtype='f32',shape=[2048]*3,
                        samples_ms=[1,2,3,4,5,6,7],warmup=2,seed=1337,max_abs=0,timing='host_synchronized')
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
        self.row['max_abs']=float('nan')
        with self.assertRaises(ValueError): validate(self.row)

    def test_no_cross_shape_claim(self):
        a = aggregate(self.runs,5)
        b = copy.deepcopy(a[0]); b['framework']='torch'; b['shape']=[512]*3
        self.assertEqual(claims(a+[b])[0]['status'],'no_data')
        b['shape']=[2048]*3; b['summary']['median_ms']=2
        self.assertEqual(claims(a+[b])[0]['status'],'failed')

    def test_duplicate_rejected(self):
        self.runs[0]['rows'].append(copy.deepcopy(self.row))
        with self.assertRaises(ValueError): aggregate(self.runs,5)

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

    def test_report_generation(self):
        document = dict(schema=1, rows=aggregate(self.runs,5))
        document['training'] = dict(status='measured',runs=[dict(model='124m',framework='nexus',repeat=0,
            summary=dict(median_ms=10,target_tokens_per_s=102300,peak_reserved_bytes=2000000000,
                         mfu_estimate=.25,hfu_full_recompute_estimate=1/3),first_step_ms=100,wall_s=2)])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            report(path,document)
            before = (path/'REPORT.md').read_bytes()
            text = before.decode('utf-8')
            for metric in ['Hedef token/s','Tepe ayrılmış GB','MFU tahmini','HFU tahmini','Parity',
                           '102300.000000','2.000000','Doğrulama kaybına erişme süresi ölçülmedi']:
                self.assertIn(metric,text)
            report(path,document)
            self.assertEqual(before,(path/'REPORT.md').read_bytes())
            ET.parse(path/'timings.svg')
            self.assertTrue(json.loads((path/'claims.json').read_text())[0]['status']=='no_data')
            document['rows'][0]['summary']['median_ms']=1
            with self.assertRaises(ValueError): report(path,document)


if __name__=='__main__': unittest.main()
