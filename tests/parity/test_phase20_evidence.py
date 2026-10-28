"""Phase 20: evidence must fail closed on missing, nonfinite or misaligned data.
Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf
öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.
"""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("phase20", ROOT / "benchmarks/runner/faz20_validate.py")
phase20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase20)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "run.jsonl"
        self.ref = self.root / "ref.json"
        self.config = dict(type="config", model="124m", precision="f32", parameters=124439808,
                           dim=768, layers=12, micro=1, seq=1024, peak_flops=0)
        self.rows = [dict(type="step", step=i, loss=2.0-i/100, lr=1e-4, ms=10+i,
                          target_tokens=1023, peak_allocated_bytes=100,
                          peak_reserved_bytes=200, reserved_bytes=200, inactive_split_bytes=20) for i in range(10)]
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.run.write_text("\n".join(json.dumps(r) for r in [self.config] + self.rows))
        self.ref.write_text(json.dumps({"losses": [r["loss"] for r in self.rows]}))

    def test_exact_and_threshold(self):
        self.assertTrue(phase20.compare(self.run, self.ref, 10)["passed"])
        losses = [r["loss"] for r in self.rows]
        losses[4] += .002
        self.ref.write_text(json.dumps({"losses": losses}))
        result = phase20.compare(self.run, self.ref, 10)
        self.assertFalse(result["passed"])
        self.assertEqual(result["worst_step"], 4)

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError): phase20.compare(self.run, self.ref, 100)
        self.rows[4]["step"] = 6
        self.save()
        with self.assertRaises(ValueError): phase20.read_run(self.run)

    def test_nonfinite_cannot_pass(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            self.rows[2]["loss"] = value
            self.save()
            with self.assertRaises(ValueError): phase20.compare(self.run, self.ref, 10)

    def test_metrics_denominators(self):
        result = phase20.summarize(self.run)
        self.assertEqual(result["median_ms"], 17)
        self.assertEqual(result["target_tokens_per_s"], 1023 * 1000 / 17)
        self.assertIsNone(result["mfu_estimate"])
        self.assertEqual(result["max_inactive_split_fraction"], .1)
        with self.assertRaises(ValueError): phase20.summarize(self.run, 6)

    def test_real_shard_extraction_and_integrity(self):
        shard = self.root / "shard.nxtk"
        shard.write_bytes(struct.pack("<4sIIIQQ32s", b"NXTK", 1, 2, 0, 8, 1, bytes(32))
                          + struct.pack("<8H", *range(8)) + struct.pack("<2Q", 0, 8))
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(dict(version=1, tokenizer=dict(name="gpt2", digest="00"*32, vocab_size=10),
            shards=[dict(path=shard.name, tokens=8, docs=1, sha256=phase20.digest(shard))])))
        target = self.root / "tokens.bin"
        phase20.extract(manifest, target, 4, 2)
        self.assertEqual(struct.unpack("<4i", target.read_bytes()), (2, 3, 4, 5))
        with self.assertRaises(ValueError): phase20.extract(manifest, target, 4)
        shard.write_bytes(shard.read_bytes()[:-1])
        with self.assertRaises(ValueError): phase20.extract(manifest, self.root / "bad.bin", 4)


if __name__ == "__main__":
    unittest.main()
