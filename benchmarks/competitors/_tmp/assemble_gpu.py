import json
from pathlib import Path
root = Path(r"C:\Users\muham\Desktop\Nexus\docs\benchmarks")
nexus = json.loads((root / "phase-6-nexus-gpu.json").read_text(encoding="utf-8"))
doc = {
    "status": "GPU tekeri yok, koşulmadı",
    "parts": [
        nexus,
        {"library": "torch", "device": "gpu", "threads": "device", "status": "GPU tekeri yok, koşulmadı", "rows": []},
        {"library": "tensorflow", "device": "gpu", "threads": "device", "status": "GPU tekeri yok, koşulmadı", "rows": []},
    ],
}
(root / "phase-6-gpu.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("phase6", len(nexus["rows"]))
