#!/usr/bin/env python3
"""Rebuild the checked-in SPIR-V headers. Uses glslangValidator from PATH or C:/VulkanSDK."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / "src" / "backend" / "spirv"
OUT.mkdir(parents=True, exist_ok=True)

validator = shutil.which("glslangValidator")
if not validator:
    candidates = sorted(Path("C:/VulkanSDK").glob("*/Bin/glslangValidator.exe"), reverse=True)
    if candidates:
        validator = str(candidates[0])
    else:
        raise RuntimeError("glslangValidator not found on PATH or C:/VulkanSDK")

for source in sorted(ROOT.glob("*.comp")):
    binary = OUT / (source.stem + ".spv")
    subprocess.run([validator, "-V", "--target-env", "vulkan1.1", "-O",
                    str(source), "-o", str(binary)], check=True)
    data = binary.read_bytes()
    words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    lines = [f"0x{word:08x}u," for word in words]
    header = OUT / (source.stem + ".inc")
    header.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"{source.name}: {len(data)} bytes -> {header.name}")
