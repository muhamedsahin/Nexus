#!/usr/bin/env python3
"""Compile and execute a Nexus StableHLO module through independent XLA/PJRT CPU."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import tempfile

MODULE = r'''module @nexus_phase19 {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>
    } : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--producer", type=Path, help="Nexus test executable exporting its real IR and native result")
    args = parser.parse_args()
    try:
        import jax
        import jaxlib
        from jaxlib import xla_client
        import numpy as np
    except ImportError as exc:
        print(f"SKIP: jax/jaxlib/numpy required: {exc}")
        return 77
    fixture = None
    if args.producer:
        with tempfile.TemporaryDirectory(prefix="nexus-xla-") as tmp:
            module = Path(tmp) / "nexus.mlir"
            subprocess.run([str(args.producer.resolve()), "export-xla", str(module)], check=True, capture_output=True)
            source = module.read_text(encoding="utf-8")
            fixture = json.loads(Path(str(module) + ".json").read_text(encoding="utf-8"))
    else:
        source = args.module.read_text(encoding="utf-8") if args.module else MODULE
    client = jax.devices("cpu")[0].client
    device = jax.devices("cpu")[0]
    # Same single-device CompileOptionsProto as the C++ PJRT adapter.
    options_bytes = bytes.fromhex("1a0f08ffffffffffffffffff0120012801")
    options = xla_client.CompileOptions.ParseFromString(options_bytes)
    executable = client.compile_and_load(source, [device], options)
    if args.module and not fixture:
        result = {"status": "compiled", "backend": client.platform, "jax": jax.__version__}
    else:
        if fixture:
            host_inputs = [np.asarray(values, dtype=np.float32).reshape(shape)
                           for values, shape in zip(fixture["inputs"], fixture["shapes"], strict=True)]
            expected_outputs = [np.asarray(values, dtype=np.float32) for values in fixture["expected"]]
        else:
            a = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
            b = np.arange(6, dtype=np.float32).reshape(3, 2) / 20.0
            host_inputs, expected_outputs = [a, b], [(a @ b).reshape(-1)]
        outputs = executable.execute([jax.device_put(a, device) for a in host_inputs])
        if len(outputs) != len(expected_outputs):
            raise RuntimeError("XLA output count differs from the Nexus fixture")
        error = 0.0
        for actual_output, expected_output in zip(outputs, expected_outputs, strict=True):
            actual = np.asarray(actual_output).reshape(-1)
            expected = expected_output.reshape(-1)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            error = max(error, float(np.max(np.abs(actual - expected), initial=0)))
        result = {"status": "compiled_and_executed", "backend": client.platform,
                  "jax": jax.__version__, "max_abs": error,
                  "module_origin": "nexus_ir_export" if fixture else "self_test",
                  "actual": actual.tolist(), "expected": expected.tolist()}
    result.update({"jaxlib": jaxlib.__version__, "numpy": np.__version__,
                   "compile_options_hex": options_bytes.hex()})
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
