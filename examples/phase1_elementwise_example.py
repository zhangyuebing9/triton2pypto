"""
Phase 1 elementwise operation conversion - CPU runnable.

Demonstrates:
1. Parse TTIR MLIR text
2. Convert to adapter IR
3. Execute on CPU via NumPy runtime
"""

from triton_adapter import convert_ttir_to_pypto, MLIRParser, TTIRToPyptoConverter
from triton_adapter.runtime import run_on_numpy

# Sample TTIR for vector addition (from Triton kernel)
TTIR_ADD = """
module {
  tt.func @add_kernel(%arg0: !tt.ptr<fp32>, %arg1: !tt.ptr<fp32>, %arg2: !tt.ptr<fp32>) {
    %0 = tt.load %arg0 : tensor<128xf32>
    %1 = tt.load %arg1 : tensor<128xf32>
    %2 = arith.addf %0, %1 : tensor<128xf32>
    tt.store %arg2, %2 : tensor<128xf32>
  }
}
"""


def main() -> None:
    import numpy as np

    print("=" * 70)
    print("Phase 1 Example: TTIR → Adapter IR → NumPy (CPU)")
    print("=" * 70)

    # Step 1: Parse TTIR
    print("\nStep 1: Parse TTIR")
    parser = MLIRParser()
    operations = parser.parse_module(TTIR_ADD)
    print(f"  Parsed {len(operations)} operations")
    for op in operations:
        print(f"    - {op.name}")

    # Step 2: Convert to adapter IR
    print("\nStep 2: Convert to adapter IR")
    program = convert_ttir_to_pypto(TTIR_ADD)
    print(f"  Program has {len(program.functions)} function(s)")
    for fn in program.functions:
        print(f"    - {fn.name} with {len(fn.params)} params, {len(fn.body)} statements")

    # Step 3: Run on NumPy (CPU)
    print("\nStep 3: Execute on CPU via NumPy")
    a = np.ones(128, dtype=np.float32)
    b = np.ones(128, dtype=np.float32) * 2
    out = np.zeros(128, dtype=np.float32)
    run_on_numpy(program, a, b, out)
    expected = a + b
    match = np.allclose(out, expected)
    print(f"  Result: {'PASS' if match else 'FAIL'}")
    print(f"  out[0] = {out[0]}, expected = {expected[0]}")

    print("\n" + "=" * 70)
    print("Phase 1 complete: TTIR conversion runs on CPU-only environment!")
    print("=" * 70)


if __name__ == "__main__":
    main()
