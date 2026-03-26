#!/usr/bin/env python3
"""Controlled test to verify NPU actually executes computation."""

import os
import sys

# Setup paths
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace, "src"))
sys.path.insert(0, workspace)
simpler_path = os.path.join(workspace, "third_party", "simpler")
if os.path.exists(simpler_path):
    os.environ["SIMPLER_ROOT"] = simpler_path
    for sub in ("examples/scripts", "python"):
        p = os.path.join(simpler_path, sub)
        if p not in sys.path:
            sys.path.insert(0, p)

import torch
from pypto.runtime import run, TensorSpec

from e2e_common import make_pypto_run_config
from triton_adapter import convert_ttir_to_pypto


def verify_npu_execution():
    """Run add kernel with controlled inputs to verify NPU actually computes."""
    print("=" * 60)
    print("CONTROLLED NPU EXECUTION VERIFICATION")
    print("=" * 60)

    # Create fixed, non-random input data
    size = 128

    # Input tensors with known values
    x = torch.full((size,), 2.0, dtype=torch.float32)
    y = torch.full((size,), 3.0, dtype=torch.float32)

    # Output tensor initialized to zeros (to detect if it gets modified)
    out = torch.zeros((size,), dtype=torch.float32)

    print(f"\nInput x (first 5): {x.flatten()[:5].tolist()}")
    print(f"Input y (first 5): {y.flatten()[:5].tolist()}")
    print(f"Output BEFORE (first 5): {out.flatten()[:5].tolist()}")

    # Store initial output state
    out_before = out.clone()

    # Create tensor specs
    tensor_specs = [
        TensorSpec("x", [size], torch.float32, init_value=x),
        TensorSpec("y", [size], torch.float32, init_value=y),
        TensorSpec("out", [size], torch.float32, is_output=True),
    ]

    # Get TTIR from Triton add kernel
    from examples.add_kernel import add_kernel
    import triton
    from triton.backends.compiler import GPUTarget

    src = triton.compiler.ASTSource(
        fn=add_kernel, signature={"x": "*fp32", "y": "*fp32", "out": "*fp32"}, constexprs={"n": 128}
    )
    k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
    ttir = k.asm["ttir"]
    print(f"\nTTIR extracted (length {len(ttir)} chars)")

    # Convert to PyPTO
    program = convert_ttir_to_pypto(ttir, program_name="add_kernel")
    print(
        f"PyPTO program: {program.name}, functions: {[f.name for f in program.functions.values()]}"
    )

    # Run on NPU
    config = make_pypto_run_config(
        platform="a2a3",
        device_id=0,
        enable_profiling=False,
    )

    # Golden function
    def golden_fn(tensors):
        tensors["out"][:] = tensors["x"] + tensors["y"]

    try:
        result = run(
            program=program,
            tensor_specs=tensor_specs,
            golden=golden_fn,
            config=config,
        )

        # The output tensor should have been updated by NPU
        # tensor_specs returns tensors from the runner
        out_after = None
        for spec in tensor_specs:
            if spec.name == "out":
                out_after = spec.init_value if spec.init_value is not None else None
                break

        print(f"\nResult passed: {result.passed}")
        print(f"Result error: {result.error}")

        # Check the actual output tensor from the result
        # The 'run' function should have modified the output tensor
        # Let's check the result output

        # Check if output changed
        if torch.allclose(out, out_before):
            print("\n❌ FAIL: Output tensor was NOT modified - NPU did NOT execute!")
            print("   This means the orchestration code is empty (no pto2_rt_submit_task calls)")
            return False

        # Check if output is correct (x + y = 2 + 3 = 5)
        expected = torch.full((size,), 5.0, dtype=torch.float32)
        if torch.allclose(out, expected, rtol=1e-4, atol=1e-4):
            print("\n✅ PASS: Output is correct (x+y=5) - NPU executed correctly!")
            return True
        else:
            print(f"\n⚠️ PARTIAL: Output was modified but values are wrong")
            print(f"   Got: {out.flatten()[:5].tolist()}")
            print(f"   Expected: {expected.flatten()[:5].tolist()}")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False

        # Check if output is correct (x + y = 2 + 3 = 5)
        expected = torch.full((size,), 5.0, dtype=torch.float32)
        if torch.allclose(out, expected, rtol=1e-4, atol=1e-4):
            print("\n✅ PASS: Output is correct (x+y=5) - NPU executed correctly!")
            return True
        else:
            print(f"\n⚠️ PARTIAL: Output was modified but values are wrong")
            print(f"   Got: {out.flatten()[:5].tolist()}")
            print(f"   Expected: {expected.flatten()[:5].tolist()}")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_npu_execution()
    sys.exit(0 if success else 1)
