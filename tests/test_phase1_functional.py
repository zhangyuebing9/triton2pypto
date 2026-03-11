"""Phase 1 functional tests for TTIR to PyPTO conversion.

These tests validate:
1. TTIR parsing and conversion to PyPTO IR
2. CPU functional testing with a2a3sim (when SIMPLER_ROOT is set)

Run with SIMPLER_ROOT for full CPU simulation:
  export SIMPLER_ROOT=/path/to/third_party/simpler
  pytest tests/test_phase1_functional.py -v
"""

import os
from pathlib import Path

import pytest


# Sample TTIR for simple vector add (simplified Phase 1 pattern)
SAMPLE_TTIR_ADD = """
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : tensor<128x128xf32>
    %1 = tt.load %arg1 : tensor<128x128xf32>
    %2 = arith.addf %0, %1 : tensor<128x128xf32>
    tt.store %arg2, %2 : tensor<128x128xf32>
    tt.return
  }
}
"""


class TestTTIRConversion:
    """Test TTIR to PyPTO conversion."""

    def test_parse_ttir(self) -> None:
        """Test that we can parse sample TTIR."""
        from triton_adapter.mlir_parser import MLIRParser

        parser = MLIRParser()
        ops = parser.parse_module(SAMPLE_TTIR_ADD)
        assert len(ops) >= 4
        op_names = [o.name for o in ops]
        assert "tt.load" in op_names or "arith.addf" in op_names

    def test_convert_ttir_to_pypto_object(self) -> None:
        """Test conversion produces ir.Program."""
        pytest.importorskip("pypto")

        from triton_adapter import convert_ttir_to_pypto

        program = convert_ttir_to_pypto(SAMPLE_TTIR_ADD, output_format="object")
        assert program is not None
        assert hasattr(program, "name")
        assert hasattr(program, "functions")

    def test_convert_ttir_to_pypto_text(self) -> None:
        """Test conversion to text format."""
        pytest.importorskip("pypto")

        from triton_adapter import convert_ttir_to_pypto

        text = convert_ttir_to_pypto(SAMPLE_TTIR_ADD, output_format="text")
        assert isinstance(text, str)
        assert len(text) > 0


class TestCPUFunctional:
    """CPU functional tests using a2a3sim.

    Requires SIMPLER_ROOT to be set to third_party/simpler for execution.
    """

    @pytest.fixture(autouse=True)
    def setup_simpler_path(self) -> None:
        """Set SIMPLER_ROOT to submodule if not set."""
        if "SIMPLER_ROOT" not in os.environ:
            simpler_path = Path(__file__).resolve().parent.parent / "third_party" / "simpler"
            if simpler_path.exists():
                os.environ["SIMPLER_ROOT"] = str(simpler_path)

    def test_pypto_vector_add_compilation(self) -> None:
        """Test that a simple PyPTO vector add program compiles.

        Does not require execution - only compilation.
        """
        pypto = pytest.importorskip("pypto")
        torch = pytest.importorskip("torch")

        import pypto.language as pl
        from pypto.backend import BackendType
        from pypto.ir import compile as ir_compile
        from pypto.ir.pass_manager import OptimizationStrategy

        @pl.program
        class VectorAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_incore(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
                tile_c: pl.Tile[[128, 128], pl.FP32] = pl.add(tile_a, tile_b)
                return pl.store(tile_c, [0, 0], c)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = pl.create_tensor([128, 128], dtype=pl.FP32)
                return self.add_incore(a, b, c)

        output_dir = ir_compile(
            VectorAddProgram,
            output_dir="/tmp/triton2pypto_test_build",
            strategy=OptimizationStrategy.Default,
            dump_passes=False,
            backend_type=BackendType.CCE,
        )
        assert output_dir is not None
        assert Path(output_dir).exists()

    @pytest.mark.skipif(
        "SIMPLER_ROOT" not in os.environ,
        reason="SIMPLER_ROOT not set - skip CPU execution test",
    )
    def test_pypto_vector_add_run_cpu(self) -> None:
        """Test PyPTO vector add runs on a2a3sim (CPU simulation).

        Requires SIMPLER_ROOT pointing to third_party/simpler.
        """
        pypto = pytest.importorskip("pypto")
        torch = pytest.importorskip("torch")

        import pypto.language as pl
        from pypto.backend import BackendType
        from pypto.ir.pass_manager import OptimizationStrategy
        from pypto.runtime import RunConfig, TensorSpec, run

        @pl.program
        class VectorAddProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def add_incore(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a: pl.Tile[[128, 128], pl.FP32] = pl.load(a, [0, 0], [128, 128])
                tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [0, 0], [128, 128])
                tile_c: pl.Tile[[128, 128], pl.FP32] = pl.add(tile_a, tile_b)
                return pl.store(tile_c, [0, 0], c)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                c = pl.create_tensor([128, 128], dtype=pl.FP32)
                return self.add_incore(a, b, c)

        def golden(tensors: dict, params: dict | None = None) -> None:
            tensors["out"][:] = tensors["a"] + tensors["b"]

        # Use random init (same as run_elementwise_e2e) for consistent behavior
        a_init = torch.randn(128, 128, dtype=torch.float32)
        b_init = torch.randn(128, 128, dtype=torch.float32)
        tensor_specs = [
            TensorSpec("a", [128, 128], torch.float32, init_value=a_init),
            TensorSpec("b", [128, 128], torch.float32, init_value=b_init),
            TensorSpec("out", [128, 128], torch.float32, is_output=True),
        ]

        result = run(
            program=VectorAddProgram,
            tensor_specs=tensor_specs,
            golden=golden,
            config=RunConfig(
                platform="a2a3sim",
                backend_type=BackendType.CCE,
                strategy=OptimizationStrategy.Default,
            ),
        )
        assert result.passed, result.error or "Unknown error"
