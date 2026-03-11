"""端到端测试: Triton add kernel 源码 -> TTIR -> PyPTO IR -> 编译。

验证 add 类型 elementwise 算子从 Triton 源码（非手写简化 IR）完整转换流程。
运行 simpler CPU 仿真需要 SIMPLER_ROOT 且 simpler 环境正确配置。
"""

import os
from pathlib import Path

import pytest


@pytest.fixture
def workspace_path():
    return Path(__file__).resolve().parent.parent


class TestTritonToPyPTOConversion:
    """测试 Triton 源码 -> TTIR -> PyPTO 转换与编译。"""

    def test_extract_ttir_from_triton_source(self, workspace_path):
        """从 Triton add kernel 源码提取 TTIR（compile-only，无需 GPU）。"""
        triton = pytest.importorskip("triton")
        torch = pytest.importorskip("torch")
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        if str(workspace_path) not in os.sys.path:
            os.sys.path.insert(0, str(workspace_path))
        from examples.add_kernel_simple import add_kernel_simple

        sig = {"a": "*fp32", "b": "*fp32", "out": "*fp32"}
        src = ASTSource(fn=add_kernel_simple, signature=sig, constexprs={})
        k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
        ttir = k.asm["ttir"]

        assert "tt.func" in ttir
        assert "arith.addf" in ttir
        assert "tt.load" in ttir
        assert "tt.store" in ttir

    def test_extract_ttir_api(self, workspace_path):
        """extract_ttir API 可从 kernel + args 提取 TTIR。"""
        triton = pytest.importorskip("triton")
        torch = pytest.importorskip("torch")
        from triton_adapter import extract_ttir, convert_ttir_to_pypto

        if str(workspace_path) not in os.sys.path:
            os.sys.path.insert(0, str(workspace_path))
        from examples.add_kernel_simple import add_kernel_simple

        a = torch.randn(128, dtype=torch.float32)
        b = torch.randn(128, dtype=torch.float32)
        out = torch.empty(128, dtype=torch.float32)

        ttir = extract_ttir(add_kernel_simple, a, b, out, grid=(1,))
        assert "tt.func" in ttir
        assert "arith.addf" in ttir

        program = convert_ttir_to_pypto(ttir, program_name="add_from_extract")
        assert program is not None
        assert len(list(program.functions.values())) >= 2

    def test_convert_real_ttir_to_pypto(self, workspace_path):
        """转换真实 Triton TTIR 到 PyPTO IR 并编译。"""
        triton = pytest.importorskip("triton")
        pypto = pytest.importorskip("pypto")
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        if str(workspace_path) not in os.sys.path:
            os.sys.path.insert(0, str(workspace_path))
        from examples.add_kernel_simple import add_kernel_simple

        from triton_adapter import convert_ttir_to_pypto

        sig = {"a": "*fp32", "b": "*fp32", "out": "*fp32"}
        src = ASTSource(fn=add_kernel_simple, signature=sig, constexprs={})
        k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
        ttir = k.asm["ttir"]

        program = convert_ttir_to_pypto(ttir, program_name="add_kernel_simple")
        assert program is not None
        assert program.name == "add_kernel_simple"

        funcs = list(program.functions.values())
        assert len(funcs) >= 2
        func_names = [f.name for f in funcs]
        assert "kernel_incore" in func_names or "add_kernel_simple_incore" in func_names
        assert "main" in func_names

        # PyPTO 编译应成功
        from pypto.backend import BackendType
        from pypto.ir import compile as ir_compile
        from pypto.ir.pass_manager import OptimizationStrategy

        output_dir = ir_compile(
            program,
            output_dir="/tmp/triton2pypto_e2e_test",
            strategy=OptimizationStrategy.Default,
            dump_passes=False,
            backend_type=BackendType.CCE,
        )
        assert output_dir is not None
        assert Path(output_dir).exists()
        assert (Path(output_dir) / "kernels").exists()


@pytest.mark.skipif(
    "SIMPLER_ROOT" not in os.environ,
    reason="SIMPLER_ROOT not set - skip CPU execution test",
)
class TestTritonToPyPTOExecution:
    """完整执行测试（需要 SIMPLER_ROOT 且 simpler 环境可用）。"""

    def test_triton_add_to_pypto_run_cpu(self, workspace_path):
        """Triton add -> PyPTO -> simpler a2a3sim 执行并与 golden 校验。"""
        triton = pytest.importorskip("triton")
        torch = pytest.importorskip("torch")
        pypto = pytest.importorskip("pypto")

        if str(workspace_path) not in os.sys.path:
            os.sys.path.insert(0, str(workspace_path))
        from examples.add_kernel_simple import add_kernel_simple

        from triton_adapter import convert_ttir_to_pypto
        from pypto.backend import BackendType
        from pypto.ir.pass_manager import OptimizationStrategy
        from pypto.runtime import RunConfig, TensorSpec, run

        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        sig = {"a": "*fp32", "b": "*fp32", "out": "*fp32"}
        src = ASTSource(fn=add_kernel_simple, signature=sig, constexprs={})
        k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
        ttir = k.asm["ttir"]

        program = convert_ttir_to_pypto(ttir, program_name="add_simple")
        a_2d = torch.randn(128, 1, dtype=torch.float32)
        b_2d = torch.randn(128, 1, dtype=torch.float32)

        def golden(tensors: dict, params: dict | None = None) -> None:
            tensors["out"][:] = tensors["a"] + tensors["b"]

        tensor_specs = [
            TensorSpec("a", [128, 1], torch.float32, init_value=a_2d),
            TensorSpec("b", [128, 1], torch.float32, init_value=b_2d),
            TensorSpec("out", [128, 1], torch.float32, is_output=True),
        ]

        result = run(
            program=program,
            tensor_specs=tensor_specs,
            golden=golden,
            config=RunConfig(
                platform="a2a3sim",
                backend_type=BackendType.CCE,
                strategy=OptimizationStrategy.Default,
            ),
        )
        assert result.passed, result.error or "Run failed"
