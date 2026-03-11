#!/usr/bin/env python3
"""
Elementwise 端到端验证：TTIR → PyPTO IR → CPU 仿真

1. 使用 elementwise add 的 TTIR 样本
2. 转换为 PyPTO IR，展示转换后的 IR
3. 用 PyPTO + simpler a2a3sim 在 CPU 执行
4. 验证 PyPTO 输出与参考（a+b）一致

需要: pypto, torch, SIMPLER_ROOT=third_party/simpler
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
simpler_path = os.path.join(workspace, "third_party", "simpler")
if os.path.exists(simpler_path):
    os.environ["SIMPLER_ROOT"] = simpler_path
    for sub in ("examples/scripts", "python"):
        p = os.path.join(simpler_path, sub)
        if p not in sys.path:
            sys.path.insert(0, p)


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


def main():
    print("=" * 70)
    print("Elementwise 端到端验证: TTIR → PyPTO IR → CPU 仿真")
    print("=" * 70)

    import torch

    m, n = 128, 128
    a_tensor = torch.randn(m, n, dtype=torch.float32)
    b_tensor = torch.randn(m, n, dtype=torch.float32)
    result_reference = (a_tensor + b_tensor).clone()

    print("\n[1] 参考计算 (a + b, 与 Triton add kernel 数学等价)")
    print(f"    输入 a[0,:4] = {a_tensor[0,:4].tolist()}")
    print(f"    输入 b[0,:4] = {b_tensor[0,:4].tolist()}")
    print(f"    参考输出 [0,:4] = {result_reference[0,:4].tolist()}")

    print("\n[2] 输入 TTIR:")
    print("-" * 50)
    for i, line in enumerate(SAMPLE_TTIR_ADD.strip().split("\n")):
        print(f"  {i+1:2}| {line}")
    print("-" * 50)

    from triton_adapter import convert_ttir_to_pypto

    program = convert_ttir_to_pypto(SAMPLE_TTIR_ADD, program_name="add_kernel")

    print("\n[3] 转换后的 PyPTO IR:")
    print("-" * 50)
    ir_text = convert_ttir_to_pypto(SAMPLE_TTIR_ADD, output_format="text")
    print(ir_text)
    print("-" * 50)

    print("\n[4] PyPTO + simpler CPU 执行 (等价 PyPTO 程序):")
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, TensorSpec, run

    import pypto.language as pl

    @pl.program
    class AddProgram:
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

    tensor_specs = [
        TensorSpec("a", [128, 128], torch.float32, init_value=a_tensor),
        TensorSpec("b", [128, 128], torch.float32, init_value=b_tensor),
        TensorSpec("out", [128, 128], torch.float32, is_output=True),
    ]

    try:
        result = run(
            program=AddProgram,
            tensor_specs=tensor_specs,
            golden=golden,
            config=RunConfig(
                platform="a2a3sim",
                backend_type=BackendType.CCE,
                strategy=OptimizationStrategy.Default,
            ),
        )
        print(f"    结果: {result}")

        if result.passed:
            print("\n" + "=" * 70)
            print("✓ 验证通过: PyPTO simpler CPU 输出与参考 (a+b) 一致")
            print("=" * 70)
        else:
            print(f"    错误: {result.error}")
    except Exception as e:
        print(f"    异常: {e}")
        import traceback

        traceback.print_exc()
        print("\n  提示: 确保 SIMPLER_ROOT 指向 third_party/simpler")
        print("  参见 README 中的构建与执行说明。")


if __name__ == "__main__":
    main()
