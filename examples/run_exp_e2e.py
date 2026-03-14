#!/usr/bin/env python3
"""
exp 端到端验证：Triton 源码 -> TTIR -> PyPTO IR -> simpler CPU 仿真 -> 结果对比

1. 从 Triton exp kernel 源码提取 TTIR（含 pid 和 mask）
2. 转换为 PyPTO IR
3. PyPTO + simpler a2a3sim CPU 仿真执行
4. 与参考 (torch.exp) 对比

需要: pypto, torch, triton, SIMPLER_ROOT=third_party/simpler
"""

import os
import sys

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


def main():
    print("=" * 70)
    print("Triton -> PyPTO 端到端验证: exp kernel (带 mask, CPU 仿真)")
    print("=" * 70)

    import torch

    n = 128
    x_tensor = torch.randn(n, dtype=torch.float32) * 0.1
    reference = torch.exp(x_tensor).clone()

    print("\n[1] 参考计算 (exp(x))")
    print(f"    x[:4] = {x_tensor[:4].tolist()}")
    print(f"    参考[:4] = {reference[:4].tolist()}")

    print("\n[2] 从 Triton 源码提取 TTIR（exp_kernel 含 pid/mask）")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from examples.exp_kernel import exp_kernel

    sig = {"x": "*fp32", "out": "*fp32"}
    src = ASTSource(fn=exp_kernel, signature=sig, constexprs={"n": 128})
    k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
    ttir = k.asm["ttir"]
    print("    TTIR 提取成功")

    print("\n[3] 转换为 PyPTO IR")
    from triton_adapter import convert_ttir_to_pypto

    program = convert_ttir_to_pypto(ttir, program_name="exp_kernel")
    print(f"    Program: {program.name}")

    print("\n[4] PyPTO + simpler a2a3sim 执行")
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, TensorSpec, run

    x_2d = x_tensor.reshape(128, 1)

    def golden(tensors: dict, params: dict | None = None) -> None:
        tensors["out"][:] = torch.exp(tensors["x"])

    tensor_specs = [
        TensorSpec("x", [128, 1], torch.float32, init_value=x_2d),
        TensorSpec("out", [128, 1], torch.float32, is_output=True),
    ]

    try:
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
        print(f"    PyPTO 运行结果: {result}")
        if result.passed:
            print("\n" + "=" * 70)
            print("✓ 验证通过: exp kernel Triton->PyPTO 转换与 CPU 仿真执行正确")
            print("=" * 70)
        else:
            print(f"    错误: {result.error}")
    except Exception as e:
        print(f"    异常: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示: export SIMPLER_ROOT=$(pwd)/third_party/simpler")


if __name__ == "__main__":
    main()
