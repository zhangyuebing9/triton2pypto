#!/usr/bin/env python3
"""
端到端验证：Triton 源码 -> TTIR -> PyPTO IR -> 编译 -> simpler CPU 仿真 -> 结果对比

1. 从 Triton add kernel 源码提取 TTIR（compile-only，无需 GPU）
2. 转换为 PyPTO IR
3. PyPTO 编译 + simpler a2a3sim CPU 仿真执行
4. Triton TRITON_INTERPRET=1 CPU 执行
5. 对比两者结果一致

需要: pypto, torch, triton, SIMPLER_ROOT=third_party/simpler
"""

import os
import sys

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace, "src"))
sys.path.insert(0, workspace)  # for examples.add_kernel_simple
simpler_path = os.path.join(workspace, "third_party", "simpler")
if os.path.exists(simpler_path):
    os.environ["SIMPLER_ROOT"] = simpler_path
    for sub in ("examples/scripts", "python"):
        p = os.path.join(simpler_path, sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def main():
    print("=" * 70)
    print("Triton -> PyPTO 端到端验证: add kernel (CPU 仿真)")
    print("=" * 70)

    import torch

    # 使用 128 元素以匹配 add_kernel_simple 的 BLOCK
    n = 128
    a_tensor = torch.randn(n, dtype=torch.float32)
    b_tensor = torch.randn(n, dtype=torch.float32)
    reference = (a_tensor + b_tensor).clone()

    print("\n[1] 参考计算 (a + b)")
    print(f"    a[:4] = {a_tensor[:4].tolist()}")
    print(f"    b[:4] = {b_tensor[:4].tolist()}")
    print(f"    参考[:4] = {reference[:4].tolist()}")

    # 提取 TTIR（compile-only，无需 GPU）
    print("\n[2] 从 Triton 源码提取 TTIR")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from examples.add_kernel_simple import add_kernel_simple

    sig = {"a": "*fp32", "b": "*fp32", "out": "*fp32"}
    src = ASTSource(fn=add_kernel_simple, signature=sig, constexprs={})
    k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
    ttir = k.asm["ttir"]
    print("    TTIR 提取成功 (长度 %d 字符)" % len(ttir))

    # 转换为 PyPTO IR
    print("\n[3] 转换为 PyPTO IR")
    from triton_adapter import convert_ttir_to_pypto

    program = convert_ttir_to_pypto(ttir, program_name="add_kernel_simple")
    funcs = list(program.functions.values())
    print(f"    Program: {program.name}, Functions: {[f.name for f in funcs]}")

    # PyPTO + simpler 执行
    print("\n[4] PyPTO + simpler a2a3sim 执行")

    # Reshape to 128x1 for tile ops (我们的 converter 转成 2D)
    a_2d = a_tensor.reshape(128, 1)
    b_2d = b_tensor.reshape(128, 1)
    out_2d = torch.zeros(128, 1, dtype=torch.float32)

    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, TensorSpec, run

    def golden(tensors: dict, params: dict | None = None) -> None:
        tensors["out"][:] = tensors["a"] + tensors["b"]

    tensor_specs = [
        TensorSpec("a", [128, 1], torch.float32, init_value=a_2d),
        TensorSpec("b", [128, 1], torch.float32, init_value=b_2d),
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
            # Triton TRITON_INTERPRET 执行（子进程，避免与 compile 冲突）
            print("\n[5] Triton TRITON_INTERPRET 执行与结果对比")
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                torch.save({"a": a_tensor, "b": b_tensor}, f.name)
                data_path = f.name
            try:
                code = f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
import sys
sys.path.insert(0, {repr(workspace)})
import torch
from examples.add_kernel_simple import add_kernel_simple

data = torch.load({repr(data_path)})
a, b = data["a"], data["b"]
out = torch.empty_like(a)
add_kernel_simple[(1,)](a, b, out)
torch.save({{"out": out}}, {repr(data_path + ".out")})
"""
                r = subprocess.run(
                    [sys.executable, "-c", code],
                    env={**os.environ, "TRITON_INTERPRET": "1"},
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                )
                if r.returncode == 0:
                    out_triton = torch.load(data_path + ".out")["out"]
                    os.unlink(data_path + ".out")
                    triton_vs_ref = (out_triton - reference).abs().max().item()
                    print(f"    Triton 输出[:4] = {out_triton[:4].tolist()}")
                    print(f"    参考 a+b[:4]     = {reference[:4].tolist()}")
                    print(f"    Triton vs 参考 max diff: {triton_vs_ref:.2e}")
                else:
                    print(f"    Triton 执行失败: {r.stderr or r.stdout}")
                    triton_vs_ref = float("inf")
            finally:
                os.unlink(data_path)

            # PyPTO 通过 golden(a+b) 校验，等价于与 Triton 一致
            print("\n[6] 综合验证")
            print("    - PyPTO 输出 = golden(a,b) = a + b (result.passed)")
            if triton_vs_ref < 1e-4:
                print("    - Triton 输出 = a + b (triton_vs_ref < 1e-4)")
                print("    -> 转换后 PyPTO 执行结果与 Triton 一致")
                print("\n" + "=" * 70)
                print("✓ 验证通过: add 类型 elementwise 算子 Triton->PyPTO 转换与 CPU 仿真执行正确")
                print("=" * 70)
            else:
                print("    - Triton 输出校验:", "通过" if triton_vs_ref < 1e-4 else f"diff={triton_vs_ref}")
                print("\n" + "=" * 70)
                print("✓ PyPTO 验证通过 (golden 校验); Triton 对比:", "通过" if triton_vs_ref < 1e-4 else "未运行/失败")
                print("=" * 70)
        else:
            print(f"    PyPTO 运行失败: {result.error}")
    except Exception as e:
        print(f"    异常: {e}")
        import traceback

        traceback.print_exc()
        print("\n提示: 确保 SIMPLER_ROOT 指向 third_party/simpler")
        print("  export SIMPLER_ROOT=$(pwd)/third_party/simpler")


if __name__ == "__main__":
    main()
