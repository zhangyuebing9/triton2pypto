#!/usr/bin/env python3
"""统一端到端验证入口：Triton 源码 -> TTIR -> PyPTO IR -> simpler 执行 -> 与 golden 对比。

支持 kernel: add, sub, mul, div, exp, reduce_sum, matmul

默认在 **CPU 仿真**（``a2a3sim``）上运行；在昇腾机器上可用 ``--platform a2a3`` 或环境变量
``TRITON2PYPTO_PLATFORM=a2a3`` 做 **NPU 上板** 验证（与 golden 数值对比，golden 与 Triton 参考一致）。

用法:
  python examples/run_e2e.py [--kernel NAME]
  python examples/run_e2e.py --kernel add   # 默认
  python examples/run_e2e.py --platform a2a3 --device-id 0   # 真机（需 CANN / npu-smi）
  python examples/run_e2e.py --list        # 列出所有支持的 kernel

需要: pypto, torch, triton, SIMPLER_ROOT=third_party/simpler；真机另需 CANN（ASCEND_HOME_PATH 等）
"""

import argparse
import os
import subprocess
import sys
import tempfile

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace, "src"))
sys.path.insert(0, workspace)
sys.path.insert(0, os.path.join(workspace, "examples"))
simpler_path = os.path.join(workspace, "third_party", "simpler")
if os.path.exists(simpler_path):
    os.environ["SIMPLER_ROOT"] = simpler_path
    for sub in ("examples/scripts", "python"):
        p = os.path.join(simpler_path, sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def _golden_add(tensors: dict, _params=None) -> None:
    tensors["out"][:] = tensors["x"] + tensors["y"]


def _golden_sub(tensors: dict, _params=None) -> None:
    tensors["out"][:] = tensors["x"] - tensors["y"]


def _golden_mul(tensors: dict, _params=None) -> None:
    tensors["out"][:] = tensors["x"] * tensors["y"]


def _golden_div(tensors: dict, _params=None) -> None:
    tensors["out"][:] = tensors["x"] / tensors["y"]


def _golden_exp(tensors: dict, _params=None) -> None:
    import torch

    tensors["out"][:] = torch.exp(tensors["x"])


def _golden_reduce_sum(tensors: dict, _params=None) -> None:
    tensors["out"][:] = tensors["x"].sum(dim=1, keepdim=True)


def _golden_matmul(tensors: dict, _params=None) -> None:
    tensors["C"][:] = tensors["A"] @ tensors["B"]


def _get_kernel_config(name: str) -> dict | None:
    """返回 kernel 的配置：sig, constexprs, tensor_specs_fn, golden_fn, program_name。"""
    import torch
    from pypto.runtime import TensorSpec

    def _specs_add(t):
        return [
            TensorSpec("x", [128, 1], torch.float32, init_value=t["x"]),
            TensorSpec("y", [128, 1], torch.float32, init_value=t["y"]),
            TensorSpec("out", [128, 1], torch.float32, is_output=True),
        ]

    def _specs_exp(t):
        return [
            TensorSpec("x", [128, 1], torch.float32, init_value=t["x"]),
            TensorSpec("out", [128, 1], torch.float32, is_output=True),
        ]

    def _specs_reduce_sum(t):
        return [
            TensorSpec("x", [128, 128], torch.float32, init_value=t["x"]),
            TensorSpec("out", [128, 1], torch.float32, is_output=True),
        ]

    def _specs_matmul(t):
        return [
            TensorSpec("A", [16, 16], torch.float32, init_value=t["A"]),
            TensorSpec("B", [16, 16], torch.float32, init_value=t["B"]),
            TensorSpec("C", [16, 16], torch.float32, is_output=True),
        ]

    configs = {
        "add": {
            "sig": {"x": "*fp32", "y": "*fp32", "out": "*fp32"},
            "constexprs": {"n": 128},
            "tensor_specs_fn": _specs_add,
            "golden_fn": _golden_add,
            "reference_fn": lambda t: t["x"] + t["y"],
            "program_name": "add_kernel",
            "supports_triton_compare": True,
        },
        "sub": {
            "sig": {"x": "*fp32", "y": "*fp32", "out": "*fp32"},
            "constexprs": {"n": 128},
            "tensor_specs_fn": _specs_add,
            "golden_fn": _golden_sub,
            "reference_fn": lambda t: t["x"] - t["y"],
            "program_name": "sub_kernel",
            "supports_triton_compare": False,
        },
        "mul": {
            "sig": {"x": "*fp32", "y": "*fp32", "out": "*fp32"},
            "constexprs": {"n": 128},
            "tensor_specs_fn": _specs_add,
            "golden_fn": _golden_mul,
            "reference_fn": lambda t: t["x"] * t["y"],
            "program_name": "mul_kernel",
            "supports_triton_compare": False,
        },
        "div": {
            "sig": {"x": "*fp32", "y": "*fp32", "out": "*fp32"},
            "constexprs": {"n": 128},
            "tensor_specs_fn": _specs_add,
            "golden_fn": _golden_div,
            "reference_fn": lambda t: t["x"] / t["y"],
            "program_name": "div_kernel",
            "supports_triton_compare": False,
        },
        "exp": {
            "sig": {"x": "*fp32", "out": "*fp32"},
            "constexprs": {"n": 128},
            "tensor_specs_fn": _specs_exp,
            "golden_fn": _golden_exp,
            "reference_fn": lambda t: torch.exp(t["x"]),
            "program_name": "exp_kernel",
            "supports_triton_compare": False,
        },
        "reduce_sum": {
            "sig": {"x": "*fp32", "out": "*fp32"},
            "constexprs": {"BLOCK": 128, "n_cols": 128},
            "tensor_specs_fn": _specs_reduce_sum,
            "golden_fn": _golden_reduce_sum,
            "reference_fn": lambda t: t["x"].sum(dim=1, keepdim=True),
            "program_name": "reduce_sum_kernel",
            "supports_triton_compare": False,
        },
        "matmul": {
            "sig": {"A": "*fp32", "B": "*fp32", "C": "*fp32"},
            "constexprs": {"BLOCK": 16, "M": 16, "N": 16, "K": 16},
            "tensor_specs_fn": _specs_matmul,
            "golden_fn": _golden_matmul,
            "reference_fn": lambda t: t["A"] @ t["B"],
            "program_name": "matmul_kernel",
            "supports_triton_compare": False,
        },
    }
    return configs.get(name)


def _prepare_tensors(name: str):
    """准备 kernel 所需张量。"""
    import torch

    if name in ("add", "sub", "mul"):
        a = torch.randn(128, 1, dtype=torch.float32)
        b = torch.randn(128, 1, dtype=torch.float32)
        return {"x": a, "y": b}
    if name == "div":
        a = torch.randn(128, 1, dtype=torch.float32)
        b = torch.ones(128, 1, dtype=torch.float32)  # 避免除零
        return {"x": a, "y": b}
    if name == "exp":
        x = torch.randn(128, 1, dtype=torch.float32) * 0.1
        return {"x": x}
    if name == "reduce_sum":
        x = torch.randn(128, 128, dtype=torch.float32)
        return {"x": x}
    if name == "matmul":
        A = torch.randn(16, 16, dtype=torch.float32) * 0.1
        B = torch.randn(16, 16, dtype=torch.float32) * 0.1
        return {"A": A, "B": B}
    raise ValueError(f"Unknown kernel: {name}")


def _get_kernel_module_and_name(name: str):
    """返回 (kernel_module, kernel_fn_name, data_keys_for_triton)。"""
    mapping = {
        "add": ("examples.add_kernel", "add_kernel", ["x", "y"]),
        "sub": ("examples.sub_kernel", "sub_kernel", ["x", "y"]),
        "mul": ("examples.mul_kernel", "mul_kernel", ["x", "y"]),
        "div": ("examples.div_kernel", "div_kernel", ["x", "y"]),
        "exp": ("examples.exp_kernel", "exp_kernel", ["x"]),
        "reduce_sum": ("examples.reduce_sum_kernel", "reduce_sum_kernel", ["x"]),
        "matmul": ("examples.matmul_kernel", "matmul_kernel", ["A", "B"]),
    }
    return mapping[name]


def run_e2e(
    kernel_name: str,
    triton_compare: bool = False,
    *,
    platform: str | None = None,
    device_id: int | None = None,
    enable_profiling: bool = False,
) -> int:
    """执行端到端验证，返回 0 成功，非 0 失败。"""
    import torch
    from pypto.runtime import run
    from triton.backends.compiler import GPUTarget

    from e2e_common import get_e2e_device_id, get_e2e_platform, make_pypto_run_config
    from triton_adapter import convert_ttir_to_pypto

    plat = platform if platform is not None else get_e2e_platform()
    dev = device_id if device_id is not None else get_e2e_device_id()

    cfg = _get_kernel_config(kernel_name)
    if not cfg:
        print(f"未知 kernel: {kernel_name}")
        return 1

    mod_name, fn_name, _ = _get_kernel_module_and_name(kernel_name)
    mod = __import__(mod_name, fromlist=[fn_name])
    kernel_fn = getattr(mod, fn_name)

    print("=" * 70)
    run_mode = "NPU 真机 (a2a3)" if plat == "a2a3" else "CPU 仿真 (a2a3sim)"
    profiling_status = " [PROFILING ON]" if enable_profiling else ""
    print(
        f"Triton -> PyPTO 端到端验证：{kernel_name} kernel (带 mask, {run_mode}{profiling_status}, device_id={dev})"
    )
    print("=" * 70)

    tensors = _prepare_tensors(kernel_name)
    reference = cfg["reference_fn"](tensors)

    print(f"\n[1] 参考计算 ({kernel_name})")
    for k, v in tensors.items():
        arr = v.flatten()[:4]
        print(f"    {k}[:4] = {arr.tolist()}")
    print(f"    参考[:4] = {reference.flatten()[:4].tolist()}")

    print(f"\n[2] 从 Triton 源码提取 TTIR（{fn_name} 含 pid/mask）")
    import triton

    src = __import__("triton").compiler.ASTSource(
        fn=kernel_fn, signature=cfg["sig"], constexprs=cfg["constexprs"]
    )
    k = triton.compile(src, target=GPUTarget("cuda", 80, 32))
    ttir = k.asm["ttir"]
    print(f"    TTIR 提取成功 (长度 {len(ttir)} 字符)")

    print("\n[3] 转换为 PyPTO IR")
    program = convert_ttir_to_pypto(ttir, program_name=cfg["program_name"])
    funcs = list(program.functions.values())
    print(f"    Program: {program.name}, Functions: {[f.name for f in funcs]}")

    print(f"\n[4] PyPTO + simpler 执行 (platform={plat})")

    tensor_specs = cfg["tensor_specs_fn"](tensors)

    # a2a3sim 上 tile.exp 与 torch.exp 的浮点误差可能略高于默认 1e-5；exp 用例单独放宽
    run_kw: dict = {"platform": plat, "device_id": dev, "enable_profiling": enable_profiling}
    if kernel_name == "exp":
        run_kw["rtol"] = 5e-4
        run_kw["atol"] = 1e-5

    try:
        result = run(
            program=program,
            tensor_specs=tensor_specs,
            golden=cfg["golden_fn"],
            config=make_pypto_run_config(**run_kw),
        )
        print(f"    PyPTO 运行结果: {result}")

        if not result.passed:
            print(f"    PyPTO 运行失败: {result.error}")
            return 1

        triton_vs_ref = None
        if triton_compare and cfg.get("supports_triton_compare") and kernel_name == "add":
            print("\n[5] Triton TRITON_INTERPRET 执行与结果对比")
            a_flat = tensors["x"].flatten()
            b_flat = tensors["y"].flatten()
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                torch.save({"a": a_flat, "b": b_flat}, f.name)
                data_path = f.name
            try:
                code = f"""
import os
os.environ["TRITON_INTERPRET"] = "1"
import sys
sys.path.insert(0, {repr(workspace)})
import torch
from examples.add_kernel import add_kernel

data = torch.load({repr(data_path)})
a, b = data["a"], data["b"]
out = torch.empty_like(a)
add_kernel[(1,)](a, b, out, n=128)
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
                    ref_flat = reference.flatten()
                    triton_vs_ref = (out_triton - ref_flat).abs().max().item()
                    print(f"    Triton vs 参考 max diff: {triton_vs_ref:.2e}")
                else:
                    print(f"    Triton 执行失败: {r.stderr or r.stdout}")
                    triton_vs_ref = float("inf")
            finally:
                os.unlink(data_path)

        print("\n[6] 综合验证")
        print("    - PyPTO 输出 = golden (result.passed)")
        if triton_vs_ref is not None:
            if triton_vs_ref < 1e-4:
                print("    - Triton 输出与参考一致")
                print("\n" + "=" * 70)
                print("✓ 验证通过: Triton->PyPTO 转换与执行正确（含 Triton 对比）")
            else:
                print(f"    - Triton 对比: diff={triton_vs_ref}")
                print("\n" + "=" * 70)
                print("✓ PyPTO 验证通过; Triton 对比: 未通过")
        else:
            print("\n" + "=" * 70)
            print("✓ 验证通过: Triton->PyPTO 转换与执行正确")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"    异常: {e}")
        import traceback

        traceback.print_exc()
        print("\n提示: 确保 SIMPLER_ROOT 指向 third_party/simpler")
        print("  export SIMPLER_ROOT=$(pwd)/third_party/simpler")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Triton->PyPTO 端到端验证")
    parser.add_argument(
        "--kernel",
        "-k",
        default="add",
        choices=["add", "sub", "mul", "div", "exp", "reduce_sum", "matmul"],
        help="要验证的 kernel",
    )
    parser.add_argument(
        "--triton-compare",
        action="store_true",
        help="与 Triton TRITON_INTERPRET 结果对比（仅 add 支持）",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="列出所有支持的 kernel",
    )
    parser.add_argument(
        "--platform",
        "-p",
        choices=["a2a3sim", "a2a3"],
        default=None,
        help="simpler/PyPTO 执行平台：a2a3sim=CPU 仿真（默认），a2a3=昇腾真机。也可用环境变量 TRITON2PYPTO_PLATFORM。",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="NPU 设备号（仅 a2a3 有效）。默认 0。也可用 TRITON2PYPTO_DEVICE_ID。",
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="启用 profiling（仅 a2a3 有效）",
    )
    args = parser.parse_args()

    if args.list:
        print("支持的 kernel:")
        for k in ["add", "sub", "mul", "div", "exp", "reduce_sum", "matmul"]:
            print(f"  - {k}")
        return 0

    return run_e2e(
        args.kernel,
        triton_compare=args.triton_compare,
        platform=args.platform,
        device_id=args.device_id,
        enable_profiling=args.enable_profiling,
    )


if __name__ == "__main__":
    sys.exit(main())
