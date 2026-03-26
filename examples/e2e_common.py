"""E2E 运行配置：CPU 仿真（a2a3sim）与 NPU 真机（a2a3）。

`run_e2e.py` 与 `tests/test_triton_to_pypto_e2e.py` 共用此模块，保证同一套 golden
与 tensor 规格在仿真与上板时一致。

环境变量（真机验证时在 NPU 机器上设置）：

- ``TRITON2PYPTO_PLATFORM``: ``a2a3sim``（默认）或 ``a2a3``（昇腾真机）。
- ``TRITON2PYPTO_DEVICE_ID``: 设备号，默认 ``0``。

命令行 ``--platform`` / ``--device-id`` 会覆盖上述环境变量（仅 ``run_e2e.py``）。
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig


def npu_runtime_available() -> bool:
    """若 ``npu-smi`` 在 PATH 中则认为可能具备 NPU 运行时（与 simpler 文档一致）。"""
    return shutil.which("npu-smi") is not None


def get_e2e_platform() -> str:
    """从环境变量读取目标平台，默认 ``a2a3sim``。"""
    raw = os.environ.get("TRITON2PYPTO_PLATFORM", "a2a3sim").strip().lower()
    if raw not in ("a2a3", "a2a3sim"):
        raise ValueError(f"TRITON2PYPTO_PLATFORM must be 'a2a3' or 'a2a3sim', got {raw!r}")
    return raw


def get_e2e_device_id() -> int:
    """从环境变量读取设备 ID，默认 0。"""
    return int(os.environ.get("TRITON2PYPTO_DEVICE_ID", "0"))


def make_pypto_run_config(
    *,
    platform: str | None = None,
    device_id: int | None = None,
    backend_type: BackendType | None = None,
    strategy: OptimizationStrategy | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    enable_profiling: bool = False,
) -> RunConfig:
    """构建与当前 E2E 策略一致的 :class:`RunConfig`。"""
    from pypto.backend import BackendType as BT
    from pypto.ir.pass_manager import OptimizationStrategy as OS
    from pypto.runtime import RunConfig

    plat = platform if platform is not None else get_e2e_platform()
    dev = device_id if device_id is not None else get_e2e_device_id()
    bt = backend_type if backend_type is not None else BT.CCE
    st = strategy if strategy is not None else OS.Default
    rc = RunConfig(
        platform=plat,
        device_id=dev,
        backend_type=bt,
        strategy=st,
    )
    if rtol is not None:
        rc.rtol = rtol
    if atol is not None:
        rc.atol = atol
    return rc


def pytest_skip_if_npu_platform_unavailable(platform: str) -> None:
    """在 pytest 中若请求 ``a2a3`` 但本机无 ``npu-smi`` 则 skip。"""
    import pytest

    if platform == "a2a3" and not npu_runtime_available():
        pytest.skip("TRITON2PYPTO_PLATFORM=a2a3 but npu-smi not in PATH (no NPU driver?)")
