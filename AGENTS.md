# Working on triton2pypto

Triton to PyPTO adapter - enables Triton kernels on Huawei AI accelerators.

## Project Overview

This project provides an adapter layer between Triton's MLIR-based IR and PyPTO's compilation pipeline:
- **triton_adapter**: Extracts and transforms Triton IR (TTIR/TTGIR)
- **passes**: IR transformation passes
- **pypto_backend**: PyPTO backend integration

---

## 执行环境选择

### 有 NPU 环境 → 使用 a2a3 上板执行

当系统有昇腾 NPU 设备（能执行 `npu-smi info`）时，**必须使用 NPU 上板方式**：

```bash
# ✅ 正确：使用 a2a3 平台（NPU 上板）
python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0

# ❌ 错误：在有 NPU 时使用 CPU 仿真
python examples/run_e2e.py --kernel add --platform a2a3sim  # 禁止！
```

### 无 NPU 环境 → 使用 a2a3sim CPU 仿真

当系统没有昇腾 NPU 设备时，使用 CPU 仿真：

```bash
# 开发机无 NPU，使用 CPU 仿真
python examples/run_e2e.py --kernel add --platform a2a3sim
# 或使用默认（a2a3sim）
python examples/run_e2e.py --kernel add
```

### 当前 NPU 设备信息（参考）

- 物理设备 ID: 4（npu-smi 显示）
- **逻辑设备 ID: 0（CANN runtime 使用，从 0 开始编号）**
- 芯片型号: 910B3
- CANN 版本: 9.0.0
- ASCEND_HOME_PATH: /home/developer/Ascend/cann-9.0.0

---

## NPU 执行环境配置指南

### 1. 环境准备

```bash
# 初始化子模块
git submodule update --init --recursive

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装 PyPTO
pip install -e third_party/pypto

# 设置环境变量
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH=$(pwd)/src:$(pwd)
```

### 2. PyPTO 兼容性补丁（重要）

**⚠️ 以下修改必须手动应用到 third_party/pypto 子模块，这些修改无法上传到 triton2pypto 代码仓。**

#### 修改 1: orchestration 函数签名

**文件**: `third_party/pypto/src/codegen/orchestration/orchestration_codegen.cpp`

**位置**: 第 904-909 行

**问题**: simpler 的 tensormap_and_ringbuffer runtime 期望 5 个参数，但 pypto 只生成 3 个参数

**修改前**:
```cpp
// 6. Entry function
oss << "__attribute__((visibility(\"default\")))\n";
oss << "void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {\n";
oss << "    (void)arg_count;\n\n";
```

**修改后**:
```cpp
// 6. Entry function (5 parameters to match DeviceOrchestrationFunc signature)
oss << "__attribute__((visibility(\"default\")))\n";
oss << "void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count, int orch_thread_num, int orch_thread_index) {\n";
oss << "    (void)arg_count;\n";
oss << "    (void)orch_thread_num;\n";
oss << "    (void)orch_thread_index;\n\n";
```

#### 修改 2: block_dim 配置（C++ 版本）

**文件**: `third_party/pypto/src/codegen/cce/cce_codegen.cpp`

**位置**: 第 150 行

**问题**: block_dim=24 与 simpler tensormap_and_ringbuffer runtime 不兼容

**修改前**:
```cpp
oss << "\t\"block_dim\": 24,\n";
```

**修改后**:
```cpp
oss << "\t\"block_dim\": 18,\n";
```

#### 修改 3: block_dim 配置（Python 版本）

**文件**: `third_party/pypto/python/pypto/ir/pto_codegen.py`

**位置**: 第 231 行

**修改前**:
```python
'\t"block_dim": 3,',
```

**修改后**:
```python
'\t"block_dim": 18,',
```

### 3. 重新编译 PyPTO

```bash
cd third_party/pypto
pip install -e .
cd ../..
```

### 4. 验证修改

```bash
# 检查 orchestration 函数签名
grep -n "aicpu_orchestration_entry" third_party/pypto/src/codegen/orchestration/orchestration_codegen.cpp

# 应输出: 包含 "int orch_thread_num, int orch_thread_index"

# 检查 block_dim
grep -n "block_dim" third_party/pypto/src/codegen/cce/cce_codegen.cpp
grep -n "block_dim" third_party/pypto/python/pypto/ir/pto_codegen.py

# 应输出: block_dim": 18
```

### 5. 运行 NPU 测试

```bash
# 设置环境
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH=$(pwd)/src:$(pwd)

# 运行 triton2pypto E2E 测试
python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0

# 运行 pypto 测试
cd third_party/pypto
pytest tests/st/runtime/test_elementwise.py -v --forked --platform=a2a3 --device=0
```

---

## NPU 验证结果（2026-03-26）

### pypto 测试结果

| 测试文件 | 通过 | 失败 | 备注 |
|---------|------|------|------|
| test_elementwise.py | 4 | 2 | 失败均为 ptoas binary 缺失（非核心依赖） |
| test_matmul.py | 7 | 7 | 失败均为 ptoas binary 缺失（非核心依赖） |
| test_dag.py | 1 | 1 | 失败为 ptoas binary 缺失（非核心依赖） |

**结论**: 所有核心功能测试通过。`*_ptoas_strategy` 测试需要外部 ptoas 汇编器，不影响核心代码生成和执行流程。

### triton2pypto E2E 测试结果

| Kernel | 结果 | 备注 |
|--------|------|------|
| add | ✅ PASS | |
| sub | ✅ PASS | |
| mul | ✅ PASS | |
| div | ✅ PASS | |
| exp | ⚠️ 超时 | 可能是设备状态问题 |
| matmul | ✅ PASS | |
| reduce_sum | ❌ FAIL | RuntimeError: 507018 |

**结论**: 5/7 kernel 在 NPU 上成功执行并通过数值验证。

---

## PyPTO 修改汇总（供其他环境复现）

| 文件 | 行号 | 修改内容 | 原值 | 新值 |
|------|------|---------|------|------|
| `pypto/.../orchestration_codegen.cpp` | 904-909 | 函数签名 | 3 参数 | 5 参数 |
| `pypto/.../cce_codegen.cpp` | 150 | block_dim | 24 | 18 |
| `pypto/.../pto_codegen.py` | 231 | block_dim | 3 | 18 |

---

## 后续待办事项

### 高优先级

1. **调查 reduce_sum 执行失败** (RuntimeError: 507018)
   - 可能是 reduce 操作的 tile 语义映射问题
   - 需要检查 tile.row_sum 的实现

2. **调查 exp 超时问题**
   - 可能是设备状态或 kernel 实现问题
   - 需要在干净环境下重试

### 中优先级

3. **统一 block_dim 配置**
   - 当前 hardcode 为 18
   - 应该根据 tile 大小动态计算

4. **完善错误处理**
   - 添加更详细的错误信息
   - 改善调试体验

### 低优先级

5. **支持 ptoas 策略测试**
   - 安装 ptoas binary
   - 验证优化后的 kernel 执行

---

## Prerequisites

- Python 3.10+
- CMake 3.15+
- C++17 compiler

## Build Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Build with specific CMake build type
CMAKE_BUILD_TYPE=Release pip install -e .

# Build with ccache (auto-detected if available)
CMAKE_BUILD_TYPE=Release pip install -e .
```

## Test Commands

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_triton_adapter.py

# Run single test
pytest tests/test_triton_adapter.py::test_ir_extraction

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v --tb=short tests/
```

## Lint and Type Check

```bash
# Run ruff linter
ruff check src/

# Run ruff formatter
ruff format src/

# Run mypy type check
mypy src/

# Run all checks
ruff check src/ && ruff format --check src/ && mypy src/
```

## Submodule Management

```bash
# Initialize submodules
git submodule update --init --recursive

# Update submodules to latest
git submodule update --remote

# Work inside a submodule
cd third_party/triton
git checkout main
git pull origin main
```

## Code Style Guidelines

### Imports

```python
# Standard library first
import os
import sys
from typing import Optional, List, Dict

# Third-party imports
import torch
from triton import ir as tir

# Local imports (absolute)
from triton_adapter.ir_extractor import extract_ttir
from passes.transform import convert_layout
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- **Type aliases**: `PascalCase` (e.g., `IRNode = tir.Operation`)

### Type Annotations

Always use type annotations:

```python
def transform_ir(
    ir_module: tir.Module,
    options: TransformOptions | None = None,
) -> tir.Module:
    ...

class IRTransformer:
    def __init__(self, config: Config) -> None:
        self.config = config
```

### Error Handling

```python
# Use custom exceptions
class IRConversionError(Exception):
    """Raised when IR conversion fails."""

def convert_ir(ir: tir.Module) -> pto.Module:
    if not ir.is_valid():
        raise IRConversionError(f"Invalid IR: {ir}")
    ...
```

### Documentation

```python
def extract_kernel_ir(kernel: triton.JITFunction) -> tir.Module:
    """Extract MLIR from a Triton kernel.

    Args:
        kernel: Compiled Triton kernel function.

    Returns:
        The extracted MLIR module.

    Raises:
        IRExtractionError: If kernel has not been compiled.
    """
    ...
```

## Working with Triton IR

### Debug Environment Variables

```bash
# Dump MLIR IR at each pass
MLIR_ENABLE_DUMP=1 pytest tests/

# Use Triton interpreter (no GPU needed)
TRITON_INTERPRET=1 pytest tests/

# Dump kernel IR to file
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/ir_dump pytest tests/
```

### IR Extraction Pattern

```python
import triton
from triton import ir as tir

def get_triton_ir(kernel: triton.JITFunction, *args, **kwargs) -> tir.Module:
    """Extract TTIR from a compiled Triton kernel."""
    # Compile to get the IR
    compiled = kernel.run(*args, **kwargs)
    # Access the internal IR
    return compiled.asm["ttir"]
```

## Working with PyPTO

### Build PyPTO from Source

```bash
cd third_party/pypto
pip install -e ".[dev]"
```

### Integration Pattern

```python
from pypto import TensorGraph, TileGraph

def triton_to_pypto(ttir: tir.Module) -> TensorGraph:
    """Convert Triton IR to PyPTO TensorGraph."""
    graph = TensorGraph()
    # Convert operations...
    return graph
```

## Git Workflow

- Branch from `main` for features
- Use descriptive branch names: `feature/add-matmul-pass`, `fix/layout-conversion`
- Squash merge PRs
- Reference issues in commits: `fix #123: handle empty IR`

## Related Documentation

- Triton: `third_party/triton/README.md`, `third_party/triton/AGENTS.md`
- PyPTO: `third_party/pypto/README.md`
- Simpler: `third_party/simpler/README.md`

## Cursor Cloud specific instructions

### Environment overview

This is a Python library project (no running services). A virtualenv at `/workspace/.venv` contains all dependencies. Always activate it before running commands:

```bash
source /workspace/.venv/bin/activate
```

The main package cannot be `pip install -e .` because `pyproject.toml` uses scikit-build-core but no `CMakeLists.txt` exists yet (C++ extension layer not implemented). Instead, use `PYTHONPATH=/workspace/src` to make the source importable.

### Known compatibility issues

1. **`DataType.FP64` AttributeError**: The pinned pypto submodule does not expose `DataType.FP64` (reserved in C++ header but not bound to Python). The `ttir_converter.py` references it at class-definition time, and `triton_adapter/__init__.py` catches `ImportError` but not `AttributeError`. When pypto is installed, importing `triton_adapter` fails. Workaround: to run MLIR parser / IR extractor tests, temporarily `pip uninstall pypto -y` so the `except ImportError` path in `__init__.py` activates.

2. **`triton.ir` ImportError**: Triton 3.6.0 does not expose a top-level `ir` submodule. `passes/layout_pass.py` imports `from triton import ir as tir` unconditionally, so `test_layout_pass.py` fails at collection. `ir_extractor.py` handles this gracefully with try/except.

### Running tests

```bash
source /workspace/.venv/bin/activate

# Run MLIR parser and IR extractor tests (requires pypto NOT installed):
pip uninstall pypto -y
PYTHONPATH=/workspace/src pytest tests/test_mlir_parser.py tests/test_ir_extractor.py -v

# Reinstall pypto after testing:
cd /workspace/third_party/pypto && pip install -e . && cd /workspace

# Run Triton->PyPTO E2E 测试（NPU 上板，使用 SIMPLER_ROOT）:
export SIMPLER_ROOT=$(pwd)/third_party/simpler
PYTHONPATH=/workspace/src:/workspace pytest tests/test_triton_to_pypto_e2e.py -v
```

Pre-existing test failures (8 pass / 8 fail out of 16) are due to MLIR parser op-name extraction bugs and triton.ir unavailability—not environment issues.

### Lint and type check

```bash
source /workspace/.venv/bin/activate
ruff check src/           # linter (40 pre-existing warnings)
ruff format --check src/  # format check
mypy src/                 # type check (passes clean)
```

### Building pypto from source

pypto is a C++ extension built with scikit-build-core + nanobind. Building takes ~70s. Requires `python3-dev` system package. The update script handles this automatically.