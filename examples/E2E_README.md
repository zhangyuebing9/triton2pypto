# 端到端测试说明

## 概述

端到端验证流程：**Triton 源码 → TTIR → PyPTO IR → simpler 执行 → 与 golden 对比**。

- **a2a3sim**：主机线程仿真，无需 NPU（默认）。
- **a2a3**：昇腾真机，与 [simpler README](../third_party/simpler/README.md) 中 `RuntimeCompiler(platform="a2a3")` / `run_example.py -p a2a3` 同属硬件路径；PyPTO `run(..., RunConfig(platform="a2a3"))` 经 `CodeRunner` 完成编译、上板与数值校验。

配置集中在 [`e2e_common.py`](e2e_common.py)，与 `run_e2e.py`、pytest 共用。

## 文件结构

| 文件 | 说明 |
|------|------|
| `run_e2e.py` | **统一入口**，支持所有 kernel，通过 `--kernel` 选择 |
| `add_kernel.py`, `sub_kernel.py`, ... | Triton kernel 定义 |

## 支持的 Kernel

| Kernel | 类型 | 说明 |
|--------|------|------|
| add | elementwise | a + b |
| sub | elementwise | a - b |
| mul | elementwise | a * b |
| div | elementwise | a / b |
| exp | elementwise | exp(x) |
| reduce_sum | reduce | 行内求和 |
| matmul | matmul | 分块矩阵乘 |

## 运行方式

### 统一入口（推荐）

```bash
source .venv/bin/activate
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 列出所有支持的 kernel
python examples/run_e2e.py --list

# 运行指定 kernel（默认 add）
python examples/run_e2e.py
python examples/run_e2e.py --kernel add
python examples/run_e2e.py --kernel exp
python examples/run_e2e.py --kernel sub
python examples/run_e2e.py --kernel mul
python examples/run_e2e.py --kernel div
python examples/run_e2e.py --kernel reduce_sum
python examples/run_e2e.py --kernel matmul

# add kernel 额外与 Triton TRITON_INTERPRET 结果对比
python examples/run_e2e.py --kernel add --triton-compare

# NPU 真机（需 CANN、npu-smi；设备号可改）
export TRITON2PYPTO_PLATFORM=a2a3
export TRITON2PYPTO_DEVICE_ID=0
python examples/run_e2e.py --kernel add
# 或命令行覆盖环境变量：
python examples/run_e2e.py --kernel matmul --platform a2a3 --device-id 0
```

### 环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `TRITON2PYPTO_PLATFORM` | `a2a3sim` 或 `a2a3` | `a2a3sim` |
| `TRITON2PYPTO_DEVICE_ID` | 真机设备号 | `0` |

## pytest 测试

`tests/test_triton_to_pypto_e2e.py` 包含：

- **TestTritonToPyPTOConversion**：add/sub/mul/div/exp/reduce_sum/matmul 的 TTIR→PyPTO 转换与编译（无需 SIMPLER_ROOT）
- **TestTritonToPyPTOExecution**：add/sub/mul/div/reduce_sum/matmul 的执行与 golden 对比（需 SIMPLER_ROOT）；默认平台由 `TRITON2PYPTO_PLATFORM` 决定（未设置时为 `a2a3sim`）；若设为 `a2a3` 则需本机有 `npu-smi`；exp 因已知问题被 skip
- **TestTritonToPyPTONPUExecution**：与上一类相同的 golden 与张量规格，但 **固定 `platform=a2a3`**；仅当 `SIMPLER_ROOT` 已设置且 `npu-smi` 在 PATH 中时才运行，便于在 CI（无 NPU）上自动跳过

```bash
# 仅转换与编译测试
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTOConversion -v

# 仿真执行（需 SIMPLER_ROOT）
export SIMPLER_ROOT=$(pwd)/third_party/simpler
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTOExecution -v

# 真机执行（需 SIMPLER_ROOT + CANN + npu-smi）
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTONPUExecution -v
```

## 对应关系

| 测试层级 | examples/ | tests/ |
|----------|-----------|--------|
| add | run_e2e.py -k add | Conversion, Execution, NPUExecution |
| exp | run_e2e.py -k exp | Conversion（Execution / NPU 中 exp 被 skip） |
| sub/mul/div/reduce_sum/matmul | run_e2e.py -k \<name\> | Conversion, Execution, NPUExecution |

所有 kernel 均有 pytest 覆盖；examples 下统一通过 `run_e2e.py` 运行。
