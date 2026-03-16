# 端到端测试说明

## 概述

端到端验证流程：**Triton 源码 → TTIR → PyPTO IR → simpler CPU 仿真 → 与参考结果对比**。

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
```

## pytest 测试

`tests/test_triton_to_pypto_e2e.py` 包含：

- **TestTritonToPyPTOConversion**：add/sub/mul/div/exp/reduce_sum/matmul 的 TTIR→PyPTO 转换与编译（无需 SIMPLER_ROOT）
- **TestTritonToPyPTOExecution**：add/sub/mul/div/reduce_sum/matmul 的 simpler 执行与 golden 对比（需 SIMPLER_ROOT）；exp 因已知问题被 skip

```bash
# 仅转换与编译测试
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTOConversion -v

# 完整执行测试（需 SIMPLER_ROOT）
export SIMPLER_ROOT=$(pwd)/third_party/simpler
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTOExecution -v
```

## 对应关系

| 测试层级 | examples/ | tests/ |
|----------|-----------|--------|
| add | run_e2e.py -k add | TestTritonToPyPTOConversion, TestTritonToPyPTOExecution |
| exp | run_e2e.py -k exp | TestTritonToPyPTOConversion（Execution 中 exp 被 skip） |
| sub/mul/div/reduce_sum/matmul | run_e2e.py -k \<name\> | TestTritonToPyPTOConversion, TestTritonToPyPTOExecution |

所有 kernel 均有 pytest 覆盖；examples 下统一通过 `run_e2e.py` 运行。
