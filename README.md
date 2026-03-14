# Triton to PyPTO Adapter

将 Triton IR 适配到 PyPTO 后端，实现在华为 AI 加速器上运行 Triton 内核。

## 项目结构

```
triton2pypto/
├── third_party/
│   ├── triton/          # submodule: triton-lang/triton
│   ├── pypto/           # submodule: hw-native-sys/pypto
│   └── simpler/         # submodule: CPU 仿真运行时
├── src/
│   ├── triton_adapter/  # Triton IR 提取/转换
│   ├── pypto_backend/   # PyPTO 后端接入
│   └── passes/          # IR 转换 pass
├── examples/            # 示例（从 Triton 源码出发）
├── tests/
├── scripts/
└── tasks/
```

## 示例说明

所有示例均**从 Triton 源码提取 TTIR**，遵循标准实现模式（含 `program_id` 和 `mask`）：

- `add_kernel.py`, `sub_kernel.py`, `mul_kernel.py`, `div_kernel.py`, `exp_kernel.py` - elementwise 算子
- `reduce_sum_kernel.py` - 行内 reduce
- `matmul_kernel.py` - 分块矩阵乘

每个 Triton kernel 均包含：

```python
pid = tl.program_id(0)
blk = pid * BLOCK
offs = blk + tl.arange(0, BLOCK)
mask = offs < n
# 使用 mask 进行 load/store
```

## 构建环境

### 前置要求

- Python 3.10+
- CMake 3.15+（PyPTO 构建）
- C++17 编译器（g++）
- Git（用于 submodule）

### 1. 初始化 Submodules

```bash
git submodule update --init --recursive
```

### 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS

pip install -e third_party/pypto
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

### 3. 设置 CPU 仿真环境变量

```bash
export SIMPLER_ROOT=$(pwd)/third_party/simpler
```

## 运行端到端示例

所有示例从 Triton 源码提取 TTIR，经 PyPTO 转换后在 simpler CPU 仿真中执行并与参考结果对比。

```bash
source .venv/bin/activate
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# add kernel 端到端（带 mask）
python examples/run_triton_to_pypto_e2e.py

# exp kernel 端到端
python examples/run_exp_e2e.py
```

## 运行测试

```bash
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

pytest tests/ -v

# 仅转换与编译测试（无需 SIMPLER_ROOT）
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTOConversion -v

# 完整执行测试（需 SIMPLER_ROOT）
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTOExecution -v
```

## 快速开始

```bash
git submodule update --init --recursive
python3 -m venv .venv && source .venv/bin/activate
pip install -e third_party/pypto
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python examples/run_triton_to_pypto_e2e.py
```

## 开发

详见 [AGENTS.md](AGENTS.md)

## 故障排除

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| `SIMPLER_ROOT not set` | 未设置环境变量 | `export SIMPLER_ROOT=$(pwd)/third_party/simpler` |
| `ImportError: triton_adapter` | 未加入 src 路径 | `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"` |
| PyPTO 安装失败 | CMake/编译器缺失 | 安装 CMake 3.15+、g++、ninja |
| simpler 子模块为空 | 未初始化 submodule | `git submodule update --init --recursive` |
