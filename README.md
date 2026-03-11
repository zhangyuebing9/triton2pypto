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
├── examples/            # 示例脚本
├── tests/
├── scripts/
└── tasks/
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
# Windows: .venv\Scripts\activate

# 安装 PyPTO（使用 submodule 中的源码）
pip install -e third_party/pypto

# 安装 triton2pypto（纯 Python，通过 PYTHONPATH 或 pip）
# 方式 A：开发模式（若项目有 CMake 则需完整构建）
pip install -e .

# 方式 B：仅将 src 加入路径运行示例
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

### 3. 设置 CPU 仿真环境变量

PyPTO 通过 simpler 的 a2a3sim 平台在 CPU 上仿真执行。需设置：

```bash
export SIMPLER_ROOT=$(pwd)/third_party/simpler
```

## 运行 Elementwise 示例

Elementwise 端到端验证：TTIR → PyPTO IR → CPU 仿真 → 精度对比。

```bash
# 确保在项目根目录，已激活 venv，并设置 SIMPLER_ROOT
source .venv/bin/activate
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

python examples/run_elementwise_e2e.py
```

该脚本将：

1. 使用 elementwise add 的 TTIR 样本
2. 转换为 PyPTO IR 并展示
3. 在 a2a3sim（CPU 仿真）上执行等价 PyPTO 程序
4. 验证 PyPTO 输出与参考 (a+b) 在数值上一致

成功输出示例：

```
======================================================================
✓ 验证通过: PyPTO simpler CPU 输出与参考 (a+b) 一致
======================================================================
```

## 运行测试

```bash
# 设置环境变量
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 运行所有测试
pytest tests/ -v

# 仅运行 Phase 1 功能测试
pytest tests/test_phase1_functional.py -v

# 运行 TTIR 转换器单元测试
pytest tests/test_ttir_converter.py -v
```

带 `SIMPLER_ROOT` 时，`test_pypto_vector_add_run_cpu` 会执行完整的 CPU 仿真；未设置时该测试会被跳过。

## 快速开始（精简）

```bash
git submodule update --init --recursive
python3 -m venv .venv && source .venv/bin/activate
pip install -e third_party/pypto
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python examples/run_elementwise_e2e.py
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
