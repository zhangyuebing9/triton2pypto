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

所有示例从 Triton 源码提取 TTIR，经 PyPTO 转换后在 **simpler** 上执行，并与 **golden**（PyTorch 参考，与 Triton 数学语义一致）逐元素比对。

- **CPU 仿真**（默认）：`platform=a2a3sim`，无需昇腾设备，仅需 g++ 等主机工具链。
- **NPU 上板**：`platform=a2a3`，走真实 Ascend 路径（CANN、`ccec`、AICPU/AICore 二进制），与 simpler 文档中硬件模式一致。

支持 kernel：add, sub, mul, div, exp, reduce_sum, matmul。详见 [examples/E2E_README.md](examples/E2E_README.md)。

### CPU 仿真（开发机常用）

```bash
source .venv/bin/activate
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 可选：显式指定（与默认 a2a3sim 等价）
export TRITON2PYPTO_PLATFORM=a2a3sim

python examples/run_e2e.py --kernel add
python examples/run_e2e.py --kernel exp
python examples/run_e2e.py --list
```

### NPU 真机验证（昇腾环境）

在已安装 **CANN / 驱动** 的机器上（能执行 `npu-smi info`），除上述依赖外还需：

1. **加载 CANN 环境**（路径以本机安装为准，以下为常见示例）：
   ```bash
   source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
   ```
2. **子模块与 PyPTO**：与 CPU 流程相同（`git submodule update --init`、`pip install -e third_party/pypto`）。若使用 simpler 的 a2a3sim 编排补丁，按 [AGENTS.md](AGENTS.md) 中「Simpler 环境配置」处理 `apply_pypto_patches.sh`。
3. **设置工程环境变量**：
   ```bash
   cd /path/to/triton2pypto
   source .venv/bin/activate
   export SIMPLER_ROOT=$(pwd)/third_party/simpler
   export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
   export TRITON2PYPTO_PLATFORM=a2a3
   export TRITON2PYPTO_DEVICE_ID=0   # 多卡时改为空闲卡号，可先 npu-smi info 查看
   ```
4. **运行**（二选一）：
   ```bash
   # 命令行指定平台（会覆盖 TRITON2PYPTO_PLATFORM）
   python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0
   python examples/run_e2e.py --kernel matmul -p a2a3 -d 0
   ```
5. **pytest 真机用例**：类 `TestTritonToPyPTONPUExecution` 在检测到 `npu-smi` 可用时运行，与 `run_e2e.py --platform a2a3` 使用同一套 golden；无驱动时整类跳过。
   ```bash
   export SIMPLER_ROOT=$(pwd)/third_party/simpler
   pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTONPUExecution -v
   ```
6. **结果含义**：`pypto.runtime.run` 通过 simpler 的 `CodeRunner` 在设备上执行后，将输出与 golden 按 `RunConfig` 的 `rtol`/`atol` 比较；通过即表示 **NPU 结果与 Triton 参考（golden）一致**。若失败，可查看终端栈迹，并参考 simpler 的 `~/ascend/log/debug/device-<id>/` 或 `ASCEND_WORK_PATH` 下设备日志。

**说明**：golden 为 CPU 上 PyTorch 计算，与 `run_e2e.py --triton-compare`（仅 add）使用的 Triton 解释器路径互补；上板验证的核心是 **PyPTO→simpler(a2a3) 链路数值与 golden 一致**。

## 运行测试

```bash
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

pytest tests/ -v

# 仅转换与编译测试（无需 SIMPLER_ROOT）
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTOConversion -v

# 完整执行测试（需 SIMPLER_ROOT；默认 a2a3sim）
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTOExecution -v

# NPU 上板执行与 golden 对比（需 SIMPLER_ROOT + npu-smi + CANN）
pytest tests/test_triton_to_pypto_e2e.py::TestTritonToPyPTONPUExecution -v
```

## 快速开始

```bash
git submodule update --init --recursive
python3 -m venv .venv && source .venv/bin/activate
pip install -e third_party/pypto
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python examples/run_e2e.py --kernel add
```

## 开发

详见 [AGENTS.md](AGENTS.md)

## 故障排除

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| `SIMPLER_ROOT not set` | 未设置环境变量 | `export SIMPLER_ROOT=$(pwd)/third_party/simpler` |
| NPU 测试全跳过 | 无 `npu-smi` 或类被 skipif | 在昇腾机器加载 CANN 后重试；开发机只跑 `TestTritonToPyPTOExecution` |
| `a2a3` 编译失败 | 缺少 `ccec` 或 `ASCEND_HOME_PATH` | 安装完整 CANN toolkit，并 `source setenv.bash` |
| `ImportError: triton_adapter` | 未加入 src 路径 | `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"` |
| PyPTO 安装失败 | CMake/编译器缺失 | 安装 CMake 3.15+、g++、ninja |
| simpler 子模块为空 | 未初始化 submodule | `git submodule update --init --recursive` |
