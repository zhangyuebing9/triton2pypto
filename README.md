# Triton to PyPTO Adapter

将 Triton IR 适配到 PyPTO 后端，实现在华为 AI 加速器上运行 Triton 内核。

## 项目结构

```
triton2pypto/
├── third_party/
│   ├── triton/          # submodule: triton-lang/triton
│   ├── pypto/           # submodule: hw-native-sys/pypto
│   └── simpler/         # submodule: 运行时（支持 NPU 和 CPU 仿真）
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

### 3. 设置环境变量

```bash
export SIMPLER_ROOT=$(pwd)/third_party/simpler
```

## 运行端到端示例

所有示例从 Triton 源码提取 TTIR，经 PyPTO 转换后在 **simpler** 上执行，并与 **golden**（PyTorch 参考，与 Triton 数学语义一致）逐元素比对。

支持 kernel：add, sub, mul, div, exp, reduce_sum, matmul。详见 [examples/E2E_README.md](examples/E2E_README.md)。

### 执行环境选择

- **有 NPU 环境**：使用 `--platform a2a3` 真机执行
- **无 NPU 环境**：使用 `--platform a2a3sim`（默认）CPU 仿真

```bash
# 有 NPU 环境（能执行 npu-smi info）
python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0

# 无 NPU 环境（开发机）
python examples/run_e2e.py --kernel add  # 默认 a2a3sim
```

---

## NPU 真机验证（昇腾环境）

### 环境准备

在已安装 **CANN / 驱动** 的机器上（能执行 `npu-smi info`），除上述依赖外还需：

1. **加载 CANN 环境**（路径以本机安装为准）：
   ```bash
   source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
   ```

2. **子模块与 PyPTO**：
   ```bash
   git submodule update --init --recursive
   pip install -e third_party/pypto
   ```

3. **设置工程环境变量**：
   ```bash
   cd /path/to/triton2pypto
   source .venv/bin/activate
   export SIMPLER_ROOT=$(pwd)/third_party/simpler
   export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
   ```

### PyPTO 兼容性补丁（重要）

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

### 重新编译 PyPTO

```bash
cd third_party/pypto
pip install -e .
cd ../..
```

### 验证修改

```bash
# 检查 orchestration 函数签名
grep -n "aicpu_orchestration_entry" third_party/pypto/src/codegen/orchestration/orchestration_codegen.cpp
# 应输出: 包含 "int orch_thread_num, int orch_thread_index"

# 检查 block_dim
grep -n "block_dim" third_party/pypto/src/codegen/cce/cce_codegen.cpp
grep -n "block_dim" third_party/pypto/python/pypto/ir/pto_codegen.py
# 应输出: block_dim": 18
```

### 运行 NPU 测试

```bash
# 设置环境
export SIMPLER_ROOT=$(pwd)/third_party/simpler
export PYTHONPATH=$(pwd)/src:$(pwd)

# 运行单个 kernel
python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0
python examples/run_e2e.py --kernel matmul --platform a2a3 --device-id 0

# 列出所有支持的 kernel
python examples/run_e2e.py --list
```

### NPU 验证结果（2026-03-26）

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

### 设备 ID 说明

- `npu-smi info` 显示的是**物理设备 ID**
- CANN runtime 使用**逻辑设备 ID**（从 0 开始编号）
- 例如：物理设备 ID 4 对应逻辑设备 ID 0

---

## PyPTO 修改汇总（供其他环境复现）

| 文件 | 行号 | 修改内容 | 原值 | 新值 |
|------|------|---------|------|------|
| `pypto/.../orchestration_codegen.cpp` | 904-909 | 函数签名 | 3 参数 | 5 参数 |
| `pypto/.../cce_codegen.cpp` | 150 | block_dim | 24 | 18 |
| `pypto/.../pto_codegen.py` | 231 | block_dim | 3 | 18 |

---

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
| NPU 测试全跳过 | 无 `npu-smi` 或类被 skipif | 在昇腾机器加载 CANN 后重试 |
| `a2a3` 编译失败 | 缺少 `ccec` 或 `ASCEND_HOME_PATH` | 安装完整 CANN toolkit，并 `source setenv.bash` |
| `ImportError: triton_adapter` | 未加入 src 路径 | `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"` |
| PyPTO 安装失败 | CMake/编译器缺失 | 安装 CMake 3.15+、g++、ninja |
| simpler 子模块为空 | 未初始化 submodule | `git submodule update --init --recursive` |
| NPU 执行卡住 | PyPTO 补丁未应用 | 按上述「PyPTO 兼容性补丁」修改并重新编译 |
| RuntimeError: 507018 | kernel 执行失败 | 检查 tile 操作语义是否正确映射 |

## 后续待办事项

### 高优先级

1. **调查 reduce_sum 执行失败** (RuntimeError: 507018)
   - 可能是 reduce 操作的 tile 语义映射问题

2. **调查 exp 超时问题**
   - 可能是设备状态或 kernel 实现问题

### 中优先级

3. **统一 block_dim 配置**
   - 当前 hardcode 为 18
   - 应该根据 tile 大小动态计算

4. **完善错误处理**
   - 添加更详细的错误信息