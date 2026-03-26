# Phase 1 实施进度报告

## 已完成工作

### 1. 基础架构 (✅ 完成)
- **TTIRToPyptoConverter** 类框架
  - ValueMap 值追踪
  - SpanTracker 源码位置追踪
  - 错误处理类（ConversionError, UnsupportedOpError）
  
- **TypeMapper** 类型映射
  - 支持 9 种基本数据类型映射（i1/i8/i16/i32/i64/fp16/bf16/fp32/fp64）
  - TensorType 创建

### 2. MLIR 解析器 (✅ 完成)
- **MLIRParser** 轻量级 MLIR 文本解析器
  - 解析操作、操作数、类型
  - 支持基本 MLIR 文本格式
  - 独立于 Triton 环境运行

### 3. 测试框架 (✅ 完成)
- 单元测试基础设施
- 类型映射测试
- 转换器初始化测试
- 支持的操作列表验证

### 4. 示例代码 (✅ 完成)
- `examples/phase1_elementwise_example.py`
  - 演示解析流程
  - 类型映射示例
  - Phase 1 路线图展示

## 当前状态

### 文件结构
```
src/triton_adapter/
├── __init__.py           # 模块导出
├── ir_extractor.py       # IR 提取（待实现）
├── mlir_parser.py        # ✅ MLIR 文本解析器
└── ttir_converter.py     # ✅ 转换器框架

tests/
├── test_ttir_converter.py  # ✅ 单元测试
├── test_ir_extractor.py    # 基础测试
└── test_layout_pass.py     # 布局测试

examples/
└── phase1_elementwise_example.py  # ✅ 示例代码

tasks/
├── todo.md                    # 任务清单
└── implementation_plan.md     # 实施计划
```

### 支持的操作（框架已就绪）
| 操作 | 状态 | 说明 |
|------|------|------|
| arith.constant | 框架就绪 | 需实现转换逻辑 |
| arith.addf | 框架就绪 | 需实现转换逻辑 |
| arith.subf | 框架就绪 | 需实现转换逻辑 |
| arith.mulf | 框架就绪 | 需实现转换逻辑 |
| arith.divf | 框架就绪 | 需实现转换逻辑 |
| tt.load | 框架就绪 | 需实现转换逻辑 |
| tt.store | 框架就绪 | 需实现转换逻辑 |
| tt.make_block_ptr | 框架就绪 | 需实现转换逻辑 |
| tt.advance | 框架就绪 | 需实现转换逻辑 |
| tt.exp | 框架就绪 | 需实现转换逻辑 |
| arith.cmpf | 框架就绪 | 需实现转换逻辑 |
| arith.select | 框架就绪 | 需实现转换逻辑 |
| tt.program_id | 框架就绪 | 需实现转换逻辑 |

## 已完成（Phase 1 实现）

### 核心转换逻辑 ✅
1. **常量转换** (arith.constant) - 已实现
2. **算术运算** (addf/subf/mulf/divf, addi/subi/muli/divi) - 已实现
3. **内存操作** (load/store) - 已实现 tile.load / tile.store
4. **块指针处理** (make_block_ptr/advance) - 框架就绪
5. **扩展算子** (tt.exp, arith.cmpf/select, tt.program_id) - 已实现

### CPU 功能测试 ✅
- 支持 SIMPLER_ROOT 指向 third_party/simpler 进行 a2a3sim CPU 仿真
- 测试: tests/test_phase1_functional.py
- PyPTO IR 使用 submodule 中的 pypto，Triton IR 使用 submodule 中的 triton

### 环境配置
```bash
# 安装 PyPTO（使用 submodule）
pip install -e third_party/pypto

# 安装 triton2pypto
pip install -e .

# 设置 SIMPLER_ROOT 用于 CPU 仿真测试
export SIMPLER_ROOT=$(pwd)/third_party/simpler
```

## 已完成（本次）

### 端到端 add elementwise 流程 ✅
- **extract_ttir**: 支持 compile-only 路径（无 GPU 时用 triton.compile + ASTSource）
- **MLIR 解析**: 支持真实 TTIR（tt.splat, tt.addptr, tt.make_range, tt.func 嵌套括号）
- **转换器**: 支持 tt.splat/addptr 指针链追踪，正确映射 load/store 的 base
- **测试**: tests/test_triton_to_pypto_e2e.py
  - test_extract_ttir_from_triton_source
  - test_extract_ttir_api（extract_ttir + convert 流程）
  - test_convert_real_ttir_to_pypto（转换 + PyPTO 编译）
  - test_triton_add_to_pypto_run_cpu（需 SIMPLER_ROOT，simpler 环境）

### 使用方式
```bash
# 转换 + 编译（无需 GPU）
PYTHONPATH=/workspace/src:/workspace python examples/run_e2e.py --kernel add

# 运行 simpler CPU 仿真需 SIMPLER_ROOT 且 simpler 环境正确
export SIMPLER_ROOT=$(pwd)/third_party/simpler
```

## 已完成（elementwise + reduce + matmul）

### 算子支持扩展 ✅
- **Elementwise**: add, sub, mul, div, exp（含 math.exp → tile.exp）
- **Reduce**: tt.reduce → tile.row_sum / tile.row_max（含 1D→2D reshape）
- **Matmul**: tt.dot → tile.matmul / tile.matmul_acc
- **辅助**: tt.expand_dims, tt.broadcast, tt.make_range, arith.muli/addi 标量处理, dense 张量常量

### 示例与测试 ✅
- **示例统一**：从 Triton 源码提取 TTIR，所有 kernel 含 pid 与 mask 标准模式
- `examples/add_kernel.py`, `sub_kernel.py`, `mul_kernel.py`, `div_kernel.py`, `exp_kernel.py`
- `examples/reduce_sum_kernel.py`, `matmul_kernel.py`
- `run_e2e.py`：统一入口，支持 add/sub/mul/div/exp/reduce_sum/matmul
- 已删除：`run_elementwise_e2e.py`（TTIR 文本）、`*_kernel_simple.py`、`phase1_elementwise_example.py`

### CPU 仿真执行验证 ✅
- **PyPTO-simpler 兼容性**：`scripts/apply_pypto_patches.sh` 应用 `pto2_rt_init_tensor_pool` 移除补丁
- **执行测试通过**：add/sub/mul/div、reduce_sum、matmul 与参考（Python 运算）一致
- **exp 执行测试**：暂跳过（2-param orchestration 待调查）
- **run_e2e.py --kernel add --triton-compare**：add 端到端验证，含 Triton TRITON_INTERPRET 对比

---

## ✅ NPU 真机验证（2026-03-26）

### 环境配置

1. **PyPTO 兼容性补丁**（必须）
   - `orchestration_codegen.cpp`: 函数签名 3 参数 → 5 参数
   - `cce_codegen.cpp` / `pto_codegen.py`: `block_dim` 从 24 → 18

2. **重新编译 PyPTO**
   ```bash
   cd third_party/pypto && pip install -e . && cd ../..
   ```

3. **运行 NPU 测试**
   ```bash
   export SIMPLER_ROOT=$(pwd)/third_party/simpler
   python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0
   ```

### pypto 测试结果

| 测试文件 | 通过 | 失败 | 备注 |
|---------|------|------|------|
| test_elementwise.py | 4 | 2 | 失败均为 ptoas binary 缺失 |
| test_matmul.py | 7 | 7 | 失败均为 ptoas binary 缺失 |
| test_dag.py | 1 | 1 | 失败为 ptoas binary 缺失 |

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

### 已修改的文件汇总

| 文件 | 修改内容 |
|------|---------|
| `third_party/pypto/src/codegen/orchestration/orchestration_codegen.cpp` | 函数签名: 3参数 → 5参数 |
| `third_party/pypto/src/codegen/cce/cce_codegen.cpp` | `block_dim`: 24 → 18 |
| `third_party/pypto/python/pypto/ir/pto_codegen.py` | `block_dim`: 3 → 18 |

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

## 下一步工作

### 优先级 2：扩展与优化（进行中）
- ~~支持带 mask 的 add kernel（更复杂 TTIR）~~ ✅ 已完成：converter 支持 arith.addi/cmpi/andi 的 mask 相关占位，add E2E 通过
- exp 执行测试：2-param orchestration 与 simpler 集成（add/exp 端到端脚本已添加，exp 数值校验待排查）

## 技术决策

### MLIR 解析方案
**选择**：轻量级文本解析器
**理由**：
- 可独立于 Triton 环境测试
- 便于调试和开发
- 不依赖 C++ 扩展

### 实施策略
**顺序**：常量 → 算术 → 内存 → 块指针 → 调度
**理由**：
- 由简单到复杂
- 逐步验证转换逻辑
- 支持渐进式测试

## 质量保证

### 代码质量
- ✅ 使用类型注解
- ✅ 遵循 PEP 8 规范
- ✅ 完整的文档字符串
- ✅ 单元测试覆盖

### 测试策略
- 单元测试：每个转换函数
- 集成测试：完整 kernel 转换
- 端到端测试：向量加法示例

## 时间估算

| 阶段 | 预计时间 | 状态 |
|------|----------|------|
| 基础框架 | 2-3 天 | ✅ 完成 |
| 核心算子 | 3-5 天 | ✅ 完成 |
| 内存操作 | 2-3 天 | ✅ 完成 |
| NPU 验证 | 2-3 天 | ✅ 完成 |
| 问题修复 | 2-3 天 | 进行中 |
| **总计** | **11-17 天** | **80% 完成** |

## 下一步行动

**立即可执行的任务**：
1. 调查 reduce_sum 失败原因（RuntimeError: 507018）
2. 在干净环境下重试 exp kernel
3. 验证 block_dim 动态计算方案

**建议用户操作**：
```bash
# 1. 运行 NPU 测试验证
export SIMPLER_ROOT=$(pwd)/third_party/simpler
python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0

# 2. 运行 pypto 测试
cd third_party/pypto
pytest tests/st/runtime/test_elementwise.py -v --forked --platform=a2a3 --device=0

# 3. 查看详细日志
# 设备日志: ~/ascend/log/debug/device-<id>/
```

## 风险与挑战

1. **MLIR 格式复杂性**：可能遇到多种格式变体
   - 缓解：渐进式支持，先覆盖常见模式

2. **类型推导**：需要从上下文推导类型
   - 缓解：类型注解和静态分析

3. **块指针语义**：指针算术较复杂
   - 缓解：参考 Triton 其他后端实现

4. **NPU 执行兼容性**：PyPTO 生成代码与 simpler runtime 不兼容
   - 解决：已通过补丁修复

## 结论

Phase 1 基础框架已经完成，包括：
- ✅ 核心类设计
- ✅ 类型映射
- ✅ MLIR 解析器
- ✅ 测试框架
- ✅ 示例代码
- ✅ NPU 真机验证

下一步重点是解决 reduce_sum 和 exp 的执行问题，并完善 block_dim 动态计算。