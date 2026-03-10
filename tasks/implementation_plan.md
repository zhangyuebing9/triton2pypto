# Phase 1 实施计划：Elementwise 算子转换

## 当前进度

### 已完成
- ✅ TTIRToPyptoConverter 基础框架
- ✅ TypeMapper 类型映射
- ✅ 错误处理类（ConversionError, UnsupportedOpError）
- ✅ SpanTracker 源码位置追踪
- ✅ 基础测试框架

### 待实施

## 下一步：实现算子转换逻辑

### 优先级 1：核心算子（必须）

#### 1. 常量处理
- `arith.constant` → PyPTO 常量表达式
- 支持整数和浮点数常量

#### 2. 内存操作
- `tt.make_block_ptr` → BlockPtrInfo 追踪
- `tt.advance` → 更新 BlockPtrInfo
- `tt.load` → `tile.load`
- `tt.store` → `tile.store`

#### 3. 算术运算
- `arith.addf` → `tile.add`
- `arith.subf` → `tile.sub`
- `arith.mulf` → `tile.mul`
- `arith.divf` → `tile.div`

#### 4. 调度参数
- `tt.program_id` → 函数参数

### 优先级 2：扩展算子（可选）

#### 5. 数学函数
- `tt.exp` → `tile.exp`
- `arith.cmpf` → `tile.cmp`
- `arith.select` → `tile.select`

## 实施策略

### 方案 A：基于 MLIR 文本解析（推荐）
**优点**：
- 不依赖 Triton C++ 扩展
- 可以独立测试
- 更容易调试

**步骤**：
1. 创建简单的 MLIR 文本解析器
2. 解析 TTIR 操作和类型
3. 转换为 PyPTO IR

### 方案 B：使用 Triton 内部 API
**优点**：
- 直接使用 Triton 的 MLIR 解析
- 类型安全

**缺点**：
- 需要 Triton 安装
- 依赖 Triton 内部 API

## 技术决策

### 决策 1：MLIR 解析方式
**选择**：方案 A - 基于 MLIR 文本解析

**理由**：
- 可以在没有 Triton 的环境中开发
- 更容易编写单元测试
- 可以使用示例 TTIR 文本进行测试

### 决策 2：实现顺序
**顺序**：常量 → 算术运算 → 内存操作 → 调度参数

**理由**：
- 常量是最基础的，其他算子都依赖它
- 算术运算最简单，可以快速验证转换逻辑
- 内存操作需要处理 BlockPtr，相对复杂
- 调度参数需要修改函数签名

## 下一步行动

1. **创建 MLIR 文本解析器**（新文件：`mlir_parser.py`）
   - 解析 MLIR 模块结构
   - 解析操作和操作数
   - 解析类型信息

2. **实现常量转换**
   - 解析 `arith.constant` 操作
   - 创建 PyPTO 常量表达式

3. **实现算术运算转换**
   - 实现 addf, subf, mulf, divf
   - 测试基本运算

4. **创建端到端测试**
   - 向量加法 kernel
   - 简单 elementwise 操作

## 预计时间

- MLIR 解析器：1-2 天
- 常量和算术运算：1 天
- 内存操作：2-3 天
- 测试和验证：1-2 天

**总计**：5-8 天

## 风险

1. **MLIR 文本格式复杂性**：可能需要处理多种格式变体
2. **类型推导**：需要从上下文推导类型信息
3. **BlockPtr 语义**：需要正确处理指针算术

## 验证标准

- [ ] 可以解析简单的 TTIR 文本
- [ ] 可以转换常量操作
- [ ] 可以转换算术运算
- [ ] 可以转换内存操作
- [ ] 端到端测试通过（向量加法）