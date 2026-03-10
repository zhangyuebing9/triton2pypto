# TTIR 到 PyPTO IR 转换方案

> 版本: v0.2
> 日期: 2025-03-10
> 状态: 已确认方案

## 一、背景与目标

### 1.1 项目目标
将 Triton 的 TTIR（Triton Tensor IR）转换为 PyPTO IR，实现在华为 AI 加速器上运行 Triton 内核。

### 1.2 拦截点选择（已确认）

经过评估，采用**分阶段混合策略**：

**当前阶段（Phase 1）：方案 A - 独立转换器**
- 在 `make_ttir()` 之后拦截 `.ttir.mlir` 文件进行转换
- 作为独立工具，不侵入 Triton 源码
- 快速验证转换逻辑正确性

**后续阶段（Phase 2）：方案 B - 自定义后端**
- 基于已验证的转换逻辑，实现完整的 PyPTO 后端
- 通过 Triton entry_points 注册，提供透明用户体验
- 复用 Triton 缓存、调试等基础设施

#### 方案 A 详细说明

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton 编译流程                           │
│  Python AST → make_ttir() → make_ttgir() → make_llir()...  │
│                    ↓                                        │
│              .ttir.mlir 文件                                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            TTIRToPyptoConverter (独立工具)                   │
│  1. 读取 .ttir.mlir 文件                                     │
│  2. 使用 MLIR Python API 解析                                │
│  3. 遍历并转换为 PyPTO IR                                     │
│  4. 输出 PyPTO IR Program                                    │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   PyPTO 编译流程                              │
│  PyPTO IR → Tile Graph → Block Graph → CodeGen → Binary    │
└─────────────────────────────────────────────────────────────┘
```

**用户使用方式：**
```python
import triton
from triton2pypto import convert_ttir_to_pypto

@triton.jit
def matmul_kernel(...):
    ...

# 编译获取 TTIR
compiled = matmul_kernel.compile(...)
ttir_text = compiled.asm["ttir"]

# 转换为 PyPTO IR
pypto_program = convert_ttir_to_pypto(ttir_text)
```

**方案 A 优势：**
- 开发简单，快速原型验证
- 无侵入性，不修改 Triton 源码
- 调试方便，转换过程独立
- MLIR 文本格式稳定，版本兼容性好

---

## 二、TTIR 与 PyPTO IR 特性差异分析

| 特性维度 | Triton TTIR | PyPTO IR | 差异分析与适配策略 |
|:---|:---|:---|:---|
| **核心抽象** | **块（Block）**：逻辑上的 N 维张量切片 | **切片（Tile）**：硬件感知的 2D 数据块，必须适配片上存储（如 UB） | TTIR 较为抽象，PyPTO IR 强调硬件资源约束。需通过分析推断数据应放置的存储层级 |
| **内存表示** | **指针模型**：通过 `tt.ptr` 和地址算术（`tt.advance`）管理内存 | **MemRef 模型**：显式指定 `MemorySpace`（DDR/UB/L0）和起始地址表达式 | 需从 TTIR 的指针算术中推导 PyPTO 的 `MemRef` 结构 |
| **类型系统** | **静态形状张量**：shape 必须在编译时确定 | **分层类型**：区分 `TensorType`（全局存储）和 `TileType`（片上存储） | 需通过分析判定变量应归属于 DDR 还是 UB |
| **指令映射** | **通用计算**：如 `tt.dot` 映射到逻辑矩阵乘法 | **硬件单元映射**：明确划分为 `Cube`（矩阵）和 `Vector`（矢量）操作 | 根据算子语义分流至 `cube.matmul` 或 `vector.add` |
| **实现基础** | **MLIR Dialect**：基于 SSA 形式的 MLIR 框架 | **Python IRBuilder**：提供 context manager 风格的构建 API | 实现 MLIR Visitor 遍历 TTIR，调用 PyPTO IRBuilder 构建节点 |

---

## 三、转换架构设计

### 3.1 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                     Triton 编译流水线                        │
│  ... → make_ttir() → make_ttgir() → make_llir() → ...      │
│              ↓ 拦截点                                        │
│         .ttir.mlir 文件                                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   TTIRToPyptoConverter                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ MLIR Visitor│→ │ Type Mapper │→ │ IRBuilder   │         │
│  │ (遍历 TTIR) │  │ (类型映射)  │  │ (构建 IR)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         ↓                ↓                ↓                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ ValueMap    │  │ MemRefInf   │  │ SpanTracker │         │
│  │ (值映射表)  │  │ (内存推导)  │  │ (源码追踪)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                     PyPTO IR Program                         │
│  Program { Function { Stmt* } }                            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 类设计

```python
# src/triton_adapter/ttir_converter.py

class TTIRToPyptoConverter:
    """TTIR 到 PyPTO IR 转换器"""
    
    def __init__(self):
        self.ib = IRBuilder()                           # PyPTO IRBuilder
        self.value_map: dict[Value, ir.Var] = {}        # MLIR Value → PyPTO Var
        self.block_ptr_map: dict[Value, BlockPtrInfo] = {}  # 块指针信息
        self.span = ir.Span.unknown()                   # 默认 Span
        
    def convert(self, ttir_module: Module) -> ir.Program:
        """入口：转换 MLIR Module 到 PyPTO Program"""
        ...
    
    def convert_function(self, func: FuncOp) -> ir.Function:
        """转换 tt.func"""
        ...
    
    def convert_operation(self, op: Operation) -> None:
        """分发操作转换"""
        ...


class TypeMapper:
    """TTIR 类型到 PyPTO 类型映射"""
    
    def map_tensor_type(self, ttir_type: RankedTensorType) -> ir.TensorType:
        """tt.tensor → ir.TensorType (DDR)"""
        ...
    
    def map_pointer_type(self, ttir_type: PointerType) -> ir.TensorType:
        """tt.ptr → ir.TensorType with DDR MemRef"""
        ...
    
    def map_block_type(self, ttir_type: BlockType) -> ir.TileType:
        """tt.block → ir.TileType (UB)"""
        ...


class MemRefInference:
    """内存引用推导"""
    
    def infer_memref(self, op: Operation, context: ConversionContext) -> ir.MemRef:
        """从操作上下文推导 MemRef"""
        ...
    
    def analyze_memory_space(self, value: Value) -> ir.MemorySpace:
        """分析值应放置的存储空间"""
        ...
```

---

## 四、详细转换规则

### 4.1 模块与函数映射

| TTIR 元素 | PyPTO IR | 示例代码 |
|:---|:---|:---|
| `builtin.module` | `ir.Program` | `with ib.program("name") as p:` |
| `tt.func` | `ir.Function` | `with ib.function("kernel") as f:` |
| 函数参数 `tt.ptr` | `ir.TensorType` + DDR MemRef | `f.param("x", ir.TensorType([M, K], dt, memref=ddr_memref))` |
| 函数返回值 | `f.return_type()` | `f.return_type(ir.TensorType(...))` |

```python
def convert_function(self, func: FuncOp) -> ir.Function:
    func_name = func.attributes["sym_name"].value
    
    with self.ib.function(func_name) as f:
        # 转换参数
        for i, arg in enumerate(func.arguments):
            pypto_type = self.type_mapper.map_type(arg.type)
            param = f.param(f"arg{i}", pypto_type)
            self.value_map[arg] = param
        
        # 设置返回类型
        return_types = self.extract_return_types(func)
        for rt in return_types:
            f.return_type(rt)
        
        # 转换函数体
        for block in func.body.blocks:
            for op in block.operations:
                self.convert_operation(op)
    
    return f.get_result()
```

### 4.2 内存与指针适配

#### 4.2.1 块指针处理

TTIR 使用 `tt.make_block_ptr` 和 `tt.advance` 管理块迭代：

```mlir
%ptr = tt.make_block_ptr %base, %shape, %strides, %offsets, %tensor_shape, %order
%ptr_next = tt.advance %ptr, %offsets_delta
%tile = tt.load %ptr, %mask, %other
```

转换策略：

```python
@dataclass
class BlockPtrInfo:
    """块指针信息追踪"""
    base: ir.Var              # 基地址
    shape: list[ir.Expr]      # 形状
    strides: list[ir.Expr]    # 步长
    current_offset: list[ir.Expr]  # 当前偏移


def convert_make_block_ptr(self, op: Operation) -> None:
    """转换 tt.make_block_ptr"""
    base = self.value_map[op.operands[0]]
    shape = [self.convert_expr(s) for s in op.operands[1]]
    strides = [self.convert_expr(s) for s in op.operands[2]]
    offsets = [self.convert_expr(s) for s in op.operands[3]]
    
    info = BlockPtrInfo(base=base, shape=shape, strides=strides, current_offset=offsets)
    self.block_ptr_map[op.result] = info


def convert_advance(self, op: Operation) -> None:
    """转换 tt.advance"""
    ptr_info = self.block_ptr_map[op.operands[0]]
    delta = [self.convert_expr(d) for d in op.operands[1]]
    
    # 更新偏移
    new_offsets = [
        ir.Add(ptr_info.current_offset[i], delta[i], DataType.INT64, self.span)
        for i in range(len(delta))
    ]
    
    new_info = BlockPtrInfo(
        base=ptr_info.base,
        shape=ptr_info.shape,
        strides=ptr_info.strides,
        current_offset=new_offsets
    )
    self.block_ptr_map[op.result] = new_info
```

#### 4.2.2 MemRef 构造

```python
def create_memref_for_load(self, ptr_info: BlockPtrInfo, tile_shape: list[int]) -> ir.MemRef:
    """为 load 操作创建 MemRef"""
    # 计算线性地址
    addr = ptr_info.base
    for i, (offset, stride) in enumerate(zip(ptr_info.current_offset, ptr_info.strides)):
        offset_expr = self.convert_expr(offset)
        stride_expr = self.convert_expr(stride)
        addr = ir.Add(addr, ir.Mul(offset_expr, stride_expr, DataType.INT64, self.span), 
                      DataType.INT64, self.span)
    
    # 创建 DDR MemRef
    size = math.prod(tile_shape) * dtype_bytes
    return self.ib.memref(ir.MemorySpace.DDR, addr, size, id=self._next_memref_id())
```

### 4.3 计算算子映射

| TTIR 算子 | PyPTO 操作层级 | PyPTO 指令 | 转换要点 |
|:---|:---|:---|:---|
| `tt.load` | Tile Memory | `ir.op.tensor.slice` | DDR → UB，创建 TileType + UB MemRef |
| `tt.store` | Tile Memory | `ir.op.tensor.assemble` | UB → DDR |
| `tt.dot` | Block (Cube Core) | `ir.op.tensor.matmul` | 矩阵乘法，支持转置参数 |
| `arith.addf/subf/mulf` | Block (Vector Core) | `ir.op.tensor.add/sub/mul` | 逐元素操作 |
| `tt.reduce` | Tile Compute | `ir.op.tensor.reduce` | 归约操作 |
| `tt.exp` | Block (Vector) | `ir.op.tensor.exp` | 逐元素数学函数 |

```python
def convert_load(self, op: Operation) -> None:
    """转换 tt.load → tensor.slice"""
    ptr = op.operands[0]
    mask = op.operands[1] if len(op.operands) > 1 else None
    other = op.operands[2] if len(op.operands) > 2 else None
    
    ptr_info = self.block_ptr_map.get(ptr)
    if ptr_info is None:
        raise ConversionError(f"No block pointer info for {ptr}")
    
    # 提取 tile 形状
    result_type = op.result.type
    tile_shape = self.extract_shape(result_type)
    dtype = self.map_dtype(result_type.element_type)
    
    # 创建 MemRef (DDR)
    memref = self.create_memref_for_load(ptr_info, tile_shape)
    
    # 创建 TileType (UB)
    tile_type = self.ib.tile_type(
        shape=tile_shape,
        dtype=dtype,
        memref=self.ib.memref(ir.MemorySpace.UB, 0, self.calc_tile_size(tile_shape, dtype))
    )
    
    # 处理 mask（如需要）
    if mask is not None:
        # 转换为 if 语句保护
        self.convert_masked_load(op, ptr_info, tile_type, mask, other)
    else:
        # 直接 load
        result = self.ib.let(
            op.result.name or f"load_{self._tmp_id()}",
            ir.op.tensor.slice(ptr_info.base, tile_shape, ptr_info.current_offset)
        )
        self.value_map[op.result] = result


def convert_dot(self, op: Operation) -> None:
    """转换 tt.dot → tensor.matmul"""
    a = self.value_map[op.operands[0]]
    b = self.value_map[op.operands[1]]
    c = self.value_map[op.operands[2]] if len(op.operands) > 2 else None
    
    # 提取属性
    allow_tf32 = op.attributes.get("allow_tf32", True)
    max_num_imprecise_acc = op.attributes.get("max_num_imprecise_acc", 0)
    
    # 确定输出类型
    result_type = op.result.type
    out_dtype = self.map_dtype(result_type.element_type)
    
    # 创建 matmul 操作
    matmul_result = self.ib.let(
        op.result.name or f"dot_{self._tmp_id()}",
        ir.op.tensor.matmul(a, b, out_dtype=out_dtype)
    )
    
    # 如果有累加器 c，生成 add 操作
    if c is not None:
        result = self.ib.let(
            f"dot_acc_{self._tmp_id()}",
            ir.op.tensor.add(matmul_result, c)
        )
    else:
        result = matmul_result
    
    self.value_map[op.result] = result
```

### 4.4 控制流与掩码处理

#### 4.4.1 循环转换

```mlir
%result = scf.for %i = 0 to %N step 1 iter_args(%sum = %init) -> (tensor<...>) {
  %next = ... // 使用 %sum
  scf.yield %next
}
```

转换为：

```python
def convert_for(self, op: Operation) -> None:
    """转换 scf.for → for_loop"""
    loop_var = self.ib.var("i", ir.ScalarType(DataType.INT64))
    start = self.convert_expr(op.operands[0])
    stop = self.convert_expr(op.operands[1])
    step = self.convert_expr(op.operands[2])
    
    with self.ib.for_loop(loop_var, start, stop, step) as loop:
        # 处理 iter_args
        iter_args = op.attributes.get("iter_args", [])
        for i, (arg, init_val) in enumerate(iter_args):
            init_expr = self.value_map[init_val]
            iter_var = loop.iter_arg(arg.name, init_expr)
            self.value_map[arg] = iter_var
        
        # 声明 return_vars
        for i in range(len(iter_args)):
            loop.return_var(f"result_{i}")
        
        # 转换循环体
        for body_op in op.region.blocks[0].operations:
            if body_op.name == "scf.yield":
                # 收集 yield 值
                yield_values = [self.value_map[v] for v in body_op.operands]
                self.ib.emit(ir.YieldStmt(yield_values, self.span))
            else:
                self.convert_operation(body_op)
    
    # 获取循环结果
    results = loop.outputs()
    for i, result in enumerate(results):
        self.value_map[op.results[i]] = result
```

#### 4.4.2 条件转换

```python
def convert_if(self, op: Operation) -> None:
    """转换 scf.if → if_stmt"""
    condition = self.convert_expr(op.operands[0])
    
    with self.ib.if_stmt(condition) as if_builder:
        # 声明 return_vars（如果有返回值）
        if op.results:
            for i, result in enumerate(op.results):
                result_type = self.type_mapper.map_type(result.type)
                if_builder.return_var(f"if_result_{i}", result_type)
        
        # then 分支
        then_block = op.regions[0].blocks[0]
        for body_op in then_block.operations:
            if body_op.name == "scf.yield":
                yield_values = [self.value_map[v] for v in body_op.operands]
                self.ib.emit(ir.YieldStmt(yield_values, self.span))
            else:
                self.convert_operation(body_op)
        
        # else 分支（如果存在）
        if len(op.regions) > 1:
            if_builder.else_()
            else_block = op.regions[1].blocks[0]
            for body_op in else_block.operations:
                if body_op.name == "scf.yield":
                    yield_values = [self.value_map[v] for v in body_op.operands]
                    self.ib.emit(ir.YieldStmt(yield_values, self.span))
                else:
                    self.convert_operation(body_op)
    
    # 获取结果
    results = if_builder.outputs()
    for i, result in enumerate(results):
        self.value_map[op.results[i]] = result
```

#### 4.4.3 掩码处理

参考 Triton Ascend 的 `DiscreteMaskConversion`：

```python
def convert_masked_load(self, op: Operation, ptr_info: BlockPtrInfo, 
                        tile_type: ir.TileType, mask: Value, other: Value) -> None:
    """转换带掩码的 load 操作"""
    mask_var = self.value_map[mask]
    
    # 创建条件表达式
    # mask 通常是一个 Block<Value>，需要逐元素检查
    
    with self.ib.if_stmt(mask_var) as if_builder:
        if_builder.return_var("masked_result", tile_type)
        
        # then: 执行 load
        load_result = self.ib.let(
            "masked_load",
            ir.op.tensor.slice(ptr_info.base, tile_type.shape, ptr_info.current_offset)
        )
        self.ib.emit(ir.YieldStmt([load_result], self.span))
        
        # else: 使用 other 值
        if_builder.else_()
        other_val = self.value_map.get(other) or self.default_value(tile_type.dtype)
        self.ib.emit(ir.YieldStmt([other_val], self.span))
    
    result = if_builder.output()
    self.value_map[op.result] = result
```

### 4.5 执行调度适配

TTIR 的 `tt.program_id` 用于获取网格坐标：

```mlir
%pid = tt.program_id %axis  // axis: 0, 1, or 2
```

转换策略：

```python
def convert_program_id(self, op: Operation) -> None:
    """转换 tt.program_id"""
    axis = op.attributes["axis"].value
    
    # 在 PyPTO 中，program_id 映射到 MPMD 调度的核心坐标
    # 生成一个 Scalar 变量表示当前核心的坐标
    result = self.ib.var(
        f"program_id_{axis}",
        ir.ScalarType(DataType.INT64)
    )
    
    # 注意：实际的核心坐标分配在 PyPTO CodeGen 阶段处理
    # 这里只创建变量占位符
    self.value_map[op.result] = result
```

---

## 五、类型映射表

### 5.1 数据类型映射

| Triton dtype | PyPTO DataType |
|:---|:---|
| `i1` | `DataType.BOOL` |
| `i8` | `DataType.INT8` |
| `i16` | `DataType.INT16` |
| `i32` | `DataType.INT32` |
| `i64` | `DataType.INT64` |
| `fp16` | `DataType.FP16` |
| `bf16` | `DataType.BF16` |
| `fp32` | `DataType.FP32` |
| `fp64` | `DataType.FP64` |

### 5.2 存储空间映射

| 数据位置 | PyPTO MemorySpace | 说明 |
|:---|:---|:---|
| 全局内存（函数参数） | `MemorySpace.DDR` | 输入/输出张量 |
| 加载后的 Tile | `MemorySpace.UB` | 统一缓冲区 |
| 矩阵计算左矩阵 | `MemorySpace.Left` | Cube 单元左矩阵缓冲 |
| 矩阵计算右矩阵 | `MemorySpace.Right` | Cube 单元右矩阵缓冲 |
| 矩阵计算累加器 | `MemorySpace.Acc` | Cube 单元累加器缓冲 |
| 向量计算 | `MemorySpace.Vec` | Vector 单元缓冲 |

---

## 六、算子支持分阶段规划

### 6.1 Phase 1: Elementwise（最小集）

| 类别 | TTIR Operation | PyPTO 对应 | 说明 |
|:---|:---|:---|:---|
| **内存操作** | `tt.make_block_ptr` | 内部处理 | 创建块指针 |
| | `tt.advance` | 内部处理 | 前进指针 |
| | `tt.load` | `tile.load` | 加载 Tile |
| | `tt.store` | `tile.store` | 存储 Tile |
| **计算** | `arith.addf` | `tile.add` | 加法 |
| | `arith.subf` | `tile.sub` | 减法 |
| | `arith.mulf` | `tile.mul` | 乘法 |
| | `arith.divf` | `tile.div` | 除法 |
| | `tt.exp` | `tile.exp` | 指数函数 |
| | `arith.cmpf` | `tile.cmp` | 比较 |
| | `arith.select` | `tile.select` | 条件选择 |
| **常量** | `arith.constant` | `ConstFloat/ConstInt` | 常量值 |
| **调度** | `tt.program_id` | 函数参数 `pid_0/1/2` | 获取网格坐标 |

**目标示例**：向量加法、逐元素运算、简单激活函数

**预计周期**：2-3 周

---

### 6.2 Phase 2: Reduce

| 类别 | TTIR Operation | PyPTO 对应 | 说明 |
|:---|:---|:---|:---|
| **归约** | `tt.reduce` | `tile.reduce` | 通用归约 |
| | (axis, combine) | `tile.row_max/row_sum/...` | 行归约 |

**目标示例**：softmax、layer norm、reduce_sum/max

**预计周期**：1-2 周

---

### 6.3 Phase 3: Matmul

| 类别 | TTIR Operation | PyPTO 对应 | 说明 |
|:---|:---|:---|:---|
| **矩阵乘** | `tt.dot` | `tile.matmul` | 矩阵乘法 |

**目标示例**：简单矩阵乘法 GEMM

**预计周期**：1 周

---

### 6.4 Phase 4: Cube + Vector 混合操作

**场景**：同时包含 Cube 操作（matmul）和 Vector 操作（elementwise/reduce）

| 组合模式 | 示例 | 验证目标 |
|:---|:---|:---|
| dot + elementwise | matmul + bias + activation | Cube 与 Vector 协同 |
| dot + reduce | matmul + row_sum | 触发 Group(AIC+AIV) 分离 |

**目标示例**：带 bias 和激活的 matmul、matmul + softmax

**预计周期**：1-2 周

---

### 6.5 Phase 5: 带控制流

| 类别 | TTIR Operation | PyPTO 对应 | 说明 |
|:---|:---|:---|:---|
| **循环** | `scf.for` | `ForStmt` | 循环 |
| | `scf.yield` | `YieldStmt` | 循环返回值 |
| **条件** | `scf.if` | `IfStmt` | 条件分支 |
| **掩码** | `tt.load` (with mask) | `IfStmt` + load | 带掩码加载 |

**目标示例**：Flash Attention、带循环的 kernel

**预计周期**：2 周

---

### 6.6 算子清单汇总

| Phase | 新增 Operation | 累计 | 核心能力 |
|:---|:---|:---|:---|
| **Phase 1** | 10 个 | 10 | Elementwise 计算 |
| **Phase 2** | +2 个 | 12 | 归约操作 |
| **Phase 3** | +1 个 | 13 | 矩阵乘法 |
| **Phase 4** | 已覆盖 | 13 | Cube+Vector 协同验证 |
| **Phase 5** | +4 个 | 17 | 控制流与掩码 |

---

## 七、实现路径

### 7.1 基础框架搭建（第 1 周）

- [ ] 创建 `TTIRToPyptoConverter` 类骨架
- [ ] 实现 `TypeMapper` 类型映射
- [ ] 实现 `ValueMap` 值追踪
- [ ] 实现 `SpanTracker` 源码位置追踪
- [ ] 编写单元测试框架

### 7.2 Phase 1: Elementwise（第 2-4 周）

- [ ] `tt.make_block_ptr` / `tt.advance` 指针处理
- [ ] `tt.load` / `tt.store` 转换
- [ ] `arith.addf/subf/mulf/divf` 转换
- [ ] `arith.constant` 常量处理
- [ ] `tt.exp` / `arith.cmpf` / `arith.select` 转换
- [ ] `tt.program_id` 参数映射
- [ ] Elementwise kernel 端到端验证

### 7.3 Phase 2: Reduce（第 5-6 周）

- [ ] `tt.reduce` 转换
- [ ] softmax kernel 端到端验证

### 7.4 Phase 3: Matmul（第 7 周）

- [ ] `tt.dot` 转换
- [ ] GEMM kernel 端到端验证

### 7.5 Phase 4: Cube + Vector（第 8-9 周）

- [ ] dot + elementwise 组合验证
- [ ] dot + reduce 组合验证
- [ ] Group(AIC+AIV) 分离验证

### 7.6 Phase 5: 控制流（第 10-11 周）

- [ ] `scf.for` / `scf.yield` 转换
- [ ] `scf.if` 转换
- [ ] mask 处理逻辑
- [ ] Flash Attention 端到端验证

---

## 七、风险与挑战

### 7.1 技术风险

| 风险项 | 影响 | 缓解措施 |
|:---|:---|:---|
| Triton MLIR API 不稳定 | 中 | 锁定 Triton 版本，关注上游变更 |
| 指针分析复杂度高 | 高 | 先支持常见模式，逐步扩展 |
| MemRef 推导不完整 | 中 | 结合静态分析和运行时信息 |
| mask 语义差异 | 中 | 参考已实现的后端（AMDGPU、Ascend） |

### 7.2 待讨论问题

1. **拦截点选择**：~~是在 `make_ttir()` 之后立即转换，还是在 Triton 编译流程中注册为自定义 Pass？~~ **已确认：采用方案 A**

2. **动态形状处理**：~~TTIR 支持动态形状，PyPTO IR 也支持符号表达式，但具体的符号映射策略需要进一步明确。~~ **已确认：对于 TTIR 中的每个动态轴(?)，新建一个符号变量来承接**

3. **分布式并行（Grid vs MPMD）**：~~Triton 的 grid 语义与 PyPTO 的 MPMD 调度如何精确对应？~~ **已确认，详见下方**

4. **错误处理**：~~转换失败时如何提供有意义的错误信息，帮助用户定位问题？~~ **已确认：详见下方**

---

### 7.4 错误处理方案（已确认）

#### 设计原则

1. **详细错误信息**：提供尽量详细的错误信息供调试
2. **源码定位**：错误信息需指向原始 Python 代码位置
3. **严格模式**：不支持的算子直接报错（不跳过/不警告）

#### 错误信息格式

```
ConversionError: [文件名:行号:列号] 错误描述

Context:
  - TTIR Operation: tt.unsupported_op
  - Location: kernel.py:15:5
  - Function: matmul_kernel

Suggestion: 错误修复建议（如有）
```

#### 实现方案

```python
class ConversionError(Exception):
    """TTIR 转换错误"""
    
    def __init__(self, message: str, op: Operation = None, span: ir.Span = None):
        self.op = op
        self.span = span
        
        # 构建详细错误信息
        if span and span != ir.Span.unknown():
            location = f"{span.filename}:{span.line}:{span.column}"
        else:
            location = "<unknown>"
        
        full_message = f"[{location}] {message}"
        if op:
            full_message += f"\n  Operation: {op.name}"
        
        super().__init__(full_message)


class UnsupportedOpError(ConversionError):
    """不支持的算子错误"""
    
    def __init__(self, op: Operation):
        message = f"Unsupported operation: {op.name}"
        suggestion = f"Operation '{op.name}' is not yet supported. " \
                     f"Please check the supported operations list."
        super().__init__(message, op=op)
        self.suggestion = suggestion
```

#### 源码位置追踪

TTIR 中的 `location` 属性包含源码位置信息：

```mlir
%0 = tt.load %ptr {location = "kernel.py:15:5"} : tensor<16x16xf32>
```

转换器需要：
1. 解析 TTIR 操作的 `location` 属性
2. 转换为 PyPTO IR 的 `Span` 对象
3. 错误时输出完整的位置信息

```python
class SpanTracker:
    """追踪源码位置"""
    
    def get_span(self, op: Operation) -> ir.Span:
        """从 TTIR 操作提取源码位置"""
        if "location" in op.attributes:
            loc = op.attributes["location"]
            # 解析 "file.py:line:col" 格式
            parts = loc.split(":")
            if len(parts) >= 3:
                return ir.Span(parts[0], int(parts[1]), int(parts[2]))
        return ir.Span.unknown()
```

#### 错误处理流程

```python
def convert_operation(self, op: Operation) -> None:
    """转换操作，包含错误处理"""
    span = self.span_tracker.get_span(op)
    
    op_name = op.name
    
    # 检查是否支持
    if op_name not in self.SUPPORTED_OPS:
        raise UnsupportedOpError(op)
    
    try:
        # 分发到具体转换函数
        handler = self.SUPPORTED_OPS[op_name]
        handler(op, span)
    except ConversionError:
        raise  # 直接传递转换错误
    except Exception as e:
        # 包装其他异常
        raise ConversionError(
            f"Internal error during conversion: {e}",
            op=op,
            span=span
        ) from e
```

---

### 7.5 MLIR 解析方式（已确认）

**采用选项 A：Triton 内部 API**

使用 Triton 内置的 MLIR 解析能力（`triton._C.libtriton.ir`），理由：
- 已随 Triton 安装，无需额外依赖
- 版本与 Triton 完全一致
- 可直接复用 Triton 的 MLIR 解析逻辑

```python
from triton._C.libtriton import ir

# 解析 TTIR 文本
context = ir.context()
ir.load_dialects(context)
module = ir.parse_mlir_module(ttir_text, context)

# 遍历模块
for func in module.body.operations:
    func_name = func.attributes["sym_name"].value
    ...
```

---

### 7.6 存储空间分配策略（已确认）

**复用 PyPTO 的 `InitMemRef` Pass 自动推导 Memory Space**

PyPTO 具备自动推导 memory 层次位置的能力，转换器无需显式指定。

#### 自动推导规则（由 PyPTO InitMemRef Pass 处理）

| 变量来源 | 自动分配的 Memory Space |
|:---|:---|
| 函数参数 | DDR |
| `tile.load` / `tile.move` | Vec（或从 `target_memory` kwarg 提取） |
| `tile.store` | DDR |
| `tile.matmul` / `tile.matmul_acc` | Acc |
| 其他 tile 操作 | Vec |
| 其他变量 | DDR（默认）|

#### 转换器职责

转换器只需生成操作，无需关心 MemRef 和 MemorySpace：

```python
# 转换器输出（无需指定 memory space）
tile_a = pl.load(a, [0, 0], [64, 64])
tile_c = pl.matmul(tile_a, tile_b)
output = pl.store(tile_c, [0, 0], output)

# PyPTO 编译流水线自动执行 InitMemRef Pass
# 为每个变量分配合适的 MemorySpace
```

---

### 7.7 测试验证策略（已确认）

#### 环境配置

| 环境 | 能力 |
|:---|:---|
| **CPU 环境** | Triton Interpreter + PyPTO simpler runtime（仿真对比） |
| **NPU 环境** | PyPTO 真实执行 |

#### 验证流程

```
┌─────────────────────────────────────────────────────────────┐
│                    CPU 环境（本地开发）                       │
│                                                             │
│  1. Triton kernel → TTIR → PyPTO IR                         │
│  2. TRITON_INTERPRET=1 执行 Triton kernel → 结果 A          │
│  3. PyPTO simpler runtime 执行 PyPTO IR → 结果 B            │
│  4. 对比 A 和 B（数值误差在容许范围内）                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓ 验证通过
┌─────────────────────────────────────────────────────────────┐
│                    NPU 环境（真机验证）                       │
│                                                             │
│  1. PyPTO IR → CodeGen → Binary                             │
│  2. NPU 执行 → 结果 C                                        │
│  3. 对比 C 与 A/B                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Triton Interpreter 模式

通过环境变量启用 CPU 仿真：

```bash
# CPU 仿真模式执行 Triton kernel
TRITON_INTERPRET=1 python test_kernel.py
```

原理：使用 NumPy 模拟 Triton kernel 执行，完全在 CPU 上运行。

#### 验证容差

| 数据类型 | 容许误差 |
|:---|:---|
| FP32 | 1e-5 (relative) |
| FP16 | 1e-3 (relative) |
| BF16 | 1e-2 (relative) |
| INT | 精确匹配 |

---

### 7.8 输出格式（已确认）

**支持多种输出格式**：

| 格式 | 用途 | 说明 |
|:---|:---|:---|
| **Python 对象** | 直接传递给 PyPTO 编译流程 | 默认输出，无需序列化 |
| **序列化文件** | 持久化存储 | `.pypto` 二进制格式 |
| **文本格式** | 调试查看 | 人类可读的 IR 表示 |

```python
# 转换器 API 设计
def convert_ttir_to_pypto(
    ttir_text: str,
    output_format: str = "object"  # "object" | "file" | "text"
) -> ir.Program | str | bytes:
    ...

# 使用示例
# 1. Python 对象（默认）
program = convert_ttir_to_pypto(ttir_text)

# 2. 文本格式（调试）
ir_text = convert_ttir_to_pypto(ttir_text, output_format="text")
print(ir_text)

# 3. 序列化文件（持久化）
serialized = convert_ttir_to_pypto(ttir_text, output_format="file")
with open("kernel.pypto", "wb") as f:
    f.write(serialized)
```

---

### 7.9 编译流程入口（已确认）

**使用完整编译流程入口**：`ir.compile(program)`

```python
from pypto import ir
from pypto.backend import BackendType

# 转换器生成 Program
program = convert_ttir_to_pypto(ttir_text)

# 完整编译流程
output_dir = ir.compile(
    program,
    strategy=ir.OptimizationStrategy.PTOAS,
    backend_type=BackendType.PTO
)
```

**目标硬件**：Ascend 910B

**编译流程**：

```
Program → PassManager.run_passes() → InitMemRef → ... → CodeGen → Binary
```

所有 Pass 自动执行，包括：
- InitMemRef（Memory Space 自动分配）
- MemoryReuse
- AllocateMemoryAddr
- 等

---

### 7.3 Grid 与 MPMD 映射方案（已确认）

#### 核心映射关系

| Triton 概念 | PyPTO 概念 | 说明 |
|:---|:---|:---|
| `@triton.jit` kernel | `Program` 包含多个 `Function` | 一个 Triton kernel 映射为一个 PyPTO Program |
| 单个 kernel 实例 | `InCore` 或 `Group(AIC+AIV)` 函数 | 根据操作类型自动推断 |
| `grid = (X, Y, Z)` | 运行时 Cluster 调度 | Grid 维度是运行时配置，不属于 IR 层面 |
| `tt.program_id(axis)` | 函数参数 `pid_0/1/2` | 作为额外参数传入 |

#### 函数类型自动推断规则

| TTIR 操作特征 | PyPTO 函数类型 | 说明 |
|:---|:---|:---|
| 包含 `tt.dot` / 矩阵乘法 | `Group(AIC + AIV)` | Cube 负责矩阵计算，Vector 负责其他 |
| 纯 Vector 操作（无 matmul） | `AIV` | 单独的 Vector 核心内核 |
| 复杂控制流 + 需要调度 | `Orchestration` + `InCore` | 主控函数 + AICore 子函数 |

#### program_id 参数处理

```python
# Triton 源码
@triton.jit
def kernel(x, y, out):
    pid_x = tl.program_id(0)  # grid 维度 0
    pid_y = tl.program_id(1)  # grid 维度 1
    ...

# 转换后的 PyPTO 函数签名
def kernel(x: Tensor, y: Tensor, out: Tensor, 
           pid_0: Scalar, pid_1: Scalar) -> Tensor:
    ...
```

#### Orchestration 函数自动生成

转换器输出包含两个函数：

1. **InCore/Group 函数**：实际计算逻辑
2. **Orchestration 函数**：调度入口，负责：
   - 创建输出张量
   - 调用 InCore 函数
   - 返回结果

```python
# 转换器输出示例
@pl.program
class MatmulKernel:
    # 1. InCore 函数（实际计算）
    @pl.function(type=pl.FunctionType.InCore)
    def matmul_incore(self, 
                      a: pl.Tensor[[M, K], pl.FP16],
                      b: pl.Tensor[[K, N], pl.FP16],
                      c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                      pid_0: pl.Scalar,
                      pid_1: pl.Scalar) -> None:
        # 实际的 load/matmul/store 逻辑
        ...
    
    # 2. Orchestration 函数（调度入口）
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self,
             a: pl.Tensor[[M, K], pl.FP16],
             b: pl.Tensor[[K, N], pl.FP16]) -> pl.Tensor[[M, N], pl.FP32]:
        c = pl.create_tensor([M, N], dtype=pl.FP32)
        self.matmul_incore(a, b, c, pid_0=0, pid_1=0)  # 运行时注入真实值
        return c
```

**Orchestration 函数职责**：

| 职责 | 说明 |
|:---|:---|
| 创建输出张量 | `pl.create_tensor()` |
| 调用 InCore 函数 | 传递输入参数和输出缓冲区 |
| 参数注入 | `pid_0/1/2` 由运行时系统注入实际值 |
| 返回结果 | 返回输出张量 |

---

## 八、参考资源

- Triton 源码：`third_party/triton/`
- PyPTO 源码：`third_party/pypto/`
- Triton MLIR Dialect：`third_party/triton/include/triton/Dialect/`
- PyPTO IRBuilder：`third_party/pypto/python/pypto/ir/builder.py`
- Flash Attention 示例：`third_party/pypto/examples/ir_builder/flash_attention_builder.py`

---

## 九、修订记录

| 版本 | 日期 | 修改内容 | 作者 |
|:---|:---|:---|:---|
| v0.1 | 2025-03-10 | 初稿 | - |
| v0.2 | 2025-03-10 | 确认拦截点选择（方案 A） | - |
| v1.0 | 2025-03-10 | 完成所有问题讨论确认，包括：<br>• 动态形状处理：符号变量承接<br>• Grid vs MPMD 映射方案<br>• 错误处理：详细错误信息+源码定位<br>• MLIR 解析：Triton 内部 API<br>• 存储空间：复用 PyPTO InitMemRef<br>• 算子支持：五阶段规划<br>• 测试验证：CPU 仿真 + NPU 真机<br>• 输出格式：多格式支持<br>• 编译入口：ir.compile()，目标 910B | - |