# NPU 执行验证问题分析

## 问题总结

在尝试验证 Triton add kernel 在 NPU 上的执行时，发现**NPU 实际上没有执行任何计算**，但测试却显示"通过"。这是一个**假阳性**结果。

## 验证证据

### 1. Orchestration 代码为空

生成的 orchestration C++ 代码 (`/tmp/pypto_run_*/orchestration/main.cpp`)：

```cpp
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count, ...) {
    (void)arg_count;
    // Extract device pointers
    void* arg_x_ptr = reinterpret_cast<void*>(args[ARG_PTR_X]);
    void* arg_y_ptr = reinterpret_cast<void*>(args[ARG_PTR_Y]);
    void* arg_out_ptr = reinterpret_cast<void*>(args[ARG_PTR_OUT]);

    // External tensors - 只创建了 Tensor 对象
    Tensor ext_x = make_tensor_external(arg_x_ptr, x_shapes, 2, DataType::INT32);
    Tensor ext_y = make_tensor_external(arg_y_ptr, y_shapes, 2, DataType::INT32);
    Tensor ext_out = make_tensor_external(arg_out_ptr, out_shapes, 2, DataType::INT32);

    // ❌ 没有 pto2_rt_submit_task 调用！
    // ❌ 没有 kernel 提交！
}
```

### 2. Profiling 确认无任务执行

```
[INFO] Profiling enabled
[INFO] poll_and_collect: Waiting for AICPU to write total_tasks in PerfDataHeader...
[ERROR] poll_and_collect: Timeout waiting for AICPU task count after 30 seconds
[INFO] poll_and_collect: AICPU finally reported task count: 0
```

### 3. Controlled Test 失败

使用非零输入的 `verify_npu_execution.py` 明确失败：

```
Input x: 2.0, Input y: 3.0, Expected: 5.0
❌ FAIL: Output tensor was NOT modified - NPU did NOT execute!
```

### 4. 假阳性原因

`run_e2e.py` 使用零输入：
- `generate_inputs()` 生成 `x=0, y=0, out=0`
- `compute_golden()` 计算 `0 + 0 = 0`
- NPU 没有执行，`out` 保持为 `0`
- 比较：`0 == 0` → "通过" ❌

## 根本原因

### PyPTO IR 转换失败

从 TTIR 到 PyPTO IR 的转换生成了 `tile.xxx` 操作：

```python
tile.full(...)
tile.load(...)
tile.add(...)
tile.store(...)
```

但 PyPTO 的 `ConvertTensorToBlockOps` pass (`third_party/pypto/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp:567`) 没有为这些 tile 操作注册 converter：

```
[ConvertTensorToBlockOps] No converter for op: tile.full
[ConvertTensorToBlockOps] No converter for op: tile.load
[ConvertTensorToBlockOps] No converter for op: tile.add
[ConvertTensorToBlockOps] No converter for op: tile.store
```

导致：
1. TensorGraph 无法转换为 BlockGraph
2. 没有 task 被创建
3. Orchestration 代码生成时没有 `pto2_rt_submit_task` 调用
4. NPU 没有执行任何计算

### 数据类型错误

生成的 orchestration 代码使用错误的数据类型：

```cpp
// ❌ 应该是 FLOAT32，但生成的是 INT32
Tensor ext_x = make_tensor_external(arg_x_ptr, x_shapes, 2, DataType::INT32);
```

## 修复方案

### 短期（避免假阳性）

修复 `golden_writer.py` 使用非零输入：

```python
# 当前（错误）
x = torch.zeros((128, 1), dtype=torch.float32)

# 修复后
x = torch.randn((128, 1), dtype=torch.float32) * 0.1  # 小随机数
y = torch.randn((128, 1), dtype=torch.float32) * 0.1
```

### 中期（修复 PyPTO converter）

在 PyPTO 的 converter registry 中注册 tile 操作：

```cpp
// third_party/pypto/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp
REGISTER_TILE_OP_CONVERTER("tile.full", TileFullConverter);
REGISTER_TILE_OP_CONVERTER("tile.load", TileLoadConverter);
REGISTER_TILE_OP_CONVERTER("tile.add", TileAddConverter);
REGISTER_TILE_OP_CONVERTER("tile.store", TileStoreConverter);
```

或者修改 pass 逻辑，对于已经是 tile 操作的表达式直接跳过转换。

### 长期（改进 triton2pypto）

1. 改进 `ttir_converter.py` 生成更符合 PyPTO 期望的 IR
2. 添加数据类型验证，确保 FLOAT32 不被转换为 INT32
3. 添加端到端验证，使用非零输入检测假阳性

## 当前状态

| 组件 | 状态 | 备注 |
|------|------|------|
| TTIR 提取 | ✅ 工作 | Triton IR 可正确提取 |
| TTIR → PyPTO IR | ⚠️ 部分工作 | 生成 tile 操作但类型有误 |
| PyPTO Pass | ❌ 失败 | ConvertTensorToBlockOps 无法处理 tile 操作 |
| C++ Codegen | ⚠️ 生成但为空 | Orchestration 无 task 提交 |
| NPU 执行 | ❌ 未执行 | 无 kernel 提交 |
| 结果验证 | ❌ 假阳性 | 零输入掩盖问题 |

## 下一步行动

1. **立即**：修改 `run_e2e.py` 使用非零输入，避免假阳性
2. **优先**：修复 PyPTO converter registry 或 pass 逻辑
3. **验证**：使用 `verify_npu_execution.py` 确认 NPU 真正执行

## 参考

- 验证脚本：`examples/verify_npu_execution.py`
- E2E 测试：`examples/run_e2e.py`
- PyPTO Pass: `third_party/pypto/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`
- TTIR Converter: `src/triton_adapter/ttir_converter.py`
