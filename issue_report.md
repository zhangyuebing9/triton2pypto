# Progress Report: NPU Execution Verification Issue

**Date**: 2026-03-26  
**Author**: CANNBot  
**Status**: 🚨 Critical Issue Identified  

---

## Executive Summary

During NPU execution verification of the Triton add kernel, we discovered a **critical false positive** in our test results. The tests reported "PASS" but **NPU was not actually executing any computation**.

---

## Critical Finding: False Positive Test Results

### The Problem

Our E2E test (`examples/run_e2e.py`) reported successful NPU execution, but this was a **false positive** caused by:

1. **Zero-initialized inputs**: The test used `x=0, y=0` as inputs
2. **Empty orchestration**: No kernel was submitted to NPU
3. **Coincidental match**: Golden computation `0 + 0 = 0` matched unmodified output `0`

### Evidence

#### 1. Empty Orchestration Code

Generated orchestration (`/tmp/pypto_run_*/orchestration/main.cpp`):

```cpp
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count, ...) {
    // Extract pointers
    void* arg_x_ptr = reinterpret_cast<void*>(args[ARG_PTR_X]);
    void* arg_y_ptr = reinterpret_cast<void*>(args[ARG_PTR_Y]);
    void* arg_out_ptr = reinterpret_cast<void*>(args[ARG_PTR_OUT]);

    // Create Tensor objects
    Tensor ext_x = make_tensor_external(arg_x_ptr, x_shapes, 2, DataType::INT32);  // ❌ Wrong dtype!
    Tensor ext_y = make_tensor_external(arg_y_ptr, y_shapes, 2, DataType::INT32);
    Tensor ext_out = make_tensor_external(arg_out_ptr, out_shapes, 2, DataType::INT32);

    // ❌ NO pto2_rt_submit_task calls!
    // ❌ NO kernel submission!
}
```

#### 2. Profiling Confirms Zero Tasks

```
[INFO] Profiling enabled
[INFO] poll_and_collect: Waiting for AICPU to write total_tasks in PerfDataHeader...
[ERROR] poll_and_collect: Timeout waiting for AICPU task count after 30 seconds
[INFO] poll_and_collect: AICPU finally reported task count: 0
```

#### 3. Controlled Test with Non-Zero Inputs FAILS

```bash
$ python examples/verify_npu_execution.py
Input x: 2.0, Input y: 3.0, Expected: 5.0
❌ FAIL: Output tensor was NOT modified - NPU did NOT execute!
```

#### 4. Test Comparison

| Test | Input Values | Result | Validity |
|------|-------------|--------|----------|
| `verify_npu_execution.py` | x=2.0, y=3.0 | ❌ FAIL | ✅ **Real** |
| `run_e2e.py` | x=0.0, y=0.0 | ✓ PASS | ❌ **False Positive** |

---

## Root Cause Analysis

### Primary Cause: PyPTO Pass Failure

The `ConvertTensorToBlockOps` pass in PyPTO cannot handle `tile.xxx` operations:

```
[ConvertTensorToBlockOps] No converter for op: tile.full
[ConvertTensorToBlockOps] No converter for op: tile.load
[ConvertTensorToBlockOps] No converter for op: tile.add
[ConvertTensorToBlockOps] No converter for op: tile.store
```

**Location**: `third_party/pypto/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp:567`

### Impact Chain

1. ✅ **TTIR Extraction**: Triton IR correctly extracted
2. ⚠️ **TTIR → PyPTO IR**: Generates `tile.xxx` ops but with wrong dtypes (INT32 instead of FLOAT32)
3. ❌ **PyPTO Pass**: `ConvertTensorToBlockOps` fails to process tile operations
4. ❌ **TensorGraph → BlockGraph**: Conversion fails, no tasks created
5. ❌ **C++ Codegen**: Orchestration generated but empty (no `pto2_rt_submit_task`)
6. ❌ **NPU Execution**: No kernel submitted, no computation performed
7. ❌ **Validation**: Zero-input test produces false positive

### Secondary Issue: Wrong Data Types

Generated orchestration uses incorrect data types:

```cpp
// ❌ Should be FLOAT32, generated as INT32
Tensor ext_x = make_tensor_external(arg_x_ptr, x_shapes, 2, DataType::INT32);
```

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| TTIR Extraction | ✅ Working | Triton IR correctly extracted |
| TTIR → PyPTO IR | ⚠️ Partial | Generates tile ops but wrong dtypes |
| PyPTO Pass | ❌ Failing | ConvertTensorToBlockOps can't handle tile ops |
| C++ Codegen | ⚠️ Empty | Orchestration has no task submission |
| NPU Execution | ❌ None | No kernel submitted |
| Result Validation | ❌ False Positive | Zero inputs mask the issue |

---

## Action Items

### Immediate (Priority: Critical)

- [ ] **Fix test to use non-zero inputs** to prevent false positives
  - File: `examples/run_e2e.py`, `src/triton_adapter/ttir_converter.py`
  - Change: Use `torch.randn()` instead of `torch.zeros()`
  - OR: Integrate `verify_npu_execution.py` into main test flow

### Short Term (Priority: High)

- [ ] **Fix PyPTO converter registry**
  - File: `third_party/pypto/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`
  - Action: Register converters for `tile.full`, `tile.load`, `tile.add`, `tile.store`
  - OR: Skip conversion for operations that are already tile ops

- [ ] **Fix data type mapping**
  - File: `src/triton_adapter/ttir_converter.py`
  - Issue: FLOAT32 tensors being mapped to INT32 in orchestration

### Medium Term (Priority: Medium)

- [ ] **Add validation safeguards**
  - Require non-zero/random inputs for all tests
  - Add profiling check: fail if task count = 0
  - Add orchestration check: verify `pto2_rt_submit_task` calls exist

- [ ] **Improve error reporting**
  - Log converter failures as errors, not warnings
  - Fail fast when no tasks are generated

---

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `examples/run_e2e.py` | Added `--enable-profiling` flag | ✅ Done |
| `examples/e2e_common.py` | Added `enable_profiling` parameter | ✅ Done |
| `NPU_EXECUTION_ANALYSIS.md` | Detailed analysis document | ✅ Done |
| `progress_report.md` | This report | 🔄 In Progress |

---

## Testing Commands

### Current (Produces False Positive)

```bash
# ❌ Uses zero inputs - will pass even if NPU doesn't execute
python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0
```

### Recommended (Detects Real Issues)

```bash
# ✅ Uses non-zero controlled inputs
python examples/verify_npu_execution.py

# ✅ Or with profiling to verify task execution
python examples/run_e2e.py --kernel add --platform a2a3 --device-id 0 --enable-profiling
# Check for: "task count: 0" in output
```

---

## References

- Analysis Document: `NPU_EXECUTION_ANALYSIS.md`
- Verification Script: `examples/verify_npu_execution.py`
- E2E Test: `examples/run_e2e.py`
- PyPTO Pass: `third_party/pypto/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`
- TTIR Converter: `src/triton_adapter/ttir_converter.py`

---

## Appendix: Full Execution Log

### Profiling Output (Zero Tasks)

```
[INFO] Profiling enabled
[INFO] initialize: Initializing performance profiling
[INFO] initialize: Performance profiling initialized
...
[INFO] poll_and_collect: Collecting performance data
[INFO] poll_and_collect: Waiting for AICPU to write total_tasks in PerfDataHeader...
[ERROR] poll_and_collect: Timeout waiting for AICPU task count after 30 seconds
[INFO] poll_and_collect: AICPU finally reported task count: 0
[WARN] export_swimlane_json: Warning: No performance data to export.
```

### Converter Warnings

```
[ConvertTensorToBlockOps] No converter for op: tile.full
[ConvertTensorToBlockOps] No converter for op: tile.full
[ConvertTensorToBlockOps] No converter for op: tile.load
[ConvertTensorToBlockOps] No converter for op: tile.load
[ConvertTensorToBlockOps] No converter for op: tile.add
[ConvertTensorToBlockOps] No converter for op: tile.store
```

---

**Next Review**: After PyPTO converter fix implementation
