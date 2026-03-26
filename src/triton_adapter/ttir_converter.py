"""TTIR to PyPTO IR converter implementation.

This module provides the main conversion logic from Triton TTIR to PyPTO IR.
Uses PyPTO from submodule for IR definitions and Triton for kernel IR extraction.
"""

import re
from dataclasses import dataclass
from typing import Any

from pypto import DataType, ir
from pypto.ir.op import tile

from .exceptions import ConversionError, UnsupportedOpError
from .mlir_parser import MLIROperation, MLIRParser, MLIRType, MLIRValue


def _ttir_op_basename(op_name: str | None) -> str:
    """First token of MLIR op name (handles ``tt.return loc`` style suffixes)."""
    if not op_name:
        return ""
    return op_name.split()[0].strip()


def _op_starts_with(op_name: str | None, prefix: str) -> bool:
    return _ttir_op_basename(op_name) == prefix


@dataclass
class BlockPtrInfo:
    """Information about a block pointer in TTIR.

    Block pointers are used in Triton to manage iteration over tensor blocks.
    """

    base: ir.Var
    shape: list[ir.Expr]
    strides: list[ir.Expr]
    current_offset: list[ir.Expr]
    tensor_shape: list[ir.Expr] | None = None
    order: list[int] | None = None


class SpanTracker:
    """Tracks source code location from TTIR operations."""

    def get_span(self, op: Any) -> ir.Span:
        """Extract source location from TTIR operation."""
        if hasattr(op, "attributes") and "location" in op.attributes:
            loc = op.attributes["location"]
            if isinstance(loc, str):
                parts = loc.split(":")
                if len(parts) >= 3:
                    try:
                        return ir.Span(parts[0], int(parts[1]), int(parts[2]))
                    except (ValueError, IndexError):
                        pass
        return ir.Span.unknown()


class TypeMapper:
    """Maps TTIR types to PyPTO types."""

    DTYPE_MAP: dict[str, DataType] = {
        "i1": DataType.BOOL,
        "i8": DataType.INT8,
        "i16": DataType.INT16,
        "i32": DataType.INT32,
        "i64": DataType.INT64,
        "fp16": DataType.FP16,
        "bf16": DataType.BF16,
        "f16": DataType.FP16,
        "fp32": DataType.FP32,
        "f32": DataType.FP32,
        "fp64": DataType.FP32,  # PyPTO 无 FP64，映射到 FP32
        "f64": DataType.FP32,
    }

    def map_dtype(self, ttir_dtype: str) -> DataType:
        """Map TTIR dtype string to PyPTO DataType."""
        dtype_str = str(ttir_dtype).lower()
        if dtype_str in self.DTYPE_MAP:
            return self.DTYPE_MAP[dtype_str]
        raise ConversionError(f"Unsupported dtype: {ttir_dtype}")

    def map_tensor_type(self, shape: list[int], dtype: DataType) -> ir.TensorType:
        """Create PyPTO TensorType from shape and dtype."""
        return ir.TensorType(shape, dtype)


class TTIRToPyptoConverter:
    """Converts Triton TTIR to PyPTO IR.

    This is the main converter class that orchestrates the conversion process.
    """

    SUPPORTED_OPS = {
        "tt.make_block_ptr",
        "tt.advance",
        "tt.splat",
        "tt.addptr",
        "tt.make_range",
        "tt.get_program_id",
        "tt.load",
        "tt.store",
        "arith.addf",
        "arith.addi",
        "arith.subf",
        "arith.subi",
        "arith.mulf",
        "arith.muli",
        "arith.divf",
        "arith.divi",
        "arith.constant",
        "arith.muli",
        "arith.addi",
        "arith.cmpi",
        "tt.exp",
        "math.exp",
        "arith.cmpf",
        "arith.cmpi",
        "arith.andi",
        "arith.select",
        "tt.program_id",
        "tt.get_program_id",
        "tt.dot",
        "tt.reshape",
        "tt.reduce",
        "tt.expand_dims",
        "tt.broadcast",
    }

    def __init__(self) -> None:
        self.ib = ir.IRBuilder()
        self.type_mapper = TypeMapper()
        self.span_tracker = SpanTracker()
        self.value_map: dict[str, ir.Var] = {}
        self.block_ptr_map: dict[str, BlockPtrInfo] = {}
        self.ptr_trace: dict[str, str] = {}  # value -> base (for splat+addptr chain)
        self._tmp_counter = 0
        self.span = ir.Span.unknown()
        self._last_store_result: ir.Expr | None = None
        self._last_store_dest_param: ir.Var | None = None
    def _tmp_id(self) -> int:
        """Generate a unique temporary ID."""
        self._tmp_counter += 1
        return self._tmp_counter

    def _value_key(self, val: MLIRValue) -> str:
        """Get key for value map."""
        return f"%{val.name}" if not val.name.startswith("%") else val.name

    def _infer_kernel_tensor_shape_dtype(
        self, body_ops: list[MLIROperation]
    ) -> tuple[list[int], DataType]:
        """Infer tile shape and element dtype from TTIR body.

        Triton may emit integer tensor constants (e.g. ``mask`` as ``tensor<128xi32>``)
        before loads; using only the *first* tensor op would misclassify params as
        INT32. Prefer the first tensor whose element type is floating-point; otherwise
        fall back to the first tensor result (legacy behavior).
        """
        default_shape: list[int] = [128, 128]
        default_dtype = DataType.FP32
        float_candidates: list[tuple[list[int], DataType]] = []
        first_any: tuple[list[int], DataType] | None = None

        for op in body_ops:
            if not op.result_types:
                continue
            rt = op.result_types[0]
            if not rt.is_tensor():
                continue
            et = rt.get_element_type()
            if not et:
                continue
            try:
                dtype = self.type_mapper.map_dtype(et)
            except ConversionError:
                continue
            s = rt.get_shape()
            shape = list(s) if s else list(default_shape)
            if len(shape) == 1:
                shape = [shape[0], 1]
            elif len(shape) > 2:
                shape = shape[:2]
            if first_any is None:
                first_any = (shape, dtype)
            if dtype in (
                DataType.FP16,
                DataType.BF16,
                DataType.FP32,
            ):
                float_candidates.append((shape, dtype))

        if float_candidates:
            return float_candidates[0]
        if first_any is not None:
            return first_any
        return default_shape, default_dtype

    def _get_operand(self, val: MLIRValue) -> ir.Expr:
        """Get PyPTO expr for an MLIR operand."""
        key = self._value_key(val)
        if key not in self.value_map:
            raise ConversionError(f"Unknown operand: {key}")
        return self.value_map[key]

    def convert(self, ttir_text: str, program_name: str = "kernel") -> ir.Program:
        """Convert TTIR text to PyPTO Program.

        Args:
            ttir_text: TTIR MLIR text.
            program_name: Name for the output program.

        Returns:
            PyPTO Program object.
        """
        parser = MLIRParser()
        operations = parser.parse_module(ttir_text)

        # Extract function info from tt.func or first operations
        func_name = "kernel"
        arg_names: list[str] = []
        body_ops: list[MLIROperation] = []

        for op in operations:
            if "tt.func" in op.name and op.operands:
                arg_names = [self._value_key(o) for o in op.operands]
                if "@" in op.name:
                    func_name = op.name.split("@")[-1].split("(")[0].strip()
                continue
            raw = (op.name or "").strip()
            if raw.startswith("}") or raw.startswith("{"):
                continue
            if _op_starts_with(op.name, "tt.return"):
                continue
            # Include ops with no SSA result (e.g. tt.store) — they were previously dropped
            # and broke kernels that only write via store with an empty tt.return.
            if "tt.func" not in op.name:
                body_ops.append(op)

        if not body_ops and operations:
            body_ops = [
                op
                for op in operations
                if "tt.func" not in op.name
                and not (op.name or "").strip().startswith("}")
                and not _op_starts_with(op.name, "tt.return")
            ]
        if not arg_names and body_ops:
            arg_names = []
            for op in body_ops:
                for o in op.operands:
                    k = self._value_key(o)
                    if k.startswith("%arg") and k not in arg_names:
                        arg_names.append(k)
            arg_names.sort(key=lambda x: int(x.replace("%arg", "")))

        with self.ib.program(program_name) as p:
            incore_func = self._build_incore_function(
                func_name, arg_names, body_ops, operations
            )
            if incore_func:
                p.add_function(incore_func)
            orch_func = self._build_orchestration_function(
                func_name, arg_names, body_ops, incore_func
            )
            if orch_func:
                p.add_function(orch_func)

        return p.get_result()

    def _build_incore_function(
        self,
        func_name: str,
        arg_names: list[str],
        body_ops: list[MLIROperation],
        all_ops: list[MLIROperation],
    ) -> ir.Function | None:
        """Build InCore function from TTIR body."""
        if not body_ops:
            return None

        # Detect program_id use - need pid params
        need_pid = set()
        for op in body_ops:
            if op.name in ("tt.get_program_id", "tt.program_id"):
                axis = op.attributes.get("axis", 0)
                if "x" in str(axis).lower():
                    axis = 0
                elif "y" in str(axis).lower():
                    axis = 1
                elif "z" in str(axis).lower():
                    axis = 2
                else:
                    try:
                        axis = int(axis)
                    except (ValueError, TypeError):
                        axis = 0
                need_pid.add(axis)

        shape, dtype = self._infer_kernel_tensor_shape_dtype(body_ops)

        self.value_map.clear()
        self.block_ptr_map.clear()
        self.ptr_trace.clear()
        self._last_store_result = None
        self._last_store_dest_param = None

        tensor_type = ir.TensorType(shape, dtype)
        with self.ib.function(
            f"{func_name}_incore", type=ir.FunctionType.InCore
        ) as f:
            params: list[ir.Var] = []
            for i, arg_name in enumerate(arg_names):
                pname = arg_name.lstrip("%")
                if i < 3:
                    param = f.param(pname, tensor_type)
                    params.append(param)
                    self.value_map[arg_name] = param
                else:
                    param = f.param(pname, ir.ScalarType(DataType.INT64))
                    params.append(param)
                    self.value_map[arg_name] = param

            for axis in sorted(need_pid):
                pid_param = f.param(f"pid_{axis}", ir.ScalarType(DataType.INT64))
                self.value_map[f"%pid_{axis}"] = pid_param

            f.return_type(tensor_type)

            for op in body_ops:
                if _op_starts_with(op.name, "tt.return"):
                    continue
                self._convert_op(op)

            return_stmt_ops = [o for o in all_ops if _op_starts_with(o.name, "tt.return")]
            if return_stmt_ops and return_stmt_ops[0].operands:
                last_val = self._get_operand(return_stmt_ops[0].operands[0])
                self.ib.return_stmt(last_val)
            elif self._last_store_dest_param is not None:
                # Triton often omits tt.return operands; we must not return the last
                # computed tile (e.g. exp result) or CCE adds a spurious output tensor
                # and mis-indexes kernel_entry args (breaks tt.store to output buffer).
                self.ib.return_stmt(self._last_store_dest_param)
            elif self._last_store_result is not None:
                self.ib.return_stmt(self._last_store_result)
            elif body_ops:
                last_op = body_ops[-1]
                if last_op.result:
                    last_val = self.value_map.get(self._value_key(last_op.result))
                    if last_val:
                        self.ib.return_stmt(last_val)

        return f.get_result()

    def _build_orchestration_function(
        self,
        func_name: str,
        arg_names: list[str],
        body_ops: list[MLIROperation],
        incore_func: ir.Function | None,
    ) -> ir.Function | None:
        """Build Orchestration function that calls InCore."""
        if not incore_func or len(arg_names) < 2:
            return None

        shape, dtype = self._infer_kernel_tensor_shape_dtype(body_ops)

        tensor_type = ir.TensorType(shape, dtype)
        num_tensor_params = min(3, len(arg_names))
        pnames = [n.lstrip("%") for n in arg_names[:num_tensor_params]]
        with self.ib.function(
            "main", type=ir.FunctionType.Orchestration
        ) as f:
            orch_params = [f.param(pnames[i] if i < len(pnames) else f"arg{i}", tensor_type) for i in range(num_tensor_params)]
            call_args: list[ir.Expr] = list(orch_params)
            # Add pid constants for incore params beyond tensor args
            for i in range(len(orch_params), len(incore_func.params)):
                call_args.append(ir.ConstInt(0, DataType.INT64, self.span))
            call_args = call_args[: len(incore_func.params)]
            out = ir.Call(ir.GlobalVar(incore_func.name), call_args, self.span)
            self.ib.return_stmt(out)
        return f.get_result()

    def _convert_op(self, op: MLIROperation) -> None:
        """Dispatch operation to handler."""
        self.span = self.span_tracker.get_span(op)
        op_name = op.name.strip('"')  # Normalize quoted names like "tt.reduce"
        if op_name not in self.SUPPORTED_OPS:
            handler_name = op_name.replace(".", "_")
            if not hasattr(self, f"_convert_{handler_name}"):
                raise UnsupportedOpError(op.name, self.span)
        handler = getattr(
            self,
            f"_convert_{op_name.replace('.', '_')}",
            self._convert_generic,
        )
        handler(op)

    def _convert_generic(self, op: MLIROperation) -> None:
        """Generic handler for unhandled ops."""
        raise UnsupportedOpError(op.name, self.span)

    def _convert_arith_constant(self, op: MLIROperation) -> None:
        """Convert arith.constant to ConstInt/ConstFloat or tile.full for tensor."""
        if not op.result:
            return
        key = self._value_key(op.result)
        attr = op.attributes.get("value")
        if attr:
            val_str = str(attr).strip()
            # Parser may leave ``loc(...)`` on the value: ``true loc(#loc)``.
            val_tok = val_str.split()[0] if val_str else ""
            if val_tok.lower() in ("true", "false"):
                b = val_tok.lower() == "true"
                expr = ir.ConstInt(1 if b else 0, DataType.BOOL, self.span)
                var = self.ib.let(key.replace("%", "cst_"), expr)
                self.value_map[key] = var
                return
            is_tensor = op.result_types and op.result_types[0].is_tensor()
            # Dense tensor constant: dense<0.0> : tensor<16x16xf32>
            if is_tensor and "dense" in val_str.lower():
                import re as _re
                m = _re.search(r"dense<([^>]+)>", val_str)
                fill_val = 0.0
                if m:
                    inner = m.group(1).strip()
                    try:
                        fill_val = float(inner)
                    except ValueError:
                        try:
                            fill_val = int(inner)
                        except ValueError:
                            pass
                shape = [16, 16]
                dtype = DataType.FP32
                if op.result_types:
                    rt = op.result_types[0]
                    if rt.get_shape():
                        shape = rt.get_shape()
                    if rt.get_element_type():
                        dtype = self.type_mapper.map_dtype(rt.get_element_type())
                if len(shape) == 1:
                    shape = [shape[0], 1]
                expr = tile.full(shape, dtype, fill_val, span=self.span)
                var = self.ib.let(key.replace("%", "cst_"), expr)
                self.value_map[key] = var
                return
            # Triton may emit splat-like FP constants: ``1.0e+00 : tensor<128x1xf32>`` (no ``dense<``).
            if is_tensor:
                rt = op.result_types[0]
                et = (rt.get_element_type() or "").lower()
                if et in ("f32", "f16", "bf16", "fp32", "fp16", "bf16"):
                    fill_val: float | None = None
                    try:
                        if "." in val_str or "e" in val_str.lower():
                            fill_val = float(val_str)
                    except ValueError:
                        fill_val = None
                    if fill_val is not None:
                        shape = [16, 16]
                        dtype = DataType.FP32
                        s = rt.get_shape()
                        if s:
                            shape = list(s)
                            if len(shape) == 1:
                                shape = [shape[0], 1]
                            elif len(shape) > 2:
                                shape = shape[:2]
                        if et:
                            dtype = self.type_mapper.map_dtype(rt.get_element_type() or "f32")
                        expr = tile.full(shape, dtype, fill_val, span=self.span)
                        var = self.ib.let(key.replace("%", "cst_"), expr)
                        self.value_map[key] = var
                        return
            if "." in val_str or "e" in val_str.lower():
                try:
                    fval = float(val_str)
                    dtype = DataType.FP32
                    expr = ir.ConstFloat(fval, dtype, self.span)
                    var = self.ib.let(key.replace("%", "cst_"), expr)
                    self.value_map[key] = var
                except ValueError:
                    pass
            else:
                try:
                    ival = int(val_str)
                    dtype = DataType.INT64
                    expr = ir.ConstInt(ival, dtype, self.span)
                    var = self.ib.let(key.replace("%", "cst_"), expr)
                    self.value_map[key] = var
                except ValueError:
                    pass

    def _convert_tt_splat(self, op: MLIROperation) -> None:
        """Track tt.splat: ptr_trace[result] = base (operand)."""
        if not op.result or not op.operands:
            return
        base_key = self._value_key(op.operands[0])
        result_key = self._value_key(op.result)
        self.ptr_trace[result_key] = base_key
        # Splat produces a tensor of ptrs; we map result to base for load/store
        self.value_map[result_key] = self._get_operand(op.operands[0])

    def _convert_tt_addptr(self, op: MLIROperation) -> None:
        """Track tt.addptr: ptr_trace[result] = base from first operand's trace."""
        if not op.result or len(op.operands) < 2:
            return
        ptr_key = self._value_key(op.operands[0])
        base_key = self.ptr_trace.get(ptr_key, ptr_key)
        result_key = self._value_key(op.result)
        self.ptr_trace[result_key] = base_key
        self.value_map[result_key] = self._get_operand(op.operands[0])

    def _convert_tt_make_range(self, op: MLIROperation) -> None:
        """tt.make_range produces indices - use placeholder for ptr chain propagation."""
        if not op.result:
            return
        # Placeholder for index computations; actual load uses fixed offsets
        c0 = ir.ConstInt(0, DataType.INT64, self.span)
        var = self.ib.let(f"range_{op.result.name}".replace("%", ""), c0)
        self.value_map[self._value_key(op.result)] = var

    def _convert_tt_get_program_id(self, op: MLIROperation) -> None:
        """Convert tt.get_program_id to pid param (like tt.program_id)."""
        self._convert_tt_program_id(op)

    def _convert_tt_load(self, op: MLIROperation) -> None:
        """Convert tt.load to tile.load."""
        if not op.result or not op.operands:
            return
        ptr = op.operands[0]
        ptr_key = self._value_key(ptr)
        base_key = self.ptr_trace.get(ptr_key, ptr_key)
        tensor_var = self.value_map.get(base_key)
        if tensor_var is None:
            tensor_var = self._get_operand(ptr)
        # Infer shape from result type or operand type
        shape = [128, 128]
        if op.result_types:
            rt = op.result_types[0]
            if rt.is_tensor():
                s = rt.get_shape()
                if s:
                    shape = s
        if not shape and op.operands:
            ot = getattr(op.operands[0], "type_str", "") or ""
            if "tensor<128x" in ot or "tensor<128 " in ot:
                shape = [128]
        if len(shape) == 1:
            # One logical dim → ``[N, 1]`` so ColMajor Vec tiles satisfy ``Rows * sizeof(dtype)``
            # multiple of 32 bytes (see pto_tile.hpp); ``[1, N]`` would have ``Rows == 1`` and fail.
            shape = [shape[0], 1]
        elif len(shape) > 2:
            shape = shape[:2]
        offsets = [0] * len(shape)
        load_call = tile.load(
            tensor_var, offsets, shape, span=self.span
        )
        result_var = self.ib.let(
            op.result.name.replace("%", "load_"),
            load_call,
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_tt_store(self, op: MLIROperation) -> None:
        """Convert tt.store to tile.store."""
        if len(op.operands) < 2:
            return
        ptr = op.operands[0]
        value = op.operands[1]
        ptr_key = self._value_key(ptr)
        base_key = self.ptr_trace.get(ptr_key, ptr_key)
        output_var = self.value_map.get(base_key)
        if output_var is None:
            output_var = self._get_operand(ptr)
        tile_var = self._get_operand(value)
        # Reduce / scalar SSA values are ``ScalarType``; ``tile.store`` needs a tile. Do not wrap
        # real tiles: ``tile.muls(one, tile)`` with two tiles would broadcast to 1×1 and break
        # PTO-ISA alignment (regression on add/mul/etc.).
        if isinstance(tile_var.type, ir.ScalarType):
            sdt = tile_var.type.dtype
            fill_one: int | float = (
                1
                if sdt
                in (
                    DataType.INT8,
                    DataType.INT16,
                    DataType.INT32,
                    DataType.INT64,
                    DataType.UINT8,
                    DataType.UINT16,
                    DataType.UINT32,
                    DataType.UINT64,
                )
                else 1.0
            )
            one = tile.full([1, 1], sdt, fill_one, span=self.span)
            one_v = self.ib.let(f"store_scalar_one_{self._tmp_id()}", one)
            tile_var = tile.muls(one_v, tile_var, span=self.span)
        shape = [128, 128]
        if op.result_types:
            rt = op.result_types[0]
            if rt and rt.is_tensor():
                s = rt.get_shape()
                if s:
                    shape = s
        if len(shape) == 1:
            shape = [shape[0], 1]
        offsets = [0] * len(shape)
        store_result = tile.store(tile_var, offsets, output_var, span=self.span)
        self._last_store_result = store_result
        dest_param = self.value_map.get(base_key)
        if dest_param is not None:
            self._last_store_dest_param = dest_param
        self.ib.let("store_result", store_result)

    def _fp_add_expr(self, lhs: ir.Expr, rhs: ir.Expr) -> ir.Expr:
        """``arith.addf``: tile+tile, tile+scalar, or scalar+scalar (unrolled reduce loops)."""
        lt = lhs.type
        rt = rhs.type
        if isinstance(lt, ir.TileType) and isinstance(rt, ir.TileType):
            return tile.add(lhs, rhs, span=self.span)
        if isinstance(lt, ir.TileType) and isinstance(rt, ir.ScalarType):
            return tile.adds(lhs, rhs, span=self.span)
        if isinstance(lt, ir.ScalarType) and isinstance(rt, ir.TileType):
            return tile.adds(rhs, lhs, span=self.span)
        return ir.Add(lhs, rhs, DataType.FP32, self.span)

    def _convert_arith_addf(self, op: MLIROperation) -> None:
        """Convert arith.addf to tile/tile or tile/scalar addition."""
        if not op.result or len(op.operands) < 2:
            return
        lhs = self._get_operand(op.operands[0])
        rhs = self._get_operand(op.operands[1])
        result_expr = self._fp_add_expr(lhs, rhs)
        result_var = self.ib.let(
            f"add_{op.result.name}".replace("%", ""),
            result_expr,
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_arith_addi(self, op: MLIROperation) -> None:
        """Convert arith.addi to tile.add or ir.Add for scalars."""
        if not op.result or len(op.operands) < 2:
            return
        is_scalar = False
        if op.result_types:
            rt = op.result_types[0]
            if rt and not rt.is_tensor():
                is_scalar = True
        if is_scalar:
            lhs = self._get_operand(op.operands[0])
            rhs = self._get_operand(op.operands[1])
            result_expr = ir.Add(lhs, rhs, DataType.INT64, self.span)
            result_var = self.ib.let(
                f"addi_{op.result.name}".replace("%", ""),
                result_expr,
            )
            self.value_map[self._value_key(op.result)] = result_var
        else:
            try:
                self._convert_binary_op(op, "add", tile.add)
            except (ValueError, TypeError) as e:
                if "TileType" in str(e) or "ScalarType" in str(e):
                    # Scalar+scalar with tensor result (e.g. splat+range): placeholder
                    shape = [128, 1]
                    if op.result_types and op.result_types[0].get_shape():
                        shape = op.result_types[0].get_shape()
                        if len(shape) == 1:
                            shape = [shape[0], 1]
                    expr = tile.full(shape, DataType.INT32, 0, span=self.span)
                    var = self.ib.let(
                        f"addi_{op.result.name}".replace("%", ""),
                        expr,
                    )
                    self.value_map[self._value_key(op.result)] = var
                else:
                    raise

    def _convert_arith_subf(self, op: MLIROperation) -> None:
        """Convert arith.subf to tile.sub."""
        self._convert_binary_op(op, "sub", tile.sub)

    def _convert_arith_subi(self, op: MLIROperation) -> None:
        """Convert arith.subi to tile.sub."""
        self._convert_binary_op(op, "sub", tile.sub)

    def _convert_arith_mulf(self, op: MLIROperation) -> None:
        """Convert arith.mulf to tile.mul."""
        self._convert_binary_op(op, "mul", tile.mul)

    def _convert_arith_muli(self, op: MLIROperation) -> None:
        """Convert arith.muli to tile.mul or ir.Mul for scalars."""
        if not op.result or len(op.operands) < 2:
            return
        # Check if scalar: result type i32/i64 or not tensor
        is_scalar = False
        if op.result_types:
            rt = op.result_types[0]
            if rt and not rt.is_tensor():
                is_scalar = True
        elif op.result and op.result.type_str and "tensor" not in op.result.type_str:
            is_scalar = True
        if is_scalar:
            lhs = self._get_operand(op.operands[0])
            rhs = self._get_operand(op.operands[1])
            result_expr = ir.Mul(lhs, rhs, DataType.INT64, self.span)
            result_var = self.ib.let(
                f"muli_{op.result.name}".replace("%", ""),
                result_expr,
            )
            self.value_map[self._value_key(op.result)] = result_var
        else:
            try:
                self._convert_binary_op(op, "mul", tile.mul)
            except (ValueError, TypeError):
                # Index tensor muli (e.g. make_range*const) - use scalar placeholder
                lhs = self._get_operand(op.operands[0])
                rhs = self._get_operand(op.operands[1])
                result_expr = ir.Mul(lhs, rhs, DataType.INT64, self.span)
                result_var = self.ib.let(
                    f"muli_{op.result.name}".replace("%", ""),
                    result_expr,
                )
                self.value_map[self._value_key(op.result)] = result_var

    def _convert_arith_divf(self, op: MLIROperation) -> None:
        """Convert arith.divf to tile.div."""
        self._convert_binary_op(op, "div", tile.div)

    def _convert_arith_divi(self, op: MLIROperation) -> None:
        """Convert arith.divi to tile.div."""
        self._convert_binary_op(op, "div", tile.div)

    def _convert_binary_op(
        self,
        op: MLIROperation,
        op_suffix: str,
        pypto_op: Any,
    ) -> None:
        """Convert binary op (addf, mulf, etc.) to tile op."""
        if not op.result or len(op.operands) < 2:
            return
        lhs = self._get_operand(op.operands[0])
        rhs = self._get_operand(op.operands[1])
        result_expr = pypto_op(lhs, rhs, span=self.span)
        result_var = self.ib.let(
            f"{op_suffix}_{op.result.name}".replace("%", ""),
            result_expr,
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_tt_make_block_ptr(self, op: MLIROperation) -> None:
        """Convert tt.make_block_ptr - track BlockPtrInfo."""
        if not op.result or len(op.operands) < 4:
            return
        base = self._get_operand(op.operands[0])
        shape = [ir.ConstInt(128, DataType.INT64, self.span)]
        strides = [ir.ConstInt(1, DataType.INT64, self.span)]
        offsets = [ir.ConstInt(0, DataType.INT64, self.span)]
        info = BlockPtrInfo(base=base, shape=shape, strides=strides, current_offset=offsets)
        self.block_ptr_map[self._value_key(op.result)] = info

    def _convert_tt_advance(self, op: MLIROperation) -> None:
        """Convert tt.advance - update BlockPtrInfo."""
        if not op.result or len(op.operands) < 2:
            return
        ptr_key = self._value_key(op.operands[0])
        if ptr_key not in self.block_ptr_map:
            return
        old_info = self.block_ptr_map[ptr_key]
        delta = [ir.ConstInt(0, DataType.INT64, self.span)]
        new_offsets = [
            ir.Add(old_info.current_offset[i], d, DataType.INT64, self.span)
            for i, d in enumerate(delta)
        ]
        new_info = BlockPtrInfo(
            base=old_info.base,
            shape=old_info.shape,
            strides=old_info.strides,
            current_offset=new_offsets,
        )
        self.block_ptr_map[self._value_key(op.result)] = new_info

    def _convert_tt_exp(self, op: MLIROperation) -> None:
        """Convert tt.exp to tile.exp."""
        if not op.result or not op.operands:
            return
        inp = self._get_operand(op.operands[0])
        result_expr = tile.exp(inp, span=self.span)
        result_var = self.ib.let(f"exp_{op.result.name}".replace("%", ""), result_expr)
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_math_exp(self, op: MLIROperation) -> None:
        """Convert math.exp to tile.exp (Triton uses math.exp for tl.exp)."""
        self._convert_tt_exp(op)

    def _convert_arith_cmpf(self, op: MLIROperation) -> None:
        """Convert arith.cmpf - use tile.cmp."""
        if not op.result or len(op.operands) < 2:
            return
        pred_map = {"olt": 0, "ole": 1, "oeq": 2, "one": 3, "oge": 4, "ogt": 5}
        pred_map.update({"slt": 0, "sle": 1, "eq": 2, "sge": 4, "sgt": 5})  # cmpi
        pred = op.attributes.get("predicate", "olt")
        cmp_type = pred_map.get(str(pred), 0)
        lhs = self._get_operand(op.operands[0])
        rhs = self._get_operand(op.operands[1])
        try:
            result_expr = tile.cmp(lhs, rhs, cmp_type=cmp_type, span=self.span)
        except (ValueError, TypeError) as e:
            if "TileType" in str(e) or "ScalarType" in str(e):
                shape = [128, 1]
                if op.result_types and op.result_types[0].get_shape():
                    shape = op.result_types[0].get_shape()
                    if len(shape) == 1:
                        shape = [shape[0], 1]
                result_expr = tile.full(shape, DataType.BOOL, 1, span=self.span)
            else:
                raise
        result_var = self.ib.let(
            f"cmp_{op.result.name}".replace("%", ""), result_expr
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_arith_cmpi(self, op: MLIROperation) -> None:
        """Convert arith.cmpi to tile.cmp."""
        self._convert_arith_cmpf(op)

    def _convert_arith_andi(self, op: MLIROperation) -> None:
        """Convert arith.andi (mask combine). tile.and needs int; bool masks use placeholder."""
        if not op.result or len(op.operands) < 2:
            return
        lhs = self._get_operand(op.operands[0])
        rhs = self._get_operand(op.operands[1])
        shape = [128, 1]
        if op.result_types and op.result_types[0].get_shape():
            shape = op.result_types[0].get_shape()
            if len(shape) == 1:
                shape = [shape[0], 1]
        try:
            result_expr = tile.and_(lhs, rhs, span=self.span)
        except (ValueError, TypeError) as e:
            err_str = str(e).lower()
            if "integer" in err_str or "bool" in err_str or "tiletype" in err_str:
                result_expr = tile.full(shape, DataType.BOOL, 1, span=self.span)
            else:
                raise
        result_var = self.ib.let(
            f"andi_{op.result.name}".replace("%", ""),
            result_expr,
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_arith_select(self, op: MLIROperation) -> None:
        """Convert arith.select to tile.sel."""
        if not op.result or len(op.operands) < 3:
            return
        cond = self._get_operand(op.operands[0])
        true_val = self._get_operand(op.operands[1])
        false_val = self._get_operand(op.operands[2])
        result_expr = tile.sel(cond, true_val, false_val, span=self.span)
        result_var = self.ib.let(
            f"sel_{op.result.name}".replace("%", ""), result_expr
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_tt_program_id(self, op: MLIROperation) -> None:
        """Convert tt.program_id - use pid param (must be added in _build_incore_function)."""
        if not op.result:
            return
        axis = op.attributes.get("axis", 0)
        if isinstance(axis, str) and "x" in axis.lower():
            axis = 0
        elif isinstance(axis, str) and "y" in axis.lower():
            axis = 1
        elif isinstance(axis, str) and "z" in axis.lower():
            axis = 2
        else:
            try:
                axis = int(axis)
            except (ValueError, TypeError):
                axis = 0
        key = f"%pid_{axis}"
        if key in self.value_map:
            self.value_map[self._value_key(op.result)] = self.value_map[key]
        else:
            var = self.ib.var(f"pid_{axis}", ir.ScalarType(DataType.INT64))
            self.value_map[self._value_key(op.result)] = var

    def _convert_tt_reshape(self, op: MLIROperation) -> None:
        """Convert ``tt.reshape`` to ``tile.reshape``."""
        if not op.result or not op.operands:
            return
        inp = self._get_operand(op.operands[0])
        shape: list[int] = [128, 128]
        if op.result_types and op.result_types[0].is_tensor():
            s = op.result_types[0].get_shape()
            if s:
                shape = list(s)
                if len(shape) == 1:
                    shape = [shape[0], 1]
                elif len(shape) > 2:
                    shape = shape[:2]
        result_expr = tile.reshape(inp, shape, span=self.span)
        result_var = self.ib.let(
            f"reshape_{op.result.name}".replace("%", ""),
            result_expr,
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_tt_dot(self, op: MLIROperation) -> None:
        """Convert ``tt.dot`` to two-argument ``tile.matmul``.

        Ignore Triton's third accumulator operand: scalar or 1×1 acc tiles break CPU sim and
        ``matmul_acc`` codegen; ``acc=0`` is equivalent to ``matmul(lhs, rhs)``.
        """
        if not op.result or len(op.operands) < 2:
            return
        lhs = self._get_operand(op.operands[0])
        rhs = self._get_operand(op.operands[1])
        result_expr = tile.matmul(lhs, rhs, span=self.span)
        result_var = self.ib.let(
            f"dot_{op.result.name}".replace("%", ""),
            result_expr,
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_tt_reduce(self, op: MLIROperation) -> None:
        """Convert ``tt.reduce`` (e.g. ``tl.sum``) to ``tile.sum`` / ``tile.max`` / ``tile.row_*``.

        Triton emits ``tensor<NxT>`` with ``axis=0`` for a row-wise ``tl.sum`` over a 1D
        block (``tensor<Nxf32>``). Map that to ``tile.row_sum`` / ``tile.row_max`` after
        ``reshape`` to ``[1, N]`` so PTO-ISA SimKernel uses TROWSUM (aligned with
        ``[N,1]`` + ``axis=1``). True column reduction stays on ``tile.sum`` / ``tile.max``.
        """
        if not op.result or not op.operands:
            return
        inp = self._get_operand(op.operands[0])

        axis = 0
        if op.attributes:
            ax = op.attributes.get("axis")
            if ax is not None:
                try:
                    axis = int(str(ax).split(":")[0].strip())
                except ValueError:
                    axis = 0

        reduce_kind = "sum"
        if op.attributes:
            attr_str = str(op.attributes)
            if "maximumf" in attr_str or "maxnumf" in attr_str:
                reduce_kind = "max"

        operand_ty = MLIRType(op.operands[0].type_str)
        in_shape = operand_ty.get_shape() if operand_ty.is_tensor() else None
        one_d_row_reduce = (
            axis == 0
            and in_shape is not None
            and len(in_shape) == 1
            and in_shape[0] > 1
        )

        if one_d_row_reduce:
            assert in_shape is not None
            n = int(in_shape[0])
            elem_dtype = DataType.FP32
            et = operand_ty.get_element_type()
            if et:
                try:
                    elem_dtype = self.type_mapper.map_dtype(et)
                except ConversionError:
                    elem_dtype = DataType.FP32
            reshaped = tile.reshape(inp, [1, n], span=self.span)
            reshaped_v = self.ib.let(f"reduce_rs_{op.result.name}".replace("%", ""), reshaped)
            tmp = tile.full([1, n], elem_dtype, 0, span=self.span)
            tmp_v = self.ib.let(f"reduce_tmp_{self._tmp_id()}", tmp)
            if reduce_kind == "max":
                result_expr = tile.row_max(reshaped_v, tmp_v, span=self.span)
            else:
                result_expr = tile.row_sum(reshaped_v, tmp_v, span=self.span)
        elif axis == 0:
            if reduce_kind == "max":
                result_expr = tile.max(inp, axis=0, keepdim=False, span=self.span)
            else:
                result_expr = tile.sum(inp, axis=0, keepdim=False, span=self.span)
        else:
            if reduce_kind == "max":
                result_expr = tile.max(inp, axis=axis, keepdim=True, span=self.span)
            else:
                result_expr = tile.sum(inp, axis=axis, keepdim=True, span=self.span)

        result_var = self.ib.let(
            f"reduce_{op.result.name}".replace("%", ""),
            result_expr,
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_tt_expand_dims(self, op: MLIROperation) -> None:
        """Convert tt.expand_dims - add dimension. Forward operand for index propagation."""
        if not op.result or not op.operands:
            return
        try:
            inp = self._get_operand(op.operands[0])
            self.value_map[self._value_key(op.result)] = inp
        except Exception:
            # Operand may be make_range etc; use placeholder
            c0 = ir.ConstInt(0, DataType.INT64, self.span)
            var = self.ib.let(f"expand_{op.result.name}".replace("%", ""), c0)
            self.value_map[self._value_key(op.result)] = var

    def _convert_tt_broadcast(self, op: MLIROperation) -> None:
        """Convert tt.broadcast - broadcast tensor. Forward operand."""
        if not op.result or not op.operands:
            return
        try:
            inp = self._get_operand(op.operands[0])
            self.value_map[self._value_key(op.result)] = inp
        except Exception:
            c0 = ir.ConstInt(0, DataType.INT64, self.span)
            var = self.ib.let(f"bcast_{op.result.name}".replace("%", ""), c0)
            self.value_map[self._value_key(op.result)] = var
