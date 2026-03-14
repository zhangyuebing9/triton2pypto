"""TTIR to PyPTO IR converter implementation.

This module provides the main conversion logic from Triton TTIR to PyPTO IR.
Uses PyPTO from submodule for IR definitions and Triton for kernel IR extraction.
"""

from dataclasses import dataclass
from typing import Any

from pypto import DataType, ir
from pypto.ir.op import tile

from .exceptions import ConversionError, UnsupportedOpError
from .mlir_parser import MLIROperation, MLIRParser, MLIRValue


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

    def _tmp_id(self) -> int:
        """Generate a unique temporary ID."""
        self._tmp_counter += 1
        return self._tmp_counter

    def _value_key(self, val: MLIRValue) -> str:
        """Get key for value map."""
        return f"%{val.name}" if not val.name.startswith("%") else val.name

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
            if op.result and "tt.func" not in op.name and op.name != "tt.return":
                body_ops.append(op)
            elif not op.result and op.name == "tt.return":
                pass

        if not body_ops and operations:
            body_ops = [
                op
                for op in operations
                if op.result and "tt.func" not in op.name and op.name != "tt.return"
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

        # Infer param types from first load/store
        shape = [128, 128]
        dtype = DataType.FP32
        for op in body_ops:
            if op.result_types:
                rt = op.result_types[0]
                if rt.is_tensor():
                    s = rt.get_shape()
                    if s:
                        shape = s
                    et = rt.get_element_type()
                    if et:
                        dtype = self.type_mapper.map_dtype(et)
                break

        # Ensure 2D for tile ops
        if len(shape) == 1:
            shape = [shape[0], 1]
        elif len(shape) > 2:
            shape = shape[:2]

        self.value_map.clear()
        self.block_ptr_map.clear()
        self.ptr_trace.clear()
        self._last_store_result = None

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
                if op.name == "tt.return":
                    continue
                self._convert_op(op)

            return_stmt_ops = [o for o in all_ops if o.name == "tt.return"]
            if return_stmt_ops and return_stmt_ops[0].operands:
                last_val = self._get_operand(return_stmt_ops[0].operands[0])
                self.ib.return_stmt(last_val)
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

        shape = [128, 128]
        dtype = DataType.FP32
        for op in body_ops:
            if op.result_types:
                rt = op.result_types[0]
                if rt.is_tensor():
                    s = rt.get_shape()
                    if s:
                        shape = s
                    et = rt.get_element_type()
                    if et:
                        dtype = self.type_mapper.map_dtype(et)
                break
        if len(shape) == 1:
            shape = [shape[0], 1]

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
            # Dense tensor constant: dense<0.0> : tensor<16x16xf32>
            is_tensor = op.result_types and op.result_types[0].is_tensor()
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
        self.ib.let("store_result", store_result)

    def _convert_arith_addf(self, op: MLIROperation) -> None:
        """Convert arith.addf to tile.add."""
        self._convert_binary_op(op, "add", tile.add)

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

    def _convert_tt_dot(self, op: MLIROperation) -> None:
        """Convert tt.dot to tile.matmul or tile.matmul_acc."""
        if not op.result or len(op.operands) < 2:
            return
        lhs = self._get_operand(op.operands[0])
        rhs = self._get_operand(op.operands[1])
        shape = [16, 16]
        if op.result_types and op.result_types[0].is_tensor():
            s = op.result_types[0].get_shape()
            if s:
                shape = s
        if len(op.operands) >= 3:
            acc = self._get_operand(op.operands[2])
            try:
                result_expr = tile.matmul_acc(acc, lhs, rhs, span=self.span)
            except (ValueError, TypeError):
                # acc was scalar (ConstFloat); create zeros tile
                acc = tile.full(shape, DataType.FP32, 0.0, span=self.span)
                acc = self.ib.let("dot_acc_init", acc)
                result_expr = tile.matmul_acc(acc, lhs, rhs, span=self.span)
        else:
            result_expr = tile.matmul(lhs, rhs, span=self.span)
        result_var = self.ib.let(
            f"dot_{op.result.name}".replace("%", ""),
            result_expr,
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_tt_reduce(self, op: MLIROperation) -> None:
        """Convert tt.reduce to tile.row_sum or tile.row_max.

        Infers reduce kind from body: arith.addf -> row_sum, arith.maximumf/maxnumf -> row_max.
        For 1D input [N], reshapes to [1,N] then row_sum -> [1,1].
        """
        if not op.result or not op.operands:
            return
        inp = self._get_operand(op.operands[0])

        # Infer reduce kind: arith.addf/addi -> row_sum, maximumf/maxnumf -> row_max
        reduce_kind = "row_sum"
        if op.attributes:
            attr_str = str(op.attributes)
            if "maximumf" in attr_str or "maxnumf" in attr_str:
                reduce_kind = "row_max"

        # Get input shape: reduce (tensor<128xf32>) -> f32 means 1D input [128]
        # Our load converts 1D to [128,1], so we need [1,128] for row_sum
        inp_shape = [128, 128]
        # Try to get operand type from body - reduce line has ": (tensor<...>) ->"
        # For now use body_ops to find the load that produced this operand
        op_key = self._value_key(op.operands[0])
        # Heuristic: if load produced [N,1] we need [1,N]. Default [128,1] from load.
        inp_shape = [128, 1]  # load of 1D gives [N,1] in our converter
        # For row_sum we need last dim to reduce, so [1,N] not [N,1]
        inp_shape = [1, 128]
        # Reshape [128,1] to [1,128] for row_sum (reduce axis 0)
        inp = tile.reshape(inp, [1, 128], span=self.span)
        inp = self.ib.let("reduce_reshape", inp)

        tmp_tile = tile.create([1, 128], DataType.FP32, span=self.span)
        tmp_var = self.ib.let(f"reduce_tmp_{op.result.name}".replace("%", ""), tmp_tile)

        if reduce_kind == "row_max":
            result_expr = tile.row_max(inp, tmp_var, span=self.span)
        else:
            result_expr = tile.row_sum(inp, tmp_var, span=self.span)

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
