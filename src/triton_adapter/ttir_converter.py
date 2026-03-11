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
        "tt.exp",
        "arith.cmpf",
        "arith.cmpi",
        "arith.select",
        "tt.program_id",
    }

    def __init__(self) -> None:
        self.ib = ir.IRBuilder()
        self.type_mapper = TypeMapper()
        self.span_tracker = SpanTracker()
        self.value_map: dict[str, ir.Var] = {}
        self.block_ptr_map: dict[str, BlockPtrInfo] = {}
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
        if not incore_func or len(arg_names) < 3:
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
        with self.ib.function(
            "main", type=ir.FunctionType.Orchestration
        ) as f:
            a = f.param("a", tensor_type)
            b = f.param("b", tensor_type)
            c = f.param("c", tensor_type)
            out = ir.Call(ir.GlobalVar(incore_func.name), [a, b, c], self.span)
            self.ib.return_stmt(out)
        return f.get_result()

    def _convert_op(self, op: MLIROperation) -> None:
        """Dispatch operation to handler."""
        self.span = self.span_tracker.get_span(op)
        if op.name not in self.SUPPORTED_OPS:
            handler_name = op.name.replace(".", "_")
            if not hasattr(self, f"_convert_{handler_name}"):
                raise UnsupportedOpError(op.name, self.span)
        handler = getattr(
            self,
            f"_convert_{op.name.replace('.', '_')}",
            self._convert_generic,
        )
        handler(op)

    def _convert_generic(self, op: MLIROperation) -> None:
        """Generic handler for unhandled ops."""
        raise UnsupportedOpError(op.name, self.span)

    def _convert_arith_constant(self, op: MLIROperation) -> None:
        """Convert arith.constant to ConstInt/ConstFloat."""
        if not op.result:
            return
        key = self._value_key(op.result)
        attr = op.attributes.get("value")
        if attr:
            val_str = str(attr).strip()
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

    def _convert_tt_load(self, op: MLIROperation) -> None:
        """Convert tt.load to tile.load."""
        if not op.result or not op.operands:
            return
        ptr = op.operands[0]
        tensor_var = self._get_operand(ptr)
        shape = [128, 128]
        if op.result_types:
            rt = op.result_types[0]
            if rt.is_tensor():
                s = rt.get_shape()
                if s:
                    shape = s
        if len(shape) == 1:
            shape = [shape[0], 1]
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
        """Convert arith.addi to tile.add (for integer tiles)."""
        self._convert_binary_op(op, "add", tile.add)

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
        """Convert arith.muli to tile.mul."""
        self._convert_binary_op(op, "mul", tile.mul)

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

    def _convert_arith_cmpf(self, op: MLIROperation) -> None:
        """Convert arith.cmpf - use tile.cmp."""
        if not op.result or len(op.operands) < 2:
            return
        # arith.cmpf predicate: olt, ole, oeq, etc. -> cmp_type 0-5
        pred_map = {"olt": 0, "ole": 1, "oeq": 2, "one": 3, "oge": 4, "ogt": 5}
        pred = op.attributes.get("predicate", "olt")
        cmp_type = pred_map.get(str(pred), 0)
        lhs = self._get_operand(op.operands[0])
        rhs = self._get_operand(op.operands[1])
        result_expr = tile.cmp(lhs, rhs, cmp_type=cmp_type, span=self.span)
        result_var = self.ib.let(
            f"cmp_{op.result.name}".replace("%", ""), result_expr
        )
        self.value_map[self._value_key(op.result)] = result_var

    def _convert_arith_cmpi(self, op: MLIROperation) -> None:
        """Convert arith.cmpi to tile.cmp."""
        self._convert_arith_cmpf(op)

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
        """Convert tt.program_id - create param placeholder."""
        if not op.result:
            return
        axis = op.attributes.get("axis", 0)
        var = self.ib.var(
            f"pid_{axis}",
            ir.ScalarType(DataType.INT64),
        )
        self.value_map[self._value_key(op.result)] = var
