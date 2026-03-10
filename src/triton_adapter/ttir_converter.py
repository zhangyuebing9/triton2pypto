"""TTIR to PyPTO IR converter implementation.

This module provides the main conversion logic from Triton TTIR to PyPTO IR.
Uses lightweight IR abstractions for CPU-only execution without full PyPTO.
"""

from dataclasses import dataclass
from typing import Any

from .ir import (
    AddExpr,
    ConstExpr,
    DataType,
    DivExpr,
    Function,
    LetStmt,
    LoadExpr,
    MulExpr,
    Program,
    Span,
    Stmt,
    StoreStmt,
    SubExpr,
    Var,
    VarExpr,
)


class ConversionError(Exception):
    """Raised when TTIR to PyPTO conversion fails."""

    def __init__(
        self,
        message: str,
        op_name: str | None = None,
        span: Span | None = None,
    ):
        self.op_name = op_name
        self.span = span

        location = "<unknown>"
        if span and span != Span.unknown():
            location = f"{span.filename}:{span.line}:{span.column}"

        full_message = f"[{location}] {message}"
        if op_name:
            full_message += f"\n  Operation: {op_name}"

        super().__init__(full_message)


class UnsupportedOpError(ConversionError):
    """Raised when encountering an unsupported TTIR operation."""

    def __init__(self, op_name: str, span: Span | None = None):
        message = f"Unsupported operation: {op_name}"
        super().__init__(message, op_name=op_name, span=span)
        self.suggestion = f"Operation '{op_name}' is not yet supported. Please check the supported operations list."


@dataclass
class BlockPtrInfo:
    """Information about a block pointer in TTIR."""

    base_var: str
    shape: list[VarExpr]
    strides: list[VarExpr]
    current_offset: list[VarExpr]


class SpanTracker:
    """Tracks source code location from TTIR operations."""

    def get_span(self, op: Any) -> Span:
        """Extract source location from TTIR operation."""
        if hasattr(op, "attributes") and op.attributes and "location" in op.attributes:
            loc = op.attributes["location"]
            if isinstance(loc, str):
                parts = loc.split(":")
                if len(parts) >= 3:
                    try:
                        return Span(parts[0], int(parts[1]), int(parts[2]))
                    except (ValueError, IndexError):
                        pass
        return Span.unknown()


class TypeMapper:
    """Maps TTIR types to PyPTO/Adapter types."""

    DTYPE_MAP: dict[str, DataType] = {
        "i1": DataType.BOOL,
        "i8": DataType.INT8,
        "i16": DataType.INT16,
        "i32": DataType.INT32,
        "i64": DataType.INT64,
        "fp16": DataType.FP16,
        "bf16": DataType.BF16,
        "fp32": DataType.FP32,
        "fp64": DataType.FP64,
    }

    def map_dtype(self, ttir_dtype: str) -> DataType:
        """Map TTIR dtype string to DataType."""
        dtype_str = str(ttir_dtype).lower()
        if dtype_str in self.DTYPE_MAP:
            return self.DTYPE_MAP[dtype_str]
        raise ConversionError(f"Unsupported dtype: {ttir_dtype}")

    def _extract_dtype_from_type_str(self, type_str: str) -> DataType:
        """Extract dtype from MLIR type string like 'f32', 'tensor<128xf32>'."""
        type_str = str(type_str).lower()
        for suffix in ["f32", "f16", "bf16", "f64", "i1", "i8", "i16", "i32", "i64"]:
            if suffix in type_str:
                mapping = {
                    "f32": DataType.FP32,
                    "f16": DataType.FP16,
                    "bf16": DataType.BF16,
                    "f64": DataType.FP64,
                    "i1": DataType.BOOL,
                    "i8": DataType.INT8,
                    "i16": DataType.INT16,
                    "i32": DataType.INT32,
                    "i64": DataType.INT64,
                }
                return mapping.get(suffix, DataType.FP32)
        return DataType.FP32

    def extract_shape_from_type(self, type_str: str) -> tuple[int, ...]:
        """Extract shape from tensor type like 'tensor<128xf32>' or 'tensor<16x16xf32>'."""
        type_str = str(type_str)
        if "tensor<" not in type_str:
            return ()
        try:
            start = type_str.index("<") + 1
            end = type_str.rindex(">")
            inner = type_str[start:end]
            parts = inner.split("x")
            shape = []
            for p in parts:
                p = p.strip()
                if p and p != "?" and p[0].isdigit():
                    shape.append(int(p))
            return tuple(shape) if shape else (1,)
        except (ValueError, IndexError):
            return (1,)


class IRBuilder:
    """Builds adapter IR Program/Function/Stmt."""

    def __init__(self) -> None:
        self._program: Program | None = None
        self._current_function: Function | None = None
        self._current_body: list[Stmt] = []

    def program(self, name: str = "main") -> "IRBuilder":
        """Start building a program."""
        self._program = Program(name=name)
        return self

    def function(self, name: str, params: list[Var] | None = None) -> "IRBuilder":
        """Start building a function."""
        self._current_function = Function(
            name=name,
            params=params or [],
            body=[],
        )
        self._current_body = self._current_function.body
        return self

    def emit(self, stmt: Stmt) -> None:
        """Emit a statement."""
        if self._current_body is not None:
            self._current_body.append(stmt)

    def let(
        self,
        name: str,
        expr: VarExpr | ConstExpr | AddExpr | SubExpr | MulExpr | DivExpr | LoadExpr,
        dtype: DataType | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> Var:
        """Create a let binding and return the variable."""
        var = Var(name=name, dtype=dtype, shape=shape)
        self.emit(LetStmt(name=name, expr=expr, dtype=dtype, shape=shape))
        return var

    def get_program(self) -> Program:
        """Get the built program."""
        if self._program is None:
            self._program = Program()
        if self._current_function is not None:
            self._program.functions.append(self._current_function)
        return self._program


class TTIRToPyptoConverter:
    """Converts Triton TTIR to adapter IR (executable on CPU via NumPy runtime)."""

    SUPPORTED_OPS = {
        "tt.make_block_ptr",
        "tt.advance",
        "tt.load",
        "tt.store",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.constant",
        "tt.exp",
        "arith.cmpf",
        "arith.select",
        "tt.program_id",
    }

    def __init__(self) -> None:
        self.ib = IRBuilder()
        self.type_mapper = TypeMapper()
        self.span_tracker = SpanTracker()
        self.value_map: dict[str, Var | VarExpr | ConstExpr] = {}
        self.block_ptr_map: dict[str, BlockPtrInfo] = {}
        self._tmp_counter = 0
        self._func_name = "kernel"

    def _tmp_id(self) -> int:
        self._tmp_counter += 1
        return self._tmp_counter

    def _op_key(self, op: Any) -> str:
        """Get unique key for operation result (for value_map)."""
        if hasattr(op, "result") and op.result:
            return op.result.name if hasattr(op.result, "name") else str(op.result)
        return f"_tmp_{id(op)}"

    def _get_operand_value(self, operand: Any) -> Var | VarExpr | ConstExpr:
        """Resolve operand to Var or Expr."""
        key = operand.name if hasattr(operand, "name") else str(operand)
        if key in self.value_map:
            return self.value_map[key]
        return VarExpr(name=key)

    def _get_result_dtype(self, op: Any) -> DataType:
        """Extract dtype from operation result type."""
        if hasattr(op, "result_types") and op.result_types:
            t = op.result_types[0]
            if hasattr(t, "get_element_type"):
                return self.type_mapper._extract_dtype_from_type_str(
                    t.get_element_type() or "f32"
                )
            if hasattr(t, "type_str"):
                return self.type_mapper._extract_dtype_from_type_str(t.type_str)
        return DataType.FP32

    def _get_result_shape(self, op: Any) -> tuple[int, ...]:
        """Extract shape from operation result type."""
        if hasattr(op, "result_types") and op.result_types:
            t = op.result_types[0]
            if hasattr(t, "get_shape") and t.get_shape():
                return tuple(t.get_shape())
            if hasattr(t, "type_str"):
                return self.type_mapper.extract_shape_from_type(t.type_str)
        return (1,)

    def convert(self, ttir_module: Any) -> Program:
        """Convert TTIR module to Program.

        Args:
            ttir_module: List of MLIROperation from parse_ttir, or module name for program.

        Returns:
            Program with converted functions.
        """
        self.ib.program("main")

        if isinstance(ttir_module, list):
            ops = ttir_module
            func_name = "kernel"
        else:
            ops = getattr(ttir_module, "operations", [])
            if not ops and hasattr(ttir_module, "body"):
                ops = getattr(ttir_module.body, "operations", [])
            func_name = getattr(ttir_module, "name", "kernel") or "kernel"

        self._func_name = func_name

        # Infer params (arg0, arg1, ...) from operands
        param_names: set[str] = set()
        for op in ops:
            for operand in getattr(op, "operands", []):
                name = getattr(operand, "name", str(operand))
                if name and name.startswith("arg") and name[3:].isdigit():
                    param_names.add(name)
                elif name and name.startswith("arg"):
                    param_names.add(name)
        def _arg_sort_key(n: str) -> int:
            suffix = n[3:] if len(n) > 3 and n.startswith("arg") else ""
            return int(suffix) if suffix.isdigit() else 999

        params = [Var(name=p) for p in sorted(param_names, key=_arg_sort_key)]

        self.ib.function(func_name, params)
        for p in params:
            self.value_map[p.name] = p

        # Convert operations
        for op in ops:
            op_name = getattr(op, "name", "")
            if not op_name or "func" in op_name or "module" in op_name:
                continue
            if op_name in self.SUPPORTED_OPS:
                self.convert_operation(op)

        return self.ib.get_program()

    def convert_operation(self, op: Any) -> None:
        """Dispatch TTIR operation to conversion handler."""
        span = self.span_tracker.get_span(op)
        op_name = getattr(op, "name", "") or "unknown"

        if op_name not in self.SUPPORTED_OPS:
            raise UnsupportedOpError(op_name, span)

        handler_name = f"_convert_{op_name.replace('.', '_')}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            raise ConversionError(f"No handler for operation: {op_name}", op_name=op_name, span=span)

        handler(op, span)

    def _convert_arith_constant(self, op: Any, span: Span) -> None:
        """Convert arith.constant."""
        key = self._op_key(op)
        val: float | int = 0.0
        dtype = self._get_result_dtype(op)

        if hasattr(op, "literal_value") and op.literal_value is not None:
            val = op.literal_value
        elif hasattr(op, "attributes") and op.attributes:
            for k, v in op.attributes.items():
                if "value" in k.lower() or k == "value":
                    try:
                        val = float(v) if "f" in str(dtype.value) else int(v)
                    except (ValueError, TypeError):
                        val = 0.0
                    break

        const_expr = ConstExpr(value=val, dtype=dtype)
        var = Var(name=key, dtype=dtype, value=val)
        self.value_map[key] = var
        self.ib.emit(LetStmt(name=key, expr=const_expr, dtype=dtype))

    def _convert_arith_addf(self, op: Any, span: Span) -> None:
        """Convert arith.addf."""
        key = self._op_key(op)
        ops_list = getattr(op, "operands", [])
        if len(ops_list) < 2:
            raise ConversionError("arith.addf requires 2 operands", op_name=op.name, span=span)

        left = self._get_operand_value(ops_list[0])
        right = self._get_operand_value(ops_list[1])
        if isinstance(left, Var):
            left = VarExpr(name=left.name)
        if isinstance(right, Var):
            right = VarExpr(name=right.name)

        dtype = self._get_result_dtype(op)
        shape = self._get_result_shape(op)
        expr = AddExpr(left=left, right=right, dtype=dtype)
        var = self.ib.let(key, expr, dtype=dtype, shape=shape)
        self.value_map[key] = var

    def _convert_arith_subf(self, op: Any, span: Span) -> None:
        """Convert arith.subf."""
        key = self._op_key(op)
        ops_list = getattr(op, "operands", [])
        if len(ops_list) < 2:
            raise ConversionError("arith.subf requires 2 operands", op_name=op.name, span=span)

        left = self._get_operand_value(ops_list[0])
        right = self._get_operand_value(ops_list[1])
        if isinstance(left, Var):
            left = VarExpr(name=left.name)
        if isinstance(right, Var):
            right = VarExpr(name=right.name)

        dtype = self._get_result_dtype(op)
        shape = self._get_result_shape(op)
        expr = SubExpr(left=left, right=right, dtype=dtype)
        var = self.ib.let(key, expr, dtype=dtype, shape=shape)
        self.value_map[key] = var

    def _convert_arith_mulf(self, op: Any, span: Span) -> None:
        """Convert arith.mulf."""
        key = self._op_key(op)
        ops_list = getattr(op, "operands", [])
        if len(ops_list) < 2:
            raise ConversionError("arith.mulf requires 2 operands", op_name=op.name, span=span)

        left = self._get_operand_value(ops_list[0])
        right = self._get_operand_value(ops_list[1])
        if isinstance(left, Var):
            left = VarExpr(name=left.name)
        if isinstance(right, Var):
            right = VarExpr(name=right.name)

        dtype = self._get_result_dtype(op)
        shape = self._get_result_shape(op)
        expr = MulExpr(left=left, right=right, dtype=dtype)
        var = self.ib.let(key, expr, dtype=dtype, shape=shape)
        self.value_map[key] = var

    def _convert_arith_divf(self, op: Any, span: Span) -> None:
        """Convert arith.divf."""
        key = self._op_key(op)
        ops_list = getattr(op, "operands", [])
        if len(ops_list) < 2:
            raise ConversionError("arith.divf requires 2 operands", op_name=op.name, span=span)

        left = self._get_operand_value(ops_list[0])
        right = self._get_operand_value(ops_list[1])
        if isinstance(left, Var):
            left = VarExpr(name=left.name)
        if isinstance(right, Var):
            right = VarExpr(name=right.name)

        dtype = self._get_result_dtype(op)
        shape = self._get_result_shape(op)
        expr = DivExpr(left=left, right=right, dtype=dtype)
        var = self.ib.let(key, expr, dtype=dtype, shape=shape)
        self.value_map[key] = var

    def _convert_tt_make_block_ptr(self, op: Any, span: Span) -> None:
        """Convert tt.make_block_ptr - track block pointer info."""
        key = self._op_key(op)
        ops_list = getattr(op, "operands", [])

        if len(ops_list) < 4:
            raise ConversionError(
                "tt.make_block_ptr requires base, shape, strides, offsets",
                op_name=op.name,
                span=span,
            )

        base = self._get_operand_value(ops_list[0])
        base_var = base.name if hasattr(base, "name") else str(ops_list[0])
        if isinstance(base, Var):
            base_var = base.name

        # For simplicity, use VarExpr for shape/strides/offsets
        shape_exprs = [VarExpr(str(o.name)) if hasattr(o, "name") else VarExpr(f"op{i}") for i, o in enumerate(ops_list[1:4])]
        strides = shape_exprs[1] if len(shape_exprs) > 1 else []
        offsets = shape_exprs[2] if len(shape_exprs) > 2 else []

        if isinstance(strides, VarExpr):
            strides = [strides]
        if isinstance(offsets, VarExpr):
            offsets = [offsets]

        def _expr(o: Any, i: int) -> VarExpr:
            return VarExpr(o.name) if hasattr(o, "name") else VarExpr(f"v{i}")

        info = BlockPtrInfo(
            base_var=base_var,
            shape=[_expr(ops_list[1], 0)] if len(ops_list) > 1 else [],
            strides=[_expr(ops_list[2], 1)] if len(ops_list) > 2 else [],
            current_offset=[_expr(ops_list[3], 2)] if len(ops_list) > 3 else [],
        )
        self.block_ptr_map[key] = info
        self.value_map[key] = Var(name=key)

    def _convert_tt_advance(self, op: Any, span: Span) -> None:
        """Convert tt.advance - update block pointer offsets."""
        key = self._op_key(op)
        ops_list = getattr(op, "operands", [])
        if len(ops_list) < 2:
            raise ConversionError("tt.advance requires ptr and delta", op_name=op.name, span=span)

        ptr_key = ops_list[0].name if hasattr(ops_list[0], "name") else str(ops_list[0])
        if ptr_key not in self.block_ptr_map:
            # Create minimal block ptr for load without make_block_ptr
            self.block_ptr_map[key] = BlockPtrInfo(
                base_var=ptr_key,
                shape=[],
                strides=[],
                current_offset=[],
            )
            self.value_map[key] = Var(name=key)
            return

        old_info = self.block_ptr_map[ptr_key]
        delta = [VarExpr(o.name) if hasattr(o, "name") else VarExpr(f"d{i}") for i, o in enumerate(ops_list[1:])]
        new_info = BlockPtrInfo(
            base_var=old_info.base_var,
            shape=old_info.shape,
            strides=old_info.strides,
            current_offset=delta if delta else old_info.current_offset,
        )
        self.block_ptr_map[key] = new_info
        self.value_map[key] = Var(name=key)

    def _convert_tt_load(self, op: Any, span: Span) -> None:
        """Convert tt.load -> LoadExpr (conceptual load for NumPy runtime)."""
        key = self._op_key(op)
        ops_list = getattr(op, "operands", [])

        if not ops_list:
            raise ConversionError("tt.load requires pointer operand", op_name=op.name, span=span)

        ptr_operand = ops_list[0]
        ptr_key = ptr_operand.name if hasattr(ptr_operand, "name") else str(ops_list[0])
        ptr_info = self.block_ptr_map.get(ptr_key)

        shape = self._get_result_shape(op)
        dtype = self._get_result_dtype(op)

        offsets = []
        if ptr_info:
            offsets = [VarExpr(o) if isinstance(o, str) else o for o in ptr_info.current_offset]

        base_var = ptr_info.base_var if ptr_info else ptr_key
        load_expr = LoadExpr(
            ptr_var=base_var,
            offsets=offsets,
            shape=shape,
            dtype=dtype,
        )
        var = self.ib.let(key, load_expr, dtype=dtype, shape=shape)
        self.value_map[key] = var

    def _convert_tt_store(self, op: Any, span: Span) -> None:
        """Convert tt.store -> StoreStmt."""
        ops_list = getattr(op, "operands", [])
        if len(ops_list) < 2:
            raise ConversionError("tt.store requires ptr and value", op_name=op.name, span=span)

        ptr_operand = ops_list[0]
        val_operand = ops_list[1]
        ptr_key = ptr_operand.name if hasattr(ptr_operand, "name") else f"arg{0}"
        val_key = val_operand.name if hasattr(val_operand, "name") else str(val_operand)

        ptr_info = self.block_ptr_map.get(ptr_key)
        base_var = ptr_info.base_var if ptr_info else ptr_key
        offsets = []
        if ptr_info:
            offsets = ptr_info.current_offset

        self.ib.emit(StoreStmt(ptr_var=base_var, value_var=val_key, offsets=offsets))

    def _convert_tt_exp(self, op: Any, span: Span) -> None:
        """Convert tt.exp - for Phase 1, raise as not yet implemented for NumPy path."""
        raise ConversionError(
            "tt.exp conversion not yet implemented for CPU runtime",
            op_name=op.name,
            span=span,
        )

    def _convert_arith_cmpf(self, op: Any, span: Span) -> None:
        """Convert arith.cmpf - deferred."""
        raise ConversionError(
            "arith.cmpf conversion not yet implemented",
            op_name=op.name,
            span=span,
        )

    def _convert_arith_select(self, op: Any, span: Span) -> None:
        """Convert arith.select - deferred."""
        raise ConversionError(
            "arith.select conversion not yet implemented",
            op_name=op.name,
            span=span,
        )

    def _convert_tt_program_id(self, op: Any, span: Span) -> None:
        """Convert tt.program_id -> param placeholder."""
        key = self._op_key(op)
        axis = 0
        if hasattr(op, "attributes") and op.attributes:
            axis = int(op.attributes.get("axis", 0))
        var = Var(name=f"pid_{axis}")
        self.value_map[key] = var
