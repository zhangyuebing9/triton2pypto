"""TTIR to PyPTO IR converter implementation.

This module provides the main conversion logic from Triton TTIR to PyPTO IR.
"""

from dataclasses import dataclass
from typing import Any

from pypto import DataType, ir


class ConversionError(Exception):
    """Raised when TTIR to PyPTO conversion fails."""

    def __init__(self, message: str, op_name: str | None = None, span: ir.Span | None = None):
        self.op_name = op_name
        self.span = span

        location = "<unknown>"
        if span and span != ir.Span.unknown():
            location = f"{span.filename}:{span.line}:{span.column}"

        full_message = f"[{location}] {message}"
        if op_name:
            full_message += f"\n  Operation: {op_name}"

        super().__init__(full_message)


class UnsupportedOpError(ConversionError):
    """Raised when encountering an unsupported TTIR operation."""

    def __init__(self, op_name: str, span: ir.Span | None = None):
        message = f"Unsupported operation: {op_name}"
        suggestion = f"Operation '{op_name}' is not yet supported. Please check the supported operations list."
        super().__init__(message, op_name=op_name, span=span)
        self.suggestion = suggestion


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
        """Extract source location from TTIR operation.

        Args:
            op: TTIR operation with optional location attribute.

        Returns:
            PyPTO Span object with source location, or unknown span.
        """
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
        "fp32": DataType.FP32,
        "fp64": DataType.FP64,
    }

    def map_dtype(self, ttir_dtype: str) -> DataType:
        """Map TTIR dtype string to PyPTO DataType.

        Args:
            ttir_dtype: TTIR data type string (e.g., "fp32", "i32").

        Returns:
            Corresponding PyPTO DataType.

        Raises:
            ConversionError: If dtype is not supported.
        """
        dtype_str = str(ttir_dtype).lower()
        if dtype_str in self.DTYPE_MAP:
            return self.DTYPE_MAP[dtype_str]

        raise ConversionError(f"Unsupported dtype: {ttir_dtype}")

    def map_tensor_type(self, shape: list[int], dtype: DataType) -> ir.TensorType:
        """Create PyPTO TensorType from shape and dtype.

        Args:
            shape: Tensor shape as list of integers.
            dtype: PyPTO DataType.

        Returns:
            PyPTO TensorType.
        """
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
        self.ib = ir.IRBuilder()
        self.type_mapper = TypeMapper()
        self.span_tracker = SpanTracker()
        self.value_map: dict[Any, ir.Var] = {}
        self.block_ptr_map: dict[Any, BlockPtrInfo] = {}
        self._tmp_counter = 0

    def _tmp_id(self) -> int:
        """Generate a unique temporary ID."""
        self._tmp_counter += 1
        return self._tmp_counter

    def convert(self, ttir_module: Any) -> ir.Program:
        """Convert TTIR module to PyPTO Program.

        Args:
            ttir_module: TTIR module (MLIR ModuleOp or parsed MLIR text).

        Returns:
            PyPTO Program object.

        Raises:
            ConversionError: If conversion fails.
        """
        raise NotImplementedError("TTIR conversion not yet implemented")

    def convert_function(self, func: Any) -> ir.Function:
        """Convert TTIR function to PyPTO Function.

        Args:
            func: TTIR FuncOp.

        Returns:
            PyPTO Function object.
        """
        raise NotImplementedError("Function conversion not yet implemented")

    def convert_operation(self, op: Any) -> None:
        """Dispatch TTIR operation to appropriate conversion handler.

        Args:
            op: TTIR operation.

        Raises:
            UnsupportedOpError: If operation is not supported.
        """
        span = self.span_tracker.get_span(op)
        op_name = getattr(op, "name", None) or getattr(op, "operation", {}).get("name", "unknown")

        if op_name not in self.SUPPORTED_OPS:
            raise UnsupportedOpError(op_name, span)

        handler_name = f"_convert_{op_name.replace('.', '_')}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            raise ConversionError(f"No handler for operation: {op_name}", op_name=op_name, span=span)

        handler(op, span)

    def _convert_tt_make_block_ptr(self, op: Any, span: ir.Span) -> None:
        """Convert tt.make_block_ptr operation."""
        raise NotImplementedError("tt.make_block_ptr conversion not yet implemented")

    def _convert_tt_advance(self, op: Any, span: ir.Span) -> None:
        """Convert tt.advance operation."""
        raise NotImplementedError("tt.advance conversion not yet implemented")

    def _convert_tt_load(self, op: Any, span: ir.Span) -> None:
        """Convert tt.load operation."""
        raise NotImplementedError("tt.load conversion not yet implemented")

    def _convert_tt_store(self, op: Any, span: ir.Span) -> None:
        """Convert tt.store operation."""
        raise NotImplementedError("tt.store conversion not yet implemented")

    def _convert_arith_addf(self, op: Any, span: ir.Span) -> None:
        """Convert arith.addf operation."""
        raise NotImplementedError("arith.addf conversion not yet implemented")

    def _convert_arith_subf(self, op: Any, span: ir.Span) -> None:
        """Convert arith.subf operation."""
        raise NotImplementedError("arith.subf conversion not yet implemented")

    def _convert_arith_mulf(self, op: Any, span: ir.Span) -> None:
        """Convert arith.mulf operation."""
        raise NotImplementedError("arith.mulf conversion not yet implemented")

    def _convert_arith_divf(self, op: Any, span: ir.Span) -> None:
        """Convert arith.divf operation."""
        raise NotImplementedError("arith.divf conversion not yet implemented")

    def _convert_arith_constant(self, op: Any, span: ir.Span) -> None:
        """Convert arith.constant operation."""
        raise NotImplementedError("arith.constant conversion not yet implemented")

    def _convert_tt_exp(self, op: Any, span: ir.Span) -> None:
        """Convert tt.exp operation."""
        raise NotImplementedError("tt.exp conversion not yet implemented")

    def _convert_arith_cmpf(self, op: Any, span: ir.Span) -> None:
        """Convert arith.cmpf operation."""
        raise NotImplementedError("arith.cmpf conversion not yet implemented")

    def _convert_arith_select(self, op: Any, span: ir.Span) -> None:
        """Convert arith.select operation."""
        raise NotImplementedError("arith.select conversion not yet implemented")

    def _convert_tt_program_id(self, op: Any, span: ir.Span) -> None:
        """Convert tt.program_id operation."""
        raise NotImplementedError("tt.program_id conversion not yet implemented")