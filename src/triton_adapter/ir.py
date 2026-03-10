"""Lightweight IR abstractions for TTIR to PyPTO conversion.

This module provides minimal IR types for the conversion pipeline,
allowing the converter to run without the full PyPTO package.
The IR can be executed via the NumPy-based simpler runtime for CPU validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataType(Enum):
    """Data types for IR values."""

    BOOL = "bool"
    INT8 = "i8"
    INT16 = "i16"
    INT32 = "i32"
    INT64 = "i64"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"


@dataclass
class Span:
    """Source code location."""

    filename: str
    line: int
    column: int

    @staticmethod
    def unknown() -> "Span":
        """Create unknown span."""
        return Span("<unknown>", 0, 0)


@dataclass
class Var:
    """IR variable - holds a name and optional value."""

    name: str
    dtype: DataType | None = None
    shape: tuple[int, ...] | None = None
    value: Any = None  # For constants


@dataclass
class Program:
    """Top-level program containing functions."""

    name: str = "main"
    functions: list["Function"] = field(default_factory=list)


@dataclass
class Function:
    """Function with parameters and body."""

    name: str
    params: list[Var] = field(default_factory=list)
    body: list["Stmt"] = field(default_factory=list)
    return_var: Var | None = None


@dataclass
class Stmt:
    """Base class for statements."""

    pass


@dataclass
class LetStmt(Stmt):
    """Let binding: name = expr."""

    name: str
    expr: "Expr"
    dtype: DataType | None = None
    shape: tuple[int, ...] | None = None


@dataclass
class StoreStmt(Stmt):
    """Store tile to memory."""

    ptr_var: str
    value_var: str
    offsets: list["Expr"] = field(default_factory=list)


@dataclass
class Expr:
    """Base class for expressions."""

    pass


@dataclass
class ConstExpr(Expr):
    """Constant expression."""

    value: float | int | bool
    dtype: DataType


@dataclass
class VarExpr(Expr):
    """Variable reference."""

    name: str


@dataclass
class AddExpr(Expr):
    """Addition: a + b."""

    left: Expr
    right: Expr
    dtype: DataType


@dataclass
class SubExpr(Expr):
    """Subtraction: a - b."""

    left: Expr
    right: Expr
    dtype: DataType


@dataclass
class MulExpr(Expr):
    """Multiplication: a * b."""

    left: Expr
    right: Expr
    dtype: DataType


@dataclass
class DivExpr(Expr):
    """Division: a / b."""

    left: Expr
    right: Expr
    dtype: DataType


@dataclass
class LoadExpr(Expr):
    """Load from memory: load(ptr, offsets, shape, dtype)."""

    ptr_var: str
    offsets: list[Expr]
    shape: tuple[int, ...]
    dtype: DataType


@dataclass
class BlockPtrInfo:
    """Block pointer metadata for load/store."""

    base_var: str
    shape: list[Expr]
    strides: list[Expr]
    offsets: list[Expr]
