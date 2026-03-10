"""NumPy-based runtime for executing converted IR on CPU.

Enables validation of TTIR→adapter conversion without NPU hardware.
"""

from typing import Any

import numpy as np

from .ir import (
    AddExpr,
    ConstExpr,
    DataType,
    DivExpr,
    LetStmt,
    LoadExpr,
    MulExpr,
    Program,
    StoreStmt,
    SubExpr,
    VarExpr,
)

# NumPy dtype mapping
DTYPE_TO_NP = {
    DataType.BOOL: np.bool_,
    DataType.INT8: np.int8,
    DataType.INT16: np.int16,
    DataType.INT32: np.int32,
    DataType.INT64: np.int64,
    DataType.FP16: np.float16,
    DataType.BF16: np.float16,
    DataType.FP32: np.float32,
    DataType.FP64: np.float64,
}


class NumPyRuntime:
    """Executes adapter IR using NumPy (CPU-only)."""

    def __init__(self) -> None:
        self._vars: dict[str, np.ndarray | float | int] = {}

    def run_program(
        self,
        program: Program,
        *args: np.ndarray,
        **kwargs: np.ndarray,
    ) -> np.ndarray | None:
        """Execute program with given input arrays.

        Args:
            program: Converted Program.
            *args: Input arrays in order (arg0, arg1, ...).

        Returns:
            Output array if function returns a value.
        """
        if not program.functions:
            return None

        func = program.functions[0]
        self._vars.clear()

        for i, param in enumerate(func.params):
            if i < len(args):
                self._vars[param.name] = args[i]
            elif f"arg{i}" in kwargs:
                self._vars[f"arg{i}"] = kwargs[f"arg{i}"]

        for i in range(max(len(args), 3)):
            arg_name = f"arg{i}"
            if arg_name not in self._vars and i < len(args):
                self._vars[arg_name] = args[i]

        for stmt in func.body:
            self._run_stmt(stmt)

        if func.return_var:
            return self._vars.get(func.return_var.name)

        return None

    def _run_stmt(self, stmt: Any) -> None:
        """Execute a single statement."""
        if isinstance(stmt, LetStmt):
            val = self._eval_expr(stmt.expr)
            self._vars[stmt.name] = val
        elif isinstance(stmt, StoreStmt):
            src = self._vars.get(stmt.value_var)
            dst_ref = stmt.ptr_var
            if dst_ref in self._vars and src is not None:
                dst = self._vars[dst_ref]
                if isinstance(dst, np.ndarray) and isinstance(src, np.ndarray):
                    np.copyto(dst, src)

    def _eval_expr(self, expr: Any) -> np.ndarray | float | int:
        """Evaluate expression to value."""
        if isinstance(expr, ConstExpr):
            return expr.value
        if isinstance(expr, VarExpr):
            return self._vars.get(expr.name, 0.0)
        if isinstance(expr, LoadExpr):
            ptr = self._vars.get(expr.ptr_var)
            if ptr is not None and isinstance(ptr, np.ndarray):
                return np.asarray(ptr, dtype=DTYPE_TO_NP.get(expr.dtype, np.float32))
            return np.zeros(expr.shape, dtype=DTYPE_TO_NP.get(expr.dtype, np.float32))
        if isinstance(expr, AddExpr):
            a, b = self._eval_expr(expr.left), self._eval_expr(expr.right)
            return np.add(np.asarray(a), np.asarray(b))
        if isinstance(expr, SubExpr):
            a, b = self._eval_expr(expr.left), self._eval_expr(expr.right)
            return np.subtract(np.asarray(a), np.asarray(b))
        if isinstance(expr, MulExpr):
            a, b = self._eval_expr(expr.left), self._eval_expr(expr.right)
            return np.multiply(np.asarray(a), np.asarray(b))
        if isinstance(expr, DivExpr):
            a, b = self._eval_expr(expr.left), self._eval_expr(expr.right)
            return np.divide(np.asarray(a), np.asarray(b))
        return 0.0


def run_on_numpy(program: Program, *args: np.ndarray) -> np.ndarray | None:
    """Convenience: run program with NumPy arrays."""
    return NumPyRuntime().run_program(program, *args)
