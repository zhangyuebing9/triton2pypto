"""Triton IR extraction utilities."""

from typing import Any

import triton
from triton import ir as tir


class IRExtractionError(Exception):
    """Raised when IR extraction fails."""


def extract_ttir(kernel: triton.JITFunction, *args: Any, **kwargs: Any) -> tir.Module:
    """Extract TTIR (Triton Tensor IR) from a compiled kernel.

    Args:
        kernel: A Triton JITFunction.
        *args: Arguments to compile the kernel with.
        **kwargs: Keyword arguments for compilation.

    Returns:
        The extracted TTIR module.

    Raises:
        IRExtractionError: If extraction fails.
    """
    raise NotImplementedError("IR extraction not yet implemented")


def extract_ttgir(kernel: triton.JITFunction, *args: Any, **kwargs: Any) -> tir.Module:
    """Extract TTGIR (Triton GPU IR) from a compiled kernel.

    Args:
        kernel: A Triton JITFunction.
        *args: Arguments to compile the kernel with.
        **kwargs: Keyword arguments for compilation.

    Returns:
        The extracted TTGIR module.

    Raises:
        IRExtractionError: If extraction fails.
    """
    raise NotImplementedError("IR extraction not yet implemented")