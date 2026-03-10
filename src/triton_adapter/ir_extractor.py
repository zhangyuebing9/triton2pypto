"""Triton IR extraction utilities."""

from typing import Any

try:
    import triton
    from triton import ir as tir
except ImportError:
    triton = None  # type: ignore
    tir = None  # type: ignore


class IRExtractionError(Exception):
    """Raised when IR extraction fails."""


def extract_ttir(kernel: Any, *args: Any, **kwargs: Any) -> Any:
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
    if triton is None:
        raise IRExtractionError("Triton is not installed. Please install triton to use IR extraction.")
    raise NotImplementedError("IR extraction not yet implemented")


def extract_ttgir(kernel: Any, *args: Any, **kwargs: Any) -> Any:
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
    if triton is None:
        raise IRExtractionError("Triton is not installed. Please install triton to use IR extraction.")
    raise NotImplementedError("IR extraction not yet implemented")