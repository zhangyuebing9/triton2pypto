"""Triton IR extraction utilities.

Uses Triton from third_party submodule to extract TTIR/TTGIR from compiled kernels.
"""

from typing import Any

try:
    import triton
except ImportError:
    triton = None  # type: ignore


class IRExtractionError(Exception):
    """Raised when IR extraction fails."""


def extract_ttir(kernel: Any, *args: Any, **kwargs: Any) -> str:
    """Extract TTIR (Triton Tensor IR) text from a compiled Triton kernel.

    The kernel must be compiled first by running it with concrete arguments.
    Use TRITON_INTERPRET=1 for CPU-only execution without GPU.

    Args:
        kernel: A Triton JITFunction (e.g. @triton.jit decorated function).
        *args: Arguments to run/compile the kernel with (tensor args, etc.).
        **kwargs: Keyword arguments for kernel launch (e.g. grid=...).

    Returns:
        TTIR MLIR text as string.

    Raises:
        IRExtractionError: If Triton is not installed or extraction fails.

    Example:
        >>> @triton.jit
        ... def add_kernel(x, y, out, n: tl.constexpr):
        ...     idx = tl.program_id(0) * 128 + tl.arange(0, 128)
        ...     mask = idx < n
        ...     a = tl.load(x + idx, mask=mask)
        ...     b = tl.load(y + idx, mask=mask)
        ...     tl.store(out + idx, a + b, mask=mask)
        >>> x = torch.randn(256)
        >>> y = torch.randn(256)
        >>> out = torch.empty(256)
        >>> ttir_text = extract_ttir(add_kernel, x, y, out, 256)
    """
    if triton is None:
        raise IRExtractionError(
            "Triton is not installed. Please install triton to use IR extraction."
        )

    try:
        # Launch kernel to trigger compilation; returns compiled binary with asm
        grid = kwargs.get("grid", (1,))

        # Call kernel to compile; use minimal grid for compilation
        # kernel[(grid)](*args, **kwargs) compiles and runs; we need the compiled object
        compiled = kernel[grid](*args, **kwargs)

        if not hasattr(compiled, "asm") or "ttir" not in compiled.asm:
            raise IRExtractionError(
                "Compiled kernel does not expose ttir. "
                "Ensure Triton version supports asm['ttir']."
            )

        return compiled.asm["ttir"]

    except IRExtractionError:
        raise
    except Exception as e:
        raise IRExtractionError(f"Failed to extract TTIR: {e}") from e


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
        raise IRExtractionError(
            "Triton is not installed. Please install triton to use IR extraction."
        )

    try:
        compiled = kernel[kwargs.get("grid", (1,))](*args, **kwargs)
        if not hasattr(compiled, "asm") or "ttgir" not in compiled.asm:
            raise IRExtractionError("Compiled kernel does not expose ttgir.")
        return compiled.asm["ttgir"]
    except IRExtractionError:
        raise
    except Exception as e:
        raise IRExtractionError(f"Failed to extract TTGIR: {e}") from e
