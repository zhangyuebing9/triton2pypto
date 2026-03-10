"""Triton IR extraction utilities."""

from typing import Any

try:
    import triton
    from triton._C.libtriton import ir as tir
except ImportError:
    triton = None  # type: ignore
    tir = None  # type: ignore


class IRExtractionError(Exception):
    """Raised when IR extraction fails."""


def extract_ttir(kernel: Any, *args: Any, **kwargs: Any) -> str:
    """Extract TTIR (Triton Tensor IR) text from a kernel.

    Compiles the kernel with the given arguments and returns the TTIR MLIR text.
    In CPU-only environments, compilation may fail; use convert_ttir_to_pypto(ttir_text)
    with manually obtained TTIR instead.

    Args:
        kernel: A Triton JITFunction (decorated with @triton.jit).
        *args: Example arguments to specialize the kernel.
        **kwargs: Optional attrs, options for compilation.

    Returns:
        TTIR MLIR text as string.

    Raises:
        IRExtractionError: If extraction fails.
    """
    if triton is None:
        raise IRExtractionError("Triton is not installed. Please install triton to use IR extraction.")

    try:
        from triton.compiler import ASTSource, compile
        from triton.runtime.driver import driver

        target = driver.active.get_current_target()
        signature = getattr(kernel, "signature", {})
        if not signature and hasattr(kernel, "params"):
            signature = {p.name: str(type(None)) for p in kernel.params}
        constexprs = {}
        attrs = kwargs.get("attrs", {})
        src = ASTSource(kernel, signature, constexprs, attrs)
        compiled = compile(src, target=target, options=kwargs.get("options", {}))
        ttir_text = compiled.asm.get("ttir", "")
        if not ttir_text:
            raise IRExtractionError("TTIR not in compiled output.")
        return ttir_text if isinstance(ttir_text, str) else ttir_text.decode("utf-8")
    except IRExtractionError:
        raise
    except Exception as e:
        raise IRExtractionError(
            f"IR extraction failed: {e}. "
            "In CPU-only environments, use convert_ttir_to_pypto(ttir_text) with TTIR from a .ttir.mlir file."
        ) from e


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
    raise NotImplementedError("TTGIR extraction not yet implemented")
