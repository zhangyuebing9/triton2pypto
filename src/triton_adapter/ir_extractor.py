"""Triton IR extraction utilities.

Uses Triton from third_party submodule to extract TTIR/TTGIR from compiled kernels.
"""

from typing import Any

try:
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
except ImportError:
    triton = None  # type: ignore
    ASTSource = None  # type: ignore
    GPUTarget = None  # type: ignore


class IRExtractionError(Exception):
    """Raised when IR extraction fails."""


def _infer_signature_and_constexprs(kernel: Any, args: tuple, kwargs: dict) -> tuple[dict, dict]:
    """Infer ASTSource signature and constexprs from kernel and call args."""
    import torch

    sig: dict[str, str] = {}
    constexprs: dict[str, Any] = {}
    arg_names = getattr(kernel, "arg_names", []) or []

    for i, name in enumerate(arg_names):
        if i >= len(args):
            break
        val = args[i]
        if type(val).__name__ == "constexpr":
            constexprs[name] = val.value
            sig[name] = "i32"  # constexpr often used as size
        elif hasattr(val, "data_ptr") and hasattr(val, "dtype"):
            # torch.Tensor
            dt = val.dtype
            if dt == torch.float32:
                sig[name] = "*fp32"
            elif dt == torch.float16:
                sig[name] = "*fp16"
            elif dt == torch.bfloat16:
                sig[name] = "*bf16"
            elif dt == torch.int32:
                sig[name] = "*i32"
            elif dt == torch.int64:
                sig[name] = "*i64"
            else:
                sig[name] = "*fp32"  # default
        elif isinstance(val, int):
            constexprs[name] = val
            sig[name] = "i32"
        else:
            sig[name] = "*fp32"  # fallback for pointers
    return sig, constexprs


def extract_ttir(kernel: Any, *args: Any, **kwargs: Any) -> str:
    """Extract TTIR (Triton Tensor IR) text from a Triton kernel.

    Two extraction paths:
    1. Run path: kernel[grid](*args) when GPU/driver is available - returns asm["ttir"]
    2. Compile-only path: triton.compile(ASTSource(...), target=GPUTarget(...)) when no GPU
       (e.g. TRITON_INTERPRET=1 or CI without GPU). Infers signature from args.

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
        ...     ...
        >>> ttir_text = extract_ttir(add_kernel, x, y, out, 256)
    """
    if triton is None:
        raise IRExtractionError(
            "Triton is not installed. Please install triton to use IR extraction."
        )

    grid = kwargs.get("grid", (1,))

    # Path 1: Try run to get compiled kernel (requires GPU)
    try:
        compiled = kernel[grid](*args, **kwargs)
        if compiled is not None and hasattr(compiled, "asm") and compiled.asm and "ttir" in compiled.asm:
            return compiled.asm["ttir"]
    except (RuntimeError, Exception):
        pass  # Fall through to compile-only path

    # Path 2: Compile-only (no GPU / TRITON_INTERPRET)
    sig, constexprs = _infer_signature_and_constexprs(kernel, args, kwargs)
    src = ASTSource(fn=kernel, signature=sig, constexprs=constexprs)
    target = GPUTarget("cuda", 80, 32)  # SM80 works for most ops
    compiled = triton.compile(src, target=target)
    if hasattr(compiled, "asm") and "ttir" in compiled.asm:
        return compiled.asm["ttir"]

    raise IRExtractionError(
        "Failed to extract TTIR. Neither run nor compile-only path produced ttir."
    )


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
