"""Public API for TTIR to PyPTO conversion."""

from __future__ import annotations

from pypto import ir

from .ttir_converter import TTIRToPyptoConverter


def convert_ttir_to_pypto(
    ttir_text: str,
    program_name: str = "converted_kernel",
    output_format: str = "object",
) -> ir.Program | str:
    """Convert TTIR MLIR text to PyPTO IR.

    Args:
        ttir_text: TTIR MLIR text (from compiled Triton kernel asm["ttir"]).
        program_name: Name for the output PyPTO program.
        output_format: Output format - "object" (default) returns ir.Program,
            "text" returns string representation.

    Returns:
        PyPTO ir.Program (default) or text representation.

    Example:
        >>> import triton
        >>> from triton_adapter import extract_ttir, convert_ttir_to_pypto
        >>> @triton.jit
        ... def add_kernel(x, y, out, n: tl.constexpr):
        ...     ...
        >>> ttir = extract_ttir(add_kernel, x, y, out, 128)
        >>> program = convert_ttir_to_pypto(ttir)
    """
    converter = TTIRToPyptoConverter()
    program = converter.convert(ttir_text, program_name=program_name)

    if output_format == "text":
        from pypto.ir.printer import python_print

        return python_print(program)
    return program
