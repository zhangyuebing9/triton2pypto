"""Triton IR extraction and transformation."""

from .ir_extractor import extract_ttgir, extract_ttir
from .mlir_parser import MLIROperation, MLIRParser, MLIRType, MLIRValue, parse_ttir

__all__ = [
    "extract_ttir",
    "extract_ttgir",
    "parse_ttir",
    "convert_ttir_to_pypto",
    "MLIRParser",
    "MLIROperation",
    "MLIRValue",
    "MLIRType",
]

from .ir import Program
from .ttir_converter import (  # noqa: F401
    BlockPtrInfo,
    ConversionError,
    TTIRToPyptoConverter,
    TypeMapper,
    UnsupportedOpError,
)

__all__.extend([
    "TTIRToPyptoConverter",
    "TypeMapper",
    "BlockPtrInfo",
    "ConversionError",
    "UnsupportedOpError",
])


def convert_ttir_to_pypto(ttir_text: str) -> Program:
    """Convert TTIR MLIR text to adapter IR Program (runnable on CPU via NumPy).

    Args:
        ttir_text: TTIR MLIR text (e.g. from compiled kernel or .ttir.mlir file).

    Returns:
        Program that can be executed with triton_adapter.runtime.run_on_numpy().
    """
    from .mlir_parser import parse_ttir

    ops = parse_ttir(ttir_text)
    converter = TTIRToPyptoConverter()
    return converter.convert(ops)
