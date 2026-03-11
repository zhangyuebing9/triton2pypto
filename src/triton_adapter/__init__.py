"""Triton IR extraction and transformation."""

from .exceptions import ConversionError, UnsupportedOpError
from .ir_extractor import IRExtractionError, extract_ttgir, extract_ttir
from .mlir_parser import MLIROperation, MLIRParser, MLIRType, MLIRValue, parse_ttir

__all__ = [
    "ConversionError",
    "UnsupportedOpError",
    "extract_ttir",
    "extract_ttgir",
    "IRExtractionError",
    "parse_ttir",
    "MLIRParser",
    "MLIROperation",
    "MLIRValue",
    "MLIRType",
]

try:
    from .converter import convert_ttir_to_pypto
    from .ttir_converter import BlockPtrInfo, SpanTracker, TTIRToPyptoConverter, TypeMapper

    __all__.extend([
        "TTIRToPyptoConverter",
        "TypeMapper",
        "SpanTracker",
        "BlockPtrInfo",
        "convert_ttir_to_pypto",
    ])
except ImportError as e:
    import warnings

    warnings.warn(
        f"triton_adapter converter not available: {e}", ImportWarning, stacklevel=2
    )

