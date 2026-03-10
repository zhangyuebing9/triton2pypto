"""Tests for Triton IR extraction."""

import pytest

from triton_adapter import extract_ttir, extract_ttgir
from triton_adapter.ir_extractor import IRExtractionError


class TestIRExtraction:
    """Test IR extraction functionality."""

    def test_extract_ttir_requires_valid_kernel(self) -> None:
        """Test that extract_ttir raises IRExtractionError with invalid input."""
        with pytest.raises(IRExtractionError):
            extract_ttir(None)  # type: ignore

    def test_extract_ttgir_not_implemented(self) -> None:
        """Test that extract_ttgir raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            extract_ttgir(None)  # type: ignore