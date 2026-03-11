"""Tests for Triton IR extraction."""

import pytest

from triton_adapter import extract_ttir, extract_ttgir, IRExtractionError


class TestIRExtraction:
    """Test IR extraction functionality."""

    def test_extract_ttir_invalid_input(self) -> None:
        """Test that extract_ttir raises IRExtractionError with invalid input."""
        with pytest.raises(IRExtractionError):
            extract_ttir(None)  # type: ignore

    def test_extract_ttgir_invalid_input(self) -> None:
        """Test that extract_ttgir raises IRExtractionError with invalid input."""
        with pytest.raises(IRExtractionError):
            extract_ttgir(None)  # type: ignore