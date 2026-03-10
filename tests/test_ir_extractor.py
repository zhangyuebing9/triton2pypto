"""Tests for Triton IR extraction."""

import pytest

from triton_adapter import extract_ttir, extract_ttgir


class TestIRExtraction:
    """Test IR extraction functionality."""

    def test_extract_ttir_not_implemented(self) -> None:
        """Test that extract_ttir raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            extract_ttir(None)  # type: ignore

    def test_extract_ttgir_not_implemented(self) -> None:
        """Test that extract_ttgir raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            extract_ttgir(None)  # type: ignore