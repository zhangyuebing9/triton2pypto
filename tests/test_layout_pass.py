"""Tests for layout conversion pass."""

import pytest

from passes import LayoutConversionPass


class TestLayoutConversionPass:
    """Test layout conversion pass."""

    def test_init(self) -> None:
        """Test pass initialization."""
        p = LayoutConversionPass()
        assert p.config == {}

    def test_init_with_config(self) -> None:
        """Test pass initialization with config."""
        config = {"target": "ascend"}
        p = LayoutConversionPass(config)
        assert p.config == config

    def test_run_not_implemented(self) -> None:
        """Test that run raises NotImplementedError."""
        p = LayoutConversionPass()
        with pytest.raises(NotImplementedError):
            p.run(None)  # type: ignore