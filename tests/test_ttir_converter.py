"""Tests for TTIR to PyPTO IR conversion."""

import pytest

pytest.importorskip("pypto", reason="PyPTO required - pip install pypto or use submodule")

from triton_adapter import ConversionError, TTIRToPyptoConverter, UnsupportedOpError


class TestTTIRConverter:
    """Test TTIR to PyPTO conversion."""

    def test_converter_initialization(self) -> None:
        """Test converter can be initialized."""
        converter = TTIRToPyptoConverter()
        assert converter.ib is not None
        assert converter.type_mapper is not None
        assert converter.span_tracker is not None

    def test_type_mapper_fp32(self) -> None:
        """Test type mapping for FP32."""
        from pypto import DataType

        converter = TTIRToPyptoConverter()
        dtype = converter.type_mapper.map_dtype("fp32")
        assert dtype == DataType.FP32

    def test_type_mapper_int32(self) -> None:
        """Test type mapping for INT32."""
        from pypto import DataType

        converter = TTIRToPyptoConverter()
        dtype = converter.type_mapper.map_dtype("i32")
        assert dtype == DataType.INT32

    def test_type_mapper_unsupported(self) -> None:
        """Test type mapping for unsupported dtype."""
        converter = TTIRToPyptoConverter()
        with pytest.raises(ConversionError):
            converter.type_mapper.map_dtype("unsupported_type")

    def test_supported_operations(self) -> None:
        """Test that supported operations list is defined."""
        converter = TTIRToPyptoConverter()
        assert "arith.addf" in converter.SUPPORTED_OPS
        assert "tt.load" in converter.SUPPORTED_OPS
        assert "tt.store" in converter.SUPPORTED_OPS

    def test_unsupported_operation_error(self) -> None:
        """Test that unsupported operations raise UnsupportedOpError."""
        converter = TTIRToPyptoConverter()
        with pytest.raises(UnsupportedOpError) as exc_info:
            converter.convert_operation(type("Op", (), {"name": "tt.unsupported"}))

        assert "tt.unsupported" in str(exc_info.value)


class TestTypeMapping:
    """Test type mapping functionality."""

    def test_all_supported_dtypes(self) -> None:
        """Test all supported dtype mappings."""
        from pypto import DataType

        converter = TTIRToPyptoConverter()
        expected = {
            "i1": DataType.BOOL,
            "i8": DataType.INT8,
            "i16": DataType.INT16,
            "i32": DataType.INT32,
            "i64": DataType.INT64,
            "fp16": DataType.FP16,
            "bf16": DataType.BF16,
            "fp32": DataType.FP32,
            "fp64": DataType.FP64,
        }

        for ttir_dtype, expected_pypto_dtype in expected.items():
            result = converter.type_mapper.map_dtype(ttir_dtype)
            assert result == expected_pypto_dtype, f"Failed for {ttir_dtype}"


class TestElementwiseConversion:
    """Test elementwise operation conversion."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_vector_add_conversion(self) -> None:
        """Test conversion of simple vector addition kernel."""
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_elementwise_mul_conversion(self) -> None:
        """Test conversion of elementwise multiplication kernel."""
        pass