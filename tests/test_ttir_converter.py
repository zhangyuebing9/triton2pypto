"""Tests for TTIR to PyPTO IR conversion."""

import pytest

from triton_adapter import ConversionError, TTIRToPyptoConverter, UnsupportedOpError
from triton_adapter.ir import DataType


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
        converter = TTIRToPyptoConverter()
        dtype = converter.type_mapper.map_dtype("fp32")
        assert dtype == DataType.FP32

    def test_type_mapper_int32(self) -> None:
        """Test type mapping for INT32."""
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

        for ttir_dtype, expected_dtype in expected.items():
            result = converter.type_mapper.map_dtype(ttir_dtype)
            assert result == expected_dtype, f"Failed for {ttir_dtype}"


class TestElementwiseConversion:
    """Test elementwise operation conversion."""

    def test_vector_add_conversion(self) -> None:
        """Test conversion and NumPy execution of vector addition kernel."""
        from triton_adapter import convert_ttir_to_pypto
        from triton_adapter.runtime import run_on_numpy
        import numpy as np

        ttir_text = """
module {
  tt.func @add_kernel(%arg0: !tt.ptr<fp32>, %arg1: !tt.ptr<fp32>, %arg2: !tt.ptr<fp32>) {
    %0 = tt.load %arg0 : tensor<128xf32>
    %1 = tt.load %arg1 : tensor<128xf32>
    %2 = arith.addf %0, %1 : tensor<128xf32>
    tt.store %arg2, %2 : tensor<128xf32>
  }
}
"""
        program = convert_ttir_to_pypto(ttir_text)
        assert program is not None
        assert len(program.functions) >= 1

        a = np.ones(128, dtype=np.float32)
        b = np.ones(128, dtype=np.float32) * 2
        out = np.zeros(128, dtype=np.float32)
        run_on_numpy(program, a, b, out)
        np.testing.assert_allclose(out, a + b, rtol=1e-5)

    @pytest.mark.skip(reason="Optional")
    def test_elementwise_mul_conversion(self) -> None:
        """Test elementwise multiplication kernel."""
        pass