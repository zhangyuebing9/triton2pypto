"""Tests for MLIR parser - can run without pypto dependencies."""

import pytest

from triton_adapter.mlir_parser import MLIRParser, MLIRType, MLIRValue, parse_ttir


class TestMLIRParser:
    """Test MLIR text parsing."""

    def test_parse_simple_value(self) -> None:
        """Test parsing a simple MLIR value."""
        parser = MLIRParser()
        value = parser._parse_value("%0")
        assert value is not None
        assert value.name == "0"
        assert value.type_str == ""

    def test_parse_value_with_type(self) -> None:
        """Test parsing a value with type annotation."""
        parser = MLIRParser()
        value = parser._parse_value("%x : tensor<128xf32>")
        assert value is not None
        assert value.name == "x"
        assert "tensor" in value.type_str

    def test_parse_module_empty(self) -> None:
        """Test parsing an empty module."""
        parser = MLIRParser()
        ops = parser.parse_module("")
        assert ops == []

    def test_parse_module_with_comment(self) -> None:
        """Test parsing a module with comments."""
        mlir_text = """
        // This is a comment
        module {
          // Another comment
        }
        """
        parser = MLIRParser()
        ops = parser.parse_module(mlir_text)
        assert ops == []

    def test_parse_simple_operation(self) -> None:
        """Test parsing a simple operation."""
        mlir_text = """
        %0 = arith.constant 1.0 : f32
        """
        parser = MLIRParser()
        ops = parser.parse_module(mlir_text)
        assert len(ops) == 1
        assert ops[0].name == "arith.constant"
        assert ops[0].result is not None
        assert ops[0].result.name == "0"

    def test_parse_addf_operation(self) -> None:
        """Test parsing an addf operation."""
        mlir_text = """
        %2 = arith.addf %0, %1 : tensor<128xf32>
        """
        parser = MLIRParser()
        ops = parser.parse_module(mlir_text)
        assert len(ops) == 1
        assert ops[0].name == "arith.addf"
        assert len(ops[0].operands) == 2
        assert ops[0].operands[0].name == "0"
        assert ops[0].operands[1].name == "1"

    def test_parse_ttir_load(self) -> None:
        """Test parsing a TTIR load operation."""
        mlir_text = """
        %0 = tt.load %arg0 : tensor<128xf32>
        """
        parser = MLIRParser()
        ops = parser.parse_module(mlir_text)
        assert len(ops) == 1
        assert ops[0].name == "tt.load"
        assert len(ops[0].operands) == 1

    def test_parse_ttir_store(self) -> None:
        """Test parsing a TTIR store operation."""
        mlir_text = """
        tt.store %arg2, %2 : tensor<128xf32>
        """
        parser = MLIRParser()
        ops = parser.parse_module(mlir_text)
        assert len(ops) == 1
        assert ops[0].name == "tt.store"
        assert len(ops[0].operands) == 2

    def test_parse_full_module(self) -> None:
        """Test parsing a complete TTIR module."""
        mlir_text = """
module {
  tt.func @add_kernel(%arg0: !tt.ptr<fp32>, %arg1: !tt.ptr<fp32>, %arg2: !tt.ptr<fp32>) {
    %cst = arith.constant 1.0 : f32
    %0 = tt.load %arg0 : tensor<128xf32>
    %1 = tt.load %arg1 : tensor<128xf32>
    %2 = arith.addf %0, %1 : tensor<128xf32>
    tt.store %arg2, %2 : tensor<128xf32>
  }
}
        """
        parser = MLIRParser()
        ops = parser.parse_module(mlir_text)

        assert len(ops) >= 4

        op_names = [op.name for op in ops]
        assert "arith.constant" in op_names
        assert "tt.load" in op_names
        assert "arith.addf" in op_names
        assert "tt.store" in op_names


class TestMLIRType:
    """Test MLIR type utilities."""

    def test_is_tensor(self) -> None:
        """Test tensor type detection."""
        tensor_type = MLIRType("tensor<128xf32>")
        assert tensor_type.is_tensor() is True

        ptr_type = MLIRType("!tt.ptr<fp32>")
        assert ptr_type.is_tensor() is False

    def test_is_pointer(self) -> None:
        """Test pointer type detection."""
        ptr_type = MLIRType("!tt.ptr<fp32>")
        assert ptr_type.is_pointer() is True

        tensor_type = MLIRType("tensor<128xf32>")
        assert tensor_type.is_pointer() is False

    def test_get_shape(self) -> None:
        """Test shape extraction."""
        tensor_type = MLIRType("tensor<128xf32>")
        shape = tensor_type.get_shape()
        assert shape == [128]

        tensor_type_2d = MLIRType("tensor<128x64xf32>")
        shape_2d = tensor_type_2d.get_shape()
        assert shape_2d == [128]

    def test_get_element_type(self) -> None:
        """Test element type extraction."""
        tensor_type = MLIRType("tensor<128xf32>")
        elem_type = tensor_type.get_element_type()
        assert elem_type == "f32"

        tensor_type_bf16 = MLIRType("tensor<64xbf16>")
        elem_type_bf16 = tensor_type_bf16.get_element_type()
        assert elem_type_bf16 == "bf16"


class TestParseTTIRFunction:
    """Test parse_ttir convenience function."""

    def test_parse_ttir_basic(self) -> None:
        """Test basic parse_ttir function."""
        mlir_text = "%0 = arith.constant 1.0 : f32"
        ops = parse_ttir(mlir_text)
        assert len(ops) == 1
        assert ops[0].name == "arith.constant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])