"""Simple MLIR text parser for TTIR operations.

This module provides a lightweight parser for MLIR text format,
specifically targeting TTIR operations needed for Phase 1.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MLIRValue:
    """Represents an MLIR value (SSA value)."""

    name: str
    type_str: str

    def __repr__(self) -> str:
        return f"%{self.name}"


@dataclass
class MLIRType:
    """Represents an MLIR type."""

    type_str: str

    def is_tensor(self) -> bool:
        """Check if this is a tensor type."""
        return "tensor" in self.type_str

    def is_pointer(self) -> bool:
        """Check if this is a pointer type."""
        return "ptr" in self.type_str

    def get_shape(self) -> list[int] | None:
        """Extract shape from tensor type if present.

        Handles tensor<128xf32>, tensor<16x16xf32>, tensor<MxNxf32> (dynamic -> 1).
        """
        if not self.is_tensor():
            return None
        try:
            start = self.type_str.index("<") + 1
            end = self.type_str.rindex(">")
            inner = self.type_str[start:end]
            # Split by 'x' but last part is dtype (e.g. f32, i32)
            parts = inner.split("x")
            if len(parts) < 2:
                return None
            shape: list[int] = []
            for p in parts[:-1]:
                p = p.strip()
                if p == "?" or not p[0].isdigit():
                    shape.append(1)  # Dynamic dim -> 1 for simplicity
                else:
                    shape.append(int(p))
            return shape if shape else [1]
        except (ValueError, IndexError):
            return None

    def get_element_type(self) -> str | None:
        """Extract element type from tensor type."""
        if not self.is_tensor():
            return None
        try:
            start = self.type_str.index("<") + 1
            end = self.type_str.rindex(">")
            inner = self.type_str[start:end]
            parts = inner.split("x")
            if len(parts) >= 2:
                return parts[-1].strip().rstrip(",")
            return None
        except (ValueError, IndexError):
            return None


@dataclass
class MLIROperation:
    """Represents an MLIR operation."""

    name: str
    result: MLIRValue | None
    operands: list[MLIRValue]
    attributes: dict[str, Any]
    result_types: list[MLIRType]

    def __repr__(self) -> str:
        result_str = f"{self.result} = " if self.result else ""
        operands_str = ", ".join(str(op) for op in self.operands)
        return f"{result_str}{self.name}({operands_str})"


class MLIRParser:
    """Simple MLIR text parser."""

    def __init__(self) -> None:
        self.current_module: list[MLIROperation] = []

    def parse_module(self, mlir_text: str) -> list[MLIROperation]:
        """Parse MLIR module text.

        Args:
            mlir_text: MLIR text to parse.

        Returns:
            List of parsed operations.
        """
        self.current_module = []
        lines = mlir_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("module") or line.startswith("}"):
                continue

            op = self._parse_operation(line)
            if op:
                self.current_module.append(op)

        return self.current_module

    def _parse_operation(self, line: str) -> MLIROperation | None:
        """Parse a single MLIR operation line.

        Args:
            line: Single line of MLIR text.

        Returns:
            Parsed operation or None if not parseable.
        """
        line = line.strip()
        if not line:
            return None

        result = None
        result_types: list[MLIRType] = []

        if "=" in line:
            result_part, op_part = line.split("=", 1)
            result = self._parse_value(result_part.strip())
            type_match = result_part.strip()
            if ":" in type_match:
                type_str = type_match.split(":")[1].strip()
                result_types = [MLIRType(type_str)]

            line = op_part.strip()

        if not line:
            return None

        op_name, operands, attributes = self._parse_op_body(line)

        # Get result type from op body if not from result: "tt.load %arg0 : tensor<128xf32>"
        if not result_types and ":" in line and "{" not in line.split(":")[0]:
            type_part = line.split(":", 1)[1].strip()
            if type_part and "<" in type_part:
                # Extract type until balancing
                end = type_part.find(">") + 1 if ">" in type_part else len(type_part)
                type_str = type_part[:end].strip()
                if type_str:
                    result_types = [MLIRType(type_str)]

        return MLIROperation(
            name=op_name, result=result, operands=operands, attributes=attributes, result_types=result_types
        )

    def _parse_value(self, value_str: str) -> MLIRValue | None:
        """Parse an MLIR value (e.g., %0, %x).

        Args:
            value_str: Value string to parse.

        Returns:
            Parsed value or None.
        """
        value_str = value_str.strip()
        if value_str.startswith("%"):
            name = value_str[1:].strip()
            if ":" in name:
                name = name.split(":")[0].strip()
            type_str = ""
            if ":" in value_str:
                type_str = value_str.split(":")[1].strip()
            return MLIRValue(name=name, type_str=type_str)
        return None

    def _parse_op_body(self, line: str) -> tuple[str, list[MLIRValue], dict[str, Any]]:
        """Parse operation name, operands, and attributes.

        Handles both op(args) and op %operand : type formats.
        """
        operands: list[MLIRValue] = []
        attributes: dict[str, Any] = {}

        # Handle attribute dict at end: {key = value}
        if "{" in line:
            attr_start = line.rfind("{")
            attr_str = line[attr_start:].strip("{}")
            attributes = self._parse_attributes(attr_str)
            line = line[:attr_start].strip()

        if "(" in line and line.find("(") < (line.find(":") if ":" in line else len(line)):
            # op(args) format
            paren = line.index("(")
            op_name = line[:paren].strip()
            body = line[paren + 1 : line.rfind(")")]
            operands = self._parse_operand_list(body)
        else:
            # op %operand : type or op value : type format
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                operand_part = parts[0].strip()
                first_space = operand_part.find(" ")
                if first_space > 0:
                    op_name = operand_part[:first_space].strip()
                    rest = operand_part[first_space:].strip()
                    # For arith.constant, rest is the value (e.g. "1.0")
                    if op_name == "arith.constant":
                        attributes["value"] = rest
                    else:
                        operands = self._parse_operand_list(rest)
                else:
                    op_name = operand_part
            else:
                first_space = line.find(" ")
                if first_space > 0:
                    op_name = line[:first_space].strip()
                    rest = line[first_space:].strip()
                    if op_name == "arith.constant":
                        attributes["value"] = rest
                    else:
                        operands = self._parse_operand_list(rest)
                else:
                    op_name = line.strip()

        return op_name, operands, attributes

    def _parse_operand_list(self, operand_str: str) -> list[MLIRValue]:
        """Parse a comma-separated list of operands.

        Args:
            operand_str: Operand list string.

        Returns:
            List of parsed values.
        """
        operands: list[MLIRValue] = []
        for part in operand_str.split(","):
            part = part.strip()
            if part:
                value = self._parse_value(part)
                if value:
                    operands.append(value)
        return operands

    def _parse_attributes(self, attr_str: str) -> dict[str, Any]:
        """Parse operation attributes.

        Args:
            attr_str: Attribute string.

        Returns:
            Dictionary of attributes.
        """
        attributes: dict[str, Any] = {}
        for pair in attr_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                attributes[key.strip()] = value.strip()
        return attributes


def parse_ttir(ttir_text: str) -> list[MLIROperation]:
    """Parse TTIR MLIR text.

    Args:
        ttir_text: TTIR MLIR text.

    Returns:
        List of parsed operations.
    """
    parser = MLIRParser()
    return parser.parse_module(ttir_text)
