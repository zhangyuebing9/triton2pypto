"""Simple MLIR text parser for TTIR operations.

This module provides a lightweight parser for MLIR text format,
specifically targeting TTIR operations needed for Phase 1.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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

    def get_shape(self) -> Optional[List[int]]:
        """Extract shape from tensor type if present."""
        if not self.is_tensor():
            return None
        try:
            start = self.type_str.index("<") + 1
            end = self.type_str.index("x", start)
            shape_str = self.type_str[start:end]
            if shape_str == "?":
                return [1]
            return [int(shape_str)]
        except (ValueError, IndexError):
            return None

    def get_element_type(self) -> Optional[str]:
        """Extract element type from tensor type."""
        if not self.is_tensor():
            return None
        try:
            parts = self.type_str.split("x")
            if len(parts) >= 2:
                return parts[-1].rstrip(">")
            return None
        except IndexError:
            return None


@dataclass
class MLIROperation:
    """Represents an MLIR operation."""

    name: str
    result: Optional[MLIRValue]
    operands: List[MLIRValue]
    attributes: Dict[str, Any]
    result_types: List[MLIRType]

    def __repr__(self) -> str:
        result_str = f"{self.result} = " if self.result else ""
        operands_str = ", ".join(str(op) for op in self.operands)
        return f"{result_str}{self.name}({operands_str})"


class MLIRParser:
    """Simple MLIR text parser."""

    def __init__(self) -> None:
        self.current_module: List[MLIROperation] = []

    def parse_module(self, mlir_text: str) -> List[MLIROperation]:
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

    def _parse_operation(self, line: str) -> Optional[MLIROperation]:
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
        result_types: List[MLIRType] = []

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

        return MLIROperation(
            name=op_name, result=result, operands=operands, attributes=attributes, result_types=result_types
        )

    def _parse_value(self, value_str: str) -> Optional[MLIRValue]:
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

    def _parse_op_body(self, line: str) -> Tuple[str, List[MLIRValue], Dict[str, Any]]:
        """Parse operation name, operands, and attributes.

        Args:
            line: Operation body text.

        Returns:
            Tuple of (op_name, operands, attributes).
        """
        parts = line.split("(", 1)
        op_name = parts[0].strip()

        operands: List[MLIRValue] = []
        attributes: Dict[str, Any] = {}

        if len(parts) > 1:
            body = parts[1].split(")")[0] if ")" in parts[1] else parts[1]
            operands = self._parse_operand_list(body)

            if "{" in line:
                attr_str = line.split("{")[1].split("}")[0]
                attributes = self._parse_attributes(attr_str)

        return op_name, operands, attributes

    def _parse_operand_list(self, operand_str: str) -> List[MLIRValue]:
        """Parse a comma-separated list of operands.

        Args:
            operand_str: Operand list string.

        Returns:
            List of parsed values.
        """
        operands: List[MLIRValue] = []
        for part in operand_str.split(","):
            part = part.strip()
            if part:
                value = self._parse_value(part)
                if value:
                    operands.append(value)
        return operands

    def _parse_attributes(self, attr_str: str) -> Dict[str, Any]:
        """Parse operation attributes.

        Args:
            attr_str: Attribute string.

        Returns:
            Dictionary of attributes.
        """
        attributes: Dict[str, Any] = {}
        for pair in attr_str.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                attributes[key.strip()] = value.strip()
        return attributes


def parse_ttir(ttir_text: str) -> List[MLIROperation]:
    """Parse TTIR MLIR text.

    Args:
        ttir_text: TTIR MLIR text.

    Returns:
        List of parsed operations.
    """
    parser = MLIRParser()
    return parser.parse_module(ttir_text)