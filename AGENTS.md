# Working on triton2pypto

Triton to PyPTO adapter - enables Triton kernels on Huawei AI accelerators.

## Project Overview

This project provides an adapter layer between Triton's MLIR-based IR and PyPTO's compilation pipeline:
- **triton_adapter**: Extracts and transforms Triton IR (TTIR/TTGIR)
- **passes**: IR transformation passes
- **pypto_backend**: PyPTO backend integration

## Prerequisites

- Python 3.10+
- CMake 3.15+
- C++17 compiler

## Build Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Build with specific CMake build type
CMAKE_BUILD_TYPE=Release pip install -e .

# Build with ccache (auto-detected if available)
CMAKE_BUILD_TYPE=Release pip install -e .
```

## Test Commands

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_triton_adapter.py

# Run single test
pytest tests/test_triton_adapter.py::test_ir_extraction

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v --tb=short tests/
```

## Lint and Type Check

```bash
# Run ruff linter
ruff check src/

# Run ruff formatter
ruff format src/

# Run mypy type check
mypy src/

# Run all checks
ruff check src/ && ruff format --check src/ && mypy src/
```

## Submodule Management

```bash
# Initialize submodules
git submodule update --init --recursive

# Update submodules to latest
git submodule update --remote

# Work inside a submodule
cd third_party/triton
git checkout main
git pull origin main
```

## Code Style Guidelines

### Imports

```python
# Standard library first
import os
import sys
from typing import Optional, List, Dict

# Third-party imports
import torch
from triton import ir as tir

# Local imports (absolute)
from triton_adapter.ir_extractor import extract_ttir
from passes.transform import convert_layout
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- **Type aliases**: `PascalCase` (e.g., `IRNode = tir.Operation`)

### Type Annotations

Always use type annotations:

```python
def transform_ir(
    ir_module: tir.Module,
    options: TransformOptions | None = None,
) -> tir.Module:
    ...

class IRTransformer:
    def __init__(self, config: Config) -> None:
        self.config = config
```

### Error Handling

```python
# Use custom exceptions
class IRConversionError(Exception):
    """Raised when IR conversion fails."""

def convert_ir(ir: tir.Module) -> pto.Module:
    if not ir.is_valid():
        raise IRConversionError(f"Invalid IR: {ir}")
    ...
```

### Documentation

```python
def extract_kernel_ir(kernel: triton.JITFunction) -> tir.Module:
    """Extract MLIR from a Triton kernel.

    Args:
        kernel: Compiled Triton kernel function.

    Returns:
        The extracted MLIR module.

    Raises:
        IRExtractionError: If kernel has not been compiled.
    """
    ...
```

## Working with Triton IR

### Debug Environment Variables

```bash
# Dump MLIR IR at each pass
MLIR_ENABLE_DUMP=1 pytest tests/

# Use Triton interpreter (no GPU needed)
TRITON_INTERPRET=1 pytest tests/

# Dump kernel IR to file
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/ir_dump pytest tests/
```

### IR Extraction Pattern

```python
import triton
from triton import ir as tir

def get_triton_ir(kernel: triton.JITFunction, *args, **kwargs) -> tir.Module:
    """Extract TTIR from a compiled Triton kernel."""
    # Compile to get the IR
    compiled = kernel.run(*args, **kwargs)
    # Access the internal IR
    return compiled.asm["ttir"]
```

## Working with PyPTO

### Build PyPTO from Source

```bash
cd third_party/pypto
pip install -e ".[dev]"
```

### Integration Pattern

```python
from pypto import TensorGraph, TileGraph

def triton_to_pypto(ttir: tir.Module) -> TensorGraph:
    """Convert Triton IR to PyPTO TensorGraph."""
    graph = TensorGraph()
    # Convert operations...
    return graph
```

## Git Workflow

- Branch from `main` for features
- Use descriptive branch names: `feature/add-matmul-pass`, `fix/layout-conversion`
- Squash merge PRs
- Reference issues in commits: `fix #123: handle empty IR`

## Related Documentation

- Triton: `third_party/triton/README.md`, `third_party/triton/AGENTS.md`
- PyPTO: `third_party/pypto/README.md`