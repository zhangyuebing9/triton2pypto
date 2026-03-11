"""Simple Triton mul kernel for elementwise multiplication."""

import triton
import triton.language as tl


@triton.jit
def mul_kernel_simple(a, b, out):
    """Multiply a * b elementwise - 128 elements, single block."""
    idx = tl.arange(0, 128)
    x = tl.load(a + idx)
    y = tl.load(b + idx)
    tl.store(out + idx, x * y)
