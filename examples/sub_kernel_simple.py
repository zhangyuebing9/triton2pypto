"""Simple Triton sub kernel for elementwise subtraction."""

import triton
import triton.language as tl


@triton.jit
def sub_kernel_simple(a, b, out):
    """Subtract b from a - 128 elements, single block."""
    idx = tl.arange(0, 128)
    x = tl.load(a + idx)
    y = tl.load(b + idx)
    tl.store(out + idx, x - y)
