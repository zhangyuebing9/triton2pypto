"""Simple Triton add kernel (no mask) for simpler TTIR - full 128-element block."""

import triton
import triton.language as tl


@triton.jit
def add_kernel_simple(a, b, out):
    """Add two 128-element vectors - no mask, single block."""
    idx = tl.arange(0, 128)
    x = tl.load(a + idx)
    y = tl.load(b + idx)
    tl.store(out + idx, x + y)
