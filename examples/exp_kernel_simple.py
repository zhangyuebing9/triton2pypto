"""Simple Triton exp kernel for elementwise exponential."""

import triton
import triton.language as tl


@triton.jit
def exp_kernel_simple(a, out):
    """Compute exp(a) elementwise - 128 elements, single block."""
    idx = tl.arange(0, 128)
    x = tl.load(a + idx)
    tl.store(out + idx, tl.exp(x))
