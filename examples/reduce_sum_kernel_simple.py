"""Simple Triton reduce_sum kernel - row-wise sum of 128x128 block."""

import triton
import triton.language as tl


@triton.jit
def reduce_sum_kernel_simple(x, out, BLOCK: tl.constexpr):
    """Row-wise sum: out[row] = sum(x[row,:]) for BLOCK x BLOCK block."""
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    ptr = x + row * BLOCK + idx
    row_data = tl.load(ptr)
    s = tl.sum(row_data, axis=0)
    tl.store(out + row, s)
