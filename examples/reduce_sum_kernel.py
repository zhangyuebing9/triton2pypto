"""Triton reduce_sum kernel - row-wise sum with pid and mask."""

import triton
import triton.language as tl


@triton.jit
def reduce_sum_kernel(x, out, BLOCK: tl.constexpr, n_cols: tl.constexpr):
    """Row-wise sum: out[row] = sum(x[row,:]) with mask for variable columns."""
    pid = tl.program_id(0)
    row = pid
    col_offs = tl.arange(0, BLOCK)
    mask = col_offs < n_cols
    ptr = x + row * n_cols + col_offs
    row_data = tl.load(ptr, mask=mask)
    s = tl.sum(row_data, axis=0)
    tl.store(out + row, s)
