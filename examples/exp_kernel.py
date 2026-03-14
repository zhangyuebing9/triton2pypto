"""Triton exp kernel for TTIR extraction - with pid and mask."""

import triton
import triton.language as tl


@triton.jit
def exp_kernel(x, out, n: tl.constexpr):
    pid = tl.program_id(0)
    blk = pid * 128
    offs = blk + tl.arange(0, 128)
    mask = offs < n
    a = tl.load(x + offs, mask=mask)
    tl.store(out + offs, tl.exp(a), mask=mask)
