"""Triton mul kernel for TTIR extraction - with pid and mask."""

import triton
import triton.language as tl


@triton.jit
def mul_kernel(x, y, out, n: tl.constexpr):
    pid = tl.program_id(0)
    blk = pid * 128
    offs = blk + tl.arange(0, 128)
    mask = offs < n
    a = tl.load(x + offs, mask=mask)
    b = tl.load(y + offs, mask=mask)
    tl.store(out + offs, a * b, mask=mask)
