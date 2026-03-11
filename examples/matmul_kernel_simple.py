"""Simple Triton matmul kernel - single 16x16 block matmul."""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel_simple(A, B, C, BLOCK: tl.constexpr):
    """C = A @ B for BLOCK x BLOCK blocks. Grid (1,1) for single block."""
    off_m = tl.arange(0, BLOCK)
    off_n = tl.arange(0, BLOCK)
    off_k = tl.arange(0, BLOCK)
    a_ptrs = A + off_m[:, None] * BLOCK + off_k[None, :]
    b_ptrs = B + off_k[:, None] * BLOCK + off_n[None, :]
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.dot(a, b)
    c_ptrs = C + off_m[:, None] * BLOCK + off_n[None, :]
    tl.store(c_ptrs, c)
