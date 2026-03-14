"""Triton matmul kernel - block matmul with pid and mask."""

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A, B, C, BLOCK: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    """C = A @ B for BLOCK x BLOCK blocks with boundary masks."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    off_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    off_k = tl.arange(0, BLOCK)
    mask_m = off_m < M
    mask_n = off_n < N
    mask_k = off_k < K
    a_ptrs = A + off_m[:, None] * K + off_k[None, :]
    b_ptrs = B + off_k[:, None] * N + off_n[None, :]
    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :])
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :])
    c = tl.dot(a, b)
    c_ptrs = C + off_m[:, None] * N + off_n[None, :]
    tl.store(c_ptrs, c, mask=mask_m[:, None] & mask_n[None, :])
