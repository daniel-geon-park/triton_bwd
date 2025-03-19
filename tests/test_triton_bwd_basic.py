import unittest

import torch
import triton
import triton.language as tl

from triton_bwd.triton_bwd import triton_bwd
from triton_bwd.triton_bwd_verify import test_run_bwd, verify_triton_fwd


# Run the following command to execute the test:
# python -m unittest tests.utils.triton_bwd.test_triton_bwd_basic
class TestTritonBwdBasic(unittest.TestCase):

    def test1(self):
        print("Test #1")
        verify_triton_fwd(
            test_func1,
            (1, 1, 1),
            torch.tensor([1.5], device="cuda"),
            torch.tensor([2.0], device="cuda"),
        )
        test_run_bwd(
            test_func1,
            (1, 1, 1),
            torch.tensor([1.5], device="cuda"),
            torch.tensor([2.0], device="cuda"),
        )

        a = torch.tensor([1.5], device="cuda")
        a.requires_grad = True
        b = torch.tensor([2.0], device="cuda")
        b.requires_grad = True

        output = test_func1.forward(
            (1, 1, 1),
            a,
            b,
            num_warps=4,
        )
        s = 0
        for out in output:
            s = s + out.sum()
        s.backward()

        print(a.grad)
        print(b.grad)

    def test2(self):
        print("Test #2")
        a = torch.randn([3, 5], device="cuda")
        b = torch.randn([3, 5], device="cuda")
        c = torch.zeros([3, 5], device="cuda")
        verify_triton_fwd(
            test_func2,
            (3, 5, 1),
            a,
            a.stride(0),
            a.stride(1),
            b,
            b.stride(0),
            b.stride(1),
            c,
            c.stride(0),
            c.stride(1),
        )
        test_run_bwd(
            test_func2,
            (3, 5, 1),
            a,
            a.stride(0),
            a.stride(1),
            b,
            b.stride(0),
            b.stride(1),
            c,
            c.stride(0),
            c.stride(1),
        )

    def test3(self):
        print("Test #3")
        M, N, K = 32, 96, 64
        a = torch.randn([M, K], device="cuda", dtype=torch.float64)
        b = torch.randn([K, N], device="cuda", dtype=torch.float64)
        c = torch.zeros([M, N], device="cuda", dtype=torch.float64)
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 16
        GROUP_SIZE_M = 8
        verify_triton_fwd(
            matmul_kernel,
            (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1),
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            # ACTIVATION="leaky_relu",
        )
        test_run_bwd(
            matmul_kernel,
            (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1),
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            # ACTIVATION="leaky_relu",
        )


@triton_bwd(["a", "b"], ["b"])
def test_func1(a, b):
    v = tl.load(a)
    w = tl.load(b)
    v = (2 * v * v + 1 * w * w) * w * v
    tl.store(b, v)


@triton_bwd(["a", "b"], ["c"])
def test_func2(
    a,
    a_stride_0,
    a_stride_1,
    b,
    b_stride_0,
    b_stride_1,
    c,
    c_stride_0,
    c_stride_1,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    a = tl.load(a + i * a_stride_0 + j * a_stride_1)
    b = tl.load(b + i * b_stride_0 + j * b_stride_1)
    r = (a * 2 + b * 3) * a / b
    tl.store(c + i * c_stride_0 + j * c_stride_1, r)


@triton_bwd(["a_ptr", "b_ptr"], ["c_ptr"])
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr = None,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.sum(a[:, :, None] * b[None, :, :], 1)
        # accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator  # .to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)
