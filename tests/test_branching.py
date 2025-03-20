import unittest

import torch
import triton
import triton.language as tl

from triton_bwd import test_run_bwd, triton_bwd, verify_triton_fwd


# Run the following command to execute the test:
# python -m unittest tests.test_branching
class TestTritonBwdBranching(unittest.TestCase):
    def test1(self):
        print("Test #1")
        verify_triton_fwd(
            test_func1,
            (2, 1, 1),
            torch.tensor([1.5, -1.0], device="cuda"),
            torch.tensor([0.0, 0.0], device="cuda"),
        )


@triton_bwd(["a", "b"], ["b"])
def test_func1(a, b):
    i = tl.program_id(0)
    v = tl.load(a + i)
    if v < 0:
        r = v * 2.0
    else:
        r = v * 3.0
    tl.store(b + i, r)
