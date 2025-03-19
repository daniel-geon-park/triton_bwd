## `triton_bwd`: Automatic differentiation for Triton Kernels

`triton_bwd` is a wrapper around `triton` that allows you to use Triton kernels in PyTorch autograd.

## Usage

```diff
+from triton_bwd import triton_bwd, autotune

-@triton.autotune(...)
-@triton.jit
+@autotune(...)
+@triton_bwd(in_args=["a", "b"], out_args=["c"])
def my_triton_kernel(a, stride_a, b, stride_b, c, stride_c):
    ...
```

```diff
def compute_something(a, b):
    c = torch.zeros_like(a)
-    my_triton_kernel[grid](a, a.stride(0), b, b.stride(0), c, c.stride(0))
+    c, = my_triton_kernel(grid, a, a.stride(0), b, b.stride(0), c, c.stride(0))

    return c  # is now differentiable!
```

## Installation
```bash
git clone <this repo>
cd triton_bwd
pip install .
```
