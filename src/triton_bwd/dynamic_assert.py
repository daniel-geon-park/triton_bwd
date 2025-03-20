import torch
import triton
import triton.language as tl


@triton.jit
def dynamic_assert_kernel(tensor, message: tl.constexpr):
    pid = tl.program_id(0)
    value = tl.load(tensor + pid)
    tl.device_assert(value, message)


class DynamicAssert(torch.autograd.Function):
    @staticmethod
    def forward(tensor: torch.Tensor, message: str):
        assert tensor.ndim == 1
        grid = (tensor.shape[0], 1, 1)
        dynamic_assert_kernel[grid](
            tensor.contiguous(),
            message,
        )
        return tensor

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def vmap(info, in_dims, tensor, message):  # noqa
        bdim, *_ = in_dims
        orig_shape = tensor.shape
        result = DynamicAssert.apply(tensor.flatten(), message)
        return result.reshape(orig_shape), bdim

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        return grad_output, None


@torch.no_grad()
def dynamic_assert(tensor: torch.Tensor, mask: torch.Tensor, message: str):
    assert tensor.dtype == torch.bool
    DynamicAssert.apply((tensor | ~mask).flatten(), message)
