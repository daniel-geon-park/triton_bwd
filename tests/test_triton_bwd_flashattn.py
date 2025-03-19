import torch

from hip_attn.utils.triton_bwd.triton_bwd_verify import print_errors
from hip_attn.v1_3.kernels.flash_attention import flash_attention, forward
from hip_research.utils.seed import seed


def test():
    seed(42)
    dtype = torch.float16
    q, k, v = (
        torch.randn([1, 32, 512, 128], device="cuda", dtype=dtype),
        torch.randn([1, 8, 256, 128], device="cuda", dtype=dtype),
        torch.randn([1, 8, 256, 128], device="cuda", dtype=dtype),
    )
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    args = dict(
        q=q,
        k=k,
        v=v,
        causal=True,
        sm_scale=0.08838834764831843,
        RETURN_SCORES=True,  # RETURN_SCORES
        EXCLUDE_LAST_WINDOW=True,  # EXCLUDE_LAST_WINDOW
        q_factor=64,  # q_factor
        kv_factor=128,  # kv_factor
    )
    n = torch.randn_like(q)

    o1, _, scores1 = forward(**args, use_torch_fwd=True)
    (o1 * n).sum().backward()

    dq_gt, dk_gt, dv_gt = q.grad, k.grad, v.grad

    q.grad = k.grad = v.grad = None

    o2, scores2 = flash_attention(**args)
    (o2 * n).sum().backward()

    dq, dk, dv = q.grad, k.grad, v.grad

    print("Outputs:")
    print_errors(o1, o2)

    print("Scores:")
    print_errors(scores1, scores2)

    print("DQ pass=", torch.allclose(dq, dq_gt))
    print_errors(dq, dq_gt)
    print("DK pass=", torch.allclose(dk, dk_gt))
    print_errors(dk, dk_gt)
    print("DV pass=", torch.allclose(dv, dv_gt))
    print_errors(dv, dv_gt)


if __name__ == "__main__":
    test()
