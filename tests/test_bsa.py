import torch
import triton
from hip_attn.v1_3.kernels.block_sparse_attention import block_sparse_attention, forward
from hip_research.utils.seed import seed

from triton_bwd.triton_bwd_verify import print_errors

triton.runtime.driver.active.utils.set_printf_fifo_size(128 * 1024)


def test(do_backward=False):
    seed(42)
    args = torch.load("bsa_args.pt", map_location="cuda")
    args["q"] = args["q"].data.to(torch.float32)
    args["k"] = args["k"].data.to(torch.float32)
    args["v"] = args["v"].data.to(torch.float32)
    q, k, v = args["q"], args["k"], args["v"]
    q.requires_grad = k.requires_grad = v.requires_grad = True
    # args["return_attention_scores"] = True
    args["args"].using_extend = False
    args["args"].need_apply_rope = False

    o2, scores2 = block_sparse_attention(
        **args,
        output_attention_scores_reduce_method="max",
    )
    if do_backward:
        n = torch.randn_like(o2)
        (o2 * n).sum().backward()
    torch.cuda.synchronize()

    if do_backward:
        dq, dk, dv = q.grad, k.grad, v.grad
        q.grad = k.grad = v.grad = None

    o1, scores1, m1 = forward(
        **args,
        output_attention_score_reduce_method="max",
        use_torch_fwd=True,
        # no_vmap=True,
    )
    torch.cuda.synchronize()

    if do_backward:
        (o1 * n).sum().backward()
    torch.cuda.synchronize()

    if do_backward:
        dq_gt, dk_gt, dv_gt = q.grad, k.grad, v.grad
        q.grad = k.grad = v.grad = None

    q.grad = k.grad = v.grad = None

    print("Outputs:")
    print_errors(o1, o2)

    # print("Scores:")
    # print_errors(scores1, scores2)

    if do_backward:
        print("DQ pass=", torch.allclose(dq, dq_gt))
        print_errors(dq, dq_gt)
        print("DK pass=", torch.allclose(dk, dk_gt))
        print_errors(dk, dk_gt)
        print("DV pass=", torch.allclose(dv, dv_gt))
        print_errors(dv, dv_gt)

    print("Done.")


if __name__ == "__main__":
    test(True)
