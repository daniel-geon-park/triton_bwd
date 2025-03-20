import torch
from hip_attn.utils.triton_bwd.triton_bwd_verify import print_errors
from hip_attn.v1_3.kernels.block_sparse_attention import block_sparse_attention, forward
from hip_research.utils.seed import seed


def test(do_backward=False):
    seed(42)
    args = torch.load("bsa_args.pt", map_location="cuda")
    q, k, v = args["q"], args["k"], args["v"]
    args["return_attention_scores"] = True

    o1, scores1, m1 = forward(
        **args,
        output_attention_score_reduce_method="max",
        use_torch_fwd=True,
    )
    torch.cuda.synchronize()

    if do_backward:
        n = torch.randn_like(o1)
        (o1 * n).sum().backward()
    torch.cuda.synchronize()

    if do_backward:
        dq_gt, dk_gt, dv_gt = q.grad, k.grad, v.grad

    q.grad = k.grad = v.grad = None

    o2, scores2 = block_sparse_attention(
        **args,
        output_attention_scores_reduce_method="max",
    )
    if do_backward:
        (o2 * n).sum().backward()
    torch.cuda.synchronize()

    if do_backward:
        dq, dk, dv = q.grad, k.grad, v.grad

    print("Outputs:")
    print_errors(o1, o2)

    print("Scores:")
    print_errors(scores1, scores2)

    if do_backward:
        print("DQ pass=", torch.allclose(dq, dq_gt))
        print_errors(dq, dq_gt)
        print("DK pass=", torch.allclose(dk, dk_gt))
        print_errors(dk, dk_gt)
        print("DV pass=", torch.allclose(dv, dv_gt))
        print_errors(dv, dv_gt)


if __name__ == "__main__":
    test()
