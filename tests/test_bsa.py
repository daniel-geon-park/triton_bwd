import torch
from hip_attn.utils.triton_bwd.triton_bwd_verify import print_errors
from hip_attn.v1_3.kernels.block_sparse_attention import block_sparse_attention, forward
from hip_research.utils.seed import seed


def test():
    seed(42)
    args = torch.load("bsa_args.pt", map_location="cuda")
    args["output_attention_score_reduce_method"] = "max"
    q, k, v = args["q"], args["k"], args["v"]

    o1, scores1, m1 = forward(
        **args,
        use_torch_fwd=True,
    )
    torch.cuda.synchronize()

    n = torch.randn_like(o1)
    (o1 * n).sum().backward()
    torch.cuda.synchronize()

    dq_gt, dk_gt, dv_gt = q.grad, k.grad, v.grad

    q.grad = k.grad = v.grad = None

    o2, scores2, m2 = block_sparse_attention(**args)
    (o2 * n).sum().backward()
    torch.cuda.synchronize()

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
