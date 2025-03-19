import copy
from typing import Callable

import torch


def verify_triton_fwd(
    _test_func: Callable,
    _grid: tuple | Callable[[dict], tuple],
    *args,
    **kwargs,
):
    bound_args = _test_func.signature.bind_partial(*args, **kwargs)
    named_args = bound_args.arguments

    in_arg_names = _test_func.in_args
    out_arg_names = _test_func.out_args

    output_tensors = {
        name: arg for name, arg in named_args.items() if name in out_arg_names
    }
    initial_outputs = copy.deepcopy(output_tensors)

    # Run triton compilation
    _test_func[_grid](**named_args)
    gt_outputs = copy.deepcopy(output_tensors)

    torch_fn = _test_func.get_torch_fn()

    # Reset output tensors
    for name in out_arg_names:
        output_tensors[name].copy_(initial_outputs[name])

    outputs = torch_fn(_grid, *args, **kwargs)

    print("Forward:")
    for name in out_arg_names:
        gt_output = gt_outputs[name]
        output = outputs[name]
        result = torch.allclose(output, gt_output)

        print(f"Output {name}", "[PASS]" if result else "[FAIL]")
        if not result:
            print_errors(gt_output, output)
            print("GT Output:", gt_output)
            print("Output:", output)


def test_run_bwd(
    _test_func: callable,
    _grid: tuple | Callable[[dict], tuple],
    *args,
    **kwargs,
):
    bound_args = _test_func.signature.bind(*args, **kwargs)
    named_args = bound_args.arguments

    in_arg_names = _test_func.in_args
    out_arg_names = _test_func.out_args

    torch_fn = _test_func.get_torch_fn()

    def forward_for_grad(*args_):
        new_args = named_args | {name: arg for name, arg in zip(in_arg_names, args_)}
        outputs_ = torch_fn(_grid, **new_args)
        outputs_ = [outputs_[name] for name in out_arg_names]
        s = 0
        for output_ in outputs_:
            s = s + output_.sum()
        return s

    for name in in_arg_names:
        named_args[name].requires_grad = True

    forward_for_grad(*[named_args[name] for name in in_arg_names]).backward()

    gradients = [named_args[name].grad for name in in_arg_names]

    if False:
        grad_fn = torch.func.grad(
            forward_for_grad,
            argnums=tuple(range(len(in_arg_names))),
        )

        gradients = grad_fn(
            *[named_args[name] for name in in_arg_names],
        )

    for name, grad in zip(in_arg_names, gradients):
        print(f"Gradient {name} ", end="")
        if not torch.isfinite(grad).all().item():
            print("[FAIL]: Gradient is not finite!")
        else:
            print("[OK]")
        print(grad)


def print_errors(a, b):
    assert a.shape == b.shape, f"Shapes do not match: {a.shape} != {b.shape}"
    a = a.float()
    b = b.float()
    if a.numel() < 16_000_000:
        quantile = 0.99
        percent = f"{quantile * 100:.0f}%"
        abs_95 = torch.quantile(
            torch.abs(a - b), torch.tensor(quantile, device=a.device)
        ).item()
        rel_95 = torch.quantile(
            (torch.abs(a - b) / (1e-6 + torch.abs(a))),
            torch.tensor(quantile, device=a.device),
        ).item()
        msg = f"abserr ({percent}): {abs_95:.6f}, relerr ({percent}): {rel_95:.6f}\n"
    else:
        abs_avg = torch.mean(torch.abs(a - b)).item()
        rel_avg = torch.mean((torch.abs(a - b) / (1e-6 + torch.abs(a)))).item()
        msg = f"abserr (avg): {abs_avg:.6f}, relerr (avg): {rel_avg:.6f}\n"
    abs_max = torch.amax(torch.abs(a - b)).item()
    rel_max = torch.amax((torch.abs(a - b) / (1e-6 + torch.abs(a)))).item()
    msg += f"abserr (max): {abs_max:.6f}, relerr (max): {rel_max:.6f}\n"
    print(msg)
    return msg
