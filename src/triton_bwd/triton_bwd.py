import ast
import builtins
import inspect
import os
import time
from typing import Any, Callable, Iterable, Optional

import torch
import torch.autograd
from triton import JITFunction
from triton.runtime.autotuner import Autotuner

from triton_bwd.code_generator import CodeGenerator, Pointer, convert_arg, underlying


class BackwardEnabledAutotuner(Autotuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _autotune(self, _callback, *args, **kwargs):
        # FIXME: this is copied from triton/runtime/autotuner.py
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = [_args[key] for key in self.keys if key in _args]
            for _, arg in _args.items():
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                # prune configs
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {
                    config: self._bench(*args, config=config, **kwargs)
                    for config in pruned_configs
                }
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = _callback(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def forward(
        self,
        grid: tuple | Callable[..., tuple],
        *args,
        device: str = None,
        batch_chunk_size: int = None,
        use_torch_fwd: bool = False,
        **kwargs,
    ):
        def callback(*args_, **kwargs_):
            grid_ = kwargs_.pop("grid")
            kwargs_.pop("warmup")
            return self.fn.forward(
                grid_,
                *args_,
                device=device,
                batch_chunk_size=batch_chunk_size,
                use_torch_fwd=use_torch_fwd,
                **kwargs_,
            )

        return self._autotune(
            callback,
            grid=grid,
            warmup=False,
            *args,
            **kwargs,
        )

    @property
    def signature(self):
        return self.fn.signature

    @property
    def in_args(self):
        return self.fn.in_args

    @property
    def out_args(self):
        return self.fn.out_args

    def get_torch_fn(
        self,
        device: str = None,
        batch_chunk_size: int = None,
    ):
        torch_fn = self.fn.get_torch_fn(device, batch_chunk_size)

        def wrapper(grid, *args, **kwargs):
            def callback(*args_, **kwargs_):
                grid_ = kwargs_.pop("grid")
                kwargs_.pop("warmup")

                launch_params = {
                    name: arg
                    for name, arg in kwargs_.items()
                    if name not in self.signature.parameters.keys()
                }
                for name in launch_params.keys():
                    kwargs_.pop(name)
                print(
                    f"WARNING: get_torch_fn: Parameters {list(launch_params.keys())} ignored."
                )

                return torch_fn(
                    grid_,
                    *args_,
                    **kwargs_,
                )

            return self._autotune(
                callback,
                grid=grid,
                warmup=False,
                *args,
                **kwargs,
            )

        return wrapper


class BackwardEnabledTritonFunc(JITFunction):
    def __init__(
        self,
        func: Callable,
        *,
        in_args=None,
        out_args=None,
        version=None,
        repr: Optional[Callable] = None,
        launch_metadata: Optional[Callable] = None,
        do_not_specialize: Optional[Iterable[int]] = None,
        debug: Optional[bool] = None,
        noinline: Optional[bool] = None,
    ):
        super().__init__(
            func,
            version=version,
            do_not_specialize=do_not_specialize,
            debug=debug,
            noinline=noinline,
            repr=repr,
            launch_metadata=launch_metadata,
        )
        self.func = func
        self.in_args = in_args
        self.out_args = out_args

    def forward(
        self,
        grid: tuple | Callable[..., tuple],
        *args,
        device: str = None,
        batch_chunk_size: int = None,
        use_torch_fwd: bool = False,
        **kwargs,
    ):
        launch_params = {
            name: arg
            for name, arg in kwargs.items()
            if name not in self.signature.parameters.keys()
        }
        for name in launch_params.keys():
            kwargs.pop(name)

        torch_fn = self.get_torch_fn(
            device=device,
            batch_chunk_size=batch_chunk_size,
        )

        bound_args = self.signature.bind(*args, **kwargs)
        named_args = bound_args.arguments

        grad_args = [named_args[name] for name in self.in_args]
        other_args = {
            name: arg for name, arg in named_args.items() if name not in self.in_args
        }
        outputs = AutogradTritonFunc.apply(
            self,
            torch_fn,
            grid,
            other_args,
            use_torch_fwd,
            launch_params,
            *grad_args,
        )
        return outputs

    def get_torch_fn(
        self,
        device: str = None,
        batch_chunk_size: int = None,
    ):
        if device is None:
            device = torch.cuda.current_device()

        source = inspect.getsource(self.func)
        tree = ast.parse(source)

        def forward(_pid, grid, *args, **kwargs):
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            named_args = bound_args.arguments

            pid0 = _pid // (grid[1] * grid[2])
            pid1 = (_pid // grid[2]) % grid[1]
            pid2 = _pid % grid[2]

            arg_dict = {name: convert_arg(arg) for name, arg in named_args.items()}
            gen = CodeGenerator(
                call_stack=[self.func.__name__],
                func_globals=self.func.__globals__,
                pids=(pid0, pid1, pid2),
                args=arg_dict,
                device=device,
            )
            gen.visit(tree)
            return {
                name: arg_dict[name].storage.updates
                for name in self.out_args
                if isinstance(arg_dict[name], Pointer)
            }

        def grid_launch(grid, *args, **kwargs):
            bound_args = self.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            named_args = bound_args.arguments

            if callable(grid):
                grid = grid(named_args)

            # Vmap over program_id's
            vmapped_forward = forward
            in_dims = (0, None) + (None,) * len(args)
            vmapped_forward = torch.func.vmap(
                vmapped_forward,
                in_dims=in_dims,
                chunk_size=batch_chunk_size,
            )

            pids = (
                torch.arange(grid[0], device=device)[:, None, None]
                * (grid[1] * grid[2])
                + torch.arange(grid[1], device=device)[None, :, None] * grid[2]
                + torch.arange(grid[2], device=device)[None, None, :]
            )
            outputs = vmapped_forward(
                pids.flatten(),
                grid,
                *args,
                **kwargs,
            )

            results = {}
            for name in self.out_args:
                arg = named_args[name]

                if name not in outputs:
                    results[name] = arg
                    continue

                updates = outputs[name]

                if len(updates) > 0:
                    orig_shape, orig_strides = arg.shape, arg.stride()
                    arg = underlying(arg)
                    arg = torch.concat([arg.new_zeros([1]), arg], dim=0)

                    for indices, values, mask in updates:
                        indices = indices.flatten() + 1
                        values = values.flatten()
                        mask = mask.flatten()
                        indices = torch.where(mask, indices, torch.zeros_like(indices))

                        arg = arg.index_put((indices,), values.to(arg.dtype))

                    arg = arg[1:].as_strided(orig_shape, orig_strides)

                results[name] = arg

            return results

        return grid_launch


class AutogradTritonFunc(torch.autograd.Function):
    def forward(
        ctx,
        func: BackwardEnabledTritonFunc,
        torch_fn: Callable,
        grid: tuple,
        other_args: dict[str, Any],
        use_torch_fwd: bool,
        launch_params: dict[str, Any],
        *grad_args,
    ):
        ctx.grid = grid
        ctx.torch_fn = torch_fn
        ctx.in_args = func.in_args
        ctx.out_args = func.out_args

        other_tensors_names = [
            name for name in other_args.keys() if torch.is_tensor(other_args[name])
        ]
        other_tensors = [other_args[name] for name in other_tensors_names]
        ctx.saved_names = func.in_args + other_tensors_names

        if any(arg.requires_grad for arg in grad_args):
            ctx.save_for_backward(*(list(grad_args) + other_tensors))

        non_tensor_args = {
            name: arg
            for name, arg in other_args.items()
            if name not in other_tensors_names
        }
        ctx.non_tensor_args = non_tensor_args

        new_args = other_args | {
            name: arg for name, arg in zip(func.in_args, grad_args)
        }
        if use_torch_fwd:
            out_tensors = ctx.torch_fn(ctx.grid, **new_args)
        else:
            out_tensors = {name: new_args[name].clone() for name in func.out_args}
            func[grid](**(new_args | out_tensors), **launch_params)

        return tuple(out_tensors[name] for name in func.out_args)

    def backward(ctx, *grad_outputs):
        in_args = ctx.saved_tensors[: len(ctx.in_args)]
        named_args = {
            name: arg for name, arg in zip(ctx.saved_names, ctx.saved_tensors)
        } | ctx.non_tensor_args

        def func(*grad_args_):
            new_args = named_args | {
                name: arg for name, arg in zip(ctx.in_args, grad_args_)
            }
            outputs = ctx.torch_fn(ctx.grid, **new_args)
            s = 0
            for name, grad in zip(ctx.out_args, grad_outputs):
                s += (outputs[name] * grad).sum()
            return s

        grad_fn = torch.func.grad(
            func,
            argnums=tuple(range(len(ctx.in_args))),
        )

        gradients = grad_fn(*in_args)

        return None, None, None, None, None, None, *gradients


def triton_bwd(in_args=None, out_args=None):
    if in_args is None:
        in_args = []
    if out_args is None:
        out_args = []

    def wrapper(func):
        return BackwardEnabledTritonFunc(
            func,
            in_args=in_args,
            out_args=out_args,
        )

    return wrapper


def autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
):

    def decorator(fn):
        return BackwardEnabledAutotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
            do_bench=do_bench,
        )

    return decorator
