import ast
import inspect
from types import FunctionType
from typing import Callable

from triton_bwd.abtract_tree import *
from triton_bwd.node_visitor import NodeVisitor
from triton_bwd.optimize_lang import *


class OptimizableFunction:
    def __init__(
        self,
        func: FunctionType,
        arg_specs: Dict[str, ArgSpec],
    ):
        self.func = func

        signature = inspect.signature(func)
        for _, param in signature.parameters.items():
            if param.name in arg_specs:
                spec = arg_specs[param.name]
                spec.name = param.name
                if isinstance(spec, ArraySpec):
                    if param.annotation is InArray:
                        spec.kind = "in"
                    elif param.annotation is OutArray:
                        spec.kind = "out"
                    elif param.annotation is InOutArray:
                        spec.kind = "in_out"
                    else:
                        raise ValueError(
                            f"Argument {param.name} must be annotated with "
                            f"InArray, OutArray, or InOutArray, but got "
                            f"{param.annotation}"
                        )
                else:
                    raise ValueError(
                        f"Argument {param.name} has unsupported spec type: "
                        f"{type(spec)}"
                    )
            else:
                if param.annotation is param.empty:
                    raise ValueError(
                        f"A type annotation is missing on argument {param.name}."
                    )
                if param.annotation is int:
                    spec = IntSpec()
                    spec.name = param.name
                    arg_specs[param.name] = spec
                elif param.annotation is float:
                    spec = FloatSpec()
                    spec.name = param.name
                    arg_specs[param.name] = spec
                elif param.annotation in [InArray, OutArray, InOutArray]:
                    # Array argument must have a spec defined
                    raise ValueError(
                        f"Array argument {param.name} must have a spec defined."
                    )
                else:
                    raise ValueError(
                        f"Argument {param.name} has unsupported type: "
                        f"{param.annotation}"
                    )

        self.arg_specs = arg_specs

        source = inspect.getsource(self.func)
        tree = ast.parse(source)

        args = {name: spec.symbol() for name, spec in self.arg_specs.items()}
        visitor = NodeVisitor(
            call_stack=[self.func.__name__],
            func_globals=self.func.__globals__,
            args=args,
        )
        self.abstract_tree: AbstractNode = visitor.visit(tree)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def optimize(
    arg_specs: Dict[str, ArgSpec]
) -> Callable[[FunctionType], OptimizableFunction]:

    def wrapper(func: FunctionType) -> OptimizableFunction:
        return OptimizableFunction(
            func=func,
            arg_specs=arg_specs,
        )

    return wrapper
