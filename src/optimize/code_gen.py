from typing import List, Tuple

import sympy

from optimize.abtract_tree import Assignment, Declaration, ForLoop
from optimize.analyzed_tree import AnalyzedNode
from optimize.code_printer import CodePrinter
from optimize.mem_access import get_mem_accesses
from optimize.sympy_utils import (
    FLOAT_TYPES,
    INT_TYPES,
    SymbolicArray,
    SymbolicScalar,
    SympyDtype,
    SympyShape,
    ceildiv,
)


def _generate_code_impl(
    node: AnalyzedNode, backend: str
) -> Tuple[List[str], List[str]]:
    if node.kind == "A":
        return _generate_code_asgn(node, backend)
    elif node.kind == "L":
        return _generate_code_loop(node, backend)
    elif node.kind == "D":
        return _generate_code_decl(node, backend)
    else:
        raise ValueError(f"Unsupported statement kind: {node.kind}")


def _generate_code_asgn(
    node: AnalyzedNode, backend: str
) -> Tuple[List[str], List[str]]:
    assert isinstance(node.obj, Assignment)
    pr = CodePrinter(backend)
    return [], [f"{pr.doprint(node.obj.target)} = {pr.doprint(node.obj.value)}"]


def _generate_code_decl(
    node: AnalyzedNode, backend: str
) -> Tuple[List[str], List[str]]:
    assert isinstance(node.obj, Declaration)
    name, symbol = node.obj.name, node.obj.symbol
    shape = SympyShape(symbol)
    dtype = SympyDtype(symbol)
    if shape == ():
        return [], []
    else:
        shape_str = ", ".join(map(str, shape.args))
        if len(shape.args) == 1:
            shape_str = f"{shape_str},"
        if backend == "torch":
            dtype = f"torch.{str(dtype)}"
            initializer = f"torch.zeros(({shape_str}), dtype={dtype})"
        elif backend == "triton":
            dtype = f"tl.{str(dtype)}"
            initializer = f"tl.zeros(({shape_str}), dtype={dtype})"
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        return [], [f"{name} = {initializer}"]


def _generate_kernel(node: AnalyzedNode, backend: str) -> Tuple[List[str], List[str]]:
    assert isinstance(node.obj, ForLoop)
    assert backend == "torch", "Nested kernels are not supported."
    triton_pr = CodePrinter("triton")

    stores, loads = get_mem_accesses(node)
    parameters = {}
    for acc in stores + loads:
        if acc.decl_stmt is None or acc.decl_stmt.level <= node.level:
            if acc.decl_stmt is node:  # Skip loop index var
                continue
            dtype = SympyDtype(acc.symbol)
            ndims = len(SympyShape(acc.symbol).args)
            parameters[acc.name] = (dtype, ndims)

    parameters = [(name, dtype, ndims) for name, (dtype, ndims) in parameters.items()]
    parameters.sort(
        key=lambda x: (-x[2], str(x[1]), x[0])
    )  # Sort by dims, dtype, then by name

    param_list = []
    for name, dtype, ndims in parameters:
        if ndims == 0:
            if dtype in INT_TYPES:
                param_list.append(f"{name}: int")
            elif dtype in FLOAT_TYPES:
                param_list.append(f"{name}: float")
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        else:
            param_list.append(f"{name}")

    kernel_function_name = f"kernel_function_L{node.num}"

    pid_var = sympy.symbols("__pid")
    index_var_value = node.obj.index_begin + pid_var * node.obj.index_step
    code_lines = [
        "@triton.jit",
        f"def {kernel_function_name}({', '.join(param_list)}):",
        f"    __pid = tl.program_id(0)",
        f"    {triton_pr.doprint(node.obj.index_var)} = {triton_pr.doprint(index_var_value)}",
    ]

    preamble = []
    for child in node.children:
        child_preamble, chld_code = _generate_code_impl(child, "triton")
        preamble.extend(child_preamble)
        for line in chld_code:
            code_lines.append(f"    {line}")
    code_lines.extend(["", ""])
    preamble.extend(code_lines)

    torch_pr = CodePrinter("torch")
    num_threads = ceildiv(
        node.obj.index_end - node.obj.index_begin, node.obj.index_step
    )
    arg_list = [name for name, dtype, ndims in parameters]
    code_lines = [
        f"{kernel_function_name}[({torch_pr.doprint(num_threads)},)]({', '.join(arg_list)})"
    ]
    return preamble, code_lines


def _generate_code_loop(
    node: AnalyzedNode, backend: str
) -> Tuple[List[str], List[str]]:
    assert isinstance(node.obj, ForLoop)
    if node.obj.is_kernel:
        return _generate_kernel(node, backend)

    pr = CodePrinter(backend)

    if node.parent is None:  # top-level loop
        assert backend == "torch"

        arguments = []
        for arg in node.obj.arguments.values():
            if isinstance(arg, SymbolicScalar):
                if arg.dtype in INT_TYPES:
                    arguments.append(f"{arg.label.name}: int")
                elif arg.dtype in FLOAT_TYPES:
                    arguments.append(f"{arg.label.name}: float")
            elif isinstance(arg, SymbolicArray):
                arguments.append(f"{arg.label.name}: torch.Tensor")

        code_lines = [f"def function({', '.join(arguments)}):"]

    else:
        i, begin, end, step, max_steps = (
            node.obj.index_var,
            node.obj.index_begin,
            node.obj.index_end,
            node.obj.index_step,
            node.obj.max_steps,
        )
        code_lines = [
            f"for {pr.doprint(i)} in range({pr.doprint(begin)}, {pr.doprint(end)}, {pr.doprint(step)}):"
        ]

    preamble = []
    for child in node.children:
        child_preamble, chld_code = _generate_code_impl(child, backend)
        preamble.extend(child_preamble)
        for line in chld_code:
            code_lines.append(f"    {line}")

    return preamble, code_lines


def generate_code(node: AnalyzedNode) -> str:
    preamble, lines = _generate_code_impl(node, backend="torch")
    return "\n".join(preamble + lines)
