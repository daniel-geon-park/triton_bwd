from typing import TYPE_CHECKING, List, Optional, Tuple

import sympy

from triton_bwd.abtract_tree import Declaration
from triton_bwd.sympy_utils import SymbolicArray, SympyIndexing, SympyShape

if TYPE_CHECKING:
    from triton_bwd.analyzed_tree import AnalyzedNode


class MemAccess:
    def __init__(
        self,
        name: str,
        decl_stmt: Optional["AnalyzedNode"],
        index: sympy.Tuple,
        flat_index: sympy.Basic,
        statement: Optional["AnalyzedNode"] = None,
    ):
        self.name = name
        self.decl_stmt = decl_stmt  # None if not declared in the loop nest (e.g. global variable or paremeter)
        self.index = index
        self.flat_index = flat_index
        self.statement = statement


def get_mem_accesses(stmt: "AnalyzedNode") -> Tuple[List[MemAccess], List[MemAccess]]:
    if stmt.kind == "A":
        nest = stmt.loop_nest()
        stores = get_expr_mem_accesses(stmt.obj.target, nest)
        loads = get_expr_mem_accesses(stmt.obj.value, nest)
        for acc in stores + loads:
            acc.statement = stmt
        return stores, loads

    elif stmt.kind == "L":
        stores, loads = [], []
        for child in stmt.children:
            cur_stores, cur_loads = get_mem_accesses(child)
            stores.extend(cur_stores)
            loads.extend(cur_loads)
        return stores, loads

    elif stmt.kind == "D":
        return [], []

    else:
        raise ValueError(f"Unsupported statement kind: {stmt.kind}")


def get_expr_mem_accesses(
    expr: sympy.Basic,
    loop_nest: List["AnalyzedNode"],
) -> List[MemAccess]:

    if isinstance(expr, sympy.Symbol):
        decl_stmt = find_decl_stmt(loop_nest, expr.name)

        return [
            MemAccess(
                name=expr.name,
                decl_stmt=decl_stmt,
                index=sympy.Tuple(),
                flat_index=sympy.Number(0),
            )
        ]

    if isinstance(expr, SympyIndexing):
        array, index = expr.args
        assert isinstance(array, SymbolicArray)
        assert isinstance(array.label, sympy.Symbol)

        array_name = array.label.name
        decl_stmt = find_decl_stmt(loop_nest, array_name)

        if not isinstance(index, sympy.Tuple):
            index = sympy.Tuple(index)

        # Flatten the multi-dimensional index to a single integer
        flat_index = sympy.Number(0)
        shape = SympyShape(array)
        for dim, idx in zip(shape.args, index.args):
            flat_index = flat_index * dim + idx

        return [
            MemAccess(
                name=array_name,
                decl_stmt=decl_stmt,
                index=index,
                flat_index=flat_index,
            )
        ]

    results = []
    for arg in expr.args:
        results.extend(get_expr_mem_accesses(arg, loop_nest))
    return results


def find_decl_stmt(
    loop_nest: List["AnalyzedNode"],
    name: str,
) -> Optional["AnalyzedNode"]:
    """Finds the declaration statement for a given variable name in the loop nest."""
    for loop in loop_nest[::-1]:
        for child in loop.children:
            if child.kind == "D":
                assert isinstance(child.obj, Declaration)
                if child.obj.name == name:
                    return child
    return None
