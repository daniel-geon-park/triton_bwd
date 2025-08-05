from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import sympy

from optimize.abtract_tree import Assignment, Declaration, ForLoop
from optimize.sympy_utils import (
    SymbolicArray,
    SymbolicScalar,
    SympyIndexing,
    sympy_slice,
)

if TYPE_CHECKING:
    from optimize.analyzed_tree import AnalyzedNode


class MemAccess:
    def __init__(
        self,
        name: str,
        symbol: Union[SymbolicScalar, SymbolicArray],
        decl_stmt: Optional["AnalyzedNode"],
        index: sympy.Tuple,
        statement: Optional["AnalyzedNode"] = None,
    ):
        self.name = name
        self.symbol = symbol
        self.decl_stmt = decl_stmt  # None if not declared in the loop nest (e.g. global variable or paremeter)
        self.index = index
        self.statement = statement


def get_mem_accesses(stmt: "AnalyzedNode") -> Tuple[List[MemAccess], List[MemAccess]]:
    if stmt.kind == "A":
        assert isinstance(stmt.obj, Assignment)
        nest = stmt.loop_nest()
        stores = get_expr_mem_accesses(stmt.obj.target, nest)
        loads = get_expr_mem_accesses(stmt.obj.value, nest)
        for acc in stores + loads:
            acc.statement = stmt
        return stores, loads

    elif stmt.kind == "L":
        assert isinstance(stmt.obj, ForLoop)
        nest = [*stmt.loop_nest(), stmt]
        stores = get_expr_mem_accesses(stmt.obj.index_var, nest)
        loads = [
            *get_expr_mem_accesses(stmt.obj.index_begin, nest),
            *get_expr_mem_accesses(stmt.obj.index_end, nest),
            *get_expr_mem_accesses(stmt.obj.index_step, nest),
        ]
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
    # Bare symbols should not be encountered here
    assert not isinstance(expr, sympy.Symbol)

    if isinstance(expr, SymbolicScalar) and isinstance(expr.label, sympy.Symbol):
        decl_stmt = find_decl_stmt(loop_nest, expr.label.name)
        return [
            MemAccess(
                name=expr.label.name,
                symbol=expr,
                decl_stmt=decl_stmt,
                index=sympy.Tuple(),
            )
        ]

    if isinstance(expr, SymbolicArray) and isinstance(expr.label, sympy.Symbol):
        decl_stmt = find_decl_stmt(loop_nest, expr.label.name)
        index = sympy.Tuple(*[sympy_slice() for _ in range(len(expr.shape.args))])
        return [
            MemAccess(
                name=expr.label.name,
                symbol=expr,
                decl_stmt=decl_stmt,
                index=index,
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

        return [
            MemAccess(
                name=array_name,
                symbol=array,
                decl_stmt=decl_stmt,
                index=index,
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
        assert isinstance(loop.obj, ForLoop)
        if name == loop.obj.index_var.label.name:
            return loop
        for child in loop.children:
            if child.kind == "D":
                assert isinstance(child.obj, Declaration)
                if child.obj.name == name:
                    return child
    return None
