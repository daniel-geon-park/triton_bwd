from typing import Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import sympy

from triton_bwd.dependence_checking import dependence_level, get_mem_accesses
from triton_bwd.sympy_utils import SympyShape


class ArraySpec:
    def __init__(self, dtype: str, dims: tuple):
        self.dtype = dtype
        self.dims = dims
        self.name: Optional[str] = None
        self.kind: Optional[str] = None

    def symbol(self) -> sympy.Symbol:
        assert self.name is not None
        shape = tuple(sympy.symbols(dim, integer=True) for dim in self.dims)
        return sympy.IndexedBase(
            self.name,
            shape=shape,
        )


class IntSpec:
    def __init__(self):
        self.name: Optional[str] = None

    def symbol(self) -> sympy.Symbol:
        assert self.name is not None
        return sympy.symbols(self.name, integer=True)


class FloatSpec:
    def __init__(self):
        self.name: Optional[str] = None

    def symbol(self) -> sympy.Symbol:
        assert self.name is not None
        return sympy.symbols(self.name, real=True)


ArgSpec = Union[ArraySpec, IntSpec, FloatSpec]
InArray = NewType("InArray", np.ndarray)
OutArray = NewType("OutArray", np.ndarray)
InOutArray = NewType("InOutArray", np.ndarray)


class ForLoop:
    def __init__(
        self,
        index_var: sympy.Symbol,
        index_begin: sympy.Basic,
        index_end: sympy.Basic,
        index_step: sympy.Basic,
        declarations: Dict[str, sympy.Basic],
        statements: List["AbstractNode"],
    ):
        assert index_step == 1, "Only step size of 1 is supported for now."
        self.index_var = index_var
        self.index_begin = index_begin
        self.index_end = index_end
        self.index_step = index_step
        self.declarations = declarations
        self.statements = statements

    def __repr__(self):
        stmt_reprs = []
        for name, decl in self.declarations.items():
            shape = SympyShape(decl)
            if shape == ():
                stmt_reprs.append(f"let {name}: scalar")
            else:
                stmt_reprs.append(
                    f"let {name}: array({', '.join(map(str, shape.args))})"
                )
        for stmt in self.statements:
            stmt_repr = repr(stmt)
            stmt_reprs.extend(stmt_repr.split("\n"))
        return (
            f"for {self.index_var} in range({self.index_begin}, {self.index_end}, {self.index_step}):\n"
            + "\n".join(f"    {stmt_repr}" for stmt_repr in stmt_reprs)
        )

    def add_numbers(self):
        stmt_idx = 0
        decl_idx = 0
        result = [
            (
                ("L", 0),
                self,
                f"for {self.index_var} in range({self.index_begin}, {self.index_end}, {self.index_step}):",
            )
        ]
        loop_idx = 1

        for name, decl in self.declarations.items():
            shape = SympyShape(decl)
            if shape == ():
                result.append((("D", decl_idx), decl, f"    let {name}: scalar"))
            else:
                result.append(
                    (
                        ("D", decl_idx),
                        decl,
                        f"    let {name}: array({', '.join(map(str, shape.args))})",
                    )
                )
            decl_idx += 1

        for stmt in self.statements:
            numbered_stmt = stmt.add_numbers()
            for (kind, num), obj, text in numbered_stmt:
                if kind == "S":
                    result.append(((kind, stmt_idx), obj, "    " + text))
                    stmt_idx += 1
                elif kind == "L":
                    result.append(((kind, loop_idx), obj, "    " + text))
                    loop_idx += 1
                elif kind == "D":
                    result.append(((kind, decl_idx), obj, "    " + text))
                    decl_idx += 1

        return result

    def get_stmt_impl(self, i: int, loop_idx: int, stmt_idx: int):
        my_loop_idx = loop_idx
        loop_idx += 1
        for stmt in self.statements:
            r, loop_idx, stmt_idx = stmt.get_stmt_impl(i, loop_idx, stmt_idx)
            if r is not None:
                S, loop_nest, loop_idx_nest = r
                new_r = (S, [self] + loop_nest, [my_loop_idx] + loop_idx_nest)
                return new_r, loop_idx, stmt_idx
        return None, loop_idx, stmt_idx


class Assignment:
    def __init__(self, target: sympy.Basic, value: sympy.Basic):
        self.target = target
        self.value = value

    def __repr__(self):
        return f"{self.target} = {self.value}"

    def add_numbers(self):
        return [(("S", 0), self, repr(self))]

    def get_stmt_impl(self, i: int, loop_idx: int, stmt_idx: int):
        if i == stmt_idx:
            return (self, [], []), loop_idx, stmt_idx + 1
        return None, loop_idx, stmt_idx + 1


class AbstractNode:
    def __init__(
        self,
        content: Union[ForLoop, Assignment],
    ):
        self.content = content

    def __repr__(self):
        return repr(self.content)

    def add_numbers(self):
        return self.content.add_numbers()

    def numbered_repr(self):
        numbered = self.add_numbers()
        return "\n".join(
            f"{f'{kind}{num}':>5}: {text}" for ((kind, num), _, text) in numbered
        )

    def get_stmt_impl(self, i: int, loop_idx: int, stmt_idx: int):
        return self.content.get_stmt_impl(i, loop_idx, stmt_idx)

    def get_stmt(self, i: int) -> Tuple[Assignment, List[ForLoop], List[int]]:
        result, _, _ = self.get_stmt_impl(i, 0, 0)
        return result

    def find_dependence(self, i: int, j: int):
        S, nest_S, nest_idx_S = self.get_stmt(i)
        T, nest_T, nest_idx_T = self.get_stmt(j)
        S_stores = get_mem_accesses(S.target)
        S_loads = get_mem_accesses(S.value)
        T_stores = get_mem_accesses(T.target)
        T_loads = get_mem_accesses(T.value)
        dependencies = []
        # Flow dependencies
        for name_s, index_s in S_stores:
            for name_t, index_t in T_loads:
                if name_s == name_t:
                    u = dependence_level(i < j, index_s, nest_S, index_t, nest_T)
                    if u is not None:
                        dependencies.append(("flow", u, name_s))
        # Antidependencies
        for name_s, index_s in S_loads:
            for name_t, index_t in T_stores:
                if name_s == name_t:
                    u = dependence_level(i < j, index_s, nest_S, index_t, nest_T)
                    if u is not None:
                        dependencies.append(("anti", u, name_s))
        # Output dependencies
        for name_s, index_s in S_stores:
            for name_t, index_t in T_stores:
                if name_s == name_t:
                    u = dependence_level(i < j, index_s, nest_S, index_t, nest_T)
                    if u is not None:
                        dependencies.append(("outp", u, name_s))
        return dependencies
