from typing import Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import sympy
from sympy.solvers.solveset import linear_coeffs


class SympyIndexing(sympy.Function):
    @classmethod
    def eval(cls, array, index):
        pass

    def _sympystr(self, printer):
        array, index = self.args
        if isinstance(index, sympy.Tuple):
            index_str = ", ".join(printer.doprint(i) for i in index)
        else:
            index_str = printer.doprint(index)
        return printer.doprint(array) + "[" + index_str + "]"


def broadcasat_shapes(shape1: sympy.Tuple, shape2: sympy.Tuple):
    """Broadcast two shapes together."""
    shape1, shape2 = shape1.args, shape2.args
    len1, len2 = len(shape1), len(shape2)
    if len1 < len2:
        shape1 = sympy.Tuple(*([1] * (len2 - len1)) + list(shape1))
    elif len2 < len1:
        shape2 = sympy.Tuple(*([1] * (len1 - len2)) + list(shape2))

    # Check for incompatible dimensions
    for idim, (dim1, dim2) in enumerate(zip(shape1, shape2)):
        if dim1 != 1 and dim2 != 1 and dim1 != dim2:
            raise ValueError(f"Incompatible {idim}th dimensions: {dim1} and {dim2}")

    return sympy.Tuple(*[sympy.Max(dim1, dim2) for dim1, dim2 in zip(shape1, shape2)])


class SympyShape(sympy.Function):
    @classmethod
    def eval(cls, array):
        if isinstance(array, sympy.Number):
            return sympy.Tuple()
        if isinstance(array, sympy.Symbol):
            return sympy.Tuple()  # scalar
        if isinstance(array, sympy.IndexedBase):
            return array.shape
        if isinstance(array, Union[sympy.Add, sympy.Mul]):
            a, b = array.as_two_terms()
            a_shape, b_shape = SympyShape(a), SympyShape(b)
            shape = broadcasat_shapes(a_shape, b_shape)
            return sympy.Tuple(*shape)
        if isinstance(array, SympyIndexing):
            # TODO: handle slices and other indexing
            return sympy.Tuple()
        # TODO: implement other array operations
        raise ValueError(f"Unsupported type for shape: {type(array)}")


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


ArgSpec = Union[ArraySpec, IntSpec]
InArray = NewType("InArray", np.ndarray)
OutArray = NewType("OutArray", np.ndarray)
InOutArray = NewType("InOutArray", np.ndarray)


class ForLoop:
    def __init__(
        self,
        index_var: sympy.Symbol,
        index_begin: int,
        index_end: int,
        index_step: int,
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
        loop_idx += 1
        for stmt in self.statements:
            r, loop_idx, stmt_idx = stmt.get_stmt_impl(i, loop_idx, stmt_idx)
            if r is not None:
                S, loop_nest = r
                return (S, [self] + loop_nest), loop_idx, stmt_idx
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
            return (self, []), loop_idx, stmt_idx + 1
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

    def get_stmt(self, i: int) -> Tuple[Assignment, List[ForLoop]]:
        (S, loop_nest), _, _ = self.get_stmt_impl(i, 0, 0)
        return S, loop_nest

    def find_dependence(self, i: int, j: int):
        S, nest_S = self.get_stmt(i)
        T, nest_T = self.get_stmt(j)
        S_stores = get_mem_accesses(S.target)
        S_loads = get_mem_accesses(S.value)
        T_stores = get_mem_accesses(T.target)
        T_loads = get_mem_accesses(T.value)
        for name_s, index_s in S_stores:
            for name_t, index_t in [*T_stores, *T_loads]:
                if name_s == name_t:
                    dependence_level(i < j, index_s, nest_S, index_t, nest_T)
        for name_s, index_s in S_loads:
            for name_t, index_t in T_stores:
                if name_s == name_t:
                    dependence_level(i < j, index_s, nest_S, index_t, nest_T)


def dependence_level(
    s_before_t: bool,
    index_s: sympy.Basic,
    nest_s: List[ForLoop],
    index_t: sympy.Basic,
    nest_t: List[ForLoop],
):
    loop_indices_s = [loop_s.index_var for loop_s in nest_s]
    loop_indices_t = [loop_t.index_var for loop_t in nest_t]
    *ai, a0 = linear_coeffs(index_s, *loop_indices_s)
    *bi, b0 = linear_coeffs(index_t, *loop_indices_t)

    num_common_loops = 0
    for loop_s, loop_t in zip(nest_s, nest_t):
        if loop_s is loop_t:
            num_common_loops += 1
        else:
            break

    for u in range(num_common_loops):
        lhs = 0
        inequalities = []
        free_vars = []
        for level in range(num_common_loops):  # FIXME: +1 ?
            i_level = sympy.Symbol(f"__i{level}")
            free_vars.append(i_level)
            inequalities.append(nest_s[level].index_begin <= i_level)
            inequalities.append(i_level < nest_s[level].index_end)
            if level < u:
                j_level = i_level
            else:
                j_level = sympy.Symbol(f"__j{level}")
                free_vars.append(j_level)
                inequalities.append(nest_t[level].index_begin <= j_level)
                inequalities.append(j_level < nest_t[level].index_end)
            if level == u:
                inequalities.append(i_level < j_level)
            lhs = lhs + ai[level] * i_level - bi[level] * j_level
        for level in range(num_common_loops, len(ai)):
            i_level = sympy.Symbol(f"__i{level}")
            free_vars.append(i_level)
            inequalities.append(nest_s[level].index_begin <= i_level)
            inequalities.append(i_level < nest_s[level].index_end)
            lhs = lhs + ai[level] * i_level
        for level in range(num_common_loops, len(bi)):
            j_level = sympy.Symbol(f"__j{level}")
            free_vars.append(j_level)
            inequalities.append(nest_t[level].index_begin <= j_level)
            inequalities.append(j_level < nest_t[level].index_end)
            lhs = lhs - bi[level] * j_level
        lhs = lhs + (a0 - b0)
        print(f"{u}:", lhs, "== 0,", inequalities)
        *ci, c0 = linear_coeffs(lhs, *free_vars)
        if not (c0 % gcd(ci)).equals(0):
            # There is no dependence at this level
            continue

    print()
    return None  # There is no dependence


def gcd(ns: List[sympy.Basic]) -> sympy.Basic:
    result = 1
    for n in ns:
        result = sympy.gcd(result, n)
    return result


def get_mem_accesses(expr: sympy.Basic) -> List[Tuple[str, sympy.Basic]]:
    if isinstance(expr, sympy.Symbol):
        return [(expr.name, sympy.Number(0))]
    if isinstance(expr, SympyIndexing):
        array, index = expr.args
        assert isinstance(array, sympy.IndexedBase)
        flat_index = sympy.Number(0)
        shape = SympyShape(array)
        for dim, idx in zip(shape.args, index.args):
            flat_index = flat_index * dim + idx
        return [(array.name, flat_index)]

    results = []
    for arg in expr.args:
        results.extend(get_mem_accesses(arg))
    return results
