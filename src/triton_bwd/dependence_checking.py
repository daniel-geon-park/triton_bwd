from typing import TYPE_CHECKING, List, Tuple

import sympy
import z3
from sympy.solvers.solveset import linear_coeffs

from triton_bwd.sympy_to_z3 import sympy_to_z3
from triton_bwd.sympy_utils import SympyIndexing, SympyShape

if TYPE_CHECKING:
    from triton_bwd.abtract_tree import ForLoop


def dependence_levels(
    s_before_t: bool,
    index_s: sympy.Basic,
    nest_s: List["ForLoop"],
    index_t: sympy.Basic,
    nest_t: List["ForLoop"],
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

    dep_levels = []
    for u in range(num_common_loops + (1 if s_before_t else 0)):
        indep_proven = prove_independence(
            u,
            num_common_loops,
            ai,
            a0,
            bi,
            b0,
            nest_s,
            nest_t,
        )
        if not indep_proven:
            dep_levels.append(u)

    return dep_levels


def prove_independence(
    u: int,
    num_common_loops: int,
    ai: List[sympy.Expr],
    a0: sympy.Expr,
    bi: List[sympy.Expr],
    b0: sympy.Expr,
    nest_s: List["ForLoop"],
    nest_t: List["ForLoop"],
):
    """Returns `True` if it can be proven that there is no dependence at level u.
    If there is a dependence, or if the lack of dependence cannot be proven, then returns `False`.
    """

    lhs = 0
    coeffs = set()
    free_vars = []
    constraints = []

    for level in range(num_common_loops):
        ik = z3.Int(f"__i{level}")
        jk = z3.Int(f"__j{level}")
        free_vars.extend([ik, jk])
        ak_vars, ak = sympy_to_z3(ai[level])
        bk_vars, bk = sympy_to_z3(bi[level])
        lhs = lhs + ak * ik - bk * jk
        pk_vars, pk = sympy_to_z3(nest_s[level].index_begin)
        qk_vars, qk = sympy_to_z3(nest_s[level].index_end - 1)
        constraints.extend([pk <= ik, ik <= qk])
        constraints.extend([pk <= jk, jk <= qk])
        if level < u:  # s = 0
            constraints.append(ik == jk)
        elif level == u:  # s = 1
            constraints.append(ik <= jk - 1)
        for var in ak_vars + bk_vars + pk_vars + qk_vars:
            coeffs.add(var)

    for level in range(num_common_loops, len(ai)):
        ik = z3.Int(f"__i{level}")
        free_vars.append(ik)
        ak_vars, ak = sympy_to_z3(ai[level])
        lhs = lhs + ak * ik
        pk_vars, pk = sympy_to_z3(nest_s[level].index_begin)
        qk_vars, qk = sympy_to_z3(nest_s[level].index_end - 1)
        constraints.extend([pk <= ik, ik <= qk])
        for var in ak_vars + pk_vars + qk_vars:
            coeffs.add(var)

    for level in range(num_common_loops, len(bi)):
        jk = z3.Int(f"__j{level}")
        free_vars.append(jk)
        bk_vars, bk = sympy_to_z3(bi[level])
        lhs = lhs - bk * jk
        pk_vars, pk = sympy_to_z3(nest_t[level].index_begin)
        qk_vars, qk = sympy_to_z3(nest_t[level].index_end - 1)
        constraints.extend([pk <= jk, jk <= qk])
        for var in bk_vars + pk_vars + qk_vars:
            coeffs.add(var)

    c0_vars, c0 = sympy_to_z3(b0 - a0)
    for var in c0_vars:
        coeffs.add(var)
    coeffs = list(coeffs)

    solver = z3.Solver()
    solver.set("timeout", 1000)  # milliseconds
    solver.add(z3.And(lhs == c0, *constraints))
    solution = solver.check()

    if solution == z3.unsat:
        return True  # Independence is proven for all possible assignments of `coeff`.

    return False


def get_mem_accesses(expr: sympy.Basic) -> List[Tuple[str, sympy.Basic]]:
    if isinstance(expr, sympy.Symbol):
        return [(expr.name, sympy.Number(0))]
    if isinstance(expr, SympyIndexing):
        array, index = expr.args
        if not isinstance(index, sympy.Tuple):
            index = sympy.Tuple(index)
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
