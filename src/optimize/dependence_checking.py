import os
import time
from typing import TYPE_CHECKING, List

import sympy
import z3
from sympy.solvers.solveset import linear_coeffs

from optimize.sympy_to_z3 import sympy_to_z3

if TYPE_CHECKING:
    from optimize.abtract_tree import ForLoop


total_time = 0


def dependence_levels(
    s_before_t: bool,
    min_level: int,
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

    start_time = time.time()

    dep_levels = prove_independence(
        u_lower=min_level,
        u_upper=num_common_loops + (1 if s_before_t else 0) - 1,
        num_common_loops=num_common_loops,
        ai=ai,
        a0=a0,
        bi=bi,
        b0=b0,
        nest_s=nest_s,
        nest_t=nest_t,
    )

    elapsed_time = time.time() - start_time
    global total_time
    total_time += elapsed_time
    if os.environ.get("OPTIMIZE_VERBOSE", "0") == "1":
        print(f"{elapsed_time:.4f} seconds / total {total_time:.4f} seconds")

    return dep_levels


def prove_independence(
    u_lower: int,
    u_upper: int,
    num_common_loops: int,
    ai: List[sympy.Expr],
    a0: sympy.Expr,
    bi: List[sympy.Expr],
    b0: sympy.Expr,
    nest_s: List["ForLoop"],
    nest_t: List["ForLoop"],
):
    """Returns the list of levels u such that the independence cannot be proven."""
    lhs = a0 - b0
    iks, jks = [], []
    constraints = []

    for level in range(num_common_loops):
        ik = sympy.symbols(f"__i{level}", integer=True)
        jk = sympy.symbols(f"__j{level}", integer=True)

        pk = nest_s[level].index_begin
        qk = nest_s[level].index_end
        sk = nest_t[level].index_step
        ik, jk = ik * sk, jk * sk

        iks.append(ik)
        jks.append(jk)

        ak, bk = ai[level], bi[level]
        lhs = lhs + ak * ik - bk * jk

        constraints.extend([pk <= ik, ik < qk])
        constraints.extend([pk <= jk, jk < qk])

    for level in range(num_common_loops, len(ai)):
        ik = sympy.symbols(f"__i{level}", integer=True)

        pk = nest_s[level].index_begin
        qk = nest_s[level].index_end
        sk = nest_s[level].index_step
        ik = ik * sk

        iks.append(ik)

        ak = ai[level]
        lhs = lhs + ak * ik

        constraints.extend([pk <= ik, ik < qk])

    for level in range(num_common_loops, len(bi)):
        jk = sympy.symbols(f"__j{level}", integer=True)

        pk = nest_t[level].index_begin
        qk = nest_t[level].index_end
        sk = nest_t[level].index_step
        jk = jk * sk

        jks.append(jk)

        bk = bi[level]
        lhs = lhs - bk * jk

        constraints.extend([pk <= jk, jk < qk])

    u = sympy.symbols("__u", integer=True)
    for level in range(num_common_loops):
        ik, jk = iks[level], jks[level]
        constraints.append(sympy.Implies(level < u, sympy.Eq(ik, jk)))  # s = 0
        constraints.append(sympy.Implies(sympy.Eq(level, u), ik < jk))  # s = 1

    lhs, _ = sympy_to_z3(lhs)
    constraints = [sympy_to_z3(c)[0] for c in constraints]
    u = sympy_to_z3(u)[0]

    solver = z3.Solver()
    solver.set("timeout", 1000)  # milliseconds
    solver.add(z3.And(lhs == 0, *constraints))

    results = []
    for u_test in range(u_lower, u_upper + 1):
        solver.push()
        solver.add(u == u_test)
        solution = solver.check()
        solver.pop()

        if solution != z3.unsat:  # possible dependence
            results.append(u_test)

    return results
