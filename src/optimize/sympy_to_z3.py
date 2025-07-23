from typing import Union

import sympy
import z3
from sympy.core import Add, Expr, Mul, Number, Pow, Symbol
from sympy.core.relational import Relational
from z3 import Int, Real, Sqrt

from optimize.sympy_utils import SymbolicScalar


def sympy_to_z3(sympy_exp: Union[Expr, Relational]):
    """convert a sympy expression to a z3 expression. This returns (z3_vars, z3_expression)"""

    if isinstance(sympy_exp, Relational):
        left, left_vars = sympy_to_z3(sympy_exp.lhs)
        right, right_vars = sympy_to_z3(sympy_exp.rhs)
        var_list = list(set(left_vars) | set(right_vars))
        if isinstance(sympy_exp, sympy.LessThan):
            return left <= right, var_list
        elif isinstance(sympy_exp, sympy.StrictLessThan):
            return left < right, var_list
        elif isinstance(sympy_exp, sympy.GreaterThan):
            return left >= right, var_list
        elif isinstance(sympy_exp, sympy.StrictGreaterThan):
            return left > right, var_list
        elif isinstance(sympy_exp, sympy.Equality):
            return left == right, var_list
        elif isinstance(sympy_exp, sympy.Unequality):
            return left != right, var_list
        else:
            raise RuntimeError(
                f"Unsupported relational expression type: {type(sympy_exp)}"
            )
    elif isinstance(sympy_exp, sympy.Implies):
        left, left_vars = sympy_to_z3(sympy_exp.args[0])
        right, right_vars = sympy_to_z3(sympy_exp.args[1])
        var_list = list(set(left_vars) | set(right_vars))
        return z3.Implies(left, right), var_list

    elif isinstance(sympy_exp, (int, float, bool)):
        return sympy_exp, []

    z3_vars = []
    z3_var_map = {}

    sympy_var_list = sympy_exp.free_symbols

    for var in sympy_var_list:
        name = var.name
        if var.is_integer:
            z3_var = Int(name)
        else:
            z3_var = Real(name)
        z3_var_map[name] = z3_var
        z3_vars.append(z3_var)

    result_exp = _sympy_to_z3_rec(z3_var_map, sympy_exp)

    return result_exp, z3_vars


def _sympy_to_z3_rec(var_map, e):
    """recursive call for sympy_to_z3()"""
    rv = None

    if not isinstance(e, Expr):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    if isinstance(e, Symbol):
        rv = var_map.get(e.name)
        if rv is None:
            raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")

    elif isinstance(e, SymbolicScalar):
        rv = _sympy_to_z3_rec(var_map, e.label)

    elif isinstance(e, Number):
        rv = int(e) if e.is_integer else float(e)

    elif isinstance(e, Mul):
        rv = _sympy_to_z3_rec(var_map, e.args[0])
        for child in e.args[1:]:
            rv *= _sympy_to_z3_rec(var_map, child)

    elif isinstance(e, Add):
        rv = _sympy_to_z3_rec(var_map, e.args[0])
        for child in e.args[1:]:
            rv += _sympy_to_z3_rec(var_map, child)

    elif isinstance(e, Pow):
        term = _sympy_to_z3_rec(var_map, e.args[0])
        exponent = _sympy_to_z3_rec(var_map, e.args[1])
        if exponent == 0.5:
            rv = Sqrt(term)
        else:
            rv = term**exponent

    if rv is None:
        raise RuntimeError(
            f"Type '{str(type(e))}' is not yet implemented for convertion to a z3 expresion. "
            f"Subexpression was '{str(e)}'."
        )

    return rv
