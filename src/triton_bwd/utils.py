from sympy.core import Add, Expr, Mul, Number, Pow, Symbol
from z3 import Int, Real, Sqrt


def sympy_to_z3(sympy_exp: Expr):
    """convert a sympy expression to a z3 expression. This returns (z3_vars, z3_expression)"""

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

    return z3_vars, result_exp


def _sympy_to_z3_rec(var_map, e):
    """recursive call for sympy_to_z3()"""

    rv = None

    if not isinstance(e, Expr):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    if isinstance(e, Symbol):
        rv = var_map.get(e.name)

        if rv is None:
            raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")

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
            # sqrt
            rv = Sqrt(term)
        else:
            rv = term**exponent

    if rv is None:
        raise RuntimeError(
            "Type '"
            + str(type(e))
            + "' is not yet implemented for convertion to a z3 expresion. "
            "Subexpression was '" + str(e) + "'."
        )

    return rv
