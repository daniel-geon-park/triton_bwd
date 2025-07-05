from typing import Dict, List, NewType, Optional, Union

import numpy as np
import sympy


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
        index_var: str,
        index_begin: int,
        index_end: int,
        index_step: int,
        declarations: Dict[str, sympy.Basic],
        statements: List["AbstractNode"],
    ):
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
                f"for {self.index_var} in range({self.index_begin}, {self.index_end}, {self.index_step}):",
            )
        ]
        loop_idx = 1

        for name, decl in self.declarations.items():
            shape = SympyShape(decl)
            if shape == ():
                result.append((("D", decl_idx), "    " + f"let {name}: scalar"))
            else:
                result.append(
                    (
                        ("D", decl_idx),
                        "    "
                        + f"let {name}: array({', '.join(map(str, shape.args))})",
                    )
                )
            decl_idx += 1

        for stmt in self.statements:
            numbered_stmt = stmt.add_numbers()
            for (kind, num), text in numbered_stmt:
                if kind == "S":
                    result.append(((kind, stmt_idx), "    " + text))
                    stmt_idx += 1
                elif kind == "L":
                    result.append(((kind, loop_idx), "    " + text))
                    loop_idx += 1
                elif kind == "D":
                    result.append(((kind, decl_idx), "    " + text))
                    decl_idx += 1

        return result


class Assignment:
    def __init__(self, target: sympy.Basic, value: sympy.Basic):
        self.target = target
        self.value = value

    def __repr__(self):
        return f"{self.target} = {self.value}"

    def add_numbers(self):
        return [(("S", 0), repr(self))]


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
            f"{f'{kind}{num}':>5}: {text}" for ((kind, num), text) in numbered
        )
