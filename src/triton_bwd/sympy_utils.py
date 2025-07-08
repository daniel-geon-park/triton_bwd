from typing import Union

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
