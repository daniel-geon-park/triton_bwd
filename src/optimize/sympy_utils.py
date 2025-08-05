from typing import Tuple, Union

import sympy
from sympy.core.assumptions import StdFactKB


class Float16(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    def _sympystr(self, p):
        return p.doprint("float16")


class Float32(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    def _sympystr(self, p):
        return p.doprint("float32")


class Float64(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    def _sympystr(self, p):
        return p.doprint("float64")


class UnknownFloat(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    pass


class Int8(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    def _sympystr(self, p):
        return p.doprint("int8")


class Int16(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    def _sympystr(self, p):
        return p.doprint("int16")


class Int32(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    def _sympystr(self, p):
        return p.doprint("int32")


class Int64(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    def _sympystr(self, p):
        return p.doprint("int64")


class UnknownInt(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    pass


float16 = Float16()
float32 = Float32()
float64 = Float64()
unknown_float = UnknownFloat()
FLOAT_TYPES = [float16, float32, float64, unknown_float]

int8 = Int8()
int16 = Int16()
int32 = Int32()
int64 = Int64()
unknown_int = UnknownInt()
INT_TYPES = [int8, int16, int32, int64, unknown_int]

UNKNOWN_TYPES = [unknown_float, unknown_int]


TYPE_MAP = {
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
}


class SymbolicScalar(sympy.Expr):
    @staticmethod
    def _set_assumptions(obj, assumptions):
        """Set assumptions on obj, making sure to apply consistent values."""
        tmp_asm_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = tmp_asm_copy  # Issue #8873

    def __new__(cls, name: Union[sympy.Basic, str], dtype: Union[sympy.Basic, str]):
        if isinstance(dtype, str):
            if dtype not in TYPE_MAP:
                raise ValueError(f"Unsupported dtype: {dtype}")
            dtype = TYPE_MAP[dtype]

        if dtype not in TYPE_MAP.values():
            raise ValueError(f"Unsupported dtype: {dtype}")

        symbol = name
        if isinstance(name, str):
            if dtype in INT_TYPES:
                symbol = sympy.symbols(name, integer=True)
            elif dtype in FLOAT_TYPES:
                symbol = sympy.symbols(name, real=True)

        obj = sympy.Expr.__new__(cls, symbol, dtype)
        cls._set_assumptions(obj, symbol._assumptions)
        return obj

    @property
    def label(self):
        return self.args[0]

    @property
    def dtype(self):
        return self.args[1]

    @property
    def shape(self):
        return sympy.Tuple()

    def _sympystr(self, p):
        return p.doprint(self.label)


class SymbolicArray(sympy.Expr):
    @staticmethod
    def _set_assumptions(obj, assumptions):
        """Set assumptions on obj, making sure to apply consistent values."""
        tmp_asm_copy = assumptions.copy()
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = tmp_asm_copy  # Issue #8873

    def __new__(
        cls,
        name: Union[sympy.Basic, str],
        dtype: Union[sympy.Basic, str],
        shape: Union[sympy.Tuple, tuple],
        is_placeholder: bool = False,
    ):
        if isinstance(dtype, str):
            if dtype not in TYPE_MAP:
                raise ValueError(f"Unsupported dtype: {dtype}")
            dtype = TYPE_MAP[dtype]

        if dtype not in TYPE_MAP.values():
            raise ValueError(f"Unsupported dtype: {dtype}")

        symbol = name
        if isinstance(name, str):
            if dtype in INT_TYPES:
                symbol = sympy.symbols(name, integer=True)
            elif dtype in FLOAT_TYPES:
                symbol = sympy.symbols(name, real=True)

        obj = sympy.Expr.__new__(
            cls,
            symbol,
            dtype,
            sympy.sympify(shape),
            is_placeholder,
        )
        cls._set_assumptions(obj, symbol._assumptions)
        return obj

    @property
    def label(self):
        return self.args[0]

    @property
    def dtype(self):
        return self.args[1]

    @property
    def shape(self):
        return self.args[2]

    @property
    def is_placeholder(self):
        return self.args[3]

    def _sympystr(self, p):
        return p.doprint(self.label)


class SympyIndexing(sympy.Function):
    is_commutative = True

    @classmethod
    def eval(cls, array, index):
        pass

    def _sympystr(self, printer):
        array, index = self.args
        if isinstance(index, sympy.Tuple):
            if len(index.args) == 0:
                index_str = "()"
            else:
                index_str = ", ".join(printer.doprint(i) for i in index)
        else:
            index_str = printer.doprint(index)
        return printer.doprint(array) + "[" + index_str + "]"

    @property
    def array(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]


class SympyDtype(sympy.Function):
    @classmethod
    def eval(cls, array):
        if isinstance(array, sympy.Number):
            if array.is_integer:
                return unknown_int
            return unknown_float
        if isinstance(array, (SymbolicScalar, SymbolicArray)):
            return array.dtype
        if isinstance(array, (sympy.Add, sympy.Mul, sympy.exp, sympy.Pow)):
            # Elementwise operations on arrays
            dtypes = [SympyDtype(arg) for arg in array.args]
            dtype = dtypes[0]
            for other_dtype in dtypes[1:]:
                if other_dtype in UNKNOWN_TYPES:
                    dtype, other_dtype = other_dtype, dtype
                if dtype == unknown_int:
                    if other_dtype in INT_TYPES:
                        dtype = other_dtype
                    elif other_dtype in FLOAT_TYPES:
                        # Promote to float
                        dtype = other_dtype
                    else:
                        raise ValueError(
                            f"Incompatible dtypes: {dtype} and {other_dtype}"
                        )
                elif dtype == unknown_float:
                    if other_dtype in FLOAT_TYPES:
                        dtype = other_dtype
                    elif other_dtype == unknown_int:
                        # Promote to float
                        pass
                    else:
                        raise ValueError(
                            f"Incompatible dtypes: {dtype} and {other_dtype}"
                        )
                elif dtype != other_dtype:
                    raise ValueError(f"Incompatible dtypes: {dtype} and {other_dtype}")
            return dtype
        if isinstance(array, SympyIndexing):
            x, index = array.args
            return SympyDtype(x)
        raise ValueError(f"Unsupported type for dtype: {type(array)}")


class SympyShape(sympy.Function):
    @classmethod
    def eval(cls, array):
        if isinstance(array, sympy.Number):
            result = sympy.Tuple()
        elif isinstance(array, (SymbolicScalar, SymbolicArray)):
            result = array.shape
        elif isinstance(array, (sympy.Add, sympy.Mul, sympy.exp, sympy.Pow)):
            # Elementwise operations on arrays
            shapes = [SympyShape(arg) for arg in array.args]
            shape = shapes[0]
            for other_shape in shapes[1:]:
                shape = broadcasat_shapes(shape, other_shape)
            result = shape
        elif isinstance(array, SympyIndexing):
            array_shape = SympyShape(array.array)
            result_shape = []
            axis = 0
            for idx in array.index.args:
                if isinstance(idx, SympySlice):
                    dim = array_shape.args[axis]
                    axis += 1
                    result_shape.append(idx.calc_dim(dim))
                # TODO: handle new axis
                else:
                    # Assume single index access
                    axis += 1
            result = sympy.Tuple(*result_shape)
        else:
            # TODO: implement other array operations
            raise ValueError(f"Unsupported type for shape: {type(array)}")

        for dim in result.args:
            assert dim.is_integer

        return result


class SympySliceSentinel(sympy.Basic, metaclass=sympy.core.singleton.Singleton):
    def _sympystr(self, p):
        return ""


SENTINEL_INDEX = SympySliceSentinel()


class SympySlice(sympy.Function):
    @classmethod
    def eval(cls, start, stop, step):
        pass

    def _sympystr(self, printer):
        start, stop, step = self.args
        result = f"{printer.doprint(start)}:{printer.doprint(stop)}"
        if step != 1:
            result = f"{result}:{printer.doprint(step)}"
        return result

    def calc_dim(self, length: sympy.Basic) -> sympy.Basic:
        """Calculate the dimension of the slice given the length of the array."""
        assert self.step.is_positive, "Only positive steps are supported."
        start, stop, step = self.args
        if start == SENTINEL_INDEX:
            start = 0
        if stop == SENTINEL_INDEX:
            stop = length
        return ceildiv(stop - start, step)

    @property
    def start(self):
        return self.args[0]

    @property
    def stop(self):
        return self.args[1]

    @property
    def step(self):
        return self.args[2]


def sympy_slice(*args):
    """Create a symbolic slice."""
    if len(args) == 0:
        return SympySlice(SENTINEL_INDEX, SENTINEL_INDEX, sympy.S.One)
    elif len(args) == 1:
        return SympySlice(SENTINEL_INDEX, args[0], sympy.S.One)
    elif len(args) == 2:
        return SympySlice(args[0], args[1], sympy.S.One)
    elif len(args) == 3:
        start, stop, step = args
        return SympySlice(start, stop, step)
    else:
        raise ValueError(f"Invalid number of arguments for sympy_slice: {len(args)}")


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


class IntegerDivision(sympy.Function):
    @classmethod
    def eval(cls, a, b):
        if isinstance(a, sympy.Number) and isinstance(b, sympy.Number):
            return int(a) // int(b)
        if b == 1:
            return a

    def _sympystr(self, printer):
        a, b = self.args
        return f"idiv({printer.doprint(a)}, {printer.doprint(b)})"


def floordiv(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return IntegerDivision(a, b)


def ceildiv(a: sympy.Basic, b: sympy.Basic) -> sympy.Basic:
    return floordiv(a + b - 1, b)
