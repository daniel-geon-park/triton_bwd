from typing import Iterable, NewType, Optional, Union

import numpy as np
import sympy

from triton_bwd.sympy_utils import SymbolicArray, SymbolicScalar


class Array:
    def __init__(self, dtype: str, dims: Iterable[int]):
        self.dtype = dtype
        self.dims = dims

    def __getitem__(self, item): ...

    def __setitem__(self, key, value): ...


class ArraySpec:
    def __init__(self, dtype: str, dims: Iterable[str]):
        self.dtype = dtype
        self.dims = dims
        self.name: Optional[str] = None
        self.kind: Optional[str] = None

    def symbol(self) -> SymbolicArray:
        assert self.name is not None
        shape = sympy.Tuple(*[sympy.symbols(dim, integer=True) for dim in self.dims])
        return SymbolicArray(self.name, self.dtype, shape)


class IntSpec:
    def __init__(self, dtype: str = "int64"):
        self.dtype = dtype
        self.name: Optional[str] = None

    def symbol(self) -> SymbolicScalar:
        assert self.name is not None
        return SymbolicScalar(self.name, self.dtype)


class FloatSpec:
    def __init__(self, dtype: str = "float32"):
        self.name: Optional[str] = None
        self.dtype = dtype

    def symbol(self) -> SymbolicScalar:
        assert self.name is not None
        return SymbolicScalar(self.name, self.dtype)


ArgSpec = Union[ArraySpec, IntSpec, FloatSpec]
InArray = NewType("InArray", np.ndarray)
OutArray = NewType("OutArray", np.ndarray)
InOutArray = NewType("InOutArray", np.ndarray)
