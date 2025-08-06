from typing import Iterable, NewType, Optional, Union

import numpy as np


class Array:
    def __init__(self, dtype: str, dims: Iterable[int]):
        self.dtype = dtype
        self.dims = dims

    def __getitem__(self, item): ...

    def __setitem__(self, key, value): ...


class ArraySpec:
    def __init__(self, dtype: str, dims: Iterable[str]):
        self.dtype = dtype
        self.dims = list(dims)
        self.name: Optional[str] = None
        self.kind: Optional[str] = None


class IntSpec:
    def __init__(self, dtype: str = "int64"):
        self.dtype = dtype
        self.name: Optional[str] = None


class FloatSpec:
    def __init__(self, dtype: str = "float32"):
        self.name: Optional[str] = None
        self.dtype = dtype


ArgSpec = Union[ArraySpec, IntSpec, FloatSpec]
InArray = NewType("InArray", np.ndarray)
OutArray = NewType("OutArray", np.ndarray)
InOutArray = NewType("InOutArray", np.ndarray)
