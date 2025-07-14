import abc
from typing import Dict, List, Optional, Union

import sympy

from triton_bwd.sympy_utils import SymbolicArray, SymbolicScalar, SympyShape


class AbstractNode(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the node."""
        ...


class Declaration(AbstractNode):
    SymbolType = Union[SymbolicScalar, SymbolicArray]

    def __init__(self, name: str, symbol: SymbolType):
        super().__init__()
        self.name = name
        self.symbol = symbol

    def __repr__(self) -> str:
        shape = SympyShape(self.symbol)
        if shape == ():
            return f"let {self.name}: scalar"
        else:
            return f"let {self.name}: array({', '.join(map(str, shape.args))})"


class ForLoop(AbstractNode):
    def __init__(
        self,
        index_var: sympy.Symbol,
        index_begin: sympy.Basic,
        index_end: sympy.Basic,
        index_step: sympy.Basic,
        declarations: Dict[str, Declaration],
        statements: List[AbstractNode],
        is_kernel: bool = False,
        arguments: Optional[Dict[str, sympy.Basic]] = None,
    ):
        super().__init__()
        assert index_step == 1, "Only step size of 1 is supported for now."
        self.index_var = index_var
        self.index_begin = index_begin
        self.index_end = index_end
        self.index_step = index_step
        self.declarations = declarations
        self.statements = statements
        self.is_kernel = is_kernel
        self.arguments = arguments

    def __repr__(self):
        stmt_reprs = []
        for decl in self.declarations.values():
            stmt_reprs.append(repr(decl))
        for stmt in self.statements:
            stmt_repr = repr(stmt)
            stmt_reprs.extend(stmt_repr.split("\n"))
        return (
            f"for {self.index_var} in range({self.index_begin}, {self.index_end}, {self.index_step}):\n"
            + "\n".join(f"    {stmt_repr}" for stmt_repr in stmt_reprs)
        )

    def rename_index_var(self, new_name: str):
        """Renames the index variable of the loop."""
        self.index_var = sympy.Symbol(new_name, integer=True)
        # TODO: check for conflicts with existing declarations in the loop
        # TODO: rename all occurrences of the index variable in the loop's statements


class Assignment(AbstractNode):
    def __init__(self, target: sympy.Basic, value: sympy.Basic):
        super().__init__()
        self.target = target
        self.value = value

    def __repr__(self):
        return f"{self.target} = {self.value}"
