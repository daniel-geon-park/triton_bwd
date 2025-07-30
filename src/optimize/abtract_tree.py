import abc
from typing import Dict, List, Optional, Union

import sympy

from optimize.sympy_utils import SymbolicArray, SymbolicScalar, SympyShape


class AbstractNode(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the node."""
        ...

    @property
    @abc.abstractmethod
    def exprs(self) -> List[sympy.Basic]:
        """Returns a list of sympy expressions associated with this node."""
        ...

    @exprs.setter
    @abc.abstractmethod
    def exprs(self, value: List[sympy.Basic]):
        """Sets the sympy expressions associated with this node."""
        ...


class Declaration(AbstractNode):
    def __init__(self, name: str, symbol: Union[SymbolicScalar, SymbolicArray]):
        super().__init__()
        self.name = name
        self.symbol = symbol

    def __repr__(self) -> str:
        if isinstance(self.symbol, SymbolicScalar):
            return f"let {self.name}: scalar"
        shape = SympyShape(self.symbol)
        return f"let {self.name}: array({', '.join(map(str, shape.args))})"

    @property
    def exprs(self) -> List[sympy.Basic]:
        return [self.symbol]

    @exprs.setter
    def exprs(self, value: List[sympy.Basic]):
        if len(value) != 1:
            raise ValueError("Expected exactly 1 expression for Declaration.")
        self.symbol = value[0]
        if not isinstance(self.symbol, (SymbolicScalar, SymbolicArray)):
            raise TypeError(
                "Declaration symbol must be a SymbolicScalar or SymbolicArray."
            )


class ForLoop(AbstractNode):
    def __init__(
        self,
        index_var: SymbolicScalar,
        index_begin: sympy.Basic,
        index_end: sympy.Basic,
        index_step: sympy.Basic,
        declarations: Dict[str, Declaration],
        statements: List[AbstractNode],
        is_kernel: bool = False,
        arguments: Optional[Dict[str, sympy.Basic]] = None,
    ):
        super().__init__()
        assert (
            index_step.is_positive is True
        ), "Only positive step size is supported for now."
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

    @property
    def exprs(self) -> List[sympy.Basic]:
        return [self.index_var, self.index_begin, self.index_end, self.index_step]

    @exprs.setter
    def exprs(self, value: List[sympy.Basic]):
        if len(value) != 4:
            raise ValueError("Expected exactly 4 expressions for ForLoop.")
        self.index_var, self.index_begin, self.index_end, self.index_step = value

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

    @property
    def exprs(self) -> List[sympy.Basic]:
        return [self.target, self.value]

    @exprs.setter
    def exprs(self, value: List[sympy.Basic]):
        if len(value) != 2:
            raise ValueError("Expected exactly 2 expressions for Assignment.")
        self.target, self.value = value
