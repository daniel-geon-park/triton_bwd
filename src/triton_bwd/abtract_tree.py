import copy
from typing import Dict, List, NewType, Optional, Tuple, Union

import numpy as np
import sympy
from litellm import success_callback

from triton_bwd.dependence_checking import dependence_level, get_mem_accesses
from triton_bwd.sympy_utils import SympyShape


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


class FloatSpec:
    def __init__(self):
        self.name: Optional[str] = None

    def symbol(self) -> sympy.Symbol:
        assert self.name is not None
        return sympy.symbols(self.name, real=True)


ArgSpec = Union[ArraySpec, IntSpec, FloatSpec]
InArray = NewType("InArray", np.ndarray)
OutArray = NewType("OutArray", np.ndarray)
InOutArray = NewType("InOutArray", np.ndarray)


class ForLoop:
    def __init__(
        self,
        index_var: sympy.Symbol,
        index_begin: sympy.Basic,
        index_end: sympy.Basic,
        index_step: sympy.Basic,
        declarations: Dict[str, sympy.Basic],
        statements: List["AbstractNode"],
    ):
        assert index_step == 1, "Only step size of 1 is supported for now."
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

        text = f"for {self.index_var} in range({self.index_begin}, {self.index_end}, {self.index_step}):"
        result = [
            NumberedStmt(
                kind="L",
                num=0,
                obj=self,
                parent=None,
                prev=None,
                succ=None,
                text=text,
            )
        ]
        loop_idx = 1

        for name, decl in self.declarations.items():
            shape = SympyShape(decl)
            if shape == ():
                result.append(
                    NumberedStmt(
                        kind="D",
                        num=decl_idx,
                        obj=(name, decl),
                        parent=self,
                        prev=None,
                        succ=None,
                        text=f"    let {name}: scalar",
                    )
                )
            else:
                result.append(
                    NumberedStmt(
                        kind="D",
                        num=decl_idx,
                        obj=(name, decl),
                        parent=self,
                        prev=None,
                        succ=None,
                        text=f"    let {name}: array({', '.join(map(str, shape.args))})",
                    )
                )
            decl_idx += 1

        for idx in range(len(self.statements)):
            stmt = self.statements[idx]
            prev_stmt = self.statements[idx - 1].content if idx > 0 else None
            succ_stmt = (
                self.statements[idx + 1].content
                if idx < len(self.statements) - 1
                else None
            )

            numbered_stmts = stmt.add_numbers()
            if len(numbered_stmts) > 0:
                numbered_stmts[0].parent = self
                numbered_stmts[0].prev = prev_stmt
                numbered_stmts[0].succ = succ_stmt

            for sub_stmt in numbered_stmts:
                sub_stmt.text = "    " + sub_stmt.text
                if sub_stmt.kind == "S":
                    sub_stmt.num = stmt_idx
                    stmt_idx += 1
                elif sub_stmt.kind == "L":
                    sub_stmt.num = loop_idx
                    loop_idx += 1
                elif sub_stmt.kind == "D":
                    sub_stmt.num = decl_idx
                    decl_idx += 1
                result.append(sub_stmt)

        return result

    def get_stmt_impl(self, i: int, loop_idx: int, stmt_idx: int):
        my_loop_idx = loop_idx
        loop_idx += 1
        for stmt in self.statements:
            r, loop_idx, stmt_idx = stmt.get_stmt_impl(i, loop_idx, stmt_idx)
            if r is not None:
                S, loop_nest, loop_idx_nest = r
                new_r = (S, [self] + loop_nest, [my_loop_idx] + loop_idx_nest)
                return new_r, loop_idx, stmt_idx
        return None, loop_idx, stmt_idx

    def rename_index_var(self, new_name: str):
        """Renames the index variable of the loop."""
        self.index_var = sympy.Symbol(new_name, integer=True)
        # TODO: check for conflicts with existing variable names in the loop
        # TODO: rename all occurrences of the index variable in the loop's statements


class Assignment:
    def __init__(self, target: sympy.Basic, value: sympy.Basic):
        self.target = target
        self.value = value

    def __repr__(self):
        return f"{self.target} = {self.value}"

    def add_numbers(self):
        return [
            NumberedStmt(
                kind="S",
                num=0,
                obj=self,
                parent=None,
                prev=None,
                succ=None,
                text=repr(self),
            )
        ]

    def get_stmt_impl(self, i: int, loop_idx: int, stmt_idx: int):
        if i == stmt_idx:
            return (self, [], []), loop_idx, stmt_idx + 1
        return None, loop_idx, stmt_idx + 1


Decl = Tuple[str, sympy.Basic]
Stmt = Union[ForLoop, Decl, Assignment]


class NumberedStmt:
    def __init__(
        self,
        kind: str,
        num: int,
        obj: Stmt,
        parent: Optional[ForLoop],
        prev: Optional[Stmt],
        succ: Optional[Stmt],
        text: str,
    ):
        self.kind = kind
        self.num = num
        self.obj = obj
        self.parent = parent
        self.prev = prev
        self.succ = succ
        self.text = text

    def __repr__(self):
        return f"{f'{self.kind}{self.num}':>5}: {self.text}"


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
            f"{f'{stmt.kind}{stmt.num}':>5}: {stmt.text}" for stmt in numbered
        )

    def get_stmt_impl(self, i: int, loop_idx: int, stmt_idx: int):
        return self.content.get_stmt_impl(i, loop_idx, stmt_idx)

    def get_stmt(self, i: int) -> Tuple[Assignment, List[ForLoop], List[int]]:
        result, _, _ = self.get_stmt_impl(i, 0, 0)
        return result

    def find_dependence(self, i: int, j: int):
        S, nest_S, nest_idx_S = self.get_stmt(i)
        T, nest_T, nest_idx_T = self.get_stmt(j)
        S_stores = get_mem_accesses(S.target)
        S_loads = get_mem_accesses(S.value)
        T_stores = get_mem_accesses(T.target)
        T_loads = get_mem_accesses(T.value)
        dependencies = []
        # Flow dependencies
        for name_s, index_s in S_stores:
            for name_t, index_t in T_loads:
                if name_s == name_t:
                    u = dependence_level(i < j, index_s, nest_S, index_t, nest_T)
                    if u is not None:
                        dependencies.append(("flow", u, name_s))
        # Antidependencies
        for name_s, index_s in S_loads:
            for name_t, index_t in T_stores:
                if name_s == name_t:
                    u = dependence_level(i < j, index_s, nest_S, index_t, nest_T)
                    if u is not None:
                        dependencies.append(("anti", u, name_s))
        # Output dependencies
        for name_s, index_s in S_stores:
            for name_t, index_t in T_stores:
                if name_s == name_t:
                    u = dependence_level(i < j, index_s, nest_S, index_t, nest_T)
                    if u is not None:
                        dependencies.append(("outp", u, name_s))
        return dependencies

    def fuse_loop(self, loop_idx_a: int, loop_idx_b: int):
        """Fuses two consecutive loops."""
        new_tree = copy.deepcopy(self)  # Ensure we don't modify the original tree

        numbered = new_tree.add_numbers()
        loop_a = loop_b = None
        for stmt in numbered:
            if stmt.kind == "L" and stmt.num == loop_idx_a:
                loop_a = stmt
            elif stmt.kind == "L" and stmt.num == loop_idx_b:
                loop_b = stmt

        if loop_a is None or loop_b is None:
            raise ValueError(f"Invalid loop indices")

        if loop_a.succ is not loop_b.obj:
            raise ValueError(f"Loops {loop_idx_a} and {loop_idx_b} are not consecutive")

        loop_a, loop_b = loop_a.obj, loop_b.obj
        loop_b.rename_index_var(loop_a.index_var.name)

        if loop_a.index_begin != loop_b.index_begin:
            raise ValueError(
                f"Loops {loop_idx_a} and {loop_idx_b} have different start indices"
            )
        if loop_a.index_end != loop_b.index_end:
            raise ValueError(
                f"Loops {loop_idx_a} and {loop_idx_b} have different end indices"
            )
        if loop_a.index_step != loop_b.index_step:
            raise ValueError(
                f"Loops {loop_idx_a} and {loop_idx_b} have different step sizes"
            )

        # Fuse the loops
        new_declarations = {**loop_a.declarations, **loop_b.declarations}
        new_statements = loop_a.statements + loop_b.statements
        new_loop = ForLoop(
            index_var=loop_a.index_var,
            index_begin=loop_a.index_begin,
            index_end=loop_a.index_end,
            index_step=loop_a.index_step,
            declarations=new_declarations,
            statements=new_statements,
        )

        print(new_loop)
