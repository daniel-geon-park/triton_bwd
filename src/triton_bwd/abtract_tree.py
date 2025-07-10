import abc
import copy
from typing import Dict, List, Optional, Set, Tuple, Union

import sympy

from triton_bwd.dependence_checking import dependence_levels
from triton_bwd.sympy_utils import SympyIndexing, SympyShape


def get_mem_accesses(
    expr: sympy.Basic,
    loop_nest: List["NumberedStmt"],
) -> List[Tuple[str, Optional["NumberedStmt"], sympy.Basic]]:

    if isinstance(expr, sympy.Symbol):
        decl_loop = None
        for loop in loop_nest[::-1]:
            if expr.name in loop.obj.declarations:
                decl_loop = loop
                break
        return [(expr.name, decl_loop, sympy.Number(0))]

    if isinstance(expr, SympyIndexing):
        array, index = expr.args
        if not isinstance(index, sympy.Tuple):
            index = sympy.Tuple(index)
        assert isinstance(array, sympy.IndexedBase)
        flat_index = sympy.Number(0)
        shape = SympyShape(array)
        for dim, idx in zip(shape.args, index.args):
            flat_index = flat_index * dim + idx

        decl_loop = None
        for loop in loop_nest[::-1]:
            if array.name in loop.obj.declarations:
                decl_loop = loop
                break

        return [(array.name, decl_loop, flat_index)]

    results = []
    for arg in expr.args:
        results.extend(get_mem_accesses(arg, loop_nest))
    return results


class AbstractNode(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Returns a string representation of the node."""
        pass

    @abc.abstractmethod
    def add_numbers(self) -> List["NumberedStmt"]:
        pass

    def numbered_repr(self) -> str:
        numbered = self.add_numbers()
        return "\n".join(
            f"{f'{stmt.kind}{stmt.num}/{stmt.level}':>6}: {stmt.text}"
            for stmt in numbered
        )

    def find_dependence(self, i: int, j: int) -> Set[Tuple[str, int, str]]:
        """Finds dependencies between two assignments."""
        numbered = self.add_numbers()

        S = T = None
        for stmt in numbered:
            if (stmt.kind, stmt.num) == ("A", i):
                S = stmt
            if (stmt.kind, stmt.num) == ("A", j):
                T = stmt

        nest_S, nest_T = [S.parent], [T.parent]
        while nest_S[0].parent is not None:
            nest_S.insert(0, nest_S[0].parent)
        while nest_T[0].parent is not None:
            nest_T.insert(0, nest_T[0].parent)

        S_stores = get_mem_accesses(S.obj.target, nest_S)
        S_loads = get_mem_accesses(S.obj.value, nest_S)
        T_stores = get_mem_accesses(T.obj.target, nest_T)
        T_loads = get_mem_accesses(T.obj.value, nest_T)

        nest_S = [stmt.obj for stmt in nest_S]
        nest_T = [stmt.obj for stmt in nest_T]

        dependencies = set()

        def add_deps(dep_kind: str, S_vars, T_vars):
            for name_s, decl_s, index_s in S_vars:
                for name_t, decl_t, index_t in T_vars:
                    if (name_s, decl_s) == (name_t, decl_t):
                        min_level = 0
                        if decl_s is not None:
                            min_level = decl_s.level + 1
                        dep_levels = dependence_levels(
                            s_before_t=i < j,
                            min_level=min_level,
                            index_s=index_s,
                            nest_s=nest_S,
                            index_t=index_t,
                            nest_t=nest_T,
                        )
                        for u in dep_levels:
                            dependencies.add((dep_kind, u, name_s))

        # Flow dependencies
        add_deps("flow", S_stores, T_loads)
        # Antidependencies
        add_deps("anti", S_loads, T_stores)
        # Output dependencies
        add_deps("outp", S_stores, T_stores)

        return dependencies

    def find_stmt_dependence(
        self,
        a: Union["AbstractNode", Tuple[str, int]],
        b: Union["AbstractNode", Tuple[str, int]],
    ) -> Set[Tuple[str, int, str]]:
        """Finds dependencies between two statements (loops or assignments)."""
        numbered = self.add_numbers()
        stmt_a = stmt_b = None
        a_idx = b_idx = None

        for idx, stmt in enumerate(numbered):
            if stmt.obj is a or (stmt.kind, stmt.num) == a:
                stmt_a = stmt
                a_idx = idx
            if stmt.obj is b or (stmt.kind, stmt.num) == b:
                stmt_b = stmt
                b_idx = idx

        if stmt_a is None:
            raise ValueError(f"Statement {a} not found in the tree")
        if stmt_b is None:
            raise ValueError(f"Statement {b} not found in the tree")

        a_len = len(stmt_a.obj.add_numbers())
        b_len = len(stmt_b.obj.add_numbers())

        a_stmts = numbered[a_idx : a_idx + a_len]
        b_stmts = numbered[b_idx : b_idx + b_len]

        a_asgn_indices = [stmt.num for stmt in a_stmts if stmt.kind == "A"]
        b_asgn_indices = [stmt.num for stmt in b_stmts if stmt.kind == "A"]

        dependencies = set()
        for i in a_asgn_indices:
            for j in b_asgn_indices:
                deps = self.find_dependence(i, j)
                dependencies.update(deps)

        return dependencies

    def find_stmt_block_dependence(
        self,
        a: List[Union["AbstractNode", Tuple[str, int]]],
        b: List[Union["AbstractNode", Tuple[str, int]]],
    ) -> Set[Tuple[str, int, str]]:
        """Finds dependencies between two blocks of statements."""
        dependencies = set()
        for stmt_a in a:
            for stmt_b in b:
                deps = self.find_stmt_dependence(stmt_a, stmt_b)
                dependencies.update(deps)
        return dependencies

    def fuse_loop(self, loop_idx_a: int, loop_idx_b: int) -> "AbstractNode":
        """Fuses two consecutive loops."""
        new_tree = copy.deepcopy(self)  # Ensure we don't modify the original tree

        if loop_idx_b < loop_idx_a:
            loop_idx_a, loop_idx_b = loop_idx_b, loop_idx_a

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
            raise ValueError(
                f"Loops L{loop_idx_a} and L{loop_idx_b} are not consecutive:\n"
                + self.numbered_repr()
            )

        loop_level, parent_loop = loop_a.level, loop_a.parent.obj

        loop_a, loop_b = loop_a.obj, loop_b.obj

        if loop_a.index_var.name != loop_b.index_var.name:
            raise ValueError(
                f"Loops L{loop_idx_a} and L{loop_idx_b} have different index variable names:\n"
                + self.numbered_repr()
            )
        if loop_a.index_begin != loop_b.index_begin:
            raise ValueError(
                f"Loops L{loop_idx_a} and L{loop_idx_b} have different start indices:\n"
                + self.numbered_repr()
            )
        if loop_a.index_end != loop_b.index_end:
            raise ValueError(
                f"Loops L{loop_idx_a} and L{loop_idx_b} have different end indices:\n"
                + self.numbered_repr()
            )
        if loop_a.index_step != loop_b.index_step:
            raise ValueError(
                f"Loops L{loop_idx_a} and L{loop_idx_b} have different step sizes:\n"
                + self.numbered_repr()
            )

        # Check for clashing declarations
        if set(loop_a.declarations.keys()) & set(loop_b.declarations.keys()):
            raise ValueError(
                f"Some declarations in Loops L{loop_idx_a} and L{loop_idx_b} clash:\n"
                + self.numbered_repr()
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

        loop_a_idx = parent_loop.statements.index(loop_a)
        parent_loop.statements[loop_a_idx] = new_loop
        parent_loop.statements.remove(loop_b)

        new_deps = new_tree.find_stmt_block_dependence(
            loop_b.statements,
            loop_a.statements,
        )
        for dep_kind, level, var_name in new_deps:
            if level == loop_level:
                raise ValueError(
                    f"Fusing loops {loop_idx_a} and {loop_idx_b} introduces "
                    f"a dependence at the same level {level} for variable `{var_name}`:\n"
                    + self.numbered_repr()
                )

        return new_tree


class ForLoop(AbstractNode):
    def __init__(
        self,
        index_var: sympy.Symbol,
        index_begin: sympy.Basic,
        index_end: sympy.Basic,
        index_step: sympy.Basic,
        declarations: Dict[str, sympy.Basic],
        statements: List[AbstractNode],
    ):
        super().__init__()
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
        asgn_idx = 0
        decl_idx = 0

        text = f"for {self.index_var} in range({self.index_begin}, {self.index_end}, {self.index_step}):"
        result = [
            NumberedStmt(
                kind="L",
                num=0,
                obj=self,
                level=0,
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
                        level=1,
                        parent=result[0],
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
                        level=1,
                        parent=result[0],
                        prev=None,
                        succ=None,
                        text=f"    let {name}: array({', '.join(map(str, shape.args))})",
                    )
                )
            decl_idx += 1

        for idx in range(len(self.statements)):
            stmt = self.statements[idx]
            prev_stmt = self.statements[idx - 1] if idx > 0 else None
            succ_stmt = (
                self.statements[idx + 1] if idx < len(self.statements) - 1 else None
            )

            numbered_stmts = stmt.add_numbers()
            if len(numbered_stmts) > 0:
                numbered_stmts[0].parent = result[0]
                numbered_stmts[0].prev = prev_stmt
                numbered_stmts[0].succ = succ_stmt

            for sub_stmt in numbered_stmts:
                sub_stmt.level += 1
                sub_stmt.text = "    " + sub_stmt.text
                if sub_stmt.kind == "A":
                    sub_stmt.num = asgn_idx
                    asgn_idx += 1
                elif sub_stmt.kind == "L":
                    sub_stmt.num = loop_idx
                    loop_idx += 1
                elif sub_stmt.kind == "D":
                    sub_stmt.num = decl_idx
                    decl_idx += 1
                result.append(sub_stmt)

        return result

    def rename_index_var(self, new_name: str):
        """Renames the index variable of the loop."""
        self.index_var = sympy.Symbol(new_name, integer=True)
        # TODO: check for conflicts with existing declarations in the loop
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
                kind="A",
                num=0,
                obj=self,
                level=0,
                parent=None,
                prev=None,
                succ=None,
                text=repr(self),
            )
        ]


Decl = Tuple[str, sympy.Basic]
Stmt = Union[ForLoop, Decl, Assignment]


class NumberedStmt:
    def __init__(
        self,
        kind: str,
        num: int,
        obj: Stmt,
        level: int,
        parent: Optional["NumberedStmt"],
        prev: Optional[Stmt],
        succ: Optional[Stmt],
        text: str,
    ):
        self.kind = kind
        self.num = num
        self.obj = obj
        self.level = level
        self.parent = parent
        self.prev = prev
        self.succ = succ
        self.text = text

    def __repr__(self):
        return f"{f'{self.kind}{self.num}':>5}: {self.text}"
