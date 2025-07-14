import copy
from typing import List, Optional, Set, Tuple, Union

import sympy

from triton_bwd.abtract_tree import AbstractNode, Assignment, Declaration, ForLoop
from triton_bwd.dependence_checking import dependence_levels
from triton_bwd.sympy_utils import (
    SymbolicArray,
    SymbolicScalar,
    SympyDtype,
    SympyIndexing,
    SympyShape,
)


class AnalyzedNode:
    def __init__(
        self,
        kind: str,
        num: int,
        obj: AbstractNode,
        level: int,
        parent: Optional["AnalyzedNode"],
        children: List["AnalyzedNode"],
        descendants: List["AnalyzedNode"],
        prev: Optional["AnalyzedNode"],
        succ: Optional["AnalyzedNode"],
        text: str,
    ):
        self.kind = kind
        self.num = num
        self.obj = obj
        self.level = level
        self.parent = parent
        self.children = children
        self.descendants = descendants
        self.prev = prev
        self.succ = succ
        self.text = text

    def __repr__(self):
        return f"{f'{self.kind}{self.num}':>5}: {self.text}"

    def numbered_repr(self) -> str:
        numbered = [self, *self.descendants]
        return "\n".join(
            f"{f'{stmt.kind}{stmt.num}/{stmt.level}':>6}: {stmt.text}"
            for stmt in numbered
        )

    def loop_nest(self) -> List["AnalyzedNode"]:
        """Returns the nesting of the current statement."""
        if self.parent is None:
            return []
        nest = [self.parent]
        while nest[0].parent is not None:
            nest.insert(0, nest[0].parent)
        return nest

    def find_dependence(
        self,
        i: int,
        j: int,
        on_vars: Optional[Set[str]] = None,
    ) -> Set[Tuple[str, int, str]]:
        """Finds dependencies between two assignments."""
        numbered = [self, *self.descendants]

        S = T = None
        for stmt in numbered:
            if (stmt.kind, stmt.num) == ("A", i):
                S = stmt
            if (stmt.kind, stmt.num) == ("A", j):
                T = stmt

        S_stores, S_loads = get_mem_accesses(S)
        T_stores, T_loads = get_mem_accesses(T)

        nest_S, nest_T = S.loop_nest(), T.loop_nest()
        nest_S = [stmt.obj for stmt in nest_S]
        nest_T = [stmt.obj for stmt in nest_T]

        dependencies = set()

        def add_deps(dep_kind: str, S_accs, T_accs):
            for s_acc in S_accs:
                s_var = (s_acc.name, s_acc.decl_loop)
                if on_vars is not None and s_acc.name not in on_vars:
                    continue
                for t_acc in T_accs:
                    if on_vars is not None and t_acc.name not in on_vars:
                        continue
                    t_var = (t_acc.name, t_acc.decl_loop)
                    if s_var == t_var:
                        min_level = 0
                        if s_acc.decl_loop is not None:
                            min_level = s_acc.decl_loop.level + 1
                        dep_levels = dependence_levels(
                            s_before_t=i < j,
                            min_level=min_level,
                            index_s=s_acc.flat_index,
                            nest_s=nest_S,
                            index_t=t_acc.flat_index,
                            nest_t=nest_T,
                        )
                        for u in dep_levels:
                            dependencies.add((dep_kind, u, s_acc.name))

        # Flow dependencies
        add_deps("flow", S_stores, T_loads)
        # Antidependencies
        add_deps("anti", S_loads, T_stores)
        # Output dependencies
        add_deps("outp", S_stores, T_stores)

        return dependencies

    def find_stmt_dependence(
        self,
        a: Union[AbstractNode, Tuple[str, int]],
        b: Union[AbstractNode, Tuple[str, int]],
        on_vars: Optional[Set[str]] = None,
    ) -> Set[Tuple[str, int, str]]:
        """Finds dependencies between two statements (loops or assignments)."""
        numbered = [self, *self.descendants]

        stmt_a = stmt_b = None
        for idx, stmt in enumerate(numbered):
            if stmt.obj is a or (stmt.kind, stmt.num) == a:
                stmt_a = stmt
            if stmt.obj is b or (stmt.kind, stmt.num) == b:
                stmt_b = stmt

        if stmt_a is None:
            raise ValueError(f"Statement {a} not found in the tree")
        if stmt_b is None:
            raise ValueError(f"Statement {b} not found in the tree")

        a_stmts = [stmt_a, *stmt_a.children]
        b_stmts = [stmt_b, *stmt_b.children]

        a_asgn_indices = [stmt.num for stmt in a_stmts if stmt.kind == "A"]
        b_asgn_indices = [stmt.num for stmt in b_stmts if stmt.kind == "A"]

        dependencies = set()
        for i in a_asgn_indices:
            for j in b_asgn_indices:
                deps = self.find_dependence(i, j, on_vars)
                dependencies.update(deps)

        return dependencies

    def find_stmt_block_dependence(
        self,
        a: List[Union[AbstractNode, Tuple[str, int]]],
        b: List[Union[AbstractNode, Tuple[str, int]]],
    ) -> Set[Tuple[str, int, str]]:
        """Finds dependencies between two blocks of statements."""
        dependencies = set()
        for stmt_a in a:
            for stmt_b in b:
                deps = self.find_stmt_dependence(stmt_a, stmt_b)
                dependencies.update(deps)
        return dependencies

    def fuse_loop(self, loop_idx_a: int, loop_idx_b: int) -> "AnalyzedNode":
        """Fuses two consecutive loops."""
        new_tree = copy.deepcopy(self.obj)  # Ensure we don't modify the original tree

        analyzed_tree = analyze_tree(new_tree)
        numbered = [analyzed_tree, *analyzed_tree.descendants]

        if loop_idx_b < loop_idx_a:
            loop_idx_a, loop_idx_b = loop_idx_b, loop_idx_a

        stmt_a = stmt_b = None
        for stmt in numbered:
            if stmt.kind == "L" and stmt.num == loop_idx_a:
                stmt_a = stmt
            elif stmt.kind == "L" and stmt.num == loop_idx_b:
                stmt_b = stmt

        if stmt_a is None or stmt_b is None:
            raise ValueError(f"Invalid loop indices:\n" + self.numbered_repr())

        if stmt_a.succ is not stmt_b:
            raise ValueError(
                f"Loops L{loop_idx_a} and L{loop_idx_b} are not consecutive:\n"
                + self.numbered_repr()
            )

        loop_level, parent_loop = stmt_a.level, stmt_a.parent.obj

        assert isinstance(parent_loop, ForLoop)
        assert isinstance(stmt_a.obj, ForLoop) and isinstance(stmt_b.obj, ForLoop)
        loop_a, loop_b = stmt_a.obj, stmt_b.obj

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

        # Need to analyze the new tree again to check for dependencies
        analyzed_tree = analyze_tree(new_tree)
        new_deps = analyzed_tree.find_stmt_block_dependence(
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

        return analyzed_tree

    def localize_array_allocation(self, decl_idx: int, loop_idx: int) -> "AnalyzedNode":
        """Moves an array declaration one level inside a loop."""
        new_tree = copy.deepcopy(self.obj)  # Ensure we don't modify the original tree

        analyzed_tree = analyze_tree(new_tree)
        numbered = [analyzed_tree, *analyzed_tree.descendants]

        decl = loop = None
        for stmt in numbered:
            if stmt.kind == "D" and stmt.num == decl_idx:
                decl = stmt
            elif stmt.kind == "L" and stmt.num == loop_idx:
                loop = stmt

        if decl is None or loop is None:
            raise ValueError(
                f"Invalid declaration or loop index:\n" + self.numbered_repr()
            )

        if loop.parent is not decl.parent:
            raise ValueError(
                f"Declaration D{decl_idx} is not in the parent loop of L{loop_idx}:\n"
                + self.numbered_repr()
            )

        assert isinstance(decl.obj, Declaration) and isinstance(loop.obj, ForLoop)

        # Make sure no other siblings contain references to the array
        array_name, array_symbol = decl.obj.name, decl.obj.symbol
        for sibling in loop.parent.children:
            if sibling is loop:
                continue

            loads, stores = get_mem_accesses(sibling)
            for acc in loads + stores:
                if (acc.name, acc.decl_loop) == (array_name, decl.parent):
                    raise ValueError(
                        f"Variable in D{decl_idx} is referenced by another sibling:\n"
                        + self.numbered_repr()
                    )

        # Make sure there is no dependence on the loop
        analyzed_tree = analyze_tree(new_tree)
        self_deps = analyzed_tree.find_stmt_dependence(
            ("L", loop_idx),
            ("L", loop_idx),
            on_vars={array_name},
        )
        for dep_kind, dep_level, dep_var in self_deps:
            if dep_var == array_name and dep_level == loop.level:
                raise ValueError(
                    f"Variable in D{decl_idx} has a dependence on L{loop_idx} at "
                    f"the same level {loop.level}:\n" + self.numbered_repr()
                )

        # Find the common array dimension of the loop's index variable
        index_var = loop.obj.index_var
        stores, loads = get_mem_accesses(loop)
        index_dim = None
        for acc in stores + loads:
            if (acc.name, acc.decl_loop) == (array_name, decl.parent):
                if index_var not in acc.index.args:
                    raise ValueError(
                        f"Variable is not indexed by the loop's index variable {index_var}"
                        f"in a statement inside the loop:\n" + self.numbered_repr()
                    )
                cur_index_dim = acc.index.args.index(index_var)
                if index_dim is not None and cur_index_dim != index_dim:
                    raise ValueError(
                        f"Variable is indexed by the loop's index variable {index_var} "
                        f"in different dimensions: {index_dim} and {cur_index_dim}:\n"
                        + self.numbered_repr()
                    )
                index_dim = cur_index_dim

        old_shape = array_symbol.shape.args
        new_shape = old_shape[:index_dim] + old_shape[index_dim + 1 :]

        # Move the declaration inside the loop
        assert isinstance(loop.parent.obj, ForLoop)
        del loop.parent.obj.declarations[array_name]
        new_symbol = SymbolicArray(
            array_name, array_symbol.dtype, sympy.Tuple(*new_shape)
        )
        loop.obj.declarations[array_name] = Declaration(array_name, new_symbol)

        # Update indices in the loop's statements
        pattern = SympyIndexing(array_symbol, sympy.Wild("index"))

        def update_index(index):
            new_index = index.args[:index_dim] + index.args[index_dim + 1 :]
            new_index = sympy.Tuple(*new_index)
            return SympyIndexing(array_symbol, new_index)

        for stmt in loop.descendants:
            if stmt.kind == "A":
                stmt.obj.target = stmt.obj.target.replace(pattern, update_index)
                stmt.obj.value = stmt.obj.value.replace(pattern, update_index)

        return analyze_tree(new_tree)

    def _generate_code_asgn(self, backend: str) -> Tuple[List[str], List[str]]:
        assert isinstance(self.obj, Assignment)
        return [], [f"{self.obj.target} = {self.obj.value}"]

    def _generate_code_decl(self, backend: str) -> Tuple[List[str], List[str]]:
        assert isinstance(self.obj, Declaration)
        name, symbol = self.obj.name, self.obj.symbol
        shape = SympyShape(symbol)
        dtype = SympyDtype(symbol)
        if shape == ():
            return [], [f'{name}: "{str(dtype)}"']
        else:
            shape_str = ", ".join(map(str, shape.args))
            if backend == "torch":
                dtype = f"torch.{str(dtype)}"
                initializer = f"torch.zeros(({shape_str}), dtype={dtype})"
            elif backend == "triton":
                dtype = f"tl.{str(dtype)}"
                initializer = f"tl.zeros(({shape_str}), dtype={dtype})"
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            return [], [f"{name} = {initializer}"]

    def _generate_code_loop(self, backend: str) -> Tuple[List[str], List[str]]:
        assert isinstance(self.obj, ForLoop)
        if self.parent is None:  # top-level loop
            arguments = []

            for arg in self.obj.arguments.values():
                if isinstance(arg, SymbolicScalar):
                    arguments.append(f"{arg.label.name}: {arg.dtype}")
                elif isinstance(arg, SymbolicArray):
                    arguments.append(f"{arg.label.name}: torch.Tensor")

            code_lines = [f"def function({', '.join(arguments)}):"]

        else:
            code_lines = [
                f"for {self.obj.index_var} in range({self.obj.index_begin}, {self.obj.index_end}, {self.obj.index_step}):"
            ]

        preamble = []
        for child in self.children:
            child_preamble, chld_code = child._generate_code_impl(backend)
            preamble.extend(child_preamble)
            for line in chld_code:
                code_lines.append(f"    {line}")

        return preamble, code_lines

    def _generate_code_impl(self, backend: str) -> Tuple[List[str], List[str]]:
        if self.kind == "A":
            return self._generate_code_asgn(backend)
        elif self.kind == "L":
            return self._generate_code_loop(backend)
        elif self.kind == "D":
            return self._generate_code_decl(backend)
        else:
            raise ValueError(f"Unsupported statement kind: {self.kind}")

    def generate_code(self) -> str:
        preamble, lines = self._generate_code_impl(backend="torch")
        return "\n".join(preamble + lines)


def analyze_tree(node: AbstractNode) -> AnalyzedNode:
    analyzed, *_ = _analyze_tree_impl(node)
    return analyzed


def _analyze_tree_impl(
    node: AbstractNode,
    level=0,
    asgn_idx=0,
    decl_idx=0,
    loop_idx=0,
) -> Tuple[AnalyzedNode, int, int, int]:

    if isinstance(node, Assignment):
        return (
            AnalyzedNode(
                kind="A",
                num=asgn_idx,
                obj=node,
                level=level,
                parent=None,
                children=[],
                descendants=[],
                prev=None,
                succ=None,
                text="    " * level + repr(node),
            ),
            asgn_idx + 1,
            decl_idx,
            loop_idx,
        )

    elif isinstance(node, Declaration):
        shape = SympyShape(node.symbol)
        if shape == ():
            text = f"let {node.name}: scalar"
        else:
            text = f"let {node.name}: array({', '.join(map(str, shape.args))})"
        return (
            AnalyzedNode(
                kind="D",
                num=decl_idx,
                obj=node,
                level=level,
                parent=None,
                children=[],
                descendants=[],
                prev=None,
                succ=None,
                text="    " * level + text,
            ),
            asgn_idx,
            decl_idx + 1,
            loop_idx,
        )

    elif isinstance(node, ForLoop):
        text = f"for {node.index_var} in range({node.index_begin}, {node.index_end}, {node.index_step}):"
        result = [
            AnalyzedNode(
                kind="L",
                num=loop_idx,
                obj=node,
                level=level,
                parent=None,
                children=[],
                descendants=[],
                prev=None,
                succ=None,
                text="    " * level + text,
            )
        ]
        loop_idx += 1

        children = []

        prev_stmt = None
        for stmt in [*node.declarations.values(), *node.statements]:
            analyzed, asgn_idx, decl_idx, loop_idx = _analyze_tree_impl(
                stmt, level + 1, asgn_idx, decl_idx, loop_idx
            )
            numbered_stmts = [analyzed, *analyzed.descendants]

            if len(numbered_stmts) > 0:
                if prev_stmt is not None:
                    prev_stmt.succ = numbered_stmts[0]
                numbered_stmts[0].parent = result[0]
                numbered_stmts[0].prev = prev_stmt
                prev_stmt = numbered_stmts[0]
                children.append(numbered_stmts[0])

            result.extend(numbered_stmts)

        result[0].children = children
        result[0].descendants = result[1:]

        return result[0], asgn_idx, decl_idx, loop_idx

    else:
        raise ValueError(f"Unsupported node type: {type(node)}")


class MemAccess:
    def __init__(
        self,
        name: str,
        decl_loop: Optional["AnalyzedNode"],
        index: sympy.Tuple,
        flat_index: sympy.Basic,
    ):
        self.name = name
        self.decl_loop = decl_loop
        self.index = index
        self.flat_index = flat_index


def get_expr_mem_accesses(
    expr: sympy.Basic,
    loop_nest: List["AnalyzedNode"],
) -> List[MemAccess]:

    if isinstance(expr, sympy.Symbol):
        decl_loop = None
        for loop in loop_nest[::-1]:
            assert isinstance(loop.obj, ForLoop)
            if expr.name in loop.obj.declarations:
                decl_loop = loop
                break

        return [
            MemAccess(
                name=expr.name,
                decl_loop=decl_loop,
                index=sympy.Tuple(),
                flat_index=sympy.Number(0),
            )
        ]

    if isinstance(expr, SympyIndexing):
        array, index = expr.args

        assert isinstance(array, SymbolicArray) and isinstance(
            array.label, sympy.Symbol
        )
        array_name = array.label.name

        if not isinstance(index, sympy.Tuple):
            index = sympy.Tuple(index)

        flat_index = sympy.Number(0)
        shape = SympyShape(array)
        for dim, idx in zip(shape.args, index.args):
            flat_index = flat_index * dim + idx

        decl_loop = None
        for loop in loop_nest[::-1]:
            assert isinstance(loop.obj, ForLoop)
            if array_name in loop.obj.declarations:
                decl_loop = loop
                break

        return [
            MemAccess(
                name=array_name, decl_loop=decl_loop, index=index, flat_index=flat_index
            )
        ]

    results = []
    for arg in expr.args:
        results.extend(get_expr_mem_accesses(arg, loop_nest))
    return results


def get_mem_accesses(stmt: "AnalyzedNode") -> Tuple[List[MemAccess], List[MemAccess]]:
    if stmt.kind == "A":
        assert isinstance(stmt.obj, Assignment)
        nest = stmt.loop_nest()
        stores = get_expr_mem_accesses(stmt.obj.target, nest)
        loads = get_expr_mem_accesses(stmt.obj.value, nest)
        return stores, loads

    elif stmt.kind == "L":
        stores, loads = [], []
        for child in stmt.children:
            cur_stores, cur_loads = get_mem_accesses(child)
            stores.extend(cur_stores)
            loads.extend(cur_loads)
        return stores, loads

    elif stmt.kind == "D":
        return [], []

    else:
        raise ValueError(f"Unsupported statement kind: {stmt.kind}")
