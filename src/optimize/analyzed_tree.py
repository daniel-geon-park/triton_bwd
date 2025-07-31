import copy
from typing import List, Optional, Set, Tuple, Union

import sympy
from sympy.solvers.solveset import NonlinearError, linear_coeffs

from optimize.abtract_tree import AbstractNode, Assignment, Declaration, ForLoop
from optimize.dependence_checking import dependence_levels
from optimize.flow_analysis import DefDict, flow_analysis
from optimize.mem_access import get_mem_accesses
from optimize.sympy_utils import (
    SymbolicArray,
    SymbolicScalar,
    SympyDtype,
    SympyIndexing,
    SympyShape,
    ceildiv,
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
        predecessors: Optional["AnalyzedNode"],
        successors: Optional["AnalyzedNode"],
        text: str,
    ):
        self.kind = kind
        self.num = num
        self.obj = obj
        self.level = level
        self.parent = parent
        self.children = children
        self.descendants = descendants
        self.prev = predecessors
        self.succ = successors
        self.text = text

        # For flow analysis
        self.in_defs: DefDict = {}
        self.out_defs: DefDict = {}

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

    def find_stmt(
        self, key: Union[AbstractNode, Tuple[str, int]]
    ) -> Optional["AnalyzedNode"]:
        """Finds a statement by its kind and number."""
        numbered = [self, *self.descendants]
        for stmt in numbered:
            if stmt.obj is key or (stmt.kind, stmt.num) == key:
                return stmt
        return None

    def find_dependence(
        self,
        i: int,
        j: int,
        on_vars: Optional[Set[str]] = None,
    ) -> Set[Tuple[str, int, str]]:
        """Finds dependencies between two assignments."""
        S = self.find_stmt(("A", i))
        T = self.find_stmt(("A", j))

        S_stores, S_loads = get_mem_accesses(S)
        T_stores, T_loads = get_mem_accesses(T)

        nest_S, nest_T = S.loop_nest(), T.loop_nest()
        nest_S = [stmt.obj for stmt in nest_S]
        nest_T = [stmt.obj for stmt in nest_T]

        dependencies = set()

        def add_deps(dep_kind: str, S_accs, T_accs):
            for s_acc in S_accs:
                s_var = (s_acc.name, s_acc.decl_stmt)
                if on_vars is not None and s_acc.name not in on_vars:
                    continue
                for t_acc in T_accs:
                    if on_vars is not None and t_acc.name not in on_vars:
                        continue
                    t_var = (t_acc.name, t_acc.decl_stmt)
                    if s_var == t_var:
                        min_level = 0
                        if s_acc.decl_stmt is not None:
                            min_level = s_acc.decl_stmt.level
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
        stmt_a = self.find_stmt(a)
        stmt_b = self.find_stmt(b)

        if stmt_a is None:
            raise ValueError(f"Statement {a} not found in the tree")
        if stmt_b is None:
            raise ValueError(f"Statement {b} not found in the tree")

        a_stmts = [stmt_a, *stmt_a.descendants]
        b_stmts = [stmt_b, *stmt_b.descendants]

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

    # Code transformation operations
    def constant_fold(
        self,
        asgn_idx: int,
        var_name: str,
        var_decl_idx: int,
        var_def_idx: int,
    ) -> "AnalyzedNode":
        """Replaces a variable in a statement with its defined value."""
        analyzed_tree = copy.deepcopy(self)
        new_tree = analyzed_tree.obj

        stmt = analyzed_tree.find_stmt(("A", asgn_idx))
        if stmt is None:
            raise ValueError(
                f"Invalid statement index: {asgn_idx}\n" + self.numbered_repr()
            )

        assert isinstance(stmt.obj, Assignment)

        decl = analyzed_tree.find_stmt(("D", var_decl_idx))
        if decl is None:
            raise ValueError(
                f"Invalid declaration index: {var_decl_idx}\n" + self.numbered_repr()
            )

        assert isinstance(decl.obj, Declaration)

        if (var_name, decl) not in stmt.in_defs:
            raise ValueError(
                f"Variable `{var_name}` is not accessible from A{asgn_idx}:\n"
                + self.numbered_repr()
            )

        in_defs: Set[AnalyzedNode] = stmt.in_defs[(var_name, decl)]

        num_replaced = 0

        if isinstance(decl.obj.symbol, SymbolicArray):
            pattern = SympyIndexing(decl.obj.symbol, sympy.Wild("index"))

            def update_expr(index: sympy.Basic) -> sympy.Basic:

                def indices_match(d: AnalyzedNode) -> bool:
                    assert isinstance(d.obj, Assignment)
                    assert isinstance(d.obj.target, SympyIndexing)
                    d_index = d.obj.target.index
                    return (
                        index == d_index
                    )  # FIXME: check if the contained variables' declarations match

                matching_defs = set(filter(indices_match, in_defs))
                if len(matching_defs) > 1:
                    raise ValueError(
                        f"Multiple reachable definitions of `{var_name}`:\n"
                        + self.numbered_repr()
                    )

                definition: AnalyzedNode = next(iter(matching_defs))
                assert isinstance(definition.obj, Assignment)

                if definition.num != var_def_idx:
                    # Keep the original symbol if definition is not the one we want
                    return SympyIndexing(decl.obj.symbol, index)

                nonlocal num_replaced
                num_replaced += 1

                return definition.obj.value

        else:
            pattern = decl.obj.symbol

            def update_expr() -> sympy.Basic:
                if len(in_defs) > 1:
                    raise ValueError(
                        f"Multiple reachable definitions of `{var_name}`:\n"
                        + self.numbered_repr()
                    )
                definition = next(iter(in_defs))
                assert isinstance(definition.obj, Assignment)

                if definition.num != var_def_idx:
                    # Keep the original symbol if definition is not the one we want
                    return decl.obj.symbol

                nonlocal num_replaced
                num_replaced += 1

                return definition.obj.value

        stmt.obj.exprs = [expr.replace(pattern, update_expr) for expr in stmt.obj.exprs]

        if num_replaced == 0:
            raise ValueError(
                f"No reference to {var_name} in target statement A{asgn_idx} refers to defintion A{var_def_idx}:\n"
                + self.numbered_repr()
            )

        return analyze_tree(new_tree)

    def fuse_loop(self, loop_idx_a: int, loop_idx_b: int) -> "AnalyzedNode":
        """Fuses two consecutive loops."""
        analyzed_tree = copy.deepcopy(self)  # Ensure we don't modify the original tree
        new_tree = analyzed_tree.obj

        if loop_idx_b < loop_idx_a:
            loop_idx_a, loop_idx_b = loop_idx_b, loop_idx_a

        stmt_a = analyzed_tree.find_stmt(("L", loop_idx_a))
        stmt_b = analyzed_tree.find_stmt(("L", loop_idx_b))

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

        if loop_a.index_var.label.name != loop_b.index_var.label.name:
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
            is_kernel=loop_a.is_kernel and loop_b.is_kernel,
            max_steps=loop_a.max_steps,
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

    def split_loop(self, split_after: Tuple[str, int]) -> "AnalyzedNode":
        """Splits a loop into two loops at the given index."""
        analyzed_tree = copy.deepcopy(self)
        new_tree = analyzed_tree.obj

        stmt = analyzed_tree.find_stmt(split_after)
        if stmt is None:
            raise ValueError(
                f"Invalid split point {split_after}:\n" + self.numbered_repr()
            )

        if stmt.level <= 1:
            raise ValueError(
                f"Cannot split top-level loop at {split_after}:\n"
                + self.numbered_repr()
            )

        parent_loop = stmt.parent
        assert isinstance(parent_loop.obj, ForLoop)
        assert isinstance(parent_loop.parent.obj, ForLoop)

        stmt_index = parent_loop.obj.statements.index(stmt.obj)
        if stmt_index == len(parent_loop.obj.statements) - 1:
            raise ValueError(
                f"Cannot split loop after the last statement at {split_after}:\n"
                + self.numbered_repr()
            )

        stmts_before = parent_loop.obj.statements[: stmt_index + 1]
        stmts_after = parent_loop.obj.statements[stmt_index + 1 :]
        deps = analyzed_tree.find_stmt_block_dependence(
            stmts_after,
            stmts_before,
        )

        for dep_kind, level, var_name in deps:
            if level == parent_loop.level:
                raise ValueError(
                    f"Cannot split loop at {split_after} due to a dependence at "
                    f"the same level {level} for variable `{var_name}`:\n"
                    + self.numbered_repr()
                )

        # Create the new loop with the statements after the split point
        parent_loop.obj.statements = stmts_before
        new_loop = ForLoop(
            index_var=parent_loop.obj.index_var,
            index_begin=parent_loop.obj.index_begin,
            index_end=parent_loop.obj.index_end,
            index_step=parent_loop.obj.index_step,
            declarations=copy.deepcopy(parent_loop.obj.declarations),
            statements=stmts_after,
            is_kernel=parent_loop.obj.is_kernel,
            max_steps=parent_loop.obj.max_steps,
        )
        loop_index = parent_loop.parent.obj.statements.index(parent_loop.obj)
        parent_loop.parent.obj.statements.insert(loop_index + 1, new_loop)

        return analyze_tree(new_tree)

    def move_array_inside(self, decl_idx: int, loop_idx: int) -> "AnalyzedNode":
        """Moves an array declaration one level inside a loop."""
        analyzed_tree = copy.deepcopy(self)  # Ensure we don't modify the original tree
        new_tree = analyzed_tree.obj

        decl = analyzed_tree.find_stmt(("D", decl_idx))
        loop = analyzed_tree.find_stmt(("L", loop_idx))

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
                if (acc.name, acc.decl_stmt) == (array_name, decl):
                    raise ValueError(
                        f"Variable in D{decl_idx} is referenced by another sibling:\n"
                        + self.numbered_repr()
                    )

        # Make sure there is no dependence on the loop
        self_deps = analyzed_tree.find_stmt_dependence(
            ("L", loop_idx),
            ("L", loop_idx),
            on_vars={array_name},
        )
        for dep_kind, dep_level, dep_var in self_deps:
            if dep_var == array_name and dep_level == loop.level:
                raise ValueError(
                    f"Variable '{array_name}' in D{decl_idx} has a dependence on L{loop_idx} at "
                    f"the same level {loop.level}:\n" + self.numbered_repr()
                )

        # Find the common array dimension of the loop's index variable
        index_var = loop.obj.index_var
        stores, loads = get_mem_accesses(loop)
        index_dim = None
        for acc in stores + loads:
            if (acc.name, acc.decl_stmt) == (array_name, decl):
                if index_var not in acc.index.args:
                    raise ValueError(
                        f"Variable is not indexed by the loop's index variable {index_var} "
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
        new_symbol = SymbolicArray(
            array_name, array_symbol.dtype, sympy.Tuple(*new_shape)
        )

        # Move the declaration inside the loop
        assert isinstance(loop.parent.obj, ForLoop)
        del loop.parent.obj.declarations[array_name]
        loop.obj.declarations[array_name] = Declaration(array_name, new_symbol)

        # Update indices in the loop's statements
        pattern = SympyIndexing(array_symbol, sympy.Wild("index"))

        def update_index(index):
            if not isinstance(index, sympy.Tuple):
                index = sympy.Tuple(index)
            new_index = index.args[:index_dim] + index.args[index_dim + 1 :]
            new_index = sympy.Tuple(*new_index)
            return SympyIndexing(array_symbol, new_index)

        for stmt in loop.descendants:
            stmt.obj.exprs = [
                expr.replace(pattern, update_index).replace(array_symbol, new_symbol)
                for expr in stmt.obj.exprs
            ]

        return analyze_tree(new_tree)

    def move_array_outside(self, decl_idx: int, new_axis: int) -> "AnalyzedNode":
        """Moves an array declaration one level outside a loop."""
        analyzed_tree = copy.deepcopy(self)  # Ensure we don't modify the original tree
        new_tree = analyzed_tree.obj

        decl = analyzed_tree.find_stmt(("D", decl_idx))

        if decl is None:
            raise ValueError(
                f"Invalid declaration index: {decl_idx}\n" + self.numbered_repr()
            )

        assert isinstance(decl.obj, Declaration)
        assert decl.parent is not None

        if decl.parent.parent is None:
            raise ValueError(
                f"Declaration D{decl_idx} is at the top level and cannot be moved outside:\n"
                + self.numbered_repr()
            )

        inner_loop = decl.parent
        outer_loop = inner_loop.parent

        assert isinstance(inner_loop.obj, ForLoop)
        assert isinstance(outer_loop.obj, ForLoop)

        array_name, array_symbol = decl.obj.name, decl.obj.symbol
        if array_name in outer_loop.obj.declarations:
            raise ValueError(
                f"Variable '{array_name}' in D{decl_idx} already exists in the outer loop L{outer_loop.num}:\n"
                + self.numbered_repr()
            )

        old_shape = array_symbol.shape.args
        if new_axis < 0 or new_axis > len(old_shape):
            raise ValueError(
                f"Invalid new axis {new_axis} for variable '{array_name}' in D{decl_idx}:\n"
                + self.numbered_repr()
            )

        outer_begin = outer_loop.obj.index_begin
        outer_step = outer_loop.obj.index_step
        outer_idx = (outer_loop.obj.index_var - outer_begin) // outer_step

        new_shape = (
            old_shape[:new_axis] + (inner_loop.obj.max_steps,) + old_shape[new_axis:]
        )
        new_symbol = SymbolicArray(
            array_name, array_symbol.dtype, sympy.Tuple(*new_shape)
        )

        del inner_loop.obj.declarations[array_name]
        outer_loop.obj.declarations[array_name] = Declaration(array_name, new_symbol)

        # Update indices in the loop's statements
        pattern = SympyIndexing(array_symbol, sympy.Wild("index"))

        def update_index(index):
            if not isinstance(index, sympy.Tuple):
                index = sympy.Tuple(index)
            new_index = (
                *index.args[:new_axis],
                outer_idx,
                *index.args[new_axis:],
            )
            new_index = sympy.Tuple(*new_index)
            return SympyIndexing(array_symbol, new_index)

        for stmt in inner_loop.descendants:
            stmt.obj.exprs = [
                expr.replace(pattern, update_index).replace(array_symbol, new_symbol)
                for expr in stmt.obj.exprs
            ]

        return analyze_tree(new_tree)

    def tile_loop(
        self, loop_idx: int, tile_size: Union[int, sympy.Basic]
    ) -> "AnalyzedNode":
        """Tiles a loop by adding a new loop with the given tile size."""
        analyzed_tree = copy.deepcopy(self)
        new_tree = analyzed_tree.obj

        loop = analyzed_tree.find_stmt(("L", loop_idx))
        if loop is None:
            raise ValueError(f"Invalid loop index: {loop_idx}\n" + self.numbered_repr())

        if loop.parent is None:
            raise ValueError(
                f"Loop L{loop_idx} is at the top level and cannot be tiled:\n"
                + self.numbered_repr()
            )

        for_loop = loop.obj
        assert isinstance(for_loop, ForLoop)

        # Check if the tile size is valid
        if not isinstance(tile_size, sympy.Basic):
            tile_size = sympy.sympify(tile_size)

        if tile_size.is_integer is not True or tile_size.is_positive is not True:
            raise ValueError(
                f"Invalid tile size: {tile_size}. It must be an positive integer expression."
            )

        loop_nest = loop.loop_nest()
        ancestor_index_vars = [anc.obj.index_var for anc in loop_nest]

        try:
            *bi, b0 = linear_coeffs(for_loop.index_begin, *ancestor_index_vars)
            *ei, e0 = linear_coeffs(for_loop.index_end, *ancestor_index_vars)
            *si, s0 = linear_coeffs(for_loop.index_step, *ancestor_index_vars)
        except NonlinearError:
            raise ValueError(
                f"Loop L{loop_idx} has a nonlinear index range, cannot tile:\n"
                + self.numbered_repr()
            )

        for sk in si:
            if sk != 0:
                raise ValueError(
                    f"Loop L{loop_idx}'s step size {sk} depends on an ancestor index, cannot tile:\n"
                    + self.numbered_repr()
                )

        orig_index_var = for_loop.index_var
        orig_step_size = for_loop.index_step
        assert orig_step_size.is_positive is True

        outer_index_begin = for_loop.index_begin
        outer_index_end = ceildiv(for_loop.index_end, tile_size)
        outer_index_var = SymbolicScalar(
            f"{orig_index_var.label.name}1", SympyDtype(orig_index_var)
        )

        inner_index_begin = sympy.Max(outer_index_var * tile_size, for_loop.index_begin)
        inner_index_end = sympy.Min(
            outer_index_var * tile_size + orig_step_size * (tile_size - 1) + 1,
            for_loop.index_end,
        )

        for_loop.index_var = outer_index_var
        for_loop.index_begin = outer_index_begin
        for_loop.index_end = outer_index_end
        for_loop.max_steps = ceildiv(for_loop.max_steps, tile_size)
        for_loop.statements = [
            ForLoop(
                index_var=orig_index_var,
                index_begin=inner_index_begin,
                index_end=inner_index_end,
                index_step=orig_step_size,
                declarations=for_loop.declarations,
                statements=for_loop.statements,
                is_kernel=for_loop.is_kernel,
                max_steps=tile_size,
            )
        ]
        for_loop.declarations = {}

        return analyze_tree(new_tree)

    def parallelize_loop(self, loop_idx: int) -> "AnalyzedNode":
        """Parallelizes a loop by adding a parallel decorator."""
        analyzed_tree = copy.deepcopy(self)  # Ensure we don't modify the original tree
        new_tree = analyzed_tree.obj

        loop = analyzed_tree.find_stmt(("L", loop_idx))

        if loop is None:
            raise ValueError(f"Invalid loop index: {loop_idx}\n" + self.numbered_repr())

        for_loop = loop.obj
        assert isinstance(for_loop, ForLoop)

        self_deps = analyzed_tree.find_stmt_block_dependence(
            for_loop.statements,
            for_loop.statements,
        )

        for dep_kind, dep_level, dep_var in self_deps:
            if dep_level == loop.level:
                raise ValueError(
                    f"Loop L{loop_idx} has a dependence at the same level {loop.level} "
                    f"for variable `{dep_var}`:\n" + self.numbered_repr()
                )

        for_loop.is_kernel = True

        return analyze_tree(new_tree)


def analyze_tree(node: AbstractNode) -> AnalyzedNode:
    analyzed, *_ = _analyze_tree_impl(node)
    flow_analysis(analyzed)
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
                predecessors=None,
                successors=None,
                text="    " * level + repr(node),
            ),
            asgn_idx + 1,
            decl_idx,
            loop_idx,
        )

    elif isinstance(node, Declaration):
        return (
            AnalyzedNode(
                kind="D",
                num=decl_idx,
                obj=node,
                level=level,
                parent=None,
                children=[],
                descendants=[],
                predecessors=None,
                successors=None,
                text="    " * level + repr(node),
            ),
            asgn_idx,
            decl_idx + 1,
            loop_idx,
        )

    elif isinstance(node, ForLoop):
        text = (
            f"for {node.index_var} in range({node.index_begin}, {node.index_end}, {node.index_step}):"
            + f"  # {node.max_steps} steps"
        )
        result = [
            AnalyzedNode(
                kind="L",
                num=loop_idx,
                obj=node,
                level=level,
                parent=None,
                children=[],
                descendants=[],
                predecessors=None,
                successors=None,
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
