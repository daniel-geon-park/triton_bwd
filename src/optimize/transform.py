import copy
from typing import Optional, Set, Tuple, Union

import sympy
from sympy.solvers.solveset import NonlinearError, linear_coeffs

from optimize.abtract_tree import Assignment, Declaration, ForLoop
from optimize.analyzed_tree import AnalyzedNode, analyze_tree
from optimize.mem_access import find_decl_stmt, get_mem_accesses
from optimize.sympy_utils import (
    SymbolicArray,
    SymbolicScalar,
    SympyDtype,
    SympyIndexing,
    SympyShape,
    ceildiv,
    indexing,
    int64,
    sympy_slice,
)


def constant_fold(
    tree: AnalyzedNode,
    asgn_idx: int,
    var_name: str,
    var_decl_idx: int,
    var_def_idx: int,
) -> AnalyzedNode:
    """Replaces a variable in a statement with its defined value."""
    analyzed_tree = copy.deepcopy(tree)
    new_tree = analyzed_tree.obj

    stmt = analyzed_tree.find_stmt(("A", asgn_idx))
    if stmt is None:
        raise ValueError(
            f"Invalid statement index: {asgn_idx}\n" + tree.numbered_repr()
        )

    assert isinstance(stmt.obj, Assignment)

    decl = analyzed_tree.find_stmt(("D", var_decl_idx))
    if decl is None:
        raise ValueError(
            f"Invalid declaration index: {var_decl_idx}\n" + tree.numbered_repr()
        )

    assert isinstance(decl.obj, Declaration)

    if (var_name, decl) not in stmt.in_defs:
        raise ValueError(
            f"Variable `{var_name}` is not accessible from A{asgn_idx}:\n"
            + tree.numbered_repr()
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
                    + tree.numbered_repr()
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
                    + tree.numbered_repr()
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
            + tree.numbered_repr()
        )

    return analyze_tree(new_tree)


def cache_array(
    tree: AnalyzedNode, var_name: str, decl_idx: Optional[int]
) -> AnalyzedNode:
    """
    Make an extra local array variable, replace all operations on var_name to the new array,
    and write the result to the original variable at the end of the original variable's scope.
    """
    analyzed_tree = copy.deepcopy(tree)
    new_tree = analyzed_tree.obj

    if decl_idx is None:  # is argument
        decl = None
        decl_loop = analyzed_tree.find_stmt(("L", 0))
        assert isinstance(analyzed_tree.obj, ForLoop)
        if var_name not in analyzed_tree.obj.arguments:
            raise ValueError(
                f"Variable `{var_name}` is not an argument of the top-level loop:\n"
                + tree.numbered_repr()
            )
        var_symbol = analyzed_tree.obj.arguments[var_name]
    else:
        decl = analyzed_tree.find_stmt(("D", decl_idx))
        if decl is None:
            raise ValueError(
                f"Invalid declaration index: {decl_idx}\n" + tree.numbered_repr()
            )
        assert isinstance(decl.obj, Declaration)
        if decl.obj.name != var_name:
            raise ValueError(
                f"Declaration D{decl_idx} does not match variable name `{var_name}`:\n"
                + tree.numbered_repr()
            )
        decl_loop = decl.parent
        var_symbol = decl.obj.symbol

    if not isinstance(var_symbol, SymbolicArray):
        raise ValueError(
            f"Variable `{var_name}` is not an array:\n" + tree.numbered_repr()
        )

    assert isinstance(decl_loop.obj, ForLoop)

    new_var_name = f"{var_name}_local"
    new_var_symbol = SymbolicArray(new_var_name, var_symbol.dtype, var_symbol.shape)

    if new_var_name in decl_loop.obj.declarations:
        raise ValueError(
            "Variable already exists in the loop's declarations:\n"
            + tree.numbered_repr()
        )

    decl_loop.obj.declarations[new_var_name] = Declaration(new_var_name, new_var_symbol)

    for stmt in decl_loop.descendants:
        stmt.obj.exprs = [
            expr.replace(var_symbol, new_var_symbol) for expr in stmt.obj.exprs
        ]

    # FIXME: copy to the new variable at the beginning of the iteration if necessary

    # Copy the original variable to the new variable at the end of the iteration
    assign_target = SympyIndexing(
        var_symbol,
        sympy.Tuple(*[sympy_slice() for _ in var_symbol.shape]),
    )
    assign_value = SympyIndexing(
        new_var_symbol,
        sympy.Tuple(*[sympy_slice() for _ in new_var_symbol.shape]),
    )
    decl_loop.obj.statements.append(Assignment(assign_target, assign_value))

    return analyze_tree(new_tree)


def expand_assignment(tree: AnalyzedNode, asgn_idx: int) -> AnalyzedNode:
    """Expands a vectorized assignment into a (nested) loop."""
    analyzed_tree = copy.deepcopy(tree)
    new_tree = analyzed_tree.obj

    stmt = analyzed_tree.find_stmt(("A", asgn_idx))
    if stmt is None:
        raise ValueError(
            f"Invalid assignment index: {asgn_idx}\n" + tree.numbered_repr()
        )

    assert isinstance(stmt.obj, Assignment)

    if not isinstance(stmt.obj.target, SympyIndexing):
        raise ValueError(
            f"Assignment A{asgn_idx} target is not an array indexing:\n"
            + tree.numbered_repr()
        )

    # Find the loop that contains the assignment
    loop = stmt.parent
    assert isinstance(loop.obj, ForLoop)

    # Create a new loop for each dimension of the array
    shape = SympyShape(stmt.obj.target.array)
    if len(shape.args) == 0:
        raise ValueError(
            f"Assignment A{asgn_idx} target has no dimensions:\n" + tree.numbered_repr()
        )

    new_loop = None
    child_loop = None
    index_vars = []
    for axis, dim in enumerate(shape.args):
        index_var = SymbolicScalar(f"{stmt.obj.target.array.label.name}_i{axis}", int64)
        index_vars.append(index_var)

        # Create the new loop
        cur_loop = ForLoop(
            index_var=index_var,
            index_begin=sympy.Integer(0),
            index_end=dim,
            index_step=sympy.Integer(1),
            declarations={},
            statements=[],
            max_steps=dim,
        )

        if new_loop is None:
            new_loop = child_loop = cur_loop
        else:
            child_loop.statements.append(cur_loop)
            child_loop = cur_loop

    assign_target = indexing(stmt.obj.target, sympy.Tuple(*index_vars))
    assign_value = indexing(stmt.obj.value, sympy.Tuple(*index_vars))

    child_loop.statements.append(Assignment(assign_target, assign_value))

    # Replace the assignment with the new loop
    loop.obj.statements[loop.obj.statements.index(stmt.obj)] = new_loop

    return analyze_tree(new_tree)


def _detect_name_conflict(
    tree: AnalyzedNode,
    parent_loop: AnalyzedNode,
    old_decl: AnalyzedNode,
    old_symbol: Union[SymbolicScalar, SymbolicArray],
    new_name: str,
):
    old_name = old_symbol.label.name
    for stmt in parent_loop.descendants:
        if stmt.kind == "D":
            continue
        if not any(len(expr.find(old_symbol)) > 0 for expr in stmt.obj.exprs):
            continue
        loop_nest = stmt.loop_nest()
        if find_decl_stmt(loop_nest, old_name) != old_decl:
            continue
        new_decl = find_decl_stmt(loop_nest, new_name)
        if new_decl is None:
            continue
        if new_decl.level > old_decl.level:
            raise ValueError(
                f"Variable `{new_name}` already exists in the loop's statements:\n"
                + tree.numbered_repr()
            )


def rename_declaration(
    tree: AnalyzedNode, decl_idx: int, new_name: str
) -> AnalyzedNode:
    """Renames a variable declaration."""
    analyzed_tree = copy.deepcopy(tree)
    new_tree = analyzed_tree.obj

    decl = analyzed_tree.find_stmt(("D", decl_idx))
    if decl is None:
        raise ValueError(
            f"Invalid declaration index: {decl_idx}\n" + tree.numbered_repr()
        )

    assert isinstance(decl.obj, Declaration)
    old_name = decl.obj.name
    old_symbol = decl.obj.symbol

    parent_loop = decl.parent
    assert isinstance(parent_loop.obj, ForLoop)

    if new_name in parent_loop.obj.declarations:
        raise ValueError(
            f"Variable `{new_name}` already exists in the loop's declarations:\n"
            + tree.numbered_repr()
        )

    _detect_name_conflict(analyzed_tree, parent_loop, decl, old_symbol, new_name)

    if isinstance(old_symbol, SymbolicScalar):
        new_symbol = SymbolicScalar(new_name, old_symbol.dtype)
    elif isinstance(old_symbol, SymbolicArray):
        new_symbol = SymbolicArray(new_name, old_symbol.dtype, old_symbol.shape)
    else:
        raise ValueError(
            f"symbol must be either SymbolicScalar or SymbolicArray, got {type(old_symbol)}"
        )

    # Rename references in the loop's statements
    for stmt in parent_loop.descendants:
        cur_decl = find_decl_stmt(stmt.loop_nest(), old_name)
        if cur_decl != decl:
            continue  # Skip variables shadowing the one we are renaming
        stmt.obj.exprs = [
            expr.replace(old_symbol, new_symbol) for expr in stmt.obj.exprs
        ]

    # Rename the declaration
    del parent_loop.obj.declarations[old_name]
    parent_loop.obj.declarations[new_name] = Declaration(new_name, new_symbol)

    return analyze_tree(new_tree)


def rename_loop_var(tree: AnalyzedNode, loop_idx: int, new_name: str) -> AnalyzedNode:
    """Renames the index variable of a loop."""
    analyzed_tree = copy.deepcopy(tree)
    new_tree = analyzed_tree.obj

    loop = analyzed_tree.find_stmt(("L", loop_idx))
    if loop is None:
        raise ValueError(f"Invalid loop index: {loop_idx}\n" + tree.numbered_repr())

    assert isinstance(loop.obj, ForLoop)
    old_symbol = loop.obj.index_var

    _detect_name_conflict(analyzed_tree, loop, loop, old_symbol, new_name)

    old_name = old_symbol.label.name
    new_symbol = SymbolicScalar(new_name, old_symbol.dtype)

    # Rename references in the loop's statements
    for stmt in loop.descendants:
        cur_decl = find_decl_stmt(stmt.loop_nest(), old_name)
        if cur_decl != loop:
            continue  # Skip variables shadowing the one we are renaming
        stmt.obj.exprs = [
            expr.replace(old_symbol, new_symbol) for expr in stmt.obj.exprs
        ]

    # Rename the index variable
    loop.obj.index_var = new_symbol

    return analyze_tree(new_tree)


def reorder_statement(
    tree: AnalyzedNode,
    stmt_idx: Tuple[str, int],
    insert_after: Optional[Tuple[str, int]],
) -> AnalyzedNode:
    """Moves a statement to a new position in the same loop.
    If `insert_after` is None, the statement is moved to the start of the loop."""
    analyzed_tree = copy.deepcopy(tree)
    new_tree = analyzed_tree.obj

    stmt = analyzed_tree.find_stmt(stmt_idx)
    if stmt is None:
        raise ValueError(
            f"Invalid statement index: {stmt_idx}\n" + tree.numbered_repr()
        )

    if stmt.kind == "D":
        raise ValueError(
            f"Cannot move declaration {stmt_idx}:\n" + tree.numbered_repr()
        )

    parent_loop = stmt.parent
    assert isinstance(parent_loop.obj, ForLoop)

    orig_idx = parent_loop.obj.statements.index(stmt.obj)

    insert_point, insert_idx = None, 0
    if insert_after is not None:
        insert_point = analyzed_tree.find_stmt(insert_after)

        if insert_point is None:
            raise ValueError(
                f"Invalid insert point {insert_after}:\n" + tree.numbered_repr()
            )

        if stmt.parent != insert_point.parent:
            raise ValueError(
                f"Cannot move statement {stmt_idx} to a different loop:\n"
                + tree.numbered_repr()
            )

        insert_idx = parent_loop.obj.statements.index(insert_point.obj) + 1

    if orig_idx == insert_idx:
        raise ValueError(
            f"Statement {stmt_idx} is already at the desired position:\n"
            + tree.numbered_repr()
        )

    if orig_idx < insert_idx:  # Moving it down
        between_stmts = parent_loop.obj.statements[orig_idx + 1 : insert_idx]

        deps = analyzed_tree.block_dependence([stmt.obj], between_stmts)

        for dep_kind, level, var_name in deps:
            if level == stmt.level:
                raise ValueError(
                    f"Moving statement {stmt_idx} down introduces "
                    f"a dependence at the same level {level} for variable `{var_name}`:\n"
                    + tree.numbered_repr()
                )

        # Remove the statement from its original position
        parent_loop.obj.statements.remove(stmt.obj)

        # Insert it at the new position
        parent_loop.obj.statements.insert(insert_idx - 1, stmt.obj)

    else:  # Moving it up
        between_stmts = parent_loop.obj.statements[insert_idx:orig_idx]

        deps = analyzed_tree.block_dependence(between_stmts, [stmt.obj])

        for dep_kind, level, var_name in deps:
            if level == stmt.level:
                raise ValueError(
                    f"Moving statement {stmt_idx} down introduces "
                    f"a dependence at the same level {level} for variable `{var_name}`:\n"
                    + tree.numbered_repr()
                )

        # Remove the statement from its original position
        parent_loop.obj.statements.remove(stmt.obj)

        # Insert it at the new position
        parent_loop.obj.statements.insert(insert_idx, stmt.obj)

    return analyze_tree(new_tree)


def fuse_loop(tree: AnalyzedNode, loop_idx_a: int, loop_idx_b: int) -> AnalyzedNode:
    """Fuses two consecutive loops."""
    analyzed_tree = copy.deepcopy(tree)  # Ensure we don't modify the original tree
    new_tree = analyzed_tree.obj

    if loop_idx_b < loop_idx_a:
        loop_idx_a, loop_idx_b = loop_idx_b, loop_idx_a

    stmt_a = analyzed_tree.find_stmt(("L", loop_idx_a))
    stmt_b = analyzed_tree.find_stmt(("L", loop_idx_b))

    if stmt_a is None or stmt_b is None:
        raise ValueError(f"Invalid loop indices:\n" + tree.numbered_repr())

    if stmt_a.succ is not stmt_b:
        raise ValueError(
            f"Loops L{loop_idx_a} and L{loop_idx_b} are not consecutive:\n"
            + tree.numbered_repr()
        )

    loop_level, parent_loop = stmt_a.level, stmt_a.parent.obj

    assert isinstance(parent_loop, ForLoop)
    assert isinstance(stmt_a.obj, ForLoop) and isinstance(stmt_b.obj, ForLoop)
    loop_a, loop_b = stmt_a.obj, stmt_b.obj

    if loop_a.index_var.label.name != loop_b.index_var.label.name:
        raise ValueError(
            f"Loops L{loop_idx_a} and L{loop_idx_b} have different index variable names:\n"
            + tree.numbered_repr()
        )
    if loop_a.index_begin != loop_b.index_begin:
        raise ValueError(
            f"Loops L{loop_idx_a} and L{loop_idx_b} have different start indices:\n"
            + tree.numbered_repr()
        )
    if loop_a.index_end != loop_b.index_end:
        raise ValueError(
            f"Loops L{loop_idx_a} and L{loop_idx_b} have different end indices:\n"
            + tree.numbered_repr()
        )
    if loop_a.index_step != loop_b.index_step:
        raise ValueError(
            f"Loops L{loop_idx_a} and L{loop_idx_b} have different step sizes:\n"
            + tree.numbered_repr()
        )

    # Check for clashing declarations
    if set(loop_a.declarations.keys()) & set(loop_b.declarations.keys()):
        raise ValueError(
            f"Some declarations in Loops L{loop_idx_a} and L{loop_idx_b} clash:\n"
            + tree.numbered_repr()
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
    new_deps = analyzed_tree.block_dependence(
        loop_b.statements,
        loop_a.statements,
    )
    for dep_kind, level, var_name in new_deps:
        if level == loop_level:
            raise ValueError(
                f"Fusing loops {loop_idx_a} and {loop_idx_b} introduces "
                f"a dependence at the same level {level} for variable `{var_name}`:\n"
                + tree.numbered_repr()
            )

    return analyzed_tree


def split_loop(tree: AnalyzedNode, split_after: Tuple[str, int]) -> AnalyzedNode:
    """Splits a loop into two loops at the given index."""
    analyzed_tree = copy.deepcopy(tree)
    new_tree = analyzed_tree.obj

    stmt = analyzed_tree.find_stmt(split_after)
    if stmt is None:
        raise ValueError(f"Invalid split point {split_after}:\n" + tree.numbered_repr())

    if stmt.level <= 1:
        raise ValueError(
            f"Cannot split top-level loop at {split_after}:\n" + tree.numbered_repr()
        )

    parent_loop = stmt.parent
    assert isinstance(parent_loop.obj, ForLoop)
    assert isinstance(parent_loop.parent.obj, ForLoop)

    stmt_index = parent_loop.obj.statements.index(stmt.obj)
    if stmt_index == len(parent_loop.obj.statements) - 1:
        raise ValueError(
            f"Cannot split loop after the last statement at {split_after}:\n"
            + tree.numbered_repr()
        )

    stmts_before = parent_loop.obj.statements[: stmt_index + 1]
    stmts_after = parent_loop.obj.statements[stmt_index + 1 :]
    deps = analyzed_tree.block_dependence(
        stmts_after,
        stmts_before,
    )

    for dep_kind, level, var_name in deps:
        if level == parent_loop.level:
            raise ValueError(
                f"Cannot split loop at {split_after} due to a dependence at "
                f"the same level {level} for variable `{var_name}`:\n"
                + tree.numbered_repr()
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


def move_array_inside(tree: AnalyzedNode, decl_idx: int, loop_idx: int) -> AnalyzedNode:
    """Moves an array declaration one level inside a loop."""
    analyzed_tree = copy.deepcopy(tree)  # Ensure we don't modify the original tree
    new_tree = analyzed_tree.obj

    decl = analyzed_tree.find_stmt(("D", decl_idx))
    loop = analyzed_tree.find_stmt(("L", loop_idx))

    if decl is None or loop is None:
        raise ValueError(f"Invalid declaration or loop index:\n" + tree.numbered_repr())

    if loop.parent is not decl.parent:
        raise ValueError(
            f"Declaration D{decl_idx} is not in the parent loop of L{loop_idx}:\n"
            + tree.numbered_repr()
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
                    + tree.numbered_repr()
                )

    # Make sure there is no dependence on the loop
    self_deps = analyzed_tree.dependence(
        ("L", loop_idx),
        ("L", loop_idx),
        on_vars={array_name},
    )
    for dep_kind, dep_level, dep_var in self_deps:
        if dep_var == array_name and dep_level == loop.level:
            raise ValueError(
                f"Variable '{array_name}' in D{decl_idx} has a dependence on L{loop_idx} at "
                f"the same level {loop.level}:\n" + tree.numbered_repr()
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
                    f"in a statement inside the loop:\n" + tree.numbered_repr()
                )
            cur_index_dim = acc.index.args.index(index_var)
            if index_dim is not None and cur_index_dim != index_dim:
                raise ValueError(
                    f"Variable is indexed by the loop's index variable {index_var} "
                    f"in different dimensions: {index_dim} and {cur_index_dim}:\n"
                    + tree.numbered_repr()
                )
            index_dim = cur_index_dim

    old_shape = array_symbol.shape.args
    new_shape = old_shape[:index_dim] + old_shape[index_dim + 1 :]
    new_symbol = SymbolicArray(array_name, array_symbol.dtype, sympy.Tuple(*new_shape))

    # Move the declaration inside the loop
    assert isinstance(loop.parent.obj, ForLoop)
    del loop.parent.obj.declarations[array_name]
    loop.obj.declarations[array_name] = Declaration(array_name, new_symbol)

    # Update indices in the loop's statements
    pattern = SympyIndexing(array_symbol, sympy.Wild("index"))

    def update_index(index):
        new_index = index.args[:index_dim] + index.args[index_dim + 1 :]
        new_index = sympy.Tuple(*new_index)
        return SympyIndexing(array_symbol, new_index)

    for stmt in loop.descendants:
        stmt.obj.exprs = [
            expr.replace(pattern, update_index).replace(array_symbol, new_symbol)
            for expr in stmt.obj.exprs
        ]

    return analyze_tree(new_tree)


def move_array_outside(
    tree: AnalyzedNode, decl_idx: int, new_axis: int
) -> AnalyzedNode:
    """Moves an array declaration one level outside a loop."""
    analyzed_tree = copy.deepcopy(tree)  # Ensure we don't modify the original tree
    new_tree = analyzed_tree.obj

    decl = analyzed_tree.find_stmt(("D", decl_idx))

    if decl is None:
        raise ValueError(
            f"Invalid declaration index: {decl_idx}\n" + tree.numbered_repr()
        )

    assert isinstance(decl.obj, Declaration)
    assert decl.parent is not None

    if decl.parent.parent is None:
        raise ValueError(
            f"Declaration D{decl_idx} is at the top level and cannot be moved outside:\n"
            + tree.numbered_repr()
        )

    inner_loop = decl.parent
    outer_loop = inner_loop.parent

    assert isinstance(inner_loop.obj, ForLoop)
    assert isinstance(outer_loop.obj, ForLoop)

    array_name, array_symbol = decl.obj.name, decl.obj.symbol
    if array_name in outer_loop.obj.declarations:
        raise ValueError(
            f"Variable '{array_name}' in D{decl_idx} already exists in the outer loop L{outer_loop.num}:\n"
            + tree.numbered_repr()
        )

    old_shape = array_symbol.shape.args
    if new_axis < 0 or new_axis > len(old_shape):
        raise ValueError(
            f"Invalid new axis {new_axis} for variable '{array_name}' in D{decl_idx}:\n"
            + tree.numbered_repr()
        )

    inner_begin = inner_loop.obj.index_begin
    inner_step = inner_loop.obj.index_step
    inner_idx = (inner_loop.obj.index_var - inner_begin) // inner_step

    new_shape = (
        old_shape[:new_axis] + (inner_loop.obj.max_steps,) + old_shape[new_axis:]
    )
    new_symbol = SymbolicArray(array_name, array_symbol.dtype, sympy.Tuple(*new_shape))

    del inner_loop.obj.declarations[array_name]
    outer_loop.obj.declarations[array_name] = Declaration(array_name, new_symbol)

    # Update indices in the loop's statements
    pattern = SympyIndexing(array_symbol, sympy.Wild("index"))

    def update_index(index):
        new_index = (
            *index.args[:new_axis],
            inner_idx,
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
    tree: AnalyzedNode, loop_idx: int, tile_size: Union[int, sympy.Basic]
) -> AnalyzedNode:
    """Tiles a loop by adding a new loop with the given tile size."""
    analyzed_tree = copy.deepcopy(tree)
    new_tree = analyzed_tree.obj

    loop = analyzed_tree.find_stmt(("L", loop_idx))
    if loop is None:
        raise ValueError(f"Invalid loop index: {loop_idx}\n" + tree.numbered_repr())

    if loop.parent is None:
        raise ValueError(
            f"Loop L{loop_idx} is at the top level and cannot be tiled:\n"
            + tree.numbered_repr()
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
            + tree.numbered_repr()
        )

    for sk in si:
        if sk != 0:
            raise ValueError(
                f"Loop L{loop_idx}'s step size {sk} depends on an ancestor index, cannot tile:\n"
                + tree.numbered_repr()
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


def parallelize_loop(tree: AnalyzedNode, loop_idx: int) -> AnalyzedNode:
    """Parallelizes a loop by adding a parallel decorator."""
    analyzed_tree = copy.deepcopy(tree)  # Ensure we don't modify the original tree
    new_tree = analyzed_tree.obj

    loop = analyzed_tree.find_stmt(("L", loop_idx))

    if loop is None:
        raise ValueError(f"Invalid loop index: {loop_idx}\n" + tree.numbered_repr())

    for_loop = loop.obj
    assert isinstance(for_loop, ForLoop)

    self_deps = analyzed_tree.block_dependence(
        for_loop.statements,
        for_loop.statements,
    )

    for dep_kind, dep_level, dep_var in self_deps:
        if dep_level == loop.level:
            raise ValueError(
                f"Loop L{loop_idx} has a dependence at the same level {loop.level} "
                f"for variable `{dep_var}`:\n" + tree.numbered_repr()
            )

    for_loop.is_kernel = True

    return analyze_tree(new_tree)
