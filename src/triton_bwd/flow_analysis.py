from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import sympy

from triton_bwd.abtract_tree import ForLoop
from triton_bwd.mem_access import MemAccess, get_expr_mem_accesses, get_mem_accesses

if TYPE_CHECKING:
    from triton_bwd.analyzed_tree import AnalyzedNode


def flow_analysis(node: "AnalyzedNode"):
    # Build the flow graph
    entry_block = BasicBlock("Entry")
    if isinstance(node.obj, ForLoop) and node.obj.arguments is not None:
        for name in node.obj.arguments.keys():
            entry_block.stores.extend(
                [
                    MemAccess(
                        name=name,
                        decl_stmt=node,
                        index=sympy.Tuple(),
                        flat_index=sympy.Number(0),
                        statement=node,
                    )
                ]
            )
    exit_block = BasicBlock("Exit")
    last_blocks = build_blocks(node, [entry_block])
    for last_block in last_blocks:
        last_block.successors.append(exit_block)
        exit_block.predecessors.append(last_block)

    # Do reachability analysis
    blocks = all_blocks(entry_block)
    iterations = 0
    while True:
        iterations += 1
        changed = False
        for block in blocks:
            block.recompute_in_defs()
            changed |= block.recompute_out_defs()
        if not changed:
            break

    # Propagate reachability info to statements
    node.in_defs = entry_block.out_defs
    node.out_defs = exit_block.in_defs
    for block in blocks:
        if block.statements is not None:
            in_defs = block.in_defs
            for stmt in block.statements:
                stmt.in_defs = in_defs
                stmt.out_defs = transfer_function(stmt, in_defs)
                in_defs = stmt.out_defs
        if block.loop_init is not None:
            block.loop_init.in_defs = block.in_defs
        if block.loop_check is not None:
            block.loop_check.out_defs = block.out_defs


DefDict = Dict[Tuple[str, "AnalyzedNode"], Set["AnalyzedNode"]]


class BasicBlock:

    def __init__(
        self,
        label: Optional[str] = None,
        assignments: Optional[List["AnalyzedNode"]] = None,
        loop_init: Optional["AnalyzedNode"] = None,
        loop_check: Optional["AnalyzedNode"] = None,
        loop_increment: Optional["AnalyzedNode"] = None,
    ):
        self.label = label
        self.statements = assignments  # List of assignments in the block
        self.loop_init = loop_init
        self.loop_check = loop_check
        self.loop_increment = loop_increment

        self.predecessors: List[BasicBlock] = []
        self.successors: List[BasicBlock] = []

        # These will get recomputed during flow analysis
        self.in_defs: DefDict = {}
        self.out_defs: DefDict = {}

        # Collect memory accesses from the assignments
        self.stores: List[MemAccess] = []
        self.loads: List[MemAccess] = []
        if assignments is not None:
            for stmt in assignments:
                stores, loads = get_mem_accesses(stmt)
                self.stores.extend(stores)
                self.loads.extend(loads)

    def __repr__(self) -> str:
        return f"BasicBlock({self.label})"

    def recompute_in_defs(self):
        """Recomputes the set of definitions that reach this block."""
        self.in_defs = {}
        for pred in self.predecessors:
            for decl, defs in pred.out_defs.items():
                if decl not in self.in_defs:
                    self.in_defs[decl] = set()
                self.in_defs[decl].update(defs)

    def recompute_out_defs(self) -> bool:
        """Recomputes the set of definitions that this block reaches."""
        orig_out_defs = self.out_defs
        self.out_defs = {
            decl: defs for decl, defs in self.in_defs.items() if decl not in self.stores
        }
        for s in self.stores:
            decl = (s.name, s.decl_stmt)
            if decl not in self.out_defs:
                self.out_defs[decl] = set()
            self.out_defs[decl].add(s.statement)
        return orig_out_defs != self.out_defs


def transfer_function(stmt: "AnalyzedNode", in_defs: DefDict) -> DefDict:
    block = BasicBlock(assignments=[stmt])
    block.in_defs = in_defs
    block.recompute_out_defs()
    return block.out_defs


def build_blocks_body(
    nodes: List["AnalyzedNode"],
    last_blocks: List[BasicBlock],
) -> List[BasicBlock]:
    child_stmts = []
    for child in nodes:
        if child.kind == "D":
            continue

        elif child.kind == "A":
            child_stmts.append(child)

        elif child.kind == "L":
            if len(child_stmts) > 0:
                child_block = BasicBlock(
                    f"A{child_stmts[0].num}", assignments=child_stmts
                )
                for last_block in last_blocks:
                    child_block.predecessors.append(last_block)
                    last_block.successors.append(child_block)
                last_blocks = [child_block]
                child_stmts = []

            last_blocks = build_blocks(child, last_blocks)

    if len(child_stmts) > 0:
        child_block = BasicBlock(f"A{child_stmts[0].num}", assignments=child_stmts)
        for last_block in last_blocks:
            child_block.predecessors.append(last_block)
            last_block.successors.append(child_block)
        last_blocks = [child_block]

    return last_blocks


def build_blocks(
    node: "AnalyzedNode",
    start_blocks: List[BasicBlock],
) -> List[BasicBlock]:

    if node.kind == "A":
        block = BasicBlock(f"A{node.num}", [node])
        for start_block in start_blocks:
            block.predecessors.append(start_block)
            start_block.successors.append(block)
        return [block]

    elif node.kind == "L":
        assert isinstance(node.obj, ForLoop)

        if node.parent is None:
            return build_blocks_body(node.children, start_blocks)

        idx_access = MemAccess(
            name=node.obj.index_var.name,
            decl_stmt=node,
            index=sympy.Tuple(),
            flat_index=sympy.Number(0),
            statement=node,
        )

        init_block = BasicBlock(f"L{node.num} Init", loop_init=node)
        init_block.stores.append(idx_access)
        init_loads = [
            *get_expr_mem_accesses(node.obj.index_begin, node.loop_nest()),
            *get_expr_mem_accesses(node.obj.index_end, node.loop_nest()),
            *get_expr_mem_accesses(node.obj.index_step, node.loop_nest()),
        ]
        for load in init_loads:
            load.statement = node
        init_block.loads.extend(init_loads)

        for start_block in start_blocks:
            init_block.predecessors.append(start_block)
            start_block.successors.append(init_block)
        last_blocks = [init_block]

        check_block = BasicBlock(f"L{node.num} Check", loop_check=node)
        check_block.loads.extend([idx_access])
        for last_block in last_blocks:
            check_block.predecessors.append(last_block)
            last_block.successors.append(check_block)
        last_blocks = [check_block]

        last_blocks = build_blocks_body(node.children, last_blocks)

        increment_block = BasicBlock(f"L{node.num} Inc", loop_increment=node)
        increment_block.stores.append(idx_access)
        increment_block.loads.extend([idx_access])

        for last_block in last_blocks:
            increment_block.predecessors.append(last_block)
            last_block.successors.append(increment_block)

        check_block.predecessors.append(increment_block)
        increment_block.successors.append(check_block)

        return [check_block]

    else:
        raise ValueError(f"Unsupported node kind for flow analysis: {node.kind}")


def all_blocks(entry_block: BasicBlock) -> List[BasicBlock]:
    """Returns all blocks reachable from the entry block."""
    visited = set()
    current_queue = [entry_block]

    while current_queue:
        block = current_queue.pop(0)
        if block in visited:
            continue
        visited.add(block)

        for succ in block.successors:
            if succ not in visited:
                current_queue.append(succ)

    return list(visited)


def visualize_flow(entry_block: "BasicBlock") -> None:
    import graphviz

    dot = graphviz.Digraph()

    visited = set()
    current_queue = [entry_block]
    while current_queue:
        block = current_queue.pop(0)
        if block in visited:
            continue
        visited.add(block)

        dot.node(block.label, label=block.label)
        for succ in block.successors:
            dot.edge(block.label, succ.label)
            if succ not in visited:
                current_queue.append(succ)

    dot.render("flow_analysis", format="png", cleanup=True, view=True)
