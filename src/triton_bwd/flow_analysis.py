from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import sympy

from triton_bwd.mem_access import MemAccess, get_expr_mem_accesses, get_mem_accesses

if TYPE_CHECKING:
    from triton_bwd.analyzed_tree import AnalyzedNode


def flow_analysis(node: "AnalyzedNode") -> Tuple["BasicBlock", "BasicBlock"]:
    # Build the flow graph
    entry_block = BasicBlock([], "Entry")
    exit_block = BasicBlock([], "Exit")
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

    return entry_block, exit_block


DefDict = Dict[Tuple[str, "AnalyzedNode"], Set["AnalyzedNode"]]


class BasicBlock:

    def __init__(self, statements: List["AnalyzedNode"], label: str):
        self.statements = statements  # List of assignments in the block
        self.label = label
        self.predecessors: List[BasicBlock] = []
        self.successors: List[BasicBlock] = []

        # These will get recomputed during flow analysis
        self.in_defs: DefDict = {}
        self.out_defs: DefDict = {}

        # Collect memory accesses from the assignments
        self.stores: List[MemAccess] = []
        self.loads: List[MemAccess] = []
        for stmt in statements:
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


def build_blocks(
    node: "AnalyzedNode",
    start_blocks: List[BasicBlock],
) -> List[BasicBlock]:

    if node.kind == "A":
        block = BasicBlock([node], node.numbered_repr())
        for start_block in start_blocks:
            start_block.successors.append(block)
        return [block]

    elif node.kind == "L":
        idx_access = MemAccess(
            name=node.obj.index_var.name,
            decl_stmt=node,
            index=sympy.Tuple(),
            flat_index=sympy.Number(0),
            statement=node,
        )

        init_block = BasicBlock([], f"L{node.num} Init {node.obj.index_var.name}")
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

        check_block = BasicBlock([], f"L{node.num} Check {node.obj.index_var.name}")
        check_block.loads.extend([idx_access])
        for last_block in last_blocks:
            check_block.predecessors.append(last_block)
            last_block.successors.append(check_block)
        last_blocks = [check_block]

        child_stmts = []
        for child in node.children:
            if child.kind == "D":
                continue

            elif child.kind == "A":
                child_stmts.append(child)

            elif child.kind == "L":
                if len(child_stmts) > 0:
                    child_block = BasicBlock(child_stmts, f"A{child_stmts[0].num}")
                    for last_block in last_blocks:
                        child_block.predecessors.append(last_block)
                        last_block.successors.append(child_block)
                    last_blocks = [child_block]

                last_blocks = build_blocks(child, last_blocks)

        if len(child_stmts) > 0:
            child_block = BasicBlock(child_stmts, f"A{child_stmts[0].num}")
            for last_block in last_blocks:
                child_block.predecessors.append(last_block)
                last_block.successors.append(child_block)
            last_blocks = [child_block]

        increment_block = BasicBlock([], f"L{node.num} Inc {node.obj.index_var.name}")
        increment_block.stores.append(idx_access)
        increment_block.loads.extend([idx_access])

        for last_block in last_blocks:
            increment_block.predecessors.append(last_block)
            last_block.successors.append(increment_block)

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
