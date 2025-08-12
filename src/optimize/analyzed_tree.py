import itertools
from typing import List, Optional, Set, Tuple, Union

from optimize.abtract_tree import AbstractNode, Assignment, Declaration, ForLoop
from optimize.dependence_checking import dependence_levels
from optimize.flow_analysis import DefDict, flow_analysis
from optimize.mem_access import MemAccess, get_mem_accesses


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

    def assignment_dependence(
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
            if on_vars is not None:
                S_accs = filter(lambda acc: acc.name in on_vars, S_accs)
                T_accs = filter(lambda acc: acc.name in on_vars, T_accs)
            for s_acc, t_acc in itertools.product(
                S_accs, T_accs
            ):  # type: MemAccess, MemAccess
                s_var = (s_acc.name, s_acc.decl_stmt)
                t_var = (t_acc.name, t_acc.decl_stmt)
                if s_var != t_var:
                    continue
                min_level = 0
                if s_acc.decl_stmt is not None:
                    min_level = s_acc.decl_stmt.level
                dep_levels = dependence_levels(
                    s_before_t=i < j,
                    min_level=min_level,
                    shape_s=s_acc.symbol.shape,
                    index_s=s_acc.index,
                    nest_s=nest_S,
                    shape_t=t_acc.symbol.shape,
                    index_t=t_acc.index,
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

    def dependence(
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
                deps = self.assignment_dependence(i, j, on_vars)
                dependencies.update(deps)

        return dependencies

    def block_dependence(
        self,
        a: List[Union[AbstractNode, Tuple[str, int]]],
        b: List[Union[AbstractNode, Tuple[str, int]]],
        on_vars: Optional[Set[str]] = None,
    ) -> Set[Tuple[str, int, str]]:
        """Finds dependencies between two blocks of statements."""
        dependencies = set()
        for stmt_a in a:
            for stmt_b in b:
                deps = self.dependence(stmt_a, stmt_b, on_vars)
                dependencies.update(deps)
        return dependencies


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
