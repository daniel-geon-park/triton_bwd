import ast
import inspect
import math
import re
from typing import Any, Dict, List, Union

import sympy

from triton_bwd.abtract_tree import AbstractNode, Assignment, Declaration, ForLoop
from triton_bwd.constexpr import Constexpr
from triton_bwd.optimize_lang import Array
from triton_bwd.sympy_utils import (
    TYPE_MAP,
    UNKNOWN_TYPES,
    SymbolicArray,
    SymbolicScalar,
    SympyDtype,
    SympyIndexing,
    SympyShape,
)


class NodeVisitor(ast.NodeVisitor):
    def __init__(
        self,
        call_stack: list[str],
        func_globals: Dict[str, Any],
        args: Dict[str, sympy.Basic],
    ):
        super().__init__()
        self.call_stack = call_stack
        self.func_globals = func_globals
        self.args = args
        self.locals = {}
        self._tmp_counter = 0

    def set_local(self, name, value):
        self.locals[name] = value

    def visit(self, node: Any):
        try:
            value = super().visit(node)
        except (ValueError, NotImplementedError) as e:
            print(f"{self.call_stack[-1]}:{getattr(node, 'lineno', None)}: {type(e)}")
            raise e
        if value is NotImplemented:
            raise NotImplementedError(f"{node} returned NotImplemented")
        return value

    def visit_Module(self, node) -> AbstractNode:
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node) -> AbstractNode:
        scope = self.scope(extra_locals={})
        statements = scope.visit_compound_statement(node.body)

        declarations = {}
        for name, target in scope.locals.items():
            declarations[name] = Declaration(name, target)

        loop_var = SymbolicScalar("__root_index", "int64")
        return ForLoop(
            index_var=loop_var,
            index_begin=sympy.Number(0),
            index_end=sympy.Number(1),
            index_step=sympy.Number(1),
            declarations=declarations,
            statements=statements,
            arguments=self.args,
        )

    def visit_Return(self, node):
        raise NotImplementedError

    def visit_Delete(self, node):
        raise NotImplementedError

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise ValueError("Only single assignment is supported.")
        value = self.visit(node.value)
        target_node = node.targets[0]
        if all_sympy(value):
            value_shape = SympyShape(value)
            value_dtype = SympyDtype(value)
            if isinstance(target_node, ast.Name):
                if value_shape.args == ():
                    target = SymbolicScalar(target_node.id, value_dtype)
                else:
                    target = SymbolicArray(target_node.id, value_dtype, value_shape)
                if target_node.id not in self.locals:
                    self.set_local(target_node.id, target)
            else:
                target = self.visit(target_node)
            if isinstance(value, SymbolicArray) and value.is_placeholder:
                return None  # Don't create an assignment for placeholder arrays
            return Assignment(target, value)
        else:
            raise ValueError("Constexpr assignment not supported yet.")

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        value = self.visit(node.value)
        if not all_sympy(target, value):
            raise ValueError(
                "Augmented assignment requires both target and value to be dynamic expressions."
            )
        target_shape = SympyShape(target)
        value_shape = SympyShape(value)
        if target_shape.args != value_shape.args:
            raise ValueError(
                f"Shape mismatch in augmented assignment: "
                f"{target_shape=} vs {value_shape=}"
            )
        result = _apply_binary_method(node.op, target, value)
        return Assignment(target, result)

    def visit_AnnAssign(self, node):
        if not node.simple:
            raise ValueError("Annotated assignment must be simple.")
        value = self.visit(node.value)
        target_node = node.target
        annotation = self.visit(node.annotation)
        if annotation not in TYPE_MAP:
            raise ValueError(f"Unsupported annotation: {annotation}")
        annot_dtype = TYPE_MAP[annotation]
        if all_sympy(value):
            value_shape = SympyShape(value)
            value_dtype = SympyDtype(value)
            if value_dtype not in UNKNOWN_TYPES and annot_dtype != value_dtype:
                raise ValueError(
                    f"Annotated type {annot_dtype} does not match value type {value_dtype}."
                )
            if value_shape.args == ():
                target = SymbolicScalar(target_node.id, annot_dtype)
            else:
                target = SymbolicArray(target_node.id, annot_dtype, value_shape)
            if target_node.id not in self.locals:
                self.set_local(target_node.id, target)
            if isinstance(value, SymbolicArray) and value.is_placeholder:
                return None  # Don't create an assignment for placeholder arrays
            return Assignment(target, value)
        else:
            raise ValueError("Constexpr annotated assignment not supported yet.")

    def scope(self, extra_locals: Dict[str, sympy.Basic] = None):
        if extra_locals is None:
            extra_locals = {}
        block = NodeVisitor(
            call_stack=self.call_stack,
            func_globals=self.func_globals,
            args=self.args | self.locals,
        )
        for name, value in extra_locals.items():
            block.set_local(name, value)
        return block

    def visit_For(self, node) -> AbstractNode:
        if not isinstance(node.target, ast.Name):
            raise ValueError("For loop index must be a variable name.")

        if not isinstance(node.iter, ast.Call):
            raise ValueError("For loop iteration must be a range.")
        if not isinstance(node.iter.func, ast.Name):
            raise ValueError("For loop iteration must be a range.")
        IteratorClass = self.dereference_name(node.iter.func.id)
        if IteratorClass is not range:
            raise ValueError("For loop iteration must be a range.")

        iter_args = [self.visit(arg) for arg in node.iter.args]
        if len(iter_args) == 1:
            begin = sympy.Number(0)
            end = iter_args[0]
            step = sympy.Number(1)
        elif len(iter_args) == 2:
            begin, end = iter_args
            step = sympy.Number(1)
        elif len(iter_args) == 3:
            begin, end, step = iter_args
        else:
            raise ValueError(f"Too many arguments for range")

        if not all_sympy(begin, end, step):
            raise ValueError("For loop range arguments must be dynamic expressions.")

        loop_var_name = node.target.id

        if loop_var_name in self.locals:
            raise ValueError(f"Loop variable {loop_var_name} already defined.")

        loop_var = SymbolicScalar(loop_var_name, "int64")
        scope = self.scope(extra_locals={loop_var_name: loop_var})
        statements = scope.visit_compound_statement(node.body)

        declarations = {}
        for name, target in scope.locals.items():
            if name != loop_var_name:
                declarations[name] = Declaration(name, target)

        return ForLoop(loop_var, begin, end, step, declarations, statements)

    def visit_compound_statement(self, stmts) -> List[AbstractNode]:
        # Ensure that stmts is iterable
        if not isinstance(stmts, (list, tuple)):
            stmts = [stmts]

        nodes = []
        for stmt in stmts:
            tree = self.visit(stmt)
            if tree is not None:
                nodes.append(tree)

        return nodes

    def visit_While(self, node):
        raise NotImplementedError

    def visit_If(self, node):
        raise NotImplementedError

    def visit_With(self, node):
        raise NotImplementedError

    def visit_Match(self, node):
        raise NotImplementedError

    def visit_Raise(self, node):
        raise NotImplementedError

    def visit_Try(self, node):
        raise NotImplementedError

    def visit_Assert(self, node):
        raise NotImplementedError

    def visit_Import(self, node):
        raise NotImplementedError

    def visit_ImportFrom(self, node):
        raise NotImplementedError

    def visit_Global(self, node):
        raise NotImplementedError

    def visit_Nonlocal(self, node):
        raise NotImplementedError

    def visit_Pass(self, node):
        raise NotImplementedError

    def visit_Break(self, node):
        raise NotImplementedError

    def visit_Continue(self, node):
        raise NotImplementedError

    # Expressions

    def visit_BoolOp(self, node):
        if len(node.values) != 2:
            raise ValueError("Only binary boolean operations are supported.")
        left = self.visit(node.values[0])
        right = self.visit(node.values[1])
        if isinstance(node.op, ast.And):
            if not all_sympy(left, right):
                return left and right
            return left & right
        if isinstance(node.op, ast.Or):
            if not all_sympy(left, right):
                return left or right
            return left | right

    def visit_NamedExpr(self, node):
        raise NotImplementedError

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        result = _apply_binary_method(node.op, lhs, rhs)
        return result

    def visit_UnaryOp(self, node):
        x = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return x
        if isinstance(node.op, ast.USub):
            return -x
        if isinstance(node.op, ast.Not):
            if all_sympy(x):
                return sympy.Not(x)
            return not x
        if isinstance(node.op, ast.Invert):
            if all_sympy(x):
                return sympy.Not(x)
            return ~x

    def visit_Lambda(self, node):
        raise NotImplementedError

    def visit_IfExp(self, node):
        raise NotImplementedError

    def visit_Dict(self, node):
        raise NotImplementedError

    def visit_Set(self, node):
        raise NotImplementedError

    def visit_ListComp(self, node):
        raise NotImplementedError

    def visit_SetComp(self, node):
        raise NotImplementedError

    def visit_DictComp(self, node):
        raise NotImplementedError

    def visit_GeneratorExp(self, node):
        raise NotImplementedError

    def visit_Await(self, node):
        raise NotImplementedError

    def visit_Yield(self, node):
        raise NotImplementedError

    def visit_YieldFrom(self, node):
        raise NotImplementedError

    def visit_Compare(self, node):
        assert len(node.ops) == 1
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        return _apply_binary_method(node.ops[0], lhs, rhs)

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}

        if func is Array:
            named_args = full_arg_dict(func, args, kwargs)
            dtype, dims = named_args["dtype"], named_args["dims"]
            return SymbolicArray(sympy.sympify(0), dtype, dims, is_placeholder=True)

        elif func is math.exp:
            if len(args) != 1:
                raise ValueError("math.exp requires exactly one argument.")
            if not all_sympy(args[0]):
                return math.exp(args[0])
            return sympy.exp(args[0])

        raise NotImplementedError

    def visit_FormattedValue(self, node):
        raise NotImplementedError

    def visit_JoinedStr(self, node):
        raise NotImplementedError

    def visit_Constant(self, node):
        if isinstance(node.value, Union[int, float]):
            return sympy.Number(node.value)
        if isinstance(node.value, bool):
            return sympy.true if node.value else sympy.false
        return node.value

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        if all_sympy(lhs):
            raise ValueError(
                "Attribute access on dynamic expressions is not supported."
            )
        return getattr(lhs, node.attr)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if all_sympy(value, index):
            return SympyIndexing(value, index)
        return value[index]

    def visit_Starred(self, node):
        raise NotImplementedError

    def visit_Name(self, node):
        return self.dereference_name(node.id)

    def visit_List(self, node):
        raise NotImplementedError

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        if all_sympy(*args):
            return sympy.Tuple(*args)
        return tuple(args)

    def visit_Slice(self, node):
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None
        return slice(lower, upper, step)

    def dereference_name(self, name, absent=None) -> Any:
        error_if_absent = False
        if absent is None:
            error_if_absent = True
            absent = object()
        val = self.func_globals.get(name, absent)
        if val is absent:
            val = self.locals.get(name, absent)
        if val is absent:
            val = self.args.get(name, absent)
        if val is absent:
            val = builtin_namespace.get(name, absent)
        if error_if_absent and val is absent:
            raise ValueError(f"Name {name} not found in globals or args")
        return val

    def next_tmp_name(self) -> str:
        name = f"__tmp{self._tmp_counter}"
        self._tmp_counter += 1
        return name


def full_arg_dict(fn, args, kwargs):
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


builtin_namespace = {
    _.__name__: _ for _ in (len, list, range, float, int, isinstance, getattr)
}
_method_name_for_bin_op = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitAnd: "__and__",
    ast.BitOr: "__or__",
    ast.BitXor: "__xor__",
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
}


def _apply_binary_method(op, lhs, rhs):
    if isinstance(op, ast.Is):
        return lhs is rhs
    if isinstance(op, ast.IsNot):
        return lhs is not rhs
    op_name = _method_name_for_bin_op.get(type(op))
    rev_op_name = re.sub(r"__(.*)__", r"__r\1__", op_name)
    assert op_name is not None
    if isinstance(lhs, sympy.Basic):
        return getattr(lhs, op_name)(rhs)
    if isinstance(rhs, sympy.Basic):
        return getattr(rhs, rev_op_name)(lhs)
    return getattr(Constexpr(lhs), op_name)(Constexpr(rhs)).value


def all_sympy(*args):
    """Check if all arguments are sympy expressions."""
    return all(isinstance(arg, sympy.Basic) for arg in args)
