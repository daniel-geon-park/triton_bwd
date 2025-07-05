import ast
import inspect
import re
from types import FunctionType
from typing import Any, Callable

from triton_bwd.abtract_tree import *
from triton_bwd.constexpr import Constexpr


class OptimizableFunction:
    def __init__(
        self,
        func: FunctionType,
        arg_specs: Dict[str, ArgSpec],
    ):
        self.func = func

        signature = inspect.signature(func)
        for _, param in signature.parameters.items():
            if param.name in arg_specs:
                spec = arg_specs[param.name]
                spec.name = param.name
                if isinstance(spec, ArraySpec):
                    if param.annotation is InArray:
                        spec.kind = "in"
                    elif param.annotation is OutArray:
                        spec.kind = "out"
                    elif param.annotation is InOutArray:
                        spec.kind = "in_out"
                    else:
                        raise ValueError(
                            f"Argument {param.name} must be annotated with "
                            f"InArray, OutArray, or InOutArray, but got "
                            f"{param.annotation}"
                        )
                else:
                    raise ValueError(
                        f"Argument {param.name} has unsupported spec type: "
                        f"{type(spec)}"
                    )
            else:
                if param.annotation is param.empty:
                    raise ValueError(
                        f"A type annotation is missing on argument {param.name}."
                    )
                if param.annotation is int:
                    spec = IntSpec()
                    spec.name = param.name
                    arg_specs[param.name] = spec
                elif param.annotation in [InArray, OutArray, InOutArray]:
                    # Array argument must have a spec defined
                    raise ValueError(
                        f"Array argument {param.name} must have a spec defined."
                    )
                else:
                    raise ValueError(
                        f"Argument {param.name} has unsupported type: "
                        f"{param.annotation}"
                    )

        self.arg_specs = arg_specs

        source = inspect.getsource(self.func)
        tree = ast.parse(source)

        args = {name: spec.symbol() for name, spec in self.arg_specs.items()}
        visitor = NodeVisitor(
            call_stack=[self.func.__name__],
            func_globals=self.func.__globals__,
            args=args,
        )
        self.abstract_tree = visitor.visit(tree)


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

    def set_local(self, name, value):
        self.locals[name] = value

    def visit(self, node: Any):
        try:
            value = super().visit(node)
            if isinstance(node, ast.expr):
                if not isinstance(value, sympy.Basic):
                    raise ValueError(
                        f"Expression {node} must return a sympy type, "
                        f"but got {type(value)}: {value}"
                    )
        except (ValueError, NotImplementedError) as e:
            print(f"{self.call_stack[-1]}:{getattr(node, 'lineno', None)}: {type(e)}")
            raise e
        if value is NotImplemented:
            raise NotImplementedError(f"{node} returned NotImplemented")
        return value

    def visit_Module(self, node) -> AbstractNode:
        return self.visit(node.body[0])

    def visit_FunctionDef(self, node) -> AbstractNode:
        nodes = self.visit_compound_statement(node.body)
        return AbstractNode(ForLoop("__root_index", 0, 1, 1, {}, nodes))

    def visit_Return(self, node):
        raise NotImplementedError

    def visit_Delete(self, node):
        raise NotImplementedError

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise ValueError("Only single assignment is supported.")
        value = self.visit(node.value)
        value_shape = SympyShape(value)
        target_node = node.targets[0]
        if isinstance(target_node, ast.Name):
            if value_shape.args == ():
                target = sympy.symbols(target_node.id)
            else:
                target = sympy.IndexedBase(
                    target_node.id,
                    shape=value_shape.args,
                )
            if target_node.id not in self.locals:
                self.set_local(target_node.id, target)
        else:
            target = self.visit(target_node)
        return Assignment(target, value)

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        target_shape = SympyShape(target)
        value = self.visit(node.value)
        value_shape = SympyShape(value)
        if target_shape.args != value_shape.args:
            raise ValueError(
                f"Shape mismatch in augmented assignment: "
                f"{target_shape=} vs {value_shape=}"
            )
        result = _apply_binary_method(node.op, target, value)
        return Assignment(target, result)

    def visit_AnnAssign(self, node):
        raise NotImplementedError

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
            begin = 0
            end = iter_args[0]
            step = 1
        elif len(iter_args) == 2:
            begin, end = iter_args
            step = 1
        elif len(iter_args) == 3:
            begin, end, step = iter_args
        else:
            raise ValueError(f"Too many arguments for range")

        loop_var_name = node.target.id

        if loop_var_name in self.locals:
            raise ValueError(f"Loop variable {loop_var_name} already defined.")

        scope = self.scope(
            extra_locals={loop_var_name: sympy.symbols(loop_var_name, integer=True)}
        )
        statements = scope.visit_compound_statement(node.body)

        declarations = {}
        for name, target in scope.locals.items():
            if name != loop_var_name:
                declarations[name] = target

        return AbstractNode(
            ForLoop(loop_var_name, begin, end, step, declarations, statements)
        )

    def visit_compound_statement(self, stmts) -> List[AbstractNode]:
        # Ensure that stmts is iterable
        if not isinstance(stmts, (list, tuple)):
            stmts = [stmts]

        nodes = []
        for stmt in stmts:
            tree = self.visit(stmt)
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
            return left & right
        if isinstance(node.op, ast.Or):
            return left & right

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
            return sympy.Not(x)
        if isinstance(node.op, ast.Invert):
            return sympy.Not(x)

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
        raise ValueError(
            f"Unsupported constant type: {type(node.value)} with value {node.value}"
        )

    def visit_Attribute(self, node):
        raise NotImplementedError

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        index = self.visit(node.slice)
        return SympyIndexing(value, index)

    def visit_Starred(self, node):
        raise NotImplementedError

    def visit_Name(self, node):
        return self.dereference_name(node.id)

    def visit_List(self, node):
        raise NotImplementedError

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        return sympy.Tuple(*args)

    def visit_Slice(self, node):
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None
        return slice(lower, upper, step)

    def dereference_name(self, name, absent=None) -> sympy.Basic:
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


def optimize(
    arg_specs: Dict[str, ArgSpec]
) -> Callable[[FunctionType], OptimizableFunction]:

    def wrapper(func: FunctionType) -> OptimizableFunction:
        return OptimizableFunction(
            func=func,
            arg_specs=arg_specs,
        )

    return wrapper
