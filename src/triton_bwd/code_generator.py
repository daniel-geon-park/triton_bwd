import ast
import inspect
import re
from contextlib import contextmanager
from typing import Any

import einx
import torch
import triton.language as tl
from triton import JITFunction
from triton.language import block_type, pointer_type

from triton_bwd.constexpr import Constexpr
from triton_bwd.dynamic_assert import dynamic_assert

RETURN_VAL = "__return_val__"


class CodeGenerator(ast.NodeVisitor):
    def __init__(self, call_stack, func_globals, pids, args, device):
        super().__init__()
        self.call_stack = call_stack
        self.func_globals = func_globals
        self.pids = pids
        self.args = args
        self.device = device
        self.locals = {}
        self.valid = True

    def set_local(self, name, value):
        self.locals[name] = value

    def visit(self, node: Any):
        value = super().visit(node)
        if value is NotImplemented:
            raise NotImplementedError(f"{node} returned NotImplemented")
        return value

    def dereference_name(self, name):
        absent = object()
        val = self.func_globals.get(name, absent)
        if val is absent:
            val = self.locals.get(name, absent)
        if val is absent:
            val = self.args.get(name, absent)
        if val is absent:
            val = builtin_namespace.get(name, absent)
        if val is absent:
            raise ValueError(f"Name {name} not found in globals or args")
        return val

    def dynamic_select(self, name, new_value, valid):
        absent = object()
        old_value = self.locals.get(name, absent)
        if old_value is absent:
            # Doesn't matter if valid is False
            self.locals[name] = new_value
        elif not is_trackable(old_value) and not is_trackable(new_value):
            if old_value != new_value:
                valid = to_trackable(valid, self.device)
                old_value = to_trackable(old_value, self.device)
                new_value = to_trackable(new_value, self.device)
                ensure_same_trackable_type(old_value, new_value)
                self.set_local(name, torch.where(valid, new_value, old_value))
            # else: old_value == new_value: do nothing
        else:
            valid = to_trackable(valid, self.device)
            old_value = to_trackable(old_value, self.device)
            new_value = to_trackable(new_value, self.device)
            ensure_same_trackable_type(old_value, new_value)
            if torch.is_tensor(new_value):
                self.set_local(name, torch.where(valid, new_value, old_value))
            elif isinstance(new_value, Pointer):
                self.set_local(
                    name,
                    Pointer(
                        old_value.storage,
                        torch.where(valid, new_value.offset, old_value.offset),
                    ),
                )
            elif isinstance(new_value, BlockPointer):
                self.set_local(
                    name,
                    BlockPointer(
                        base=old_value.base,
                        shape=old_value.shape,
                        strides=old_value.strides,
                        offsets=[
                            torch.where(
                                valid,
                                new_value.offsets[0],
                                old_value.offsets[0],
                            ),
                            torch.where(
                                valid,
                                new_value.offsets[1],
                                old_value.offsets[1],
                            ),
                        ],
                        block_shape=old_value.block_shape,
                        order=old_value.order,
                    ),
                )
            else:
                raise ValueError(f"Unsupported type {type(new_value)}")

    def visit_FunctionDef(self, node):
        # TODO: initialize default arguments
        self.visit_compound_statement(node.body)

    def visit_Return(self, node):
        self.set_local(RETURN_VAL, self.visit(node.value))
        return "return"

    def visit_AugAssign(self, node):
        name = node.target.id
        lhs = ast.Name(id=name, ctx=ast.Load())
        rhs = ast.BinOp(lhs, node.op, node.value)
        assign = ast.Assign(targets=[node.target], value=rhs)
        self.visit(assign)
        return "continue"

    def visit_AnnAssign(self, node):
        target = self.visit(node.target)
        value = self.visit(node.value)
        self.set_local(target, value)
        return "continue"

    def visit_compound_statement(self, stmts):
        # Ensure that stmts is iterable
        if not isinstance(stmts, (list, tuple)):
            stmts = [stmts]
        for stmt in stmts:
            status = self.visit(stmt)
            assert status in ("return", "continue"), f"Unexpected status {status}"

            # Stop parsing as soon as we hit a `return` statement; everything
            # after this is dead code.
            if status == "return":
                return "return"
        return "continue"

    @contextmanager
    def cond_block(self, condition, extra_locals=None):
        if extra_locals is None:
            extra_locals = {}
        valid = self.valid & condition
        block = CodeGenerator(
            call_stack=self.call_stack,
            func_globals=self.func_globals,
            pids=self.pids,
            args=self.args | self.locals,
            device=self.device,
        )
        block.valid = valid
        for name, value in extra_locals.items():
            block.set_local(name, value)
        yield block
        for name in block.locals:
            new_value = block.locals[name]
            self.dynamic_select(name, new_value, valid)

    def dynamic_assert(self, condition, message):
        dynamic_assert(
            condition,
            to_trackable(self.valid, self.device),
            message,
        )

    def visit_For(self, node):
        IteratorClass = self.visit(node.iter.func)
        iter_args = [self.visit(arg) for arg in node.iter.args]
        iter_kwargs = {
            keyword.arg: self.visit(keyword.value) for keyword in node.iter.keywords
        }
        if IteratorClass is range:
            if not any(torch.is_tensor(arg) for arg in iter_args):

                def static_iterate():
                    for i in range(*iter_args):
                        yield i, True

                iterator = static_iterate()

            else:
                if "max_iters" not in iter_kwargs:
                    raise ValueError(
                        f"{self.call_stack[-1]}:{node.lineno}: "
                        "max_iters must be provided for a dynamic range"
                    )
                max_iters = iter_kwargs["max_iters"]
                assert not torch.is_tensor(max_iters)

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

                self.dynamic_assert(
                    (end - begin + step - 1) // step <= max_iters,
                    f"{self.call_stack[-1]}:{node.lineno}: " "Range exceeds max_iters",
                )

                def dynamic_iterate():
                    it = begin
                    for i in range(max_iters):
                        valid_ = it < end
                        yield it, valid_
                        it = it + step

                iterator = dynamic_iterate()
        else:
            raise ValueError(f"Unsupported iterator {IteratorClass}")

        status = "continue"
        for iteration, valid in iterator:
            with self.cond_block(valid, {node.target.id: iteration}) as block:
                status = block.visit_compound_statement(node.body)
                if status == "return":
                    return "return"
        if status == "continue":
            status = self.visit_compound_statement(node.orelse)
        return status

    def visit_While(self, node):
        raise NotImplementedError("While not supported")

    def visit_If(self, node):
        test = self.visit(node.test)
        if torch.is_tensor(test):
            with self.cond_block(test) as block:
                status_true = block.visit_compound_statement(node.body)
            with self.cond_block(~test) as block:
                status_false = block.visit_compound_statement(node.orelse)
            if "return" in (status_true, status_false):
                raise NotImplementedError("Return not supported in dynamic conditional")
            return "continue"
        else:
            if test:
                status = self.visit_compound_statement(node.body)
            else:
                status = self.visit_compound_statement(node.orelse)
        return status

    def visit_Assert(self, node):
        test = self.visit(node.test)
        msg = self.visit(node.msg)
        self.dynamic_assert(test, f"{self.call_stack[-1]}:{node.lineno}: " + msg)
        return "continue"

    def visit_Pass(self, node):
        return "continue"

    def visit_Break(self, node):
        raise NotImplementedError("Break not supported")

    def visit_Continue(self, node):
        raise NotImplementedError("Continue not supported")

    def visit_Call(self, node):
        fn = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        if isinstance(fn, tuple):
            name, fn = fn
            assert name == "callable"
            return fn(*args, **kwargs)
        elif fn is tl.sum:
            bound_args = fn.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            named_args = bound_args.arguments
            input = named_args["input"]
            axis = named_args["axis"]
            keep_dims = named_args["keep_dims"]
            if input.dtype == torch.bfloat16:
                input = input.to(torch.float32)
            return input.sum(dim=axis, keepdim=keep_dims)
        elif fn is tl.max:
            bound_args = fn.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            named_args = bound_args.arguments
            input = named_args["input"]
            axis = named_args["axis"]
            return_indices = named_args["return_indices"]
            keep_dims = named_args["keep_dims"]
            assert not return_indices
            if input.dtype == torch.bfloat16:
                input = input.to(torch.float32)
            return torch.amax(input, dim=axis, keepdim=keep_dims)
        elif isinstance(fn, JITFunction):
            bound_args = fn.signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            tree = ast.parse(fn.src)
            gen = CodeGenerator(
                call_stack=self.call_stack[:-1]
                + [self.call_stack[-1] + f":{node.lineno}", fn.__name__],
                func_globals=fn.__globals__,
                pids=self.pids,
                args=bound_args.arguments,
                device=self.device,
            )
            gen.valid = self.valid
            gen.visit(tree)
            return gen.locals.get(RETURN_VAL, None)
        elif fn is tl.program_id:
            named_args = full_arg_dict(fn, args, kwargs)
            axis = named_args["axis"]
            return self.pids[axis]
        elif fn is tl.load:
            named_args = full_arg_dict(fn, args, kwargs)
            pointer = named_args["pointer"]
            if isinstance(pointer, Pointer):
                mask = named_args["mask"]
                other = named_args["other"]
                mask = True if mask is None else mask
                return pointer.value(mask & self.valid, other)
            elif isinstance(pointer, BlockPointer):
                boundary_check = named_args["boundary_check"]
                padding_option = named_args["padding_option"]
                return pointer.value(boundary_check, padding_option, self.valid)
        elif fn is tl.store:
            named_args = full_arg_dict(fn, args, kwargs)
            pointer = named_args["pointer"]
            if isinstance(pointer, Pointer):
                value = named_args["value"]
                mask = named_args["mask"]
                mask = True if mask is None else mask
                pointer.assign(value, mask & self.valid)
            elif isinstance(pointer, BlockPointer):
                value = named_args["value"]
                boundary_check = named_args["boundary_check"]
                pointer.assign(value, boundary_check, self.valid)
        elif fn is tl.minimum:
            a, b = args
            if torch.is_tensor(a) or torch.is_tensor(b):
                if not torch.is_tensor(a):
                    a = b.new_full((), a)
                if not torch.is_tensor(b):
                    b = a.new_full((), b)
                return torch.minimum(a, b)
            return min(a, b)
        elif fn is tl.maximum:
            a, b = args
            if torch.is_tensor(a) or torch.is_tensor(b):
                if not torch.is_tensor(a):
                    a = b.new_full((), a)
                if not torch.is_tensor(b):
                    b = a.new_full((), b)
                return torch.maximum(a, b)
            return max(a, b)
        elif fn is tl.arange:
            named_args = full_arg_dict(fn, args, kwargs)
            start = named_args["start"]
            end = named_args["end"]
            return torch.arange(start, end, device=self.device)
        elif fn is tl.full:
            named_args = full_arg_dict(fn, args, kwargs)
            shape = named_args["shape"]
            value = named_args["value"]
            dtype = named_args["dtype"]
            return torch.full(
                shape,
                fill_value=value,
                dtype=to_torch_dtype(dtype),
                device=self.device,
            )
        elif fn is tl.dot:
            named_args = full_arg_dict(fn, args, kwargs)
            input = named_args["input"]
            other = named_args["other"]
            acc = named_args["acc"]
            input_precision = named_args["input_precision"]
            out_dtype = named_args["out_dtype"]
            if input_precision is None:
                input_precision = tl.float32
            result = input.to(to_torch_dtype(input_precision)) @ other.to(
                to_torch_dtype(input_precision)
            )
            if acc is not None:
                result += acc
            if out_dtype is not None:
                result = result.to(to_torch_dtype(out_dtype))
            return result
        elif fn is tl.where:
            named_args = full_arg_dict(fn, args, kwargs)
            condition = named_args["condition"]
            x = named_args["x"]
            y = named_args["y"]
            return torch.where(condition, x, y)
        elif fn is tl.static_assert:
            named_args = full_arg_dict(fn, args, kwargs)
            cond = named_args["cond"]
            msg = named_args["msg"]
            assert cond, f"Static assert failed at {node.func} L{node.lineno}:\n{msg}"
            return None
        elif fn is tl.make_block_ptr:
            named_args = full_arg_dict(fn, args, kwargs)
            base = named_args["base"]
            shape = named_args["shape"]
            strides = named_args["strides"]
            offsets = named_args["offsets"]
            block_shape = named_args["block_shape"]
            order = named_args["order"]
            return BlockPointer(
                base,
                shape,
                strides,
                offsets,
                block_shape,
                order,
            )
        elif fn is tl.advance:
            named_args = full_arg_dict(fn, args, kwargs)
            base: BlockPointer = named_args["base"]
            offsets = named_args["offsets"]
            return base.advance(offsets)
        elif fn is tl.multiple_of:
            named_args = full_arg_dict(fn, args, kwargs)
            input = named_args["input"]
            values = named_args["values"]
            self.dynamic_assert(
                input % values == 0, f"{self.call_stack[-1]}:{node.lineno}: "
            )
            return input
        elif fn is tl.math.exp2:
            named_args = full_arg_dict(fn, args, kwargs)
            x = named_args["x"]
            return torch.exp2(x)
        elif fn is tl.math.log2:
            named_args = full_arg_dict(fn, args, kwargs)
            x = named_args["x"]
            return torch.log2(x)
        elif fn is tl.static_print:
            # TODO
            return
        elif fn is range:
            assert not any(torch.is_tensor(arg) for arg in args)
            return range(*args)
        elif fn is float:
            assert not any(torch.is_tensor(arg) for arg in args)
            return float(*args)
        else:
            raise ValueError(f"Unsupported function {fn}")

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        target = self.visit(node.targets[0])
        value = self.visit(node.value)
        if isinstance(target, tuple):
            for t, v in zip(target, value):
                self.set_local(t, v)
        else:
            self.set_local(target, value)
        return "continue"

    def visit_Expr(self, node):
        self.visit(node.value)
        return "continue"

    def visit_BoolOp(self, node):
        assert len(node.values) == 2
        left = self.visit(node.values[0])
        right = self.visit(node.values[1])
        if type(node.op) is ast.And:
            if not torch.is_tensor(left) and not torch.is_tensor(right):
                return left and right
            return left & right
        if type(node.op) is ast.Or:
            if not torch.is_tensor(left) and not torch.is_tensor(right):
                return left or right
            return left | right

    def visit_NamedExpr(self, node):
        target = self.visit(node.target)
        value = self.visit(node.value)
        self.set_local(target, value)
        return value

    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        result = _apply_binary_method(node.op, lhs, rhs)
        return result

    def visit_UnaryOp(self, node):
        x = self.visit(node.operand)
        if type(node.op) is ast.UAdd:
            return x
        if type(node.op) is ast.USub:
            return -x
        if type(node.op) is ast.Not:
            return not x
        if type(node.op) is ast.Invert:
            return ~x

    def visit_Lambda(self, node):
        raise NotImplementedError("Lambda not supported")

    def visit_IfExp(self, node):
        test = self.visit(node.test)
        assert not torch.is_tensor(test)
        if test:
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Dict(self, node):
        raise NotImplementedError("Dict not supported")

    def visit_Set(self, node):
        raise NotImplementedError("Set not supported")

    def visit_ListComp(self, node):
        raise NotImplementedError("ListComp not supported")

    def visit_SetComp(self, node):
        raise NotImplementedError("SetComp not supported")

    def visit_DictComp(self, node):
        raise NotImplementedError("DictComp not supported")

    def visit_GeneratorExp(self, node):
        raise NotImplementedError("GeneratorExp not supported")

    def visit_Compare(self, node):
        assert len(node.ops) == 1
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        return _apply_binary_method(node.ops[0], lhs, rhs)

    def visit_FormattedValue(self, node):
        raise NotImplementedError("FormattedValue not supported")

    def visit_Constant(self, node):
        return node.value

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        if torch.is_tensor(lhs):
            if node.attr == "to":
                return "callable", lambda dtype: lhs.to(to_torch_dtype(dtype))
            if node.attr == "dtype":
                return to_triton_dtype(lhs.dtype)
            else:
                raise ValueError(f"Unsupported attribute {node.attr}")
        return getattr(lhs, node.attr)

    def visit_Subscript(self, node):
        lhs = self.visit(node.value)
        slices = self.visit(node.slice)
        return lhs[slices]

    def visit_JoinedStr(self, node):
        raise NotImplementedError("JoinedStr not supported")

    def visit_Starred(self, node):
        raise NotImplementedError("Starred not supported")

    def visit_List(self, node):
        args = [self.visit(x) for x in node.elts]
        return args

    def visit_Tuple(self, node):
        args = [self.visit(x) for x in node.elts]
        return tuple(args)

    def visit_Slice(self, node):
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None
        return slice(lower, upper, step)

    def visit_Name(self, node):
        if type(node.ctx) == ast.Store:
            return node.id
        return self.dereference_name(node.id)


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

builtin_namespace = {
    _.__name__: _ for _ in (len, list, range, float, int, isinstance, getattr)
}
builtin_namespace.update(
    (
        ("print", tl.device_print),
        ("min", tl.minimum),
        ("max", tl.maximum),
    )
)

dtype_mapping = {
    tl.float8e5: torch.float8_e5m2,
    tl.bfloat16: torch.bfloat16,
    tl.float16: torch.float16,
    tl.float32: torch.float32,
    tl.float64: torch.float64,
    tl.int8: torch.int8,
    tl.int16: torch.int16,
    tl.int32: torch.int32,
    tl.int64: torch.int64,
    tl.uint8: torch.uint8,
    tl.uint16: torch.uint16,
    tl.uint32: torch.uint32,
    tl.uint64: torch.uint64,
}
dtype_rev_mapping = {v: k for k, v in dtype_mapping.items()}


def to_torch_dtype(dtype):
    return dtype_mapping[dtype]


def to_triton_dtype(dtype):
    return dtype_rev_mapping[dtype]


def ensure_same_trackable_type(lhs, rhs):
    if torch.is_tensor(lhs) and torch.is_tensor(rhs):
        assert lhs.dtype == rhs.dtype
        assert lhs.shape == rhs.shape
    elif isinstance(lhs, Pointer) and isinstance(rhs, Pointer):
        assert lhs.storage is rhs.storage
        assert lhs.offset.shape == rhs.offset.shape
    elif isinstance(lhs, BlockPointer) and isinstance(rhs, BlockPointer):
        assert lhs.base.storage is rhs.base.storage
    else:
        raise ValueError(f"Unsupported types {lhs} and {rhs}")


def is_trackable(value):
    return (
        torch.is_tensor(value)
        or isinstance(value, Pointer)
        or isinstance(value, BlockPointer)
    )


def to_trackable(value, device):
    if not is_trackable(value):
        value = torch.full((), value, device=device)
    if isinstance(value, Pointer):
        value = Pointer(
            value.storage,
            to_trackable(value.offset, device),
        )
    if isinstance(value, BlockPointer):
        value = BlockPointer(
            base=value.base,
            shape=value.shape,
            strides=value.strides,
            offsets=[
                to_trackable(value.offsets[0], device),
                to_trackable(value.offsets[1], device),
            ],
            block_shape=value.block_shape,
            order=value.order,
        )
    return value


def invalid_trackable_like(value):
    if torch.is_tensor(value):
        return torch.full_like(value, 0)
    if isinstance(value, Pointer):
        return Pointer(value.storage, torch.full_like(value.offset, 0))
    if isinstance(value, BlockPointer):
        return BlockPointer(
            base=value.base,
            shape=value.shape,
            strides=value.strides,
            offsets=[
                torch.full_like(value.offsets[0], 0),
                torch.full_like(value.offsets[1], 0),
            ],
            block_shape=value.block_shape,
            order=value.order,
        )
    raise ValueError(f"Unsupported value {value}")


def _apply_binary_method(op, lhs, rhs):
    if type(op) is ast.Is:
        return lhs is rhs
    if type(op) is ast.IsNot:
        return lhs is not rhs
    method_name = _method_name_for_bin_op.get(type(op))
    reverse_method_name = re.sub(r"__(.*)__", r"__r\1__", method_name)
    assert method_name is not None
    if isinstance(lhs, Pointer):
        return getattr(lhs, method_name)(rhs)
    if isinstance(rhs, Pointer):
        return getattr(rhs, reverse_method_name)(lhs)
    if torch.is_tensor(lhs):
        return getattr(lhs, method_name)(rhs)
    if torch.is_tensor(rhs):
        return getattr(rhs, reverse_method_name)(lhs)
    return getattr(Constexpr(lhs), method_name)(Constexpr(rhs)).value


def full_arg_dict(fn, args, kwargs):
    sig = inspect.signature(fn)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def convert_arg(arg):
    if isinstance(arg, torch.Tensor):
        storage = TensorStorage(underlying(arg))
        return Pointer(
            storage,
            torch.zeros([], dtype=torch.long, device=arg.device),
        )
    return arg


def underlying(tensor):
    return tensor.as_strided([tensor.numel()], [1])


class TensorStorage:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.updates = []


class Pointer:
    def __init__(self, storage: TensorStorage, offset):
        assert storage.tensor.ndim == 1
        self.storage = storage
        self.offset: torch.Tensor = offset

    def __add__(self, other):
        return Pointer(self.storage, self.offset + other)

    def __radd__(self, other):
        return Pointer(self.storage, other + self.offset)

    def value(self, mask, other):
        offset = self.offset
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.full(
                    offset.shape, mask, dtype=torch.bool, device=offset.device
                )
            offset = torch.where(mask, offset, torch.zeros_like(offset))

        result = einx.get_at("[i], ... -> ...", self.storage.tensor, offset)

        if mask is not None:
            if other is None:
                other = torch.zeros_like(result)
            result = torch.where(mask, result, other)

        return result

    def assign(self, value, mask):
        assert self.offset.shape == value.shape
        if mask is None:
            mask = torch.ones(value.shape, dtype=torch.bool, device=self.offset.device)
        else:
            if not torch.is_tensor(mask):
                mask = torch.full(
                    value.shape, mask, dtype=torch.bool, device=self.offset.device
                )
        assert mask is None or mask.shape == value.shape
        self.storage.updates.append((self.offset, value, mask))

    @property
    def type(self):
        return pointer_type(to_triton_dtype(self.storage.tensor.dtype))

    @property
    def dtype(self):
        return pointer_type(to_triton_dtype(self.storage.tensor.dtype))

    @property
    def element_ty(self):
        return to_triton_dtype(self.storage.tensor.dtype)


class BlockPointer:
    def __init__(
        self,
        base: Pointer,
        shape: list[int | torch.Tensor],
        strides: list[int | torch.Tensor],
        offsets: list[int | torch.Tensor],
        block_shape: list[int | torch.Tensor],
        order: list[int | torch.Tensor],
    ):
        self.base = base
        self.shape = shape
        self.strides = strides
        self.offsets = offsets
        self.block_shape = block_shape
        self.order = order

    @property
    def dtype(self):
        return block_type(self.base.element_ty, self.block_shape)

    def advance(self, offsets):
        new_offsets = [
            self.offsets[0] + offsets[0],
            self.offsets[1] + offsets[1],
        ]
        return BlockPointer(
            base=self.base,
            shape=self.shape,
            strides=self.strides,
            offsets=new_offsets,
            block_shape=self.block_shape,
            order=self.order,
        )

    def value(self, boundary_check, padding_option, valid):
        dtype = self.base.storage.tensor.dtype
        device = dict(device=self.base.offset.device)
        offset_0 = self.offsets[0] + torch.arange(self.block_shape[0], **device)
        offset_1 = self.offsets[1] + torch.arange(self.block_shape[1], **device)
        ptr = (
            self.base
            + self.strides[0] * offset_0[:, None]
            + self.strides[1] * offset_1[None, :]
        )
        if not torch.is_tensor(valid):
            mask = torch.full(ptr.offset.shape, valid, dtype=torch.bool, **device)
        else:
            mask = valid.expand(ptr.offset.shape)
        if 0 in boundary_check:
            mask = mask & (offset_0 < self.shape[0])[:, None]
        if 1 in boundary_check:
            mask = mask & (offset_1 < self.shape[1])[None, :]
        if padding_option in ("", "zero"):
            other = torch.zeros(ptr.offset.shape, dtype=dtype, **device)
        elif padding_option == "nan":
            other = torch.full(ptr.offset.shape, float("nan"), dtype=dtype, **device)
        else:
            raise ValueError(f"Unsupported padding option {padding_option}")
        result = ptr.value(mask, other)
        return result

    def assign(self, value, boundary_check, valid):
        device = dict(device=self.base.offset.device)
        offset_0 = self.offsets[0] + torch.arange(self.block_shape[0], **device)
        offset_1 = self.offsets[1] + torch.arange(self.block_shape[1], **device)
        ptr = (
            self.base
            + self.strides[0] * offset_0[:, None]
            + self.strides[1] * offset_1[None, :]
        )
        if not torch.is_tensor(valid):
            mask = torch.full(ptr.offset.shape, valid, dtype=torch.bool, **device)
        else:
            mask = valid.expand(ptr.offset.shape)
        if 0 in boundary_check:
            mask = mask & (offset_0 < self.shape[0])[:, None]
        if 1 in boundary_check:
            mask = mask & (offset_1 < self.shape[1])[None, :]
        ptr.assign(value, mask)
