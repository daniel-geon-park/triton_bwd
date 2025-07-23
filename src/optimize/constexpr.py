class Constexpr:
    """
    This class is used to store a value that is known at compile-time.
    """

    def __init__(self, value):
        if isinstance(value, Constexpr):
            self.value = value.value
        else:
            self.value = value

    def __repr__(self) -> str:
        return f"constexpr[{self.value}]"

    def __index__(self):
        return self.value

    # In interpreter mode, constant values are not wrapped in constexpr,
    # and therefore do not have a .value attribute.
    # As a result, from here and below, we need to call the _constexpr_to_value
    # function to obtain either constexpr.value or the value itself.
    def __add__(self, other):
        return Constexpr(self.value + _constexpr_to_value(other))

    def __radd__(self, other):
        return Constexpr(_constexpr_to_value(other) + self.value)

    def __sub__(self, other):
        return Constexpr(self.value - _constexpr_to_value(other))

    def __rsub__(self, other):
        return Constexpr(_constexpr_to_value(other) - self.value)

    def __mul__(self, other):
        return Constexpr(self.value * _constexpr_to_value(other))

    def __mod__(self, other):
        return Constexpr(self.value % _constexpr_to_value(other))

    def __rmul__(self, other):
        return Constexpr(_constexpr_to_value(other) * self.value)

    def __truediv__(self, other):
        return Constexpr(self.value / _constexpr_to_value(other))

    def __rtruediv__(self, other):
        return Constexpr(_constexpr_to_value(other) / self.value)

    def __floordiv__(self, other):
        return Constexpr(self.value // _constexpr_to_value(other))

    def __rfloordiv__(self, other):
        return Constexpr(_constexpr_to_value(other) // self.value)

    def __gt__(self, other):
        return Constexpr(self.value > _constexpr_to_value(other))

    def __rgt__(self, other):
        return Constexpr(_constexpr_to_value(other) > self.value)

    def __ge__(self, other):
        return Constexpr(self.value >= _constexpr_to_value(other))

    def __rge__(self, other):
        return Constexpr(_constexpr_to_value(other) >= self.value)

    def __lt__(self, other):
        return Constexpr(self.value < _constexpr_to_value(other))

    def __rlt__(self, other):
        return Constexpr(_constexpr_to_value(other) < self.value)

    def __le__(self, other):
        return Constexpr(self.value <= _constexpr_to_value(other))

    def __rle__(self, other):
        return Constexpr(_constexpr_to_value(other) <= self.value)

    def __eq__(self, other):
        return Constexpr(self.value == _constexpr_to_value(other))

    def __ne__(self, other):
        return Constexpr(self.value != _constexpr_to_value(other))

    def __bool__(self):
        return bool(self.value)

    def __neg__(self):
        return Constexpr(-self.value)

    def __and__(self, other):
        return Constexpr(self.value & _constexpr_to_value(other))

    def logical_and(self, other):
        return Constexpr(self.value and _constexpr_to_value(other))

    def __or__(self, other):
        return Constexpr(self.value | _constexpr_to_value(other))

    def __xor__(self, other):
        return Constexpr(self.value ^ _constexpr_to_value(other))

    def logical_or(self, other):
        return Constexpr(self.value or _constexpr_to_value(other))

    def __pos__(self):
        return Constexpr(+self.value)

    def __invert__(self):
        return Constexpr(~self.value)

    def __pow__(self, other):
        return Constexpr(self.value ** _constexpr_to_value(other))

    def __rpow__(self, other):
        return Constexpr(_constexpr_to_value(other) ** self.value)

    def __rshift__(self, other):
        return Constexpr(self.value >> _constexpr_to_value(other))

    def __lshift__(self, other):
        return Constexpr(self.value << _constexpr_to_value(other))

    def __not__(self):
        return Constexpr(not self.value)


def _constexpr_to_value(v):
    if isinstance(v, Constexpr):
        return v.value
    return v
