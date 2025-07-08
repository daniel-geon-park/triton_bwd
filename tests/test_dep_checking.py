from triton_bwd.abtract_tree import InOutArray
from triton_bwd.optimize import ArraySpec, InArray, OutArray, optimize


@optimize(
    {
        "a": ArraySpec(dtype="float32", dims=("N", "K")),
        "b": ArraySpec(dtype="float32", dims=("K", "M")),
        "c": ArraySpec(dtype="float32", dims=("N", "M")),
    }
)
def matrix_multiply(
    a: InArray,
    b: InArray,
    c: OutArray,
    N: int,
    K: int,
    M: int,
):
    for i in range(N):
        for j in range(M):
            for k in range(K):
                c[i, j] += a[i, k] * b[k, j]


def test_matmul():
    print(matrix_multiply.abstract_tree.numbered_repr())
    result = set(matrix_multiply.abstract_tree.find_dependence(0, 0))
    assert result == {("flow", 3, "c"), ("anti", 3, "c"), ("outp", 3, "c")}


@optimize(
    {
        "a": ArraySpec(dtype="float32", dims=("N",)),
        "b": ArraySpec(dtype="float32", dims=("N",)),
        "c": ArraySpec(dtype="float32", dims=("N",)),
        "f": ArraySpec(dtype="float32", dims=("N", "M")),
    }
)
def book_example(
    a: InOutArray,
    b: InOutArray,
    c: InOutArray,
    f: InOutArray,
    y: float,
    z: float,
    N: int,
    M: int,
):
    x = y + 1
    for i in range(1, N - 1):
        c[i] = x + b[i]
        a[i] = c[i - 1] + z
        c[i + 1] = b[i] * a[i]
        for j in range(1, M):
            f[i, j] = f[i, j - 1] + x
    z = y + 3


def test_book_example():
    print(book_example.abstract_tree.numbered_repr())
    book_example.abstract_tree.find_dependence(1, 1)
    dependencies = set()
    for i in range(6):
        for j in range(6):
            deps = book_example.abstract_tree.find_dependence(i, j)
            if deps:
                print(f"Dependence from {i+1} to {j+1}: {deps}")
            for dep in deps:
                dependencies.add(((i, j), dep))
    assert dependencies == {
        ((0, 1), ("flow", 1, "x")),
        ((0, 4), ("flow", 1, "x")),
        ((2, 5), ("anti", 1, "z")),
        ((3, 1), ("outp", 1, "c")),
        ((3, 2), ("flow", 1, "c")),
        ((4, 4), ("flow", 2, "f")),
        ((1, 2), ("flow", 1, "c")),
        ((2, 3), ("flow", 2, "a")),
    }
