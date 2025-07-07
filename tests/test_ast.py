from triton_bwd.abtract_tree import InOutArray
from triton_bwd.make_ast import ArraySpec, InArray, OutArray, optimize


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


print(matrix_multiply.abstract_tree.numbered_repr())
print(matrix_multiply.abstract_tree.find_dependence(0, 0))


@optimize(
    {
        "a": ArraySpec(dtype="float32", dims=("N",)),
        "b": ArraySpec(dtype="float32", dims=("N",)),
        "c": ArraySpec(dtype="float32", dims=("N",)),
        "f": ArraySpec(dtype="float32", dims=("N", "M")),
    }
)
def toy_example(
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


print(toy_example.abstract_tree.numbered_repr())
for i in range(6):
    for j in range(6):
        deps = toy_example.abstract_tree.find_dependence(i, j)
        if deps:
            print(f"Dependence from {i+1} to {j+1}: {deps}")
