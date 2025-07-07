from triton_bwd.make_ast import ArraySpec, InArray, OutArray, optimize


@optimize(
    {
        "a": ArraySpec(dtype="float32", dims=("N", "K")),
        "b": ArraySpec(dtype="float32", dims=("K", "M")),
        "c": ArraySpec(dtype="float32", dims=("N", "K")),
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
