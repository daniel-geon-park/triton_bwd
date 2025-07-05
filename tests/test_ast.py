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
            sum = 0.0
            for k in range(K):
                sum += a[i, k] * b[k, j]
            c[i, j] = sum


print(matrix_multiply.abstract_tree.numbered_repr())
print(matrix_multiply.abstract_tree.find_dependence(1, 2))
