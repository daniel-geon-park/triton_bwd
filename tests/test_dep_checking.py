import math

from optimize.code_gen import generate_code
from optimize.optimize import Array, ArraySpec, InArray, InOutArray, OutArray, optimize


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
    print(matrix_multiply.tree.numbered_repr())
    result = matrix_multiply.tree.find_dependence(0, 0)
    print(result)
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
    print(book_example.tree.numbered_repr())
    book_example.tree.find_dependence(1, 1)
    dependencies = set()
    for i in range(6):
        for j in range(6):
            deps = book_example.tree.find_dependence(i, j)
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


@optimize(
    {
        "a": ArraySpec(dtype="float32", dims=("N", "K")),
        "b": ArraySpec(dtype="float32", dims=("K", "M")),
        "c": ArraySpec(dtype="float32", dims=("N", "M")),
        "d": ArraySpec(dtype="float32", dims=("N", "M")),
    }
)
def example1(
    a: InArray,
    b: InArray,
    c: OutArray,
    d: OutArray,
    N: int,
    K: int,
    M: int,
):
    for i in range(N):
        for j in range(M):
            for k in range(K):
                c[i, j] += a[i, k] * b[k, j]
            for k in range(K):
                d[i, j] += a[i, k] * b[k, j]


@optimize(
    {
        "a": ArraySpec(dtype="float32", dims=("N", "K")),
        "b": ArraySpec(dtype="float32", dims=("K", "M")),
        "c": ArraySpec(dtype="float32", dims=("N", "M")),
    }
)
def example2(
    a: InArray,
    b: InArray,
    c: OutArray,
    N: int,
    K: int,
    M: int,
):
    for i in range(N):
        for k in range(K):
            for j in range(M):
                c[i, j] += a[i, k] * b[k, j]
            for j in range(M):
                c[i, j] = a[i, k] * b[k, j]


@optimize(
    {
        "a": ArraySpec(dtype="float32", dims=("N", "K")),
        "b": ArraySpec(dtype="float32", dims=("K", "M")),
        "c": ArraySpec(dtype="float32", dims=("N", "M")),
    }
)
def example3(
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
            for k in range(K):
                c[i, j] += a[i, k] * b[k, j]


def test_fuse_loop():
    print(example1.tree.numbered_repr())
    print("Fuse result:", example1.tree.fuse_loop(3, 4), sep="\n")

    print(example2.tree.numbered_repr())
    print("Fuse result:", example2.tree.fuse_loop(3, 4), sep="\n")

    print(example3.tree.numbered_repr())
    try:
        example3.tree.fuse_loop(3, 4)
    except ValueError as e:
        print(f"Expected error: {e}")
    else:
        raise AssertionError("Expected a ValueError due to non-fusible loops.")


@optimize(
    {
        "a": ArraySpec(
            dtype="float32",
            dims=(
                "M",
                "N",
            ),
        ),
        "b": ArraySpec(
            dtype="float32",
            dims=(
                "M",
                "N",
            ),
        ),
        "c": ArraySpec(
            dtype="float32",
            dims=(
                "M",
                "N",
            ),
        ),
    }
)
def example4(
    a: InOutArray,
    b: InOutArray,
    c: InOutArray,
    M: int,
    N: int,
):
    for i in range(M - 1):
        for j in range(N):
            a[i, j] = b[i, j] + 1
            c[i, j] = a[i + 1, j] + a[i, j] + b[i, j]


def test_split_loop():
    print(example4.tree.numbered_repr())
    tree = example4.tree.split_loop(("A", 0))
    print("Split result:", tree.numbered_repr(), sep="\n")


@optimize(
    {
        "a": ArraySpec(
            dtype="float32",
            dims=(
                "M",
                "N",
            ),
        ),
        "b": ArraySpec(
            dtype="float32",
            dims=(
                "M",
                "N",
            ),
        ),
        "c": ArraySpec(
            dtype="float32",
            dims=(
                "M",
                "N",
            ),
        ),
    }
)
def example5(
    a: InOutArray,
    b: InOutArray,
    c: InOutArray,
    M: int,
    N: int,
):
    for i in range(M - 1):
        for j in range(N - 1):
            a[i, j] = b[i, j] + 1
            c[i, j] = a[i + 1, j] + a[i, j + 1] + b[i, j]


def test_split_loop_2():
    print(example5.tree.numbered_repr())
    try:
        tree = example5.tree.split_loop(("A", 0))
    except ValueError as e:
        print(f"Expected error: {e}")
    else:
        raise AssertionError("Expected a ValueError due to non-splittable loops.")


@optimize(
    {
        "q": ArraySpec(dtype="float32", dims=("T_Q", "D")),
        "k": ArraySpec(dtype="float32", dims=("T_KV", "D")),
        "v": ArraySpec(dtype="float32", dims=("T_KV", "D")),
        "o": ArraySpec(dtype="float32", dims=("T_Q", "D")),
    }
)
def attention(
    q: InArray,
    k: InArray,
    v: InArray,
    o: OutArray,
    T_Q: int,
    T_KV: int,
    D: int,
):
    scores = Array(dtype="float32", dims=(T_Q, T_KV))
    exp_scores = Array(dtype="float32", dims=(T_Q, T_KV))
    exp_sum = Array(dtype="float32", dims=(T_Q,))
    probs = Array(dtype="float32", dims=(T_Q, T_KV))
    for iq in range(T_Q):
        exp_sum[iq] = 0.0
        for d in range(D):
            o[iq, d] = 0.0
    for iq in range(T_Q):
        for ik in range(T_KV):
            s: "float32" = 0
            for d in range(D):
                s += q[iq, d] * k[ik, d]
            scores[iq, ik] = s
    for iq in range(T_Q):
        for ik in range(T_KV):
            exp_scores[iq, ik] = math.exp(scores[iq, ik])
    for iq in range(T_Q):
        for ik in range(T_KV):
            exp_sum[iq] += exp_scores[iq, ik]
    for iq in range(T_Q):
        for ik in range(T_KV):
            probs[iq, ik] = exp_scores[iq, ik] / exp_sum[iq]
    for iq in range(T_Q):
        for ik in range(T_KV):
            for d in range(D):
                o[iq, d] += probs[iq, ik] * v[ik, d]


def test_optimize_attention():
    tree = attention.tree
    print(tree.numbered_repr())

    # 1 MERGE LOOPS
    tree = tree.fuse_loop(1, 3)
    tree = tree.fuse_loop(1, 5)
    tree = tree.fuse_loop(1, 6)
    tree = tree.fuse_loop(1, 7)
    tree = tree.fuse_loop(1, 8)

    # 2 TILE LOOP iq
    tree = tree.tile_loop(1, 32)

    # 3 LOCALIZE ARRAYS
    tree = tree.localize_array_allocation(0, 1)

    print("End result:")
    print(tree.numbered_repr())
