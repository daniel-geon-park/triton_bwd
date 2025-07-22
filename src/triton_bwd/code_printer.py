from sympy.printing.pycode import ArrayPrinter, PythonCodePrinter

from triton_bwd.sympy_utils import SymbolicScalar, SympyIndexing

torch_functions = {
    "exp": "torch.exp",
}

triton_functions = {
    "exp": "tl.exp",
}


class CodePrinter(ArrayPrinter, PythonCodePrinter):
    def __init__(self, backend: str, settings=None):
        if backend == "torch":
            self._kf = torch_functions
        else:
            self._kf = triton_functions
        self._kc = {}
        super().__init__(settings=settings)
        self.backend = backend

    def _print_SympyIndexing(self, expr: SympyIndexing):
        return f"{expr}"

    def _print_SymbolicScalar(self, expr: SymbolicScalar):
        return self.doprint(expr.label)

    def _print_SymbolicArray(self, expr):
        return self.doprint(expr.label)
