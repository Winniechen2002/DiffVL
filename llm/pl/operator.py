from ast import arguments
from .types import Arrow, List, Type, TypeInferenceFailure


def match_input(argnames, name2id, *args, **kwargs):
    inps = [None for i in argnames]
    for i in range(len(args)):
        inps[i] = args[i]
    for k, v in kwargs.items():
        assert inps[name2id[i]] is None
        inps[name2id[k]] = v
    for i in inps:
        assert i is not None, f"Missing input: {argnames} {args} {kwargs}"
    return inps


class Operator:
    def __init__(self, token, tp, value, argnames, code=None, default_parameters=None):
        assert isinstance(tp, Type)
        assert tp.n_defaults == len(default_parameters)
        self.token = token
        self.tp = tp
        self.value = value
        self.argnames = argnames
        self.code = code

        self.closure = None # only used in FuncDef within Func
        self.is_lambda = False
        self.code = None

        self.default_parameters = default_parameters

        self.name2id = {}
        for idx, name in enumerate(argnames):
            self.name2id[name] = idx

    def clone(self):
        op = Operator(self.token, self.tp, self.value, self.argnames, self.code, self.default_parameters)
        op.closure = self.closure
        op.is_lambda = self.is_lambda
        op.code = self.code
        return op

    @property
    def polymorphism(self):
        return self.tp.polymorphism

    def __call__(self, *args, **kwargs):
        # NOTE: we do not do the type check; instead we ensure all program returns the correct types 
        from .functool import get_ext
        return get_ext().eval(self, *args, **kwargs)

    def execute(self, *args, **kwargs):
        from .program import Program
        if isinstance(self.value, Program):
            raise NotImplementedError("do not supprot program..")
        else:
            assert len(kwargs) == 0, f"{kwargs}"
            out = self.value(*args, **kwargs)
        return out

    def __str__(self):
        return "(%s :  %s)" % (self.token, str(self.tp))

    def __repr__(self):
        return self.__str__()

    def pretty_print(self, *args, **kwargs):
        from .program import Program
        assert isinstance(self.value, Program)
        output = self.value._pretty_print(*args, **kwargs, is_return=True)

        def print_type(tp):
            if isinstance(tp, Arrow):
                return f"Arrow({', '.join(map(print_type, tp.arguments))}, {print_type(tp.out)})"
            elif isinstance(tp, List):
                return f"List({print_type(tp.base_type)})"
            else:
                return str(tp)

        head = f"def {self.token}(" + ", ".join(
            [f"{name}: {print_type(tp)}" for name, tp in 
             zip(self.argnames, self.tp.arguments)]) \
            + f") -> {print_type(self.tp.out)}:"
        output = [head] + output
        return  "\n    ".join(output)