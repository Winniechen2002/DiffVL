import inspect

from .operator import Operator
from .types import Arrow, Type, Integer, Bool, DataType, List, EnumType, String, none
from .functool import new_op
import typing


tA = Type("\'A")


def register_int(dsl: "DSL", lower=-200, upper=200):
    myint = Integer(lower, upper)
    dsl.register_constant_type(int, myint)

    #TODO: add type check for operators ..
    import torch
    def torch_bool2bool(x):
        if isinstance(x, bool): return x
        if isinstance(x, torch.Tensor) and x.dtype == torch.bool: return bool(x)
        return x

    @dsl.as_primitive
    def Add(a: tA, b: tA) -> tA:
        return a + b

    @dsl.as_primitive
    def Mult(a: tA, b: tA) -> tA:
        return a * b

    @dsl.as_primitive
    def Sub(a: tA, b: tA) -> tA:
        return a - b

    @dsl.as_primitive
    def Lt(a: tA, b: tA) -> dsl.bool:
        return torch_bool2bool(a < b)

    @dsl.as_primitive
    def USub(a: tA) -> tA:
        return -a

    @dsl.as_primitive
    def Gt(a: tA, b: tA) -> dsl.bool:
        return torch_bool2bool(a > b)

    @dsl.as_primitive
    def Eq(a: tA, b: tA) -> dsl.bool:
        return torch_bool2bool(a == b)

    @dsl.as_primitive
    def length(a: List(tA)) -> myint:
        return len(a)

    return myint


def register_bool(dsl: "DSL"):
    mybool = Bool()
    dsl.register_constant_type(bool, mybool)

    @dsl.as_primitive
    def And(a: tA, b: tA) -> tA:
        if isinstance(a, Operator):
            assert a.tp.out == mybool
            def myand(*args):
                return a(*args) & b(*args)
            op = new_op(myand)
            op.tp = a.tp
            op.argnames = b.argnames
            return op
        else:
            return a & b

    @dsl.as_primitive
    def Or(a: tA, b: tA) -> tA:
        if not isinstance(a, Operator):
            return a | b
        assert a.tp.out == mybool
        return new_op(lambda x: a(x) | b(x))

    def invert(x):
        if isinstance(x, bool):
            return not x
        return ~x

    @dsl.as_primitive
    def Not(a: tA) -> tA:
        if not isinstance(a, Operator):
            return invert(a)
        assert a.tp.out == mybool
        return new_op(lambda x: invert(a(x)))

    return mybool


class DSL:
    def __init__(self):
        super().__init__()
        self.operators: typing.Dict[str, Operator] = dict()
        self.constant_type = dict()
        self.constant_type_constructor = dict()
        self.types = dict()
        
        from . import lisp, functool
        for i in lisp.__all__ + functool.__all__:
            self.as_primitive(i)

        self.register_types(none)
        self.bool = register_bool(self)
        self.int = register_int(self)


        self.str = String()
        self.register_types(self.str)

        self.enum_types = []

        from . import Executor
        self.default_executor = Executor(self)

        from . import functool
        functool.EXECUTOR = self.default_executor

    def get_executor(self):
        from .functool import get_ext
        return get_ext()

    def register_constant_type(self, c_type, tp, constructor=None):
        self.constant_type[c_type] = tp
        self.constant_type_constructor[str(tp)] = constructor

    def register_types(self, tp: Type):
        if isinstance(tp, EnumType):
            self.enum_types.append(tp)

        children = tp.children
        if len(children) == 0:
            if tp.__class__ == Type:
                assert str(tp).startswith('\'')
                return
            self.types[str(tp)] = tp
        else:
            for i in children:
                self.register_types(i)

    def register_operator(self, primitive: Operator):
        assert primitive.token not in self.operators, f"Operator {primitive.token} has been added"
        self.operators[primitive.token] = primitive
        self.register_types(primitive.tp)

    def get_fn_tp(self, fn, check_return=True, self_type=None):
        argnames = []
        arrows = []
        defaults = {}
        signature = inspect.signature(fn)

        for k, v in signature.parameters.items():
            annotation = v.annotation
            if annotation is inspect._empty:
                annotation = self_type
            assert isinstance(annotation, Type), f"{k} {annotation} {type(annotation)}"
            arrows.append(annotation)
            argnames.append(k)
            if v.default is not inspect._empty:
                defaults[k] = v.default

        if check_return:
            annotation = signature.return_annotation
            if annotation is inspect._empty:
                annotation = self_type
            assert isinstance(annotation, Type), f"{annotation}"
            arrows.append(annotation)
        return arrows, argnames, defaults

    def build_data_type(self, cls):
        data_type =  DataType(cls)
        self.types[str(data_type)] = data_type

        for k in cls.__dict__:
            v = cls.__dict__[k]
            if inspect.isfunction(v) and hasattr(v, '__is_attr__'):
                self.as_attr(data_type)(v)

        return data_type

    def get_method_prefix(self, tp):
        if not hasattr(tp, 'BASETYPE'):
            return '__' + str(tp) + '__'
        else:
            return '__' + tp.BASETYPE + '__'

    def as_attr(self, tp):
        if inspect.isfunction(tp):
            tp.__is_attr__ = True
            return tp
        else:
            def wrapper(fn):
                return self.as_primitive(fn, prefix=self.get_method_prefix(tp), self_type=tp)
            return wrapper

    def as_primitive(self, fn, prefix=None, self_type=None):
        arrows, argnames, defaults = self.get_fn_tp(fn, self_type=self_type)

        name = fn.__name__
        if prefix is not None:
            name = prefix + name
        op = Operator(name, Arrow(*arrows, n_defaults=len(defaults)), fn, argnames=argnames, default_parameters=defaults)
        self.register_operator(op)
        return op

    def parse(self, fn, scope=None, context=None):
        if scope is None:
            if isinstance(fn, str):
                scope = self.types
            else:
                import inspect
                frame = inspect.stack()[1][0]
                if frame.f_code.co_name == 'as_func':
                    frame = frame.f_back
                scope = frame.f_locals

        from .program import Program
        prog, tp, argnames, defaults, name = Program.from_ast(fn, self, scope, context=context)

        return Operator(
            name,
            Arrow(*tp, prog.return_tp, n_defaults=len(defaults)),
            value=prog,
            argnames=argnames,
            default_parameters=defaults,
        )

    def as_func(self, fn):
        fn = self.parse(fn)
        self.register_operator(fn)
        return fn

    def __str__(self):
        outs = [str(self.operators[i]) for i in self.operators]
        #outs += [i+' = ' + str(self.funcs[i]) for i in self.funcs]
        return '\n'.join(outs)

        
    def parse_constant(self, value):
        from .program import Constant
        if not isinstance(value, str):
            tp = type(value)
            assert tp in self.constant_type, f"constant type {type(tp)} of {tp} is not registered."
            tp = self.constant_type[tp]
            return Constant(value, tp, self.constant_type_constructor[str(tp)])
        else:
            for enum_type in self.enum_types:
                if enum_type.instance(value):
                    return Constant(value, enum_type)
            #raise NotImplementedError(f"{value} does not belongs to any enum_types; We current only have {self.enum_types}")
            return Constant(value, self.str)