import numpy as np
from functools import lru_cache


class ParseFailure(Exception):
    """Objects of type Program should throw this exception in their constructor if their arguments are bad"""

class TypeInferenceFailure(Exception):
    """Error in type inference"""


class Type:
    def __init__(self, type_name):
        self._type_name = type_name

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self._type_name

    def instance(self, x):
        return True

    def match_many(self):
        return False

    @property
    def children(self):
        return ()

    @property
    def polymorphism(self):
        #TODO: accelerate this ..
        if len(self.children) > 0:
            for i in self.children:
                if i.polymorphism:
                    return True
            return False
        else:
            return hasattr(self, '_type_name')


class VariableArgs(Type):
    def __init__(self, tp):
        if isinstance(tp, str):
            self._type_name = tp
        else:
            assert isinstance(tp, Type)
            self.base_type = tp

    def match_many(self):
        return True
    
    def __str__(self):
        return "VariableArgs(" + (self._type_name if hasattr(self, '_type_name') else str(self.base_type)) + ")"

    @property
    def children(self):
        if hasattr(self, '_type_name'): return ()
        return (self.base_type,)


class NoneType(Type):
    def __init__(self):
        pass

    def __str__(self):
        return 'None'
    
    def isinstance(self, x):
        return x is None
none = NoneType()


class DataType(Type):
    # data_cls could be anything ..
    def __init__(self, data_cls, type_name=None):
        self.data_cls = data_cls
        self.type_name = type_name or self.data_cls.__name__

    def __str__(self):
        #return self.data_cls.__name__
        return self.type_name

    def instance(self, x):
        return isinstance(x, self.data_cls)

    @property
    def children(self):
        return ()


class Arrow(Type):
    def __init__(self, *args, n_defaults=0):
        assert len(args) >= 1
        for a in args:
            assert isinstance(a, Type), f"{a} is not a Type .."
        self.out = args[-1]
        self.arguments = args[:-1]
        self.n_defaults = n_defaults

    def __str__(self):
        def decorate(t):
            if isinstance(t, Arrow):
                return f"({str(t)})"
            return str(t)
        c = self.children
        if len(c) > 1:
            lists = [decorate(t) for t in self.children]
            if self.n_defaults >= 0:
                lists[-self.n_defaults -2] =  lists[-self.n_defaults - 2] + ' ['
                lists[-2] = lists[-2] + ']'
            return " -> ".join(lists)
        else:
            return "-> " + decorate(c[0])

    def instance(self, args):
        #NOTE: to check if a operator belongs to the TYPE, we can only check its tp.
        #   - we do not check the input to the function.. it should be checked by the APPLY program.
        from .operator import Operator
        assert isinstance(args, Operator), f"{args} {type(args)}"
        assert len(args.default_parameters) == self.n_defaults
        A = self.children 
        B = args.tp.children 
        assert len(A) == len(B)
        for a, b in zip(A, B):
            if a != b:
                return False
        return True

    @property
    def children(self):
        return list(self.arguments) + [self.out]

    def is_transform(self):
        assert len(self.arguments) == 1

    def is_cond(self):
        return self.is_transform and self.out == Bool()


class String(Type):
    def __init__(self):
        pass

    def __str__(self):
        return f"str"

    def instance(self, x):
        return isinstance(x, str)


class Integer(Type):
    def __init__(self, lower, upper):
        assert type(lower) is int
        assert type(upper) is int
        self.upper = upper
        self.lower = lower

    def __str__(self):
        return f"int"

    def instance(self, x):
        return isinstance(x, int) and x >= self.lower and x <= self.upper

    @property
    def N(self):
        return self.upper - self.lower + 1

class Bool(Type):
    def __init__(self):
        pass

    def instance(self, x):
        return isinstance(x, bool)

    def __str__(self):
        return "bool"

class List(Type):
    def __init__(self, base_type: Type):
        self.base_type = base_type

    def __str__(self):
        return f"List({self.base_type})"
    
    def instance(self, x):
        if not isinstance(x, list):
            return False
        for i in x:
            if not self.base_type.instance(i):
                return False
        return True

    @property
    def children(self):
        return [self.base_type]

class Tuple(Type):
    def __init__(self, *elements):
        self.elements = elements
    
    def __str__(self):
        return f"Tuple({', '.join(str(e) for e in self.elements)})"

    def instance(self, x):
        raise NotImplementedError("not support tuple type now")

    @property
    def children(self):
        return self.elements

        
class EnumType(Type):
    def __init__(self, tokens, name):
        self.tokens = tokens
        self.name = name

    def __str__(self):
        return self.name

    def instance(self, x):
        if not isinstance(x, str):
            return False
        return x in self.tokens
    
    @property
    def children(self):
        return []