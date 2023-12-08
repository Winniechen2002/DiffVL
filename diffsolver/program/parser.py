import termcolor
import numpy as np
from typing import Callable, TypeVar, ParamSpec, Optional, MutableMapping, Any, Generic
from .constant import *

import ast

def ensure_out_variable(code):
    # Parse the code into an AST
    tree = ast.parse(code)

    # Check if the last node is an expression
    if isinstance(tree.body[-1], ast.Expr):
        # If it's an expression, wrap it in an assignment to __out__
        last_expr = tree.body[-1].value
        assign_out = ast.Assign(targets=[ast.Name(id='__out__', ctx=ast.Store())], value=last_expr)
        tree.body[-1] = assign_out

        assign_out.lineno = 1
        assign_out.col_offset = 0
        assign_out.targets[0].lineno = 1
        assign_out.targets[0].col_offset = 0


        # Generate the modified code from the new AST
        modified_code = ast.unparse(tree)
        return modified_code
    else:
        return code

TIN = ParamSpec("TIN")
TOUT = TypeVar("TOUT")

class Parser(Generic[TIN, TOUT]):
    def __init__(self, **kwargs) -> None:
        self._CONSTANT = {}
        self._LIBRARIES: MutableMapping[str, Callable[..., Callable[TIN, Any]]] = {**kwargs}
        # self._LIBRARIES = _LIBRARIES

    def parse(self, _prog: str) -> Callable[TIN, TOUT]:
        program = _prog.strip()
        __locals = {**self._CONSTANT, **self._LIBRARIES}
        assert len(__locals) == len(self._CONSTANT) + len(self._LIBRARIES), "Duplicate keys"

        program = ensure_out_variable(program)

        try:
            exec(program, None, __locals)
        except Exception as e:
            print('Error in program:\n')
            print(termcolor.colored(_prog, 'red'))
            raise e
        return __locals['__out__']

    def register(self, _name: Optional[str] = None):
        P = ParamSpec("P")
        T2 = TypeVar("T2")
        def wrapper(fn: Callable[P, Callable[TIN, T2]]) -> Callable[P, Callable[TIN, T2]]:
            name = _name or fn.__name__
            assert name not in self._LIBRARIES, 'Library %s already registered' % name
            self._LIBRARIES[name] = fn
            return fn
        return wrapper

    def register_constant(self, _name: str):
        def wrapper(fn: Callable[TIN, TOUT]) -> Callable[TIN, TOUT]:
            self._CONSTANT[_name] = fn
            return fn
        return wrapper