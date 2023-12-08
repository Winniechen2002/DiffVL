from typing import Any, TypeVar, Callable
import inspect
import re

def match(key, input, start='='):
    pattern = f"{start} *{key} *[\\(\\[\\.]"
    matches = re.findall(pattern, input)
    return len(matches) > 0


def find_caller(key, trace):
    """
    look for the caller of the function with key in the stack.
    return the left value of the assignment and the trace of the caller.
    """
    key = key or 'OPERATORS'
    for frame in inspect.stack():
        frame_4 = frame[4]
        if frame_4 is None:
            break
        assert frame_4 is not None
        input = str(frame_4[0])
        equal = match(key, input, '=') 
        ret = match(key, input, 'return')
        if equal or ret:
            frame = frame[0]
            code_context = inspect.getframeinfo(frame).code_context
            assert code_context is not None
            code = code_context[0].strip()
            if ret:
                name = 'return'
            else:
                name = code.split("=")[0].strip()
            trace = ''
            return name, trace
    return "Unknown", "Unknown trace"


class Trace:
    def __init__(self, f_name, assigned_name, f, lang: str|None, default_weight: float|None=None, *inputs) -> None:
        self.f_name = f_name
        self.assigned_name = assigned_name
        self.f = f
        self.inputs =  inputs
        self.lang = lang
        self.default_weight = default_weight

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args, **kwds)

    def __str__(self) -> str:
        if self.assigned_name != "Unknown":
            return self.assigned_name
        return f"{self.f_name}({', '.join([str(i) for i in self.inputs])})"

    def synchronize(self):
        elements = [i.synchronize() if isinstance(i, Trace) else str(i) for i in self.inputs]
        if self.lang is not None:
            return self.lang.format(*elements)
        return f"{self.f_name}({', '.join(elements)})"


#_T = TypeVar('_T', bound=Callable)
def as_code(func, lang: str|None = None, default_weight: float|None = None):
    """
    wrap the returned function with a tracer class
    the goal is to trace the function call so that we can print it out.
    """
    f_name = func.__name__
    def make_tracer(*args):
        assigned_name = find_caller(f_name, '')[0]
        f = func(*args)
        tracer = Trace(f_name, assigned_name, f, lang, default_weight, *args)
        return tracer
    return make_tracer