# Linear temporal logic
from llm.pl import Executor
from llm.pl.types import Type,  Arrow, List, Integer
from llm.tiny import dsl
from llm.pl.functool import new_op


tpA = Type('\'A')
tpB = Type('\'B')


cond = Arrow(dsl.int, dsl.bool)
list_cond = Arrow(List(dsl.int), dsl.bool)


def check(x):
    assert len(x) == 2
    assert x[1] > x[0] >= 0


def TStart(A: cond) -> list_cond:
    def f(x):
        check(x)
        return A(x[0], use_cache=True)
    return new_op(f, use_cache=True)


def TLast(A: cond) -> list_cond:
    def f(x):
        check(x)
        return A(x[-1]-1, use_cache=True)
    return new_op(f, use_cache=True)


def TAll(A: cond) -> list_cond:
    def f(x):
        check(x)
        out = None
        for i in range(x[0], x[1]):
            value = A(i, use_cache=True)
            if value is False:
                return False
            if out is None:
                out = value
            else:
                out = out & value
        return out
    return new_op(f, use_cache=True)


def or_list(val):
    outputs = []
    for i in val:
        if isinstance(i, bool):
            if i:
                return True
        outputs.append(i)
    if len(outputs) == 0:
        x = dsl.get_executor().constructor.get('LogProb', bool)(False)
        return x
    if isinstance(i, bool):
        return False

    from .logprob import LogProb
    out = sum([(~i).value for i in outputs]) # all does not exists..
    return ~LogProb(out)
    # return LogProb(max([i.value for i in outputs]))


def TExists(A: cond) -> list_cond:
    def f(x):
        check(x)
        out = None
        for i in range(x[0], x[1]):
            value = A(i, use_cache=True)
            if value is True:
                return True
            if out is None:
                out = value
            else:
                out = out | value
                raise NotImplementedError("Rewrite this ..")
        return out
    return new_op(f, use_cache=True)


def Then(A: list_cond, B: list_cond) -> list_cond:
    # weak until
    # find a subsequence that, A hold; then B hold for the remaining ..
    def f(x):
        check(x)
        def gen():
            for i in range(x[0]+1, x[1]-1):
                a = A([x[0], i], use_cache=True)
                b = B([i, x[1]], use_cache=True)
                value = a & b
                yield value
        return or_list(gen())
    return new_op(f, use_cache=True)


def bind_LTL(dsl):
    for func in [TStart, TLast, TAll, TExists, Then]:
        dsl.as_primitive(func)