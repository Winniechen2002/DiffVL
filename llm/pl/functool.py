from .operator import Operator
from .types import Type, Bool, Arrow, List
import inspect
ID = 0
EXECUTOR = None

def get_ext():
    return EXECUTOR


tpA = Type('\'A')
tpB = Type('\'B')
tpC = Type('\'C')
mybool = Bool()

attr = Arrow(tpA, tpB)
relation = Arrow(tpB, tpC, tpA)
cond = Arrow(tpA, mybool)


list_A = List(tpA)
list_B = List(tpB)


def clear_op():
    global ID
    ID = 0


def new_op(value, **kwargs):
    global ID
    ID += 1

    op = Operator(f'new_op{ID}', Type('\'A'), value, list(inspect.signature(value).parameters), **kwargs)
    return op


def compose(op: Arrow(tpB, tpC), right: Arrow(tpC, tpA)) -> Arrow(tpB, tpA):
    return new_op(lambda x: EXECUTOR.eval(right, EXECUTOR.eval(op, x)))

def apply(op: Arrow(tpB, tpC), right: tpB) -> tpC:
    return new_op(lambda x: EXECUTOR.eval(right, EXECUTOR.eval(op, x)))

def lcurry(op: relation, left: tpB) -> Arrow(tpC, tpA):
    return new_op(lambda x: EXECUTOR.eval(op, left, x))


def rcurry(op: relation, right: tpC) -> Arrow(tpB, tpA):
    return new_op(lambda x: EXECUTOR.eval(op, x, right))


def lmap(a: list_A, b: attr) -> list_B:
    assert b.tp.is_transform
    out = [EXECUTOR.eval(b, i) for i in a]
    return out

def lreduce(a: List(tpA), f: Arrow(tpA, tpA, tpA)) -> tpA:
    out = a[0]
    for i in a[1:]:
        out = EXECUTOR.eval(f, out, i)
    return out


def lfilter(a: list_A, b: cond) -> list_A:
    assert b.tp.is_cond
    return [i for i in a if EXECUTOR.eval(b, i)]


def lall(a: list_A, b: cond) -> mybool:
    for i in a:
        if not EXECUTOR.eval(b, i):
            return False
    return True


def lexists(a: list_A, b: cond) -> mybool:
    for i in a:
        if EXECUTOR.eval(b, i):
            return True
    return False


__all__ = [compose, lcurry, rcurry, lmap, lfilter, lall, lexists, apply, lreduce]