from .types import Type, List, Bool, Arrow

bool = Bool()
tpA = Type('\'A')
tpB = Type('\'B')
list_A = List(tpA)
list_B = List(tpB)

transform = Arrow(tpA, tpB)
cond = Arrow(tpA, bool)

get_attr = Arrow(tpB, tpA)


def cons(a: tpA, b: list_A) -> list_A:
    return [a] + b

def concat(a: list_A, b: list_A) -> list_A:
    assert isinstance(a, list)
    return a + b

def car(a: list_A) -> tpA:
    assert isinstance(a, list)
    return a[0]

def last(a: list_A) -> tpA:
    assert isinstance(a, list)
    return a[-1]

def cdr(a: list_A) -> list_A:
    return a[1:]
    


__all__ = [cons, concat, car, cdr, last]