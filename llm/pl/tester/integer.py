from ..dsl import DSL
from ..types import Integer, List, Bool
from ..lisp import  concat, cons, car
from ..functool import lmap, lfilter

int_dsl = DSL()
myint = int_dsl.int
mybool = int_dsl.bool


@int_dsl.as_primitive
def add(a: myint, b: myint) -> myint:
    return a + b

@int_dsl.as_primitive
def delete(a: myint, b: myint) -> myint:
    return a - b

@int_dsl.as_func
def incr(a: myint) -> myint:
    b = add(a, 3)
    return add(b, 5)
    

@int_dsl.as_func
def decr(a: myint) -> myint:
    return delete(a, 1)


@int_dsl.as_func
def addd(a: myint, b: myint=1) -> myint:
    return add(a, b)


#TODO: bool filter .. and how to turn it into a better filters.
@int_dsl.as_primitive
def greater(a: myint, b: myint) -> myint:
    return a - b

@int_dsl.as_primitive
def greater_two(a: myint) -> mybool:
    return a > 1

@int_dsl.parse
def test(a: myint):
    _0 = lmap(lmap([1,2,3], decr), decr)
    return car(cons(a, lfilter(_0, greater_two)))

@int_dsl.as_func
def start():
    a = 1

@int_dsl.parse
def make_list():
    x = []
    def x2():
        y = 8
        return add(y, 5)
    start()
    b = concat(cons(1, x), [1, 2, 3, x2()])
    return b

print(make_list.pretty_print())
print(make_list())

@int_dsl.as_func
def add3(x: myint):
    return add(x, 2)
print(add3.pretty_print())
print(add3(3))


"""
print(test.pretty_print())


@int_dsl.parse
def add(a: myint, b: myint):
    c = add(a, 2)
    return add(c, b)


@int_dsl.as_attr('int')
def less(self: myint) -> mybool:
    return self < 0

@int_dsl.parse
def annd(a: myint):
    return int.less
    #return not False

@int_dsl.parse
def test(L:List(myint)):
    b = not greater_two
    return lfilter(L, b)

print(int_dsl.default_executor.eval(test,[-10, -9, 1,2,3]))


print('map', int_dsl.operators['lmap'].tp)
print('decr', decr.tp)
print("map([1,2,3], decr)   List(int(-100, 100)) -> (int(-100, 100) -> int(-100, 100)) -> int(-100, 100)")

print(int_dsl.parse(test.pretty_print()).pretty_print())
exit(0)


@int_dsl.as_func
def addall(a: List(myint), b: myint):
    c = concat(a, [1,2,3])
    # b = incr(a)
    return lmap(c, incr)

#int_dsl.add_prog(lambda (a:myint) :add(1, 2))
print(str(int_dsl))
"""

@int_dsl.as_func
def work():
    a = 5
    def add2(x: myint, y: myint=a):
        return x + y

    y = lambda : 3
    return add2(1, add2(1))

print(work.pretty_print())
print(work())