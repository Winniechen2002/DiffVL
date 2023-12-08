
from ..dsl import DSL

dsl = DSL()
myint = dsl.int
mybool = dsl.bool


@dsl.as_func
def add(a: myint):
    def add2(b: myint):
        return b + a
    return add2


op = add(1)
print(add.pretty_print())
print(op(3))

@dsl.parse
def test():
    c = add(3)
    return c(2)

print(test.pretty_print())
print(test())