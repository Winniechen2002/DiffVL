from llm.tiny import *

pcd = ArrayType(10, 3)
pcd2 = ArrayType(10, 3)

@dsl.parse
def work(a: pcd, b: pcd2):
    return as_tensor([0., 0., 0.]).get(0)

    
@dsl.as_primitive
def work2(*b: VariableArgs("\'*")) -> tA:
    return b[2][0] + 1

@dsl.as_func
def add(a:myint, b:myint) -> myint:
    return a + b

vargs = VariableArgs(myint)
@dsl.parse
def work3(x: myint, *b: vargs):
    c = b
    p = work2([1], [2.], [3])
    return p + car(c) + x

print(work3.pretty_print())
print(work3(3, 4), '== 11')