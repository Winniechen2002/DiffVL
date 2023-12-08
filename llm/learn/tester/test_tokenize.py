import os
from llm.pl.tester.integer import test, int_dsl
from llm.pl import none
from llm.learn.serialize import extract_code


# print(serialize(test))

@int_dsl.as_primitive
def val() -> none:
    pass

@int_dsl.parse
def test2():
    a = 1
    b = 2
    c =  a + b
    return c - (a-c)

    

prog = extract_code(test2)
for i in range(prog.timestep):
    os.system('clear')
    print(prog.print(i))
    input()

# print(prog.serialize())