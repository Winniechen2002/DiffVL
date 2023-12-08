from llm.pl.tester.integer import test, int_dsl
from llm.pl import none
from llm.learn.polish import polish


print(polish(test.value))

@int_dsl.as_primitive
def val() -> none:
    pass

@int_dsl.parse
def test2():
    val()
    a = 1
    b = 2
    c =  a + b
    return c - (a-c)

    
print(polish(test2.value, lineid=True))
