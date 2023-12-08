# assign an position to each new variable and its definition so that we can do auto regression ..
# https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/gpt2/modeling_gpt2.py

import typing
from llm.pl import Program
from llm.pl.program import ListExpr, MakeLisp, Constant, Input, FuncCall, Operator
from llm.pl.types import Type


special_tp = Type('SPECIAL')


class Token:
    def __init__(self, token, lineno, col_offset, types, timestep):
        assert lineno is not None
        self.token = token
        self.lineno = lineno
        self.col_offset = col_offset
        self.types = types
        self.timestep = timestep
    
    def __str__(self) -> str:
        # return f"{self.token}@{self.lineno}_{self.col_offset}_{self.timestep}"
        return self.token

class Line:
    def __init__(self, lineno, code: "Code") -> None:
        self.lineno = lineno
        self.code = code
        self.tokens: typing.List[Token] = []
        self.start_timestep = None

    def append(self, token, tp):
        offset = len(self.tokens)
        self.code.last_tuple = (token, 'L' + str(self.lineno), 'C' + str(offset))
        self.tokens.append(
            Token(token, self.lineno, offset, tp, self.code.timestep))
        if self.start_timestep is None:
            self.start_timestep = self.code.timestep
        self.code.timestep += 1

    def print(self, timestep):
        if self.start_timestep > timestep:
            return None
        x = " ".join([str(i) for i in self.tokens if i.timestep <= timestep])
        if self.tokens[-1].timestep > timestep:
            x = x + " ?"
        return "$"+ str(self.lineno) + " " + x

    def __str__(self):
        return "$"+ str(self.lineno) + " " + " ".join([str(i) for i in self.tokens])
        
        
class Code: 
    # This is an intermediate structure ..
    def __init__(self) -> None:
        self.timestep = 0
        self.lineno = 0
        self.lines: typing.List[Line] = []

        self.last_tuple = None

    def new_line(self):
        t = self.lineno
        self.lineno += 1
        line = Line(t, self)
        self.lines.append(line)
        return "$" + str(t), line

    def __str__(self) -> str:
        return "\n".join([str(i) for i in self.lines])

    def print(self, timestep):
        outs = []
        for i in self.lines:
            x = i.print(timestep)
            if x is not None:
                outs.append(x)
        return "\n".join(outs)

    def serialize(self):
        inputs_ids = [None] * self.timestep
        linenos = [None] * self.timestep
        col_offsets = [None] * self.timestep
        for line in self.lines:
            for token in line.tokens:
                t = token.timestep
                assert inputs_ids[t] is None
                inputs_ids[t] = token.token
                linenos[t] = 'L' + str(token.lineno)
                col_offsets[t] = 'C' + str(token.col_offset)
        return {
            'input_ids': inputs_ids,
            'linenos': linenos,
            'col_offsets': col_offsets
        }


def deserialize(tokens, code=None, execute=False):
    assert not execute, "we haven't implemented REPL like function yet."
    if isinstance(tokens, str):
        tokens = [tokens]

    if code is None:
        code = Code()
    
    for i in tokens:
        last_line = None
        for j in code.lines:
            if j.tokens[-1].token != '<CR>':
                last_line = j

        if i in ['=', 'lambda', 'ret']:
            _, line = code.new_line()
            line.append(i, None)
        else:
            if i == 'EOF':
                code.last_tuple = ("EOF", "EOF", "EOF")
                break
            assert last_line is not None, f"{code}"
            line = last_line
            line.append(i, None)
    return code


def extract_code(op: Operator):
    assert isinstance(op, Operator)
    code = Code()
    context = {}

    def _visit(program):
        if program in context:
            return context[program]
        if isinstance(program, Input):
            return f"i{op.name2id[program.token]}"
            
        elif isinstance(program, Constant):
            if isinstance(program.token, Operator):
                val = program.token.token
            else:
                val = str(program.token)
            return val

        elif isinstance(program, ListExpr):
            for i in program.exprs:
                _visit(i)
            return _visit(program.return_prog)

        var, line = code.new_line()
        context[program] = var


        line.append("=", special_tp)
        if isinstance(program, MakeLisp):
            line.append('list', special_tp)
            for i in program.elements:
                line.append(_visit(i), i.return_tp)

        elif isinstance(program, FuncCall):
            assert len(program.kwargs) == 0
            op_token = program.op
            if isinstance(program.op, Program):
                op_token = _visit(program.op)

            line.append(op_token, special_tp)
            for i in program.args:
                line.append(_visit(i), i.return_tp)

        else:
            raise NotImplementedError

        line.append("<CR>", special_tp)
        return var

    #head = "lambda " + ' '.join([f'i{i}' for i, _ in enumerate(op.argnames)])
    _, line = code.new_line()
    line.append("lambda", special_tp)
    for i in range(len(op.argnames)):
        line.append("i" + str(i), op.tp.arguments[i])

    var = _visit(op.value)
    _, line = code.new_line()
    line.append('ret', special_tp)
    line.append(var, op.value.return_tp)
    line.append('<CR>', special_tp)
    return code
