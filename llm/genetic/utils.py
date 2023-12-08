import os
from llm.pl import *


FILEPATH = os.path.dirname(os.path.abspath(__file__))

def get(NAME, default):
    if NAME in os.environ:
        PATH = os.environ[NAME]
    else:
        PATH = os.path.join(FILEPATH, 'dataset', default)
    return PATH

GENOMEPATH = get("GENETIC_GENOMEPATH", 'genome')
SOLPATH = get("GENETIC_SOLPATH", 'data')

import tabulate

def print_table(outputs, fn = lambda x: float(x['value']), print_mode=False):
    cols = set()
    rows = set()
    for val in outputs:
        rows.add(val['code'])

        cols.add(val['start'])
        cols.add(val['end'])

    row_id = {}
    col_id = {}
    rows = sorted(rows)
    cols = sorted(cols)

    for idx, i in enumerate(rows):
        row_id[i] = idx
    for idx, i in enumerate(cols):
        col_id[i] = idx + 1

    table = [['' for i in range(len(cols))] for j in range(len(rows))]
    for val in outputs:
        v = fn(val)

        s = col_id[val['start']]
        t = col_id[val['end']]
        #if val['mode'] == 'st':
        #    mid = t - 1
        #else:
        #    mid = (s + t - 1) // 2
        mid = s

        r = row_id[val['code']]

        def add(x):
            if len(x) == 0:
                return '|'
            if x[0] == '|':
                return x
            return '|' + x

        table[r][s] = add(table[r][s])
        if t < len(table[r]):
            table[r][t] = add(table[r][t])

        if print_mode: 
            mode = val['mode']
            stage_id = val['stage']
            v = f'{mode}: {v} @ s{stage_id} '
        table[r][mid] += v

    for i in rows:
        table[row_id[i]][0] = i
    header = [f'{cols[i]} - {cols[i+1]}'for i in range(len(cols)-1)]
    return tabulate.tabulate(table, headers=['code'] + header)


def serialize(dsl, prog: Program, context, mode='polish'):
    def _visit(program):
        if program in context:
            return context[program]
        
        if isinstance(program, Input):
            raise NotImplementedError("do not support additional input")

        elif isinstance(program, ListExpr):
            raise NotImplementedError("do not support list expr now ..")
            
        elif isinstance(program, Constant):
            if isinstance(program.token, Operator):
                val = program.token.token
            else:
                val = str(program.token)
            return val

        outs = []
        if isinstance(program, MakeLisp):
            outs.append("[")
            for i in program.elements:
                outs.append(_visit(i))
            outs.append("]")
        elif isinstance(program, FuncCall):
            assert len(program.kwargs) == 0
            outs = []
            op_token = program.op
            if isinstance(program.op, Program):
                op_token = _visit(program.op)
            outs.append(op_token)
            for i in program.args:
                outs.append(_visit(i))
        else:
            raise NotImplementedError
        if mode == 'polish':
            return " ".join(outs)
        else:
            if outs[0] == '[':
                return outs[0] + ','.join(outs[1:-1]) + outs[-1]
            else:
                return outs[0] + '(' + ','.join(outs[1:]) +')'

    return _visit(prog)
    

def deserialize(dsl: DSL, chars, context):
    if chars[0] in context:
        value = context[chars[0]]
    elif chars[0] in dsl.operators:
        op = dsl.operators[chars[0]]
        value = Constant(op, op.tp)
    elif chars[0] == '[' or chars[0] == ']':
        value = chars[0]
    else:
        value = dsl.parse_constant(eval(chars[0]))

    chars = chars[1:]
    if isinstance(value, Constant) and isinstance(value.token, Operator):
        args = []
        op = value.token
        for i in op.tp.arguments:
            next_value, chars = deserialize(dsl, chars, context)
            args.append(next_value)
        return FuncCall(value, op.tp, op.argnames, args, {}, dsl), chars 

    elif value == '[':
        outs = []
        while True:
            next_value, chars = deserialize(dsl, chars, context)
            if next_value == ']':
                break
            outs.append(next_value)
        return MakeLisp(outs), chars
    else:
        return value, chars