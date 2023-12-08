import typing
from llm.pl import DSL
from llm.pl.types import Type
from llm.pl.program import Program, FuncCall, MakeLisp, Constant, Operator, ListExpr, Name, NewVar
from llm.pl.operator import match_input


class Executor:

    def __init__(self, dsl: DSL):
        self.dsl = dsl
        self.op_bindings = {}
        self.type_bindings = {}
        self.constructor = {}
        self.clear_trace()

    def clear_trace(self):
        from .functool import clear_op
        clear_op()
        self.trace = []
        self.global_context = {} # register global variables here
        return self

    def register_type(self, type: Type, diff_type: Type):
        self.type_bindings[str(type)] = diff_type

    def register_op(self, op: Operator, diff_op: typing.Callable):
        self.op_bindings[op] = diff_op

    def register_constructor(self, tp, value):
        self.constructor[str(tp)] = value

    def _execute(self, program: Program, context):
        self.trace.append(program)

        value = None
        if isinstance(program, FuncCall):
            args = [self._execute(i, context = context) if i is not None else None for i in program.args]
            op = program.op
            if isinstance(op, Program): 
                op = self._execute(op, context)
            else:
                op = self.dsl.operators[op]
            
            if len(program.queries) == 0:
                value = self.eval(op, *args)
            else:
                def work(*input):
                    outputs = []
                    idx = 0
                    for i in args:
                        if i is None:
                            outputs.append(input[idx])
                            idx += 1
                        else:
                            outputs.append(i)
                    assert idx == len(input)
                    return self.eval(op, *outputs)
                value = Operator(None, program.return_tp, work,
                                 argnames=['i'] * len(program.queries), code=None)
        
        elif isinstance(program, Name):
            value = context[program.token]
        
        elif isinstance(program, NewVar):
            context[program.left] = self._execute(program.right, context)
            value = context[program.left]

        elif isinstance(program, Constant):
            value = program.token
            
            constructor = program.constructor
            tp = self.type_bindings.get(str(program.return_tp), program.return_tp)
            constructor = self.constructor.get(str(tp), constructor)
            if constructor is not None:
                value = constructor(program.token)

        elif isinstance(program, MakeLisp):
            value = [self._execute(i, context) for i in program.elements]

        elif isinstance(program, ListExpr):
            for i in program.exprs:
                self._execute(i, context=context)
            if program.return_prog is not None:
                value = self._execute(program.return_prog, context=context)
            else:
                value = None

        else:
            raise NotImplementedError(f"{type(program)}")

        diff_type = self.type_bindings.get(str(program.return_tp), program.return_tp)
        assert diff_type.instance(value), "the type of {} is not {} but {} in:\n{}".format(value, diff_type, type(value), '\n'.join(program._pretty_print()))

        if isinstance(value, Operator) and value.polymorphism:
            value.tp = diff_type

        if isinstance(value, Operator):
            if value.closure is not None and not hasattr(value, 'context'):
                value = value.clone() # copy the operator ..
                value.context = {}
                for k, v in value.closure.items():
                    value.context[v.token] = self._execute(v, context=context)

                for k, v in value.default_parameters.items():
                    value.context[k] = self._execute(v, context=context)

        self.trace.pop(-1)
        return value

        
    def eval(self, op: Operator, *args, return_context=False):
        from . import functool
        functool.EXECUTOR = self

        min_args =  max_args = len(op.argnames)
        if sum([int(i.match_many()) for i in op.tp.arguments]) > 0:
            min_args -= 1 
            max_args = 1e9
        #if op.default_parameters
        min_args -= len(op.default_parameters)

        if len(args) >= min_args and len(args) <= max_args:
            pass
        else:
            raise Exception(f"missing input to the function {op.token}, expected {(min_args, max_args)} but got {len(args)}")

        inps = args
        self.trace.append([op, args])


        if isinstance(op.value, Program):
            #print(op.argnames)
            #exit(0)
            context = {}
            inps = list(inps)
            for k in op.argnames:
                if not k.startswith('*'):
                    if len(inps) >= 1:
                        context[k] = inps.pop(0)
                    else:
                        assert k in op.default_parameters
                        # context[k] = self._execute(op.default_parameters[k], context)
                else:
                    context[k[1:]] = list(inps)
                    break
            if op.closure is not None:
                for k, v in op.context.items():
                    if k not in context: context[k] = v
            out = self._execute(op.value, context)
        else:
            #TODO: override
            value = None
            if op.token in ['Add', 'Lt', 'Gt', 'Eq', 'And', 'Or', 'Not']:
                for i in self.dsl.types.values():
                    if i.instance(args[0]):
                        op_name = self.dsl.get_method_prefix(i) + op.token 
                        if op_name in self.op_bindings:
                            value = self.op_bindings[op_name]
                            break

            if value is None:
                value = self.op_bindings.get(op.token, op)

            fn = value.execute if isinstance(value, Operator) else value

            try:
                out = fn(*args)
            except Exception as e:
                print("# ----------------------------------------------------- #")
                for i in self.trace:
                    print(i)
                raise e

            if isinstance(out, Operator) and out.polymorphism:
                out.tp = value.tp.out


        self.trace.pop(-1)
        if len(self.trace) == 0:
            self.clear_trace()
        if return_context:
            return out, context
        return out
