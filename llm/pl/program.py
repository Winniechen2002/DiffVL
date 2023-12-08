import  inspect
import ast

import copy
import typing
from .types import TypeInferenceFailure, List, Arrow, none, Type, VariableArgs
from .operator import Operator
from .dsl import DSL

def annotation2tp(annotation, type_scope_dict):
    if isinstance(annotation, ast.Constant):
        if annotation.value is None:
            return none
        else:
            assert annotation.value.startswith("\'"), f"{annotation.value}"
            return Type(annotation.value)

    if isinstance(annotation, ast.Call):
        if annotation.func.id == 'List':
            assert len(annotation.args) == 1
            return List(annotation2tp(annotation.args[0], type_scope_dict))
        elif annotation.func.id == 'Arrow':
            return Arrow(*[annotation2tp(i, type_scope_dict) for i in annotation.args])
        elif annotation.func.id == 'VariableArgs':
            return VariableArgs(annotation2tp(annotation.args[0], type_scope_dict))
    return type_scope_dict[annotation.id] if isinstance(annotation, ast.Name) else type_scope_dict[annotation.value]

COMMENT = True

def set_comment(comment):
    global COMMENT
    COMMENT = comment

class Analyzer(ast.NodeVisitor):
    # we only support: assign, function call and return ..
    def __init__(self, dsl: DSL, type_scope_dict, context=None):
        self.dsl = dsl
        self.type_scope_dict = type_scope_dict
        self.name = None
        self.tps = [] 
        self.argnames = []
        self.default_kws = {}
        self.context = {} if context is None else context
    
    def get(self, name):
        # reference counting
        if '__counter__' in self.context:
            self.context['__counter__'][name] = True
        return self.context[name]

    def define_new_func(self, node, is_lambda=False):
        # assert '__counter__' not in self.context
        p = copy.copy(self.context)
        p['__counter__'] = {}
        op = self.dsl.parse(ast.unparse(node), scope=self.type_scope_dict, context=p)
        op.closure = {i: Name(i, self.context[i], self.context[i].return_tp)
                        for i in p['__counter__'] if i in self.context} #copy.copy(self.context)

        # assert len(op.default_parameters) == 0, "default parameters are not supported yet"

        if '__counter__' in self.context:
            for i in p['__counter__']:
                self.context['__counter__'][i] = True

        op.code = ast.unparse(node)
        op.is_lambda = is_lambda
        return Constant(op, op.tp)

    def visit(self, node: ast.AST):
        if isinstance(node, ast.Module):
            assert len(node.body) == 1
            return self.visit(node.body[0])

        elif isinstance(node, ast.FunctionDef):
            if self.name is not None:
                self.context[node.name] = self.define_new_func(node)
                return NewVar(node.name, self.context[node.name], is_func=True)

            self.name = node.name


            assert node.args.kwarg is None, "kw for function define is not supported."

            arg_lists = []
            for i in node.args.args:
                assert i.annotation is not None, "input type must be annotated."
                tp = annotation2tp(i.annotation, self.type_scope_dict)

                self.argnames.append(i.arg)
                self.tps.append(tp)

                assert i.arg not in self.context, f"Duplicated name in func def {i.arg}."
                self.context[i.arg] = Input(i.arg, tp, False)

                arg_lists.append(i.arg)


            self.default_kws = {}
            L = len(node.args.defaults)
            for i in range(L):
                self.default_kws[arg_lists[-L+i]] = self.visit(node.args.defaults[i])

            
            varg = node.args.vararg
            if varg is not None:
                assert len(node.defaults) == 0, "default value with vararg is not supported."

                var_tp = annotation2tp(varg.annotation, self.type_scope_dict)
                self.context[varg.arg] = Input(varg.arg, List(var_tp.base_type), True)

                self.argnames.append("*" + varg.arg)
                self.tps.append(var_tp)


            body = []
            return_state = None
            for i in node.body:
                out = self.visit(i)
                if out is None:
                    assert isinstance(i, ast.FunctionDef), f"{type(i)}: {ast.unparse(i)} is not a function def."
                    continue

                if isinstance(i, ast.Return):
                    if node.returns is not None:
                        assert out.return_tp == annotation2tp(node.returns, self.type_scope_dict),\
                            f"Code \n{ast.unparse(node)}\n has a return type {out.return_tp}.."
                    assert out.return_tp != none, f"You can not return None .. in \n{ast.unparse(node)}"
                    return_state = out
                    break
                else:
                    body.append(out)
            return ListExpr(body, return_state)


        elif isinstance(node, ast.Assign):
            out = self.visit(node.value)
            assert out is not None, "assign must have an explicit return.."
            assert len(node.targets) == 1

            v = node.targets[0]
            assert isinstance(v, ast.Name)
            assert v.id not in self.context and v.id not in self.dsl.operators,\
                f"Can not assign to an existing name {v.id} in {ast.unparse(node)}."

            assignment = NewVar(v.id, out)
            self.context[v.id] = assignment
            return assignment

        elif isinstance(node, ast.Name):
            id = node.id
            if id in self.dsl.operators:
                op = self.dsl.operators[id]
                return Constant(op, op.tp)
            assert id in self.context, f"{id} is not in the current scope, {ast.unparse(node)}"
            out = self.get(id)
            return Name(id, out, out.return_tp)


        elif isinstance(node, ast.Attribute):
            #TODO: add attribute and method together ..
            assert node.value.id in self.dsl.types
            func_id = '__' + str(node.value.id) + '__' + node.attr
            op = self.dsl.operators[func_id]
            return Constant(op, op.tp)

        elif isinstance(node, ast.Call):
            args = []
            if isinstance(node.func, ast.Attribute):
                attr_node = node.func
                caller = self.visit(attr_node.value) #.id)
                func_id = self.dsl.get_method_prefix(caller.return_tp) + attr_node.attr
                args.append(caller)
            else:
                func_id = None
                caller = self.visit(node.func)

            if func_id in self.dsl.operators:
                op = self.dsl.operators[func_id]
                tp = op.tp
            else:
                assert func_id is None, f"{func_id} does not exists."
                func_id = op = caller
                tp = op.return_tp #NOTE: the function type must be determined when it is written. 

            args += list([self.visit(i) for i in node.args])
            assert len(node.keywords) == 0, "keyword arguments are not supported."

            return FuncCall(func_id, tp, args, self.dsl, node=node) 

        elif isinstance(node, ast.Lambda):
            assert len(node.args.args) == 0, "lambda function must have no arguments."
            op = self.define_new_func(node.body, is_lambda=True)
            return op

        elif isinstance(node, ast.Return):
            return self.visit(node.value)

        elif isinstance(node, ast.List):
            return MakeLisp([self.visit(i) for i in node.elts])

        elif isinstance(node, ast.Constant):
            if node.value is None:
                return None
            return self.dsl.parse_constant(node.value)

        elif isinstance(node, ast.BoolOp) or isinstance(node, ast.BinOp) or isinstance(node, ast.UnaryOp) or isinstance(node, ast.Compare):
            if isinstance(node, ast.Compare):
                assert len(node.comparators) == 1
                assert len(node.ops) == 1
                values = [node.left, node.comparators[0]]
                op = node.ops[0].__class__.__name__
            else:
                if isinstance(node, ast.BinOp):
                    values = [node.left, node.right]
                elif isinstance(node, ast.UnaryOp):
                    values = [node.operand]
                else:
                    values = node.values
                op = node.op.__class__.__name__ 
            
            args = [self.visit(i) for i in values]
            tp = self.dsl.operators[op].tp
            return FuncCall(op, tp, args, self.dsl, node=node) 

        elif isinstance(node, ast.Expr):
            return self.visit(node.value)
        else:
            raise NotImplementedError(f"{type(node)}, {ast.unparse(node)}")


class Program:
    lineno = None
    lineval = None

    def __init__(self):
        self._tp = None
        self.arg_tps = None

    @property
    def return_tp(self):
        assert self._tp is not None
        return self._tp

    @classmethod
    def from_ast(cls, fn, dsl, scope, context=None):
        # in this case, we do not need to check the program's return.. 
        # Instead we do the simple type inference.
        if not isinstance(fn, str):
            tp, argnames, defaults = dsl.get_fn_tp(fn, check_return=False) 
            source = inspect.getsource(fn)
        else:
            tp, argnames = None, None
            source = fn

        source = source.split('\n')
        while source[0][0] == '@':
            # assert source[0].endswith('.as_func') or source[0].endswith('parse') or source[0].endswith('variable')
            source = source[1:]
        source = '\n'.join(source)
        ast_tree = ast.parse(source)

        analyzer = Analyzer(dsl, type_scope_dict=scope, context=context)

        if tp is not None:
            for a, b in zip(tp, analyzer.tps):
                assert a == b
            for a, b in zip(argnames, analyzer.argnames):
                assert a == b
        tp, argnames = analyzer.tps, analyzer.argnames

        prog = analyzer.visit(ast_tree)
        return prog, tp, argnames, analyzer.default_kws, analyzer.name

    def _print_line(self, context):
        raise NotImplementedError

    def _pretty_print(self, context=None, is_expr=False, is_return=False):
        if context is None:
            context = {"__prog__": []}
        if self not in context:
            val = self._print_line(context)

            if is_return and val is not None:
                # outmost, if there is a return, we need to print the return.
                line = f"return {val}"
                context["__prog__"].append(line)
                self.lineno = len(context["__prog__"])
                self.lineval = line

            elif not isinstance(self, Constant) and not isinstance(self, Input) and not isinstance(self, Name):
                if is_expr and val is not None:
                    line = val
                    #T = max(80, len(line) + 10)
                    T = len(line) + 5
                    line = line + ' ' * (T - len(line))
                    line = line + '# ' + str(self.return_tp)

                    context["__prog__"].append(line)
                    self.lineno = len(context["__prog__"])
                    self.lineval = line

            context[self] = val

        return context["__prog__"]


class FuncCall(Program):
    def __init__(self, op, tp, args: typing.List["Program"], dsl: DSL, node=None):
        self.args = args
        self.dsl = dsl
        self.op = op

        arg_types = []
        self.queries = []
        for idx, i in enumerate(self.args):
            if i is None:
                # this is for making a partial function.
                arg_types.append(Type("\'"+str(idx)))
                self.queries.append(arg_types[-1])
            else:
                arg_types.append(i.return_tp)
        self.node = node

        if tp.n_defaults > 0:
            for i in tp.arguments:
                assert not i.match_many()
            n = len(tp.arguments)
            assert len(arg_types) <= n
            assert len(arg_types) + tp.n_defaults >= n
            if n > len(arg_types):
                arg_types = arg_types + list(tp.arguments[-(n-len(arg_types)):])
            assert len(arg_types) == n

        try:
            from .unification import type_inference
            self._tp = type_inference(tp, arg_types, queries=self.queries)
        except TypeInferenceFailure as e:
            raise TypeInferenceFailure(f"{e} in '{ast.unparse(node)}'")

        
    def _print_line(self, context):
        args = []

        for i in self.args:
            if i is not None:
                i._pretty_print(context)
                args.append(context[i])
            else:
                args.append("None")

        token = self.op
        if isinstance(self.op, Program):
            self.op._pretty_print(context)
            token = context[self.op]

        if token.startswith('__'): # classmethod
            token = token.split('__')
            return args[0] + '.' + token[-1] + '('  + ", ".join(args[1:])+ ')'
        else:
            return f"{token}(" + ", ".join(args) + ")"


class Input(Program):
    def __init__(self, token, tp, is_varg):
        #super().__init__(token, (), {}, dsl)
        self.token = token
        self._tp = tp
        self.is_varg = is_varg

    @property
    def return_tp(self):
        return self._tp

    def __str__(self):
        return f"{self.token}"

    def _print_line(self, context):
        context[self] = self.token
        return self.token


class MakeLisp(Program):
    def __init__(self, elements: typing.List["Program"]):
        self.elements = elements
        self._tp = None

    @property
    def return_tp(self):
        tps = [i.return_tp for i in self.elements]
        for i in tps[1:]:
            if i != tps[0]:
                raise TypeInferenceFailure("List must have the same types")

        if len(tps) == 0:
            from .types import Type
            return List(Type('\'A'))

        return List(tps[0])

    def __str__(self):
        return "[" + ", ".join(map(str,self.elements)) + "]"
    
    def _print_line(self, context):
        for e in self.elements:
            e._pretty_print(context)
        return "[" + ", ".join([context[i] for i in self.elements]) + "]"


class Constant(Program):
    def __init__(self, token, tp, constructor=None):
        self.token = token
        self._tp = tp
        self.show = token
        self.constructor = constructor

    def __str__(self):
        out = str(self.token)
        return out

    def _print_line(self, context):
        val = str(self.token)
        if isinstance(self.token, Operator):
            #TODO: unify the str representation ..
            val = self.token.token
            if val is not None:
                if '__' in val:
                    val = '.'.join(val.split('__')[1:])
                if self.token.closure is not None:
                    for i, v in self.token.closure.items():
                        v._pretty_print(context)
                    cc = copy.copy(context)
                    cc['__prog__'] = []
                    out = self.token.pretty_print(cc).split('\n')
                    for i in out:
                        context['__prog__'].append(i)
            else:
                # anonymous function
                for i, v in self.token.closure.items():
                    v._pretty_print(context)
                val = self.token.code
                if self.token.is_lambda:
                    val = 'lambda: ' + val

        elif isinstance(self.token, str):
            val= f"\"{val}\""

        context[self] = val
        return context[self]

class NewVar(Program):
    def __init__(self, left, right, is_func=False):
        self.left = left
        self.right = right
        self._tp = right.return_tp
        self.is_func = is_func

    def __str__(self):
        return f"{self.left} = {self.right}"

    def _print_line(self, context):
        self.right._pretty_print(context)
        if self.is_func:
            return None
        return f"{self.left} = {context[self.right]}"
        
class Name(Program):
    def __init__(self, token, assign, tp):
        self.token = token
        self.assign = assign
        self._tp = tp

    def __str__(self):
        return f"{self.token}"

    def _print_line(self, context):
        self.assign._pretty_print(context)
        return self.token


class ListExpr(Program):
    def __init__(self, exprs:typing.List[Program], return_prog: Program):
        self.exprs = exprs
        self.return_prog = return_prog
        if self.return_prog is not None:
            self._tp = self.return_prog.return_tp
        else:
            self._tp = none

    def _print_line(self, context):
        for i in self.exprs:
            i._pretty_print(context, is_expr=True)
        if self.return_prog is not None:
            return self.return_prog._print_line(context)
        else:
            return None