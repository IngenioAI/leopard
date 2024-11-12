import ast

from util import *
from ASTUtil import *

class IntFunc():
    def __init__(self, node, returns):
        self.name = node.name
        self.node = node
        self.returns = returns

class InternalFunctions(ast.NodeVisitor):
    def __init__(self):
        self.int_funcs = set()

    def visit_FunctionDef(self, node):
        returns = []
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                returns.append(stmt.value)

        retvals = []
        for retval in returns:
            if isinstance(retval, ast.Tuple):
                retvec = flatten_elts(retval)
            else:
                retvec = [retval]

            ret = []
            for flat in retvec:
                ret.append(list(map(lambda fl: expr_to_string(fl), flatten(flat))))
            retvals.append(ret)

        fn = IntFunc(node, retvals)
        self.int_funcs.add(fn)
