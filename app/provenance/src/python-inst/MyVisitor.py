import ast

from util import *
from ASTUtil import *

class MyVisitor(ast.NodeVisitor):
    def __init__(self):
        pass

    '''
    def generic_visit(self, node):
        return node
    '''

    def visit_Assign(self, node):
        # ast.Assign(targets, value, type_comment)
        # targets is a list of nodes
        # value is a single node

        lhs = node.targets
        rhs = node.value

    def visit_AugAssign(self, node):
        # ast.AugAssign(target, op, value)
        # e.g., a += 3

        lhs = node.target
        rhs = node.value

    def visit_Call(self, node):
        result = []
        result.extend(list(map(self.visit, node.args)))
        result.extend(list(map(lambda kw: self.visit(kw.value), node.keywords)))
        return result

    def visit_Attribute(self, node):
        # ast.Attribute(value, attr, ctx)

        return ("attribute", self.visit(node.value), node.attr)

    def visit_Name(self, node):
        return f"{node.id}"

    def visit_Slice(self, node):
        return [self.visit(node.lower), self.visit(node.upper)]

    def visit_Constant(self, node):
        return f"{node.value} (const)"

    def visit_Tuple(self, node):
        return list(map(self.visit, node.elts))

    def visit_Subscript(self, node):
        # ast.Subscript(value, slice, ctx)

        return ("subscript", self.visit(node.value), self.visit(node.slice))
