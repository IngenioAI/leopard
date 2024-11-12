import ast
from util import *

class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.import_as = {} 
        self.from_import_all = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname == None:
                self.import_as[alias.name] = alias.name
            else:
                self.import_as[alias.asname] = alias.name

    def visit_ImportFrom(self, node):
        # NOTE: ignore import level here
        # e.g., import ..foo.bar import a -> level: 2 (relative path depth)
        # actually it is really trivial feature, we don't need it

        module = node.module
        for alias in node.names:
            if alias.name == "*":
                self.from_import_all.append(module)
            else:
                if alias.asname == None:
                    self.import_as[alias.name] = f"{module}.{alias.name}"
                else:
                    self.import_as[alias.asname] = f"{module}.{alias.name}"
