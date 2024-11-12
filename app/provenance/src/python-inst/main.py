#!/usr/bin/env python3

import ast
import os

import argparse

from util import *
from TaintInstrument import TaintInstrument
from MyVisitor import MyVisitor
from InternalFunctions import InternalFunctions
from ImportAnalyzer import ImportAnalyzer
from Logger import Logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="/model_src/train.py")
    parser.add_argument("-o", "--output", type=str, default="/src/train.py")

    args = parser.parse_args()
    if args.output == None:
        root, ext = os.path.splitext(args.input)
        args.output = root + ".inst.py"
    return args

def main(args):
    basedir = os.getcwd()

    with open(args.input, "r") as src:
        node = ast.parse(src.read())

    curnode = node

    import_analyzer = ImportAnalyzer()
    import_analyzer.visit(curnode)
    import_as = import_analyzer.import_as
    from_import_all = import_analyzer.from_import_all

    visitor = MyVisitor()

    _internal = InternalFunctions()
    _internal.visit(curnode)
    int_funcs = _internal.int_funcs

    tainter = TaintInstrument(visitor, int_funcs, import_as)
    curnode = tainter.visit(curnode)

    logger = Logger()
    curnode = logger.visit(curnode)

    curnode = ast.fix_missing_locations(curnode)

    # os.chdir(basedir)

    with open(args.output, "w") as dst:
        dst.write(ast.unparse(curnode))

if __name__ == "__main__":
    args = parse_args()
    main(args)
