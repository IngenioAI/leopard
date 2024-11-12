import ast
from util import *

def expr_to_string(node):
    # to get variable name
    if isinstance(node, ast.Attribute):
        return expr_to_string(node.value) + "." + node.attr
    elif isinstance(node, ast.Name):
        return f"{node.id}"
    elif isinstance(node, ast.Constant):
        return f"{node.value}"
    elif isinstance(node, ast.Subscript):
        return expr_to_string(node.value) + "[" + expr_to_string(node.slice) + "]"
    elif isinstance(node, ast.Slice):
        if hasattr(node, "step"):
            return f"{node.lower}:{node.upper}:{node.step}"
        else:
            return f"{node.lower}:{node.upper}"
    elif isinstance(node, ast.UnaryOp):
        return expr_to_string(node.operand)
    elif isinstance(node, ast.Call):
        return expr_to_string(node.func) + "()"
    else:
        # print(RED("NOO"))
        # print(ast.dump(node))
        return ""

def flatten(node):
    flag = True
    worklist = [node]
    result = []

    while flag:
        flag = False
        result = []

        for worknode in worklist:
            if hasattr(worknode, "elts"):
                result.extend(flatten_elts(worknode))
                flag = True
            elif isinstance(worknode, ast.BinOp):
                result.extend(flatten_binop(worknode))
                flag = True
            else:
                result.append(worknode)

        worklist = result

    return result

def flatten_elts(node):
    # Tuple, List, Set
    # (a, b, c) -> [a, b, c]

    if hasattr(node, "elts"):
        result = []
        for elt in node.elts:
            result.append(elt)
        return result
    else:
        return [node]

def flatten_binop(node):
    # BinOp
    # a + b -> [a, b]

    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    else:
        return [node]

def extract_args_from_call(node):
    assert isinstance(node, ast.Call), RED("extract_args_from_call: node is not ast.Call")

    result = []
    for arg in node.args:
        if isinstance(arg, ast.Call):
            result.extend(extract_args_from_call(arg))
        else:
            result.append(expr_to_string(arg))
    return result
