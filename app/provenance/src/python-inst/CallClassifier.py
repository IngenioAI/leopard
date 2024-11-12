# heusristically classify methods
# same role with KB (knowledge base) from Vamsa (KDD '20)

import ast
import json
from util import *

class CallClassifier():
    def __init__(self, import_as, from_import_all, kbpath):
        self.import_as = import_as
        self.from_import_all = from_import_all
        self.kb = {}
        with open(kbpath) as kbfile:
            kb_json = json.load(kbfile)
            for kb in kb_json:
                method = kb["method"]
                action = kb["action"]
                finput = kb["input"]
                foutput = kb["output"]

                self.kb[method] = { "action": action, "input": finput, "output": foutput }

        print(self.kb)

    def getMethodFullName(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func

        try:
            if isinstance(func, ast.Attribute):
                flatten_value = ""
                value = func
                # value = func.value
                while not isinstance(value, ast.Name):
                    flatten_value = f".{value.attr}{flatten_value}"
                    value = value.value
                if value.id in self.import_as:
                    # print(f"{self.import_as[value.id]}{flatten_value}")
                    return f"{self.import_as[value.id]}{flatten_value}"
                else:
                    # always var.method
                    # print(f"{value.id}{flatten_value}")
                    return f"{value.id}{flatten_value}"
            else:
                if func.id in self.import_as:
                    # print(f"{self.import_as[func.id]}")
                    return f"{self.import_as[func.id]}"
                else:
                    # print(f"{func.id}")
                    return f"{func.id}"
        except AttributeError:
            return ""

    def getPossibleMethodFullName(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func

        fullname = self.getMethodFullName(node)

        result = []
        result.append(fullname)

        if "." not in fullname:
            for import_all in self.from_import_all:
                result.append(f"{import_all}.{fullname}")

        return result

    def extractSelf(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func
        fullname = self.getMethodFullName(node)

        return fullname.split(".")[:-1][0]

    def isSplit(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func
        fullnames = self.getMethodFullName(node)

        if "sklearn.model_selection.train_test_split" in fullnames:
            return True

        return False

    def isFit(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func
        fullnames = self.getMethodFullName(node)

        # TODO: how can we differentiate model.fit and another fit methods?
        if len(node.args) != 2:
            return False

        if "fit" in fullnames:
            return True

        return False

    def isDrop(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func
        fullname = self.getMethodFullName(node)

        if "." not in fullname:
            return False

        methodname = fullname.split(".")[-1]

        if methodname in self.kb and self.kb[methodname]["action"] == "drop":
            return True

        return False

    def isReadDataset(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func
        fullname = self.getMethodFullName(node)

        if fullname in self.kb and self.kb[fullname]["action"] == "read":
            return True

        return False

    def isProcess(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func
        fullname = self.getMethodFullName(node)

        if fullname in self.kb and self.kb[fullname]["action"] == "process":
            return True

        return False

    def isPseudonymize(self, node):
        assert isinstance(node, ast.Call), RED("Given node is not ast.Call")

        func = node.func
        fullname = self.getMethodFullName(node)

        if fullname in self.kb and self.kb[fullname]["action"] == "pseudonymize":
            return True

        return False
