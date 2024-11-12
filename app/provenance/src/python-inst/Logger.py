import ast
from util import *
import os
import csv

def create_log(dbfile, lineno, tx, desc):
    # log_code = f"logger.log({dbfile},{lineno},{tx},{desc})"
    log_code = f"ProvenanceLogger.log({dbfile},{lineno},{tx},{desc})"
    log_node = ast.parse(log_code).body[0]
    return log_node

def get_id_or_value(node):
    try:
        ret = (node.id, True)
    except AttributeError:
        ret = (node.value, False)

    return ret

class Logger(ast.NodeTransformer):
    def __init__(self):
        pass

    def visit_Module(self, node):
        insert_code = (
            # f"import time\n"
            f"from time import ctime\n"
            f"class ProvenanceLogger:\n"
            f"    callstack = [\"__main__\"]\n"
            f"    @staticmethod\n"
            f"    def log(dbfile, lineno, tx, desc):\n"
            f"        logfile = open(\"logfile\", \'a\')\n"
            f"        curtime = ctime()\n"
            f"        logfile.write(f'{{curtime}},{{__file__}},{{dbfile}},{{ProvenanceLogger.callstack[-1]}},{{lineno}},{{tx}},\"{{desc}}\"\\n')\n"
            f"        logfile.close()\n"
            f"\n"
            # f"logger = ProvenanceLogger(\"logfile\")\n"
        )
        insert_node = ast.parse(insert_code).body
        node.body = insert_node + node.body

        self.generic_visit(node)

        return node
