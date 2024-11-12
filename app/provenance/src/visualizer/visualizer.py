#!/usr/bin/env python3

import csv
import os
import argparse

set_dbfile = set()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="/log/logfile")
    parser.add_argument("-o", "--output", type=str, default="/mmd/facenet.mmd")

    args = parser.parse_args()
    if args.output == None:
        root, ext = os.path.splitext(args.input)
        args.output = root + ".mmd"
    return args

class Graph:
    def __init__(self):
        self.scripts = dict()

    def get_script(self, sname):
        if sname not in self.scripts:
            self.scripts[sname] = Script(sname, len(self.scripts) + 1)
        return self.scripts[sname]

    def visualize(self, graphfile):
        graph = ""
        graph += "graph LR\n"
        graph += "\n"

        for script in self.scripts.values():
            for (dbname, dbnode) in script.databases.items():
                graph += f"{dbnode}(\"{dbname}\")\n"
            for (modelname, modelnode) in script.models.items():
                graph += f"{modelnode}(\"{modelname}\")\n"
        graph += "\n"

        for script in self.scripts.values():
            for edge in script.edges:
                lhs = edge.split("--")[0]
                rhs = edge.split("-->")[1]
                if lhs in script.databases.values():
                    graph += edge + "\n"
                if rhs in script.models.values():
                    graph += edge + "\n"
        graph += "\n"

        for script in self.scripts.values():
            graph += f"subgraph sc{script.scriptid} [\"{script.name}\"]\n"
            for (nodename, node) in script.nodes.items():
                # if script.tags.get(nodename) != None:
                #     graph += "    " + nodename + "(\"" + node + "\")\n"
                if "DATABASE" in script.tags.get(nodename, ""):
                    graph += "    " + nodename + "(\"" + node + "\")\n"

            sensitive_edges = []
            for (i, edge) in enumerate(script.edges):
                lhs = edge.split("--")[0]
                rhs = edge.split("-->")[1]

                if "UNLEARN" in script.tags.get(lhs, "") and "MODEL" in script.tags.get(rhs, ""):
                    sensitive_edges.append(i)

                if lhs in script.databases.values() \
                    or rhs in script.models.values():
                    continue

                graph += "    " + edge + "\n"

            for sensitive in sensitive_edges:
                graph += f"    linkStyle {sensitive} stroke:red, stroke-width:2px\n"

            graph += "end\n"

        with open(graphfile, "w") as f:
            f.write(graph)
            f.close()


class Script:
    def __init__(self, name, sid):
        self.name = name
        self.nodes = dict()
        self.edges = []
        self.var_last_appear = dict()

        self.stmts = set()

        self.tags = dict()

        self.databases = dict()
        self.models = dict()

        self.scriptid = sid

    def create_node(self, node):
        pass

    def create_edge(self):
        pass

    def find_node(self, node):
        for (k, v) in self.nodes.items():
            if v == node:
                return k
        return None

    def new_node(self, action):
        find = self.find_node(action)
        if find is not None:
            return find

        nodename = f"SC{self.scriptid}N{len(self.nodes)+1}"
        self.nodes[nodename] = action

        return nodename

    def new_db(self, filename):
        # CAUTION: it is filename -> dbname relation, which is opposite of node

        if filename in self.databases:
            return self.databases[filename]

        dbname = f"D{len(self.databases)+1}"
        self.databases[filename] = dbname

        return dbname

    def new_model(self, modelname):
        # CAUTION: it is varname -> modelid relation, which is opposite of node

        if modelname in self.models:
            return self.models[modelname]

        modelid = f"SC{self.scriptid}M{len(self.models)+1}"
        self.models[modelname] = modelid

        return modelid

    def insert_dataflow(self, scope, lineno, action, dataflow):
        if (lineno, action, dataflow) in self.stmts:
            return

        if "loop_" in action:
            return

        if "fn_" in action:
            return

        self.stmts.add((lineno, action, dataflow))

        # print(dataflow, "\t", action)
        lhs, rhs = dataflow.split(" <- ", maxsplit=1)

        if "callarg" in action:
            # diff scope
            callee = action.split(": ")[1]
            self.dataflow(lineno, action, f"{lhs} ({callee})", f"{rhs} ({scope})")
            self.dataflow(lineno, action, f"{rhs} ({scope})", f"{lhs} ({callee})")
        elif "extcall" in action:
            # HACK: callee might be callable, so try callee <- arg
            callee = action.split(": ")[1]
            maybecallable = self.find_node(f"{callee} ({scope})")
            if maybecallable is not None:
                self.dataflow(lineno, "callable", f"{callee} ({scope})", f"{rhs} ({scope})")
                self.dataflow(lineno, "callable", f"{lhs} ({scope})", f"{callee} ({scope})")
            self.dataflow(lineno, action, f"{lhs} ({scope})", f"{rhs} ({scope})")
        else:
            self.dataflow(lineno, action, f"{lhs} ({scope})", f"{rhs} ({scope})")

    def insert_retedge(self, callee, caller):
        lineno = caller[4]

        calleescope = callee[3]
        callerscope = caller[3]

        callee = eval(callee[6])
        caller = eval(caller[6].split("to ")[1])

        for (lhs, rhss) in zip(caller, callee):
            for rhs in rhss:
                self.dataflow(lineno, "return", f"{lhs} ({callerscope})", f"{rhs} ({calleescope})")

    def dataflow(self, lineno, action, lhs, rhs):
        # lhs <- rhs

        if "(const)" in rhs:
            return

        newnode = self.new_node(f"{lhs}")

        if "path" in action and "extcall" not in action:
            import ast
            # Heuristic to detect list

            if "pathR" in action:
                rhs = rhs.split(" (")[0]
                if rhs[1] == "[" and rhs[-2]=="]":
                    rhs = rhs[1:-1]
                paths = ast.literal_eval(rhs)
                if isinstance(paths, list):
                    for p in paths:
                        '''
                        if not self.var_last_appear.get(p):
                            newnode_path = self.new_db(f"{p}")
                            self.var_last_appear[p] = newnode_path
                        else:
                            newnode_path = self.var_last_appear[p]
                        '''
                        newnode_path = self.new_db(f"{p}")
                        self.var_last_appear[p] = newnode_path

                        self.edges.append(f"{newnode_path}--{lineno}-->{newnode}")
                else:
                    p = paths
                    '''
                    if not self.var_last_appear.get(p):
                        newnode_path = self.new_db(f"{p}")
                        self.var_last_appear[p] = newnode_path
                    else:
                        newnode_path = self.var_last_appear[p]
                    '''
                    newnode_path = self.new_db(f"{p}")
                    self.var_last_appear[p] = newnode_path
                    self.edges.append(f"{newnode_path}--{lineno}-->{newnode}")
            else:
                # pathL
                lhs = lhs.split(" (")[0]
                p = lhs

                newnode_path = self.new_model(f"{p}")
                self.var_last_appear[lhs] = newnode_path
                rhs_appear = self.var_last_appear[rhs]

                self.edges.append(f"{rhs_appear}--{lineno}-->{newnode_path}")
        else:
            if rhs not in self.var_last_appear:
                # rhs must be const, just make lhs out of nowhere without edge
                self.var_last_appear[lhs] = newnode
            else:
                rhs_appear = self.var_last_appear[rhs]
                self.var_last_appear[lhs] = newnode

                self.edges.append(f"{rhs_appear}--{lineno}-->{newnode}")

    def taint(self):
        for database in self.databases.values():
            self.tags[database] = "DATABASE"
        for model in self.models.values():
            self.tags[model] = "MODEL"

        # SCENARIO: Won_bin is now private and needs to be unlearned
        self.tags[self.databases["Won_bin"]] += " UNLEARN"

        changed = True
        while changed:
            changed = False
            for edge in self.edges:
                lhs = edge.split("--")[0]
                rhs = edge.split("-->")[1]
                if "DATABASE" in self.tags.get(lhs, "") \
                        and "DATABASE" not in self.tags.get(rhs, ""):
                    changed = True
                    if self.tags.get(rhs) == None:
                        self.tags[rhs] = ""
                    self.tags[rhs] += " DATABASE"
                if "UNLEARN" in self.tags.get(lhs, "") \
                        and "UNLEARN" not in self.tags.get(rhs, ""):
                    changed = True
                    if self.tags.get(rhs) == None:
                        self.tags[rhs] = ""
                    self.tags[rhs] += " UNLEARN"

        changed = True
        while changed:
            changed = False
            for edge in self.edges:
                lhs = edge.split("--")[0]
                rhs = edge.split("-->")[1]
                if "MODEL" in self.tags.get(rhs, "") \
                        and "MODEL" not in self.tags.get(lhs, ""):
                    changed = True
                    if self.tags.get(lhs) == None:
                        self.tags[lhs] = ""
                    self.tags[lhs] += " MODEL"

    def sort_edges(self):
        result = []

        for edge in self.edges:
            lhs = edge.split("--")[0]
            rhs = edge.split("-->")[1]
            if lhs in self.databases.values():
                result.append(edge)

        for edge in self.edges:
            lhs = edge.split("--")[0]
            rhs = edge.split("-->")[1]
            if lhs not in self.databases.values() and rhs in self.models.values():
                result.append(edge)

        for edge in self.edges:
            lhs = edge.split("--")[0]
            rhs = edge.split("-->")[1]
            if lhs not in self.databases.values() and rhs not in self.models.values():
                result.append(edge)

        delete_idx = []
        for i, edge in enumerate(result):
            lhs = edge.split("--")[0]
            rhs = edge.split("-->")[1]
            if "DATABASE" not in self.tags.get(lhs, ""):
                delete_idx.append(i)

        for didx in reversed(delete_idx):
            result.pop(didx)

        self.edges = result

def main(args):
    logfile = args.input
    graph = Graph()

    with open(logfile, "r") as f_log:
        csvreader = csv.reader(f_log)
        prevline = ""
        for (i, line) in enumerate(csvreader):
            (time, script, dbfile, scope, lineno, action, dataflow) = line

            # for id()
            dataflow = dataflow.split(" ::")[0]

            script = graph.get_script(script)

            if "fn_return" in action and dataflow != "":
                prevline = line
                continue
            if "callret" in action:
                script.insert_retedge(prevline, line)
                continue

            script.insert_dataflow(scope, lineno, action, dataflow)

    for script in graph.scripts:
        graph.get_script(script).taint()
        graph.get_script(script).sort_edges()

    graph.visualize(args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)
