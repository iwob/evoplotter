import os
import re
import subprocess
from subprocess import PIPE, STDOUT
from evoplotter import utils



def str_to_wlist(s, par_open = '(', par_close = ')'):
    """Converts a string to a list of words, where words are delimited by whitespaces."""
    return s.replace(par_open, ' '+par_open+' ').replace(par_close, ' '+par_close+' ').split()


def index_of_closing_parenthesis(words, start, left_enc ='(', right_enc =')'):
    """Returns index of the closing parenthesis of the parenthesis indicated by start."""
    num_opened = 1
    for i in range(start + 1, len(words)):
        if words[i] == left_enc:
            num_opened += 1
        elif words[i] == right_enc:
            num_opened -= 1

        if num_opened == 0:
            return i
    return -1


def index_of_opening_parenthesis(words, start, left_enc ='(', right_enc =')'):
    """Returns index of the opening parenthesis of the parenthesis indicated by start."""
    num_opened = -1
    i = start-1
    while i >= 0:
        if words[i] == left_enc:
            num_opened += 1
        elif words[i] == right_enc:
            num_opened -= 1

        if num_opened == 0:
            return i
        i -= 1
    return -1


class NodeSmt2(object):

    def __init__(self, name, args, info = None):
        self.name = name
        self.args = args
        self.info = info if info is not None else {}
        self.is_terminal = len(self.args) == 0

    def __str__(self):
        return self.str_format(sep=', ', smt2_mode=False)

    def height(self):
        if len(self.args) == 0:
            return 0
        else:
            return 1 + max([a.height() for a in self.args])

    def size(self):
        if len(self.args) == 0:
            return 1
        else:
            return 1 + sum([a.size() for a in self.args])

    def str_smt2(self):
        return self.str_format(sep=' ', smt2_mode=True)

    def str_format(self, sep=' ', lpar='(', rpar=')', smt2_mode=False):
        if self.is_terminal:
            return self.name
        if smt2_mode:
            text = lpar + self.name + ' '
        else:
            text = self.name + lpar
        text += sep.join([a.str_format(sep, lpar, rpar, smt2_mode) for a in self.args])
        text += rpar
        if ':named' in self.info:
            return '(! ' + text + ' :named ' + self.info[':named'] + ')'
        else:
            return text

    def change_args(self, new_args):
        return NodeSmt2(self.name, new_args, self.info.copy())

    @staticmethod
    def from_str(code):
        return NodeSmt2.from_wlist(str_to_wlist(code))

    @staticmethod
    def from_wlist(words):
        if len(words) == 1:
            return NodeSmt2(words[0], [])
        else:
            assert words[0] == '('
            name = words[1]
            i = 2
            args = []
            while i < len(words)-1:
                if words[i] == '(':
                    # Term in parenthesis
                    j = index_of_closing_parenthesis(words, i)
                    args.append(NodeSmt2.from_wlist(words[i:j + 1]))
                    i = j + 1
                else:
                    # Term without parenthesis: variable or constant
                    args.append(NodeSmt2.from_wlist([words[i]]))
                    i += 1
            return NodeSmt2(name, args)







def collect_all_names(expr):
    variables = re.findall(r"\s[a-zA-Z][a-zA-Z0-9]*", expr)
    variables = [v[1:] for v in variables]
    return set(variables)


def simplifyExpressionScript(expr, logic, var_type):
    """Produces a script in SMT-LIB of the simplification query."""
    variables = collect_all_names(expr)
    text  = "(set-logic {0})\n".format(logic)
    text += "\n".join(["(declare-var {0} {1})".format(v, var_type) for v in variables]) + "\n"
    text += "(simplify {0})".format(expr)
    return text


def simplifyExpression(expr, logic, var_type):
    script = simplifyExpressionScript(expr, logic, var_type)

    # Save script
    file = open("tmp.smt2", "w")
    file.write(script)
    file.close()

    # If shell=True, the command string is interpreted as a raw shell command.
    z3_args = "pp.single_line=true pp.decimal_precision=50 pp.decimal=true pp.min-alias-size=1000000 pp.max_depth=1000000"
    completed_proc = subprocess.run("./z3 {0} tmp.smt2".format(z3_args), shell=True, universal_newlines=True, stdout=PIPE, stderr=PIPE)
    out = completed_proc.stdout #.decode("utf-8")
    err = completed_proc.stderr #.decode("utf-8")

    # Remove file
    os.remove("tmp.smt2")
    if completed_proc.returncode != 0:
        print("STDERR:\n" + str(err))
        return None
    else:
        return out.strip()



def updatePropertyFile(p, exprSimpl):
    p["result.bestOrig"] = p["result.best"]
    p["result.bestOrig.size"] = p["result.best.size"]
    p["result.bestOrig.height"] = p["result.best.height"]
    p["result.bestOrig.smtlib"] = p["result.best.smtlib"]

    tree = NodeSmt2.from_str(exprSimpl)
    p["result.best"] = exprSimpl
    p["result.best.size"] = tree.size()
    p["result.best.height"] = tree.height()
    p["result.best.smtlib"] = exprSimpl
    return p



def simplifySolutionsInDirectory(folders, logic, var_type):
    props = utils.load_properties_dirs(folders, ignoreExts=[".txt", ".error"], add_file_path=True)
    for p in props:
        if "result.best" in p:
            print("Processing: ", p["evoplotter.file"])
            exprSimpl = simplifyExpression(p["result.best.smtlib"], logic=logic, var_type=var_type)
            updatePropertyFile(p, exprSimpl)
            utils.save_properties_file(p, p["evoplotter.file"])
            print("Successfully saved")




# print(collect_all_names("(ite (and (<= a b) (<= b c) (<= c d) (<= d e)) 1 0)"))
# print(collect_all_names("(ite (and (>= k1 y1) (<= k1 y2)) 1 (ite (and (<= k1 y2) (<= k1 y1)) 0 2))"))
# print(simplifyExpressionScript("(ite (and (<= a b) (<= b c) (<= c d) (<= d e)) 1 0)", logic="LIA", var_type="Int"))


# expr = "(ite (and (<= a b) (<= b c) (<= c d) (<= d e)) 1 0)"
# exprSimpl = simplifyExpression(expr, "LIA", "Int")
#
# print("expr:", expr)
# print("exprSimpl:", exprSimpl)


simplifySolutionsInDirectory(["FORMAL_SIMPLIFIED/data_formal_lia"], logic="LIA", var_type="Int")
simplifySolutionsInDirectory(["FORMAL_SIMPLIFIED/data_formal_slia"], logic="ALL", var_type="String")