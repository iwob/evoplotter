import os
import re
import subprocess
import pandas as pd
from pathlib import Path
from subprocess import call, STDOUT
from .. import utils
from .. import printer


def extractLineFromROutput(output, key):
    i = output.rfind(key)
    return (output[i + len(key) + 1:].split("\n")[0]).split(" ")[1].strip()

def extractLinesFromROutput(output, key, numLines):
    i = output.rfind(key)
    return "\n".join(output[i + len(key) + 1:].split("\n")[0:numLines+1]).strip()

def pandasTableFromROutput(output, key, numLines, tmpCsv="tmp.csv"):
    output_extr = extractLinesFromROutput(output, key, numLines=numLines)
    utils.save_to_file(tmpCsv, output_extr)
    res = pd.read_csv(tmpCsv, header=0, index_col=0, delimiter=r"\s+")
    # call(["rm", "-f", tmpCsv])
    return res



class FriedmanResult:
    """Stores results of the Friedman test."""
    def __init__(self, output, p_value, ranks, cmp_matrix=None, cmp_method=""):
        assert isinstance(p_value, float)
        self.output = output
        self.p_value = p_value
        self.ranks = ranks
        self.cmp_matrix = cmp_matrix
        self.cmp_method = cmp_method

    def __str__(self):
        return self.output



def runFriedmanKK(table):
    """Runs a Friedman statistical test using a mysterious script provided to me by KK.
     Input is a Table."""
    assert isinstance(table, printer.Table)
    return runFriedmanKK_csv(table.renderCsv(delim=";"))


def runFriedmanKK_csv(text):
    """Runs a Friedman statistical test using a mysterious script provided to me by KK.
     Input is a CSV-formatted text."""
    csvFile = "tmp.csv"
    thisScriptPath = Path(os.path.abspath(__file__))
    print("thisScriptPath.parent", thisScriptPath.parent)

    cwd = os.getcwd()
    os.chdir(str(thisScriptPath.parent))
    utils.save_to_file(csvFile, text)

    # Running command
    try:
        output = subprocess.check_output(["Rscript", "friedman_kk.r", csvFile, "FALSE"], stderr=STDOUT,
                                universal_newlines=True)
        output = output[output.rfind("$p.value"):]
        print(output)

        print('\n\n')

        p_value = float(extractLineFromROutput(output, "$p.value"))
        cmp_method = extractLineFromROutput(output, "$cmp.method")
        print("p_value: '{0}'".format(p_value))
        print("cmp_method: '{0}'".format(cmp_method))

        ranks = pandasTableFromROutput(output, "$ranks", numLines=2, tmpCsv="tmpRanks.csv")
        print("ranks:", ranks)

        i = output.rfind("$cmp.matrix")
        if i == -1:
            cmp_matrix = None
        else:
            cmp_matrix = pandasTableFromROutput(output, "$cmp.matrix", numLines=ranks.shape[1]+1, tmpCsv="tmpCmpMatrix.csv")
            print("cmp_matrix:", cmp_matrix)

        friedmanResult = FriedmanResult(output, p_value, ranks, cmp_matrix, cmp_method=cmp_method)

    except subprocess.CalledProcessError as exc:
        output = exc.output.replace("\\n", "\n")
        print("Status: FAIL, return code: {0}, msg: {1}".format(exc.returncode, output))
        friedmanResult = None

    # call(["rm", "-f", csvFile])
    os.chdir(cwd)
    return friedmanResult
