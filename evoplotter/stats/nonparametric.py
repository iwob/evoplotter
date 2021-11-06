import os
import subprocess
import pandas as pd
import numpy as np
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
    def __init__(self, output, p_value, ranks, cmp_matrix=None, cmp_method="", binary_cmp_matrix=False):
        """
        :param output: raw output as returned by the R script.
        :param p_value: p value returned by R script.
        :param ranks: pandas DataFrame containing ranks for particular approaches.
        :param cmp_matrix: pandas DataFrame approaches x approaches, where 1 means that approach is significantly
         better, and -1 that it is significantly worse.
        :param cmp_method: method of the post-hoc test.
        """
        assert p_value is None or isinstance(p_value, float)
        self.output = output
        self.p_value = p_value
        self.ranks = ranks
        self.cmp_matrix = cmp_matrix
        self.cmp_method = cmp_method
        self.binary_cmp_matrix = binary_cmp_matrix

    def getSignificantPairs(self, p_threshold=0.05):
        """Returns a list of tuples, where the first element is significantly better than the second."""
        if self.cmp_matrix is None:
            return []
        else:
            res = []
            for i in range(self.cmp_matrix.shape[0]):
                for j in range(self.cmp_matrix.shape[1]):
                    if (self.binary_cmp_matrix and self.cmp_matrix.iat[i,j] == 1) or\
                       (not self.binary_cmp_matrix and self.cmp_matrix.iat[i,j] <= p_threshold):
                        L = self.cmp_matrix.index.values[i]
                        R = self.cmp_matrix.columns.values[j]
                        res.append((L, R))
            res.sort(key=lambda t: t[0])
            return res

    def getSignificantPairsText(self):
        """Returns a formatted text for significant pairs."""
        return "\n".join(["{0}\t>\t{1}".format(L, R) for L, R in self.getSignificantPairs()])

    def __str__(self):
        return self.output




def runFriedmanPython(table):
    """Runs a Friedman statistical test with Nemenyi posthoc test using implementation in scikit_posthocs package."""
    assert isinstance(table, printer.Table)
    try:
        data = np.array(table.content.cells, dtype=np.float32)
        return runFriedmanPython_array(data)
    except ValueError:
        return None


def runFriedmanPython_array(data):
    import scipy.stats as ss
    import scikit_posthocs as sp
    p_statistic, p_value = ss.friedmanchisquare(*data.T)
    # https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_nemenyi_friedman/#id2
    # P. Nemenyi (1963) Distribution-free Multiple Comparisons. Ph.D. thesis, Princeton University.
    pc = sp.posthoc_nemenyi_friedman(data)
    return FriedmanResult("", p_value, None, cmp_matrix=pc, binary_cmp_matrix=False, cmp_method="nemenyi")


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
    pathFriedmanScript = "friedman_kk.r"
    # pathFriedmanScript = "friedman_penn.r"
    try:
        output = subprocess.check_output(["Rscript", pathFriedmanScript, csvFile, "FALSE"], stderr=STDOUT,
                                universal_newlines=True)
        output = output[output.rfind("$p.value"):]
        print(output)

        print('\n\n')

        p_value = float(extractLineFromROutput(output, "$p.value"))
        cmp_method = extractLineFromROutput(output, "$cmp.method").replace("\"", "")
        # print("p_value: '{0}'".format(p_value))
        # print("cmp_method: '{0}'".format(cmp_method))

        ranks = pandasTableFromROutput(output, "$ranks", numLines=2, tmpCsv="tmpRanks.csv")
        # print("ranks:", ranks)

        i = output.rfind("$cmp.matrix")
        if i == -1:
            cmp_matrix = None
        else:
            cmp_matrix = pandasTableFromROutput(output, "$cmp.matrix", numLines=ranks.shape[1]+1, tmpCsv="tmpCmpMatrix.csv")
            # print("cmp_matrix:", cmp_matrix)

        friedmanResult = FriedmanResult(output, p_value, ranks, cmp_matrix, cmp_method=cmp_method, binary_cmp_matrix=True)

    except subprocess.CalledProcessError as exc:
        output = exc.output #.decode("utf-8")
        output = output.replace("\\n", "\n")
        print("Status: FAIL, return code: {0}, msg: {1}".format(exc.returncode, output))
        friedmanResult = FriedmanResult(output, None, None, None)

    call(["rm", "-f", csvFile])
    os.chdir(cwd)
    return friedmanResult



def runWilcoxonSignedRank(x, y):
    """Runs a Wilcoxon signed-rank test. x and y are the data vectors to be compared."""
    import scipy.stats as ss
    w, p_value = ss.wilcoxon(x, y)
    return p_value