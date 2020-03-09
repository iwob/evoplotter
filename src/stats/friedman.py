import os
import re
import subprocess
# import pandas as pd
from pathlib import Path
from subprocess import call, STDOUT
from .. import utils
from .. import printer


class FriedmanResult:
    def __init__(self, p_value, ranks):
        assert isinstance(p_value, float)
        assert isinstance(p_value, printer.TableContent)
        self.p_value = p_value


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

        i = output.rfind("$p.value")
        p_value = float((output[i + len("$p.value") + 1:].split("\n")[0]).split(" ")[1].strip())
        print("p_value: '{0}'".format(p_value))

        # i = output.rfind("$ranks")
        # print(re.split('\d+', s_nums))

        # pd.read("whitespace.csv", header=None, delimiter=r"\s+")


    except subprocess.CalledProcessError as exc:
        output = exc.output.replace("\\n", "\n")
        print("Status: FAIL, return code: {0}, msg: {1}".format(exc.returncode, output))

    # call(["rm", "-f", csvFile])
    os.chdir(cwd)
    return output



