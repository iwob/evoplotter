import sys
from src import utils
from src import plotter
from src import printer
from src.dims import *
import argparse


options = argparse.ArgumentParser(description="Simple parser for the smtgp results.", add_help=True)
options.add_argument("dirs", type=str, nargs="*", default=None,
		             help="Name of the file containing ")
options.add_argument("-f", "--file", type=str, default=None,
		             help="Name of the file containing ")




##################################################
#                      MAIN
##################################################

# Checking if the number of arguments is correct.
if len(sys.argv) == 1:
	print("No results directory was specified!")
	exit()

env = options.parse_args()



if env.file is None:
	folders = env.dirs
else:
	folders = [L.strip() for L in utils.read_lines(env.file)]

props = utils.load_properties_dirs(folders, exts=[".txt"])
props = [p for p in props if "benchmark" in p and ("result.best.eval" in p or "result.successRate" in p)]



def p_GP(p):
	return p["searchAlgorithm"] == "GP"
def p_Lexicase(p):
	return p["searchAlgorithm"] == "Lexicase"
def p_method0(p):
	return p["method"] == "0"
def p_method1(p):
	return p["method"] == "1"
def p_method2(p):
	return p["method"] == "2"




dim_method = Dim([Config("CDGP", p_method0),
                  Config("CDGPconservative", p_method1),
                  Config("GPR", p_method2)])
dim_sa = Dim([Config("GP", p_GP),
              Config("Lexicase", p_Lexicase)])
dim_benchmarks = Dim.from_data(props, lambda p: p["benchmark"])




def get_num_optimal(props):
	props2 = [p for p in props if ("result.best.eval" in p and p["result.best.eval"] == "-1") or \
		                          ("result.successRate" in p and p["result.successRate"] == "1.0")]
	return len(props2)

def fun1(filtered):
	if len(filtered) == 0:
		return None
	num_opt = get_num_optimal(filtered)
	return "{0}/{1}".format(str(num_opt), str(len(filtered)))
def fun2(filtered):
	if len(filtered) == 0:
		return None
	num_opt = get_num_optimal(filtered)
	sr = float(num_opt) / float(len(filtered))
	return "{0}".format("%0.2f" % sr)


# text = printer.text_table(props, dim_benchmarks.sort(), dim_method*dim_sa, fun1)
# print(text)
# print("\n\n")

text = printer.latex_table(props, dim_benchmarks.sort(), dim_method*dim_sa, fun1)
print(text)
print("\n\n")

text = printer.latex_table(props, dim_benchmarks.sort(), dim_method*dim_sa, fun2)
print(text)
print("\n\n")

text = printer.table_color_map(text, 0.0, 0.5, 1.0, "lightred", "lightyellow", "lightgreen")
print(text)
print("\n\n")