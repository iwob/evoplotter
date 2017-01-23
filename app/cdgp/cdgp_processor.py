import sys
from src import utils
from src import plotter
from src import printer
from src import reporting
from src.dims import *
import numpy
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

# Printing names of files which finished with error status.
print("Files with error status:")
props_errors = [p for p in props if p["status"] != "completed" and p["status"] != "initialized"]
for p in props_errors:
	print(p["thisFileName"])


# Filtering props so only correct ones
props = [p for p in props if "benchmark" in p and ("result.best.eval" in p or "result.successRate" in p)]



def p_GP(p):
	return p["searchAlgorithm"] == "GP"
def p_GPSteadyState(p):
	return p["searchAlgorithm"] == "GPSteadyState"
def p_Lexicase(p):
	return p["searchAlgorithm"] == "Lexicase"
def p_LexicaseSteadyState(p):
	return p["searchAlgorithm"] == "LexicaseSteadyState"
def p_method0(p):
	return p["method"] == "0"
def p_method1(p):
	return p["method"] == "1"
def p_method2(p):
	return p["method"] == "2"

print("\nFiles with > 5000 total tests:")
props_errors = [p for p in props if p["benchmark"] == "other/CountPositive2.sl" and p["status"] == "completed" and int(p["totalTests"]) > 5000]
for p in props_errors:
	print(p["thisFileName"])

dim_method = Dim([Config("CDGP", p_method0),
                  Config("CDGPconservative", p_method1),
                  Config("GPR", p_method2)])
dim_sa = Dim([Config("GP", p_GP),
			  Config("GPSteadyState", p_GPSteadyState),
              Config("Lexicase", p_Lexicase),
              Config("LexicaseSteadyState", p_LexicaseSteadyState)])
dim_benchmarks = Dim.from_data(props, lambda p: p["benchmark"])




def get_num_optimal(props):
	def is_eval_correct(e):
		return e == "-1" or e[:7] == "List(-1" or e[:9] == "Vector(-1"
	props2 = [p for p in props if ("result.best.eval" in p and is_eval_correct(p["result.best.eval"])) or \
	                              ("result.successRate" in p and p["result.successRate"] == "1.0")]
	return len(props2)

def get_num_computed(filtered):
	return len(filtered)
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
def get_stats_size(props):
	vals = [float(p["result.best.size"]) for p in props]
	if len(vals) == 0:
		return "-"#-1.0, -1.0
	else:
		return "%0.2f" % numpy.mean(vals)#, numpy.std(vals)
def get_avg_totalTests(props):
	vals = [float(p["totalTests"]) for p in props]
	if len(vals) == 0:
		return "-"  # -1.0, -1.0
	else:
		return "%0.2f" % numpy.mean(vals)  # , numpy.std(vals)


# text = printer.text_table(props, dim_benchmarks.sort(), dim_method*dim_sa, fun1)
# print(text)
# print("\n\n")

print("STATUS")
text = printer.latex_table(props, dim_benchmarks.sort(), dim_method*dim_sa, get_num_computed)
latex_status = printer.table_color_map(text, 0.0, 1, 10.0, "lightred", "lightyellow", "lightgreen")
print(text)
print("\n\n")


print("SUCCESS RATES")
text = printer.latex_table(props, dim_benchmarks.sort(), dim_method*dim_sa, fun2)
latex_successRates = printer.table_color_map(text, 0.0, 0.5, 1.0, "lightred", "lightyellow", "lightgreen")
print(text)
print("\n\n")


print("SUCCESS RATES (FULL INFO)")
text = printer.latex_table(props, dim_benchmarks.sort(), dim_method*dim_sa, fun1)
print(text)
print("\n\n")


print("AVG TOTAL TESTS")
text = printer.latex_table(props, dim_benchmarks.sort(), dim_method*dim_sa, get_avg_totalTests)
latex_avgTotalTests = printer.table_color_map(text, 0.0, 1000.0, 2000.0, "lightred", "lightyellow", "lightgreen")
print(text)
print("\n\n")


print("AVG SIZES")
text = printer.latex_table(props, dim_benchmarks.sort(), dim_method*dim_sa, get_stats_size)
latex_sizes = printer.table_color_map(text, 0.0, 100.0, 200.0, "lightred", "lightyellow", "lightgreen")
print(text)
print("\n\n")






# -------- Creating nice LaTeX report of the above results --------


report = reporting.ReportPDF()
section1 = r"""\section{Initial experiments}
\subsection{Success rates}
\definecolor{lightred}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{lightyellow}{rgb}{0.76, 0.98, 0.76} % lightgreen
\definecolor{lightgreen}{rgb}{0.66, 0.90, 0.66} % green
\begin{tabular}{ccccccc}
 & CDGP\_GP & CDGP\_Lexicase & CDGPconservative\_GP & CDGPconservative\_Lexicase & GPR\_GP & GPR\_Lexicase\\
sygus16/fg\_array\_search\_2.sl & \cellcolor{lightgreen!82.0!lightyellow}0.91 & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightred!0.0!lightyellow}0.50 & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!78.0!lightyellow}0.11\\
sygus16/fg\_array\_search\_4.sl & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
sygus16/fg\_array\_search\_6.sl & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
sygus16/fg\_array\_search\_8.sl & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
sygus16/fg\_array\_sum\_2\_15.sl & \cellcolor{lightred!80.0!lightyellow}0.10 & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightred!78.0!lightyellow}0.11 & \cellcolor{lightred!19.999999999999996!lightyellow}0.40 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
sygus16/fg\_array\_sum\_4\_15.sl & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
sygus16/fg\_array\_sum\_6\_15.sl & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
sygus16/fg\_array\_sum\_8\_15.sl & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
sygus16/fg\_max2.sl & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightgreen!100.0!lightyellow}1.00\\
sygus16/fg\_max4.sl & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightgreen!100.0!lightyellow}1.00 & \cellcolor{lightred!78.0!lightyellow}0.11 & \cellcolor{lightred!0.0!lightyellow}0.50 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!80.0!lightyellow}0.10\\
sygus16/fg\_max6.sl & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!14.000000000000002!lightyellow}0.43 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
sygus16/fg\_max8.sl & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00 & \cellcolor{lightred!100.0!lightyellow}0.00\\
\end{tabular}
\vspace{1cm}

"""
report.add(reporting.BlockLatex(section1))


section2 = reporting.BlockSection("Initial experiments 2", [])
subsects = [("Status (correctly finished processes)", latex_status, reporting.color_scheme_red_r),
			("Success rates", latex_successRates, reporting.color_scheme_green),
            ("Average sizes of $T_C$ (total tests in run)", latex_avgTotalTests, reporting.color_scheme_blue),
            ("Average sizes of best of runs (number of nodes)", latex_sizes, reporting.color_scheme_yellow)]
for title, table, cs in subsects:
	sub = reporting.BlockSubSection(title, [cs, reporting.BlockLatex(table + "\n")])
	section2.add(sub)
report.add(section2)



print("\n\nREPORT:\n")
print(report.apply())
report.save_and_compile("cdgp_results.tex")