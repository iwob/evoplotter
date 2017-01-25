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
		             help="Paths to directories containing files with results (Initial experiment 2).")
options.add_argument("dirsExp3", type=str, nargs="*", default=None,
		             help="Paths to directories containing files with results of the second run (Initial experiment 3).")




##################################################
#                      MAIN
##################################################

# Checking if the number of arguments is correct.
# if len(sys.argv) == 1:
# 	print("No results directory was specified!")
# 	exit()

env = options.parse_args()



if env.dirs is None or len(env.dirs) == 0:
	folders_exp2 = ["res1_rms", "res1_first", "res1_gprlexss"]
	folders_exp3 = ["res1_tournDeselection", "res1_tournDeselection_gprNew"]
	print("Using default results directory names.")
	print("Experiment 2:")
	for f in folders_exp2:
		print(f)
	print("Experiment 3:")
	for f in folders_exp3:
		print(f)
else:
	folders_exp2 = env.dirs
	folders_exp3 = env.dirsExp3




def load_correct_props(folders):
	props = utils.load_properties_dirs(folders, exts=[".txt"])

	# Printing names of files which finished with error status.
	print("\nFiles with error status:")
	props_errors = [p for p in props if p["status"] != "completed" and p["status"] != "initialized"]
	for p in props_errors:
		print(p["thisFileName"])

	# Filtering props so only correct ones
	props = [p for p in props if "benchmark" in p and ("result.best.eval" in p or "result.successRate" in p)]
	return props


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





# print("\nFiles with > 5000 total tests:")
# props_errors = [p for p in props if p["benchmark"] == "other/CountPositive2.sl" and p["status"] == "completed" and int(p["totalTests"]) > 5000]
# for p in props_errors:
# 	print(p["thisFileName"])



dim_method = Dim([Config("CDGP", p_method0),
                  Config("CDGPconservative", p_method1),
                  Config("GPR", p_method2)])
dim_sa = Dim([Config("GP", p_GP),
			  Config("GPSteadyState", p_GPSteadyState),
              Config("Lexicase", p_Lexicase),
              Config("LexicaseSteadyState", p_LexicaseSteadyState)])




def get_num_optimal(props):
	def is_eval_correct(e):
		return e == "-1" or e[:7] == "List(-1" or e[:9] == "Vector(-1"
	props2 = [p for p in props if ("result.best.eval" in p and is_eval_correct(p["result.best.eval"])) or \
	                              ("result.successRate" in p and p["result.successRate"] == "1.0")]
	return len(props2)

def get_num_computed(filtered):
	return len(filtered)
def fun_successRates_full(filtered):
	if len(filtered) == 0:
		return "-"
	num_opt = get_num_optimal(filtered)
	return "{0}/{1}".format(str(num_opt), str(len(filtered)))
def fun_successRates(filtered):
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
def get_avg_fitness(props):
	vals = []
	for p in props:
		if "result.best.passedTestsRatio" in p:
			ratio = float(p["result.best.passedTestsRatio"])
			vals.append(ratio)
		elif p["result.best.eval"] == "-1":
			vals.append(1.0)
		elif p["result.best.eval"].isdigit(): # exclude cases like "Vector(1,0)".
			totalTests = float(p["totalTests"])
			ratio = (totalTests - float(p["result.best.eval"])) / totalTests
			vals.append(ratio)
	if len(vals) == 0:
		return "-"  # -1.0, -1.0
	else:
		return "%0.2f" % numpy.mean(vals)  # , numpy.std(vals)




def create_section_with_results(title, desc, props):
	assert isinstance(title, str)
	assert isinstance(desc, str)
	assert isinstance(props, list)
	# text = printer.text_table(props, dim_benchmarks.sort(), dim_method*dim_sa, fun1)
	# print(text)
	# print("\n\n")

	dim_benchmarks = Dim.from_data(props, lambda p: p["benchmark"])

	print("STATUS")
	text = printer.latex_table(props, dim_benchmarks.sort(), dim_method * dim_sa, get_num_computed)
	latex_status = printer.table_color_map(text, 0.0, 1, 10.0, "colorLow", "colorMedium", "colorHigh")
	print(text)
	print("\n\n")

	print("SUCCESS RATES")
	text = printer.latex_table(props, dim_benchmarks.sort(), dim_method * dim_sa, fun_successRates)
	latex_successRates = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
	print(text)
	print("\n\n")

	print("SUCCESS RATES (FULL INFO)")
	text = printer.latex_table(props, dim_benchmarks.sort(), dim_method * dim_sa, fun_successRates_full)
	print(text)
	print("\n\n")

	print("AVG BEST-OF-RUN FITNESS")
	text = printer.latex_table(props, dim_benchmarks.sort(), dim_method * dim_sa, get_avg_fitness)
	latex_avgBestOfRunFitness = printer.table_color_map(text, 0.6, 0.98, 1.0, "colorLow", "colorMedium", "colorHigh")
	print(text)
	print("\n\n")

	print("AVG TOTAL TESTS")
	text = printer.latex_table(props, dim_benchmarks.sort(), dim_method * dim_sa, get_avg_totalTests)
	latex_avgTotalTests = printer.table_color_map(text, 0.0, 1000.0, 2000.0, "colorLow", "colorMedium", "colorHigh")
	print(text)
	print("\n\n")

	print("AVG SIZES")
	text = printer.latex_table(props, dim_benchmarks.sort(), dim_method * dim_sa, get_stats_size)
	latex_sizes = printer.table_color_map(text, 0.0, 100.0, 200.0, "colorLow", "colorMedium", "colorHigh")
	print(text)
	print("\n\n")

	section = reporting.BlockSection(title, [])
	subsects = [("Status (correctly finished processes)", latex_status, reporting.color_scheme_red_r),
	            ("Success rates", latex_successRates, reporting.color_scheme_green),
	            ("Average best-of-run ratio of passed tests", latex_avgBestOfRunFitness, reporting.color_scheme_green),
	            ("Average sizes of $T_C$ (total tests in run)", latex_avgTotalTests, reporting.color_scheme_blue),
	            ("Average sizes of best of runs (number of nodes)", latex_sizes, reporting.color_scheme_yellow)]
	bl_desc = reporting.BlockLatex(desc + "\n")
	section.add(bl_desc)
	for title, table, cs in subsects:
		sub = reporting.BlockSubSection(title, [cs, reporting.BlockLatex(table + "\n")])
		section.add(sub)
	section.add(reporting.BlockLatex(r"\vspace{1cm}" + "\n"))
	return section








props_exp2 = load_correct_props(folders_exp2)
desc_exp2 = ""
props_exp3 = load_correct_props(folders_exp3)
desc_exp3 = r"""
Changes:
\begin{itemize}
\item LexicaseSteadyState uses Tournament ($k=7$) deselection.
\item GPR uses range [-100, 100].
\item Various improvements in the code.
\end{itemize}
"""












# -------- Creating nice LaTeX report of the above results --------


report = reporting.ReportPDF()
section1 = r"""\section{Initial experiments}
\subsection{Success rates}
\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.76, 0.98, 0.76} % colorHigh
\definecolor{colorHigh}{rgb}{0.66, 0.90, 0.66} % green
\begin{tabular}{ccccccc}
 & CDGP\_GP & CDGP\_Lexicase & CDGPconservative\_GP & CDGPconservative\_Lexicase & GPR\_GP & GPR\_Lexicase\\
sygus16/fg\_array\_search\_2.sl & \cellcolor{colorHigh!82.0!colorMedium}0.91 & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorLow!0.0!colorMedium}0.50 & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!78.0!colorMedium}0.11\\
sygus16/fg\_array\_search\_4.sl & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
sygus16/fg\_array\_search\_6.sl & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
sygus16/fg\_array\_search\_8.sl & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
sygus16/fg\_array\_sum\_2\_15.sl & \cellcolor{colorLow!80.0!colorMedium}0.10 & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorLow!78.0!colorMedium}0.11 & \cellcolor{colorLow!19.999999999999996!colorMedium}0.40 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
sygus16/fg\_array\_sum\_4\_15.sl & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
sygus16/fg\_array\_sum\_6\_15.sl & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
sygus16/fg\_array\_sum\_8\_15.sl & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
sygus16/fg\_max2.sl & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorHigh!100.0!colorMedium}1.00\\
sygus16/fg\_max4.sl & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorHigh!100.0!colorMedium}1.00 & \cellcolor{colorLow!78.0!colorMedium}0.11 & \cellcolor{colorLow!0.0!colorMedium}0.50 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!80.0!colorMedium}0.10\\
sygus16/fg\_max6.sl & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!14.000000000000002!colorMedium}0.43 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
sygus16/fg\_max8.sl & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
\end{tabular}
\vspace{1cm}

"""
report.add(reporting.BlockLatex(section1))

section2 = create_section_with_results("Initial experiments 2", desc_exp2, props_exp2)
section3 = create_section_with_results("Initial experiments 3", desc_exp3, props_exp3)
report.add(section2)
report.add(section3)



print("\n\nREPORT:\n")
#print(report.apply())
report.save_and_compile("cdgp_results.tex")