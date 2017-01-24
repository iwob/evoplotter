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

props = utils.load_properties_dirs(folders)


# for p in props[:3]:
# 	print(str(p))



def p_correctness(p):
	return "result.best.eval" in p and \
	       "result.best.isOptimal" in p and \
			"smtgp.exception.stacktrace" not in p
props = [p for p in props if p_correctness(p)]





def p_onlyWithHoles(p):
	return "smtgp.holesConsts" in p or \
           "smtgp.holesVars" in p


def p_optBisecting(p):
	return p["smtgp.optimizationMode"] == "bisecting"
def p_optSolver(p):
	return p["smtgp.optimizationMode"] == "solver"


def p_cv_timed(p):
	return p["smtgp.useConstantProvider"] == "true" and \
         p["smtgp.useInputVarsAsTerminals"] == "true" and \
         "smtgp.holesConsts" not in p and \
         "smtgp.holesVars" not in p and \
         "maxTime" in p and \
		 p["populationSize"] == "250"
def p_cv(p):
	return p["smtgp.useConstantProvider"] == "true" and \
         p["smtgp.useInputVarsAsTerminals"] == "true" and \
         "smtgp.holesConsts" not in p and \
         "smtgp.holesVars" not in p and \
         "maxTime" not in p and \
		 p["populationSize"] == "250"
def p_cv5000(p):
	return p["smtgp.useConstantProvider"] == "true" and \
         p["smtgp.useInputVarsAsTerminals"] == "true" and \
         "smtgp.holesConsts" not in p and \
         "smtgp.holesVars" not in p and \
         "maxTime" not in p and \
		 p["populationSize"] == "5000"
def p_cV(p):
	return p["smtgp.useConstantProvider"] == "true" and \
         p["smtgp.useInputVarsAsTerminals"] == "false" and \
         "smtgp.holesConsts" not in p and \
         "smtgp.holesVars" in p
def p_Cv(p):
	return p["smtgp.useConstantProvider"] == "false" and \
         p["smtgp.useInputVarsAsTerminals"] == "true" and \
         "smtgp.holesConsts" in p and \
         "smtgp.holesVars" not in p
def p_CV(p):
	return p["smtgp.useConstantProvider"] == "false" and \
         p["smtgp.useInputVarsAsTerminals"] == "false" and \
         "smtgp.holesConsts" in p and \
         "smtgp.holesVars" in p


def p_fill(p):
	return "smtgp.fillHoles" in p and p["smtgp.fillHoles"] == "true"
def p_notFill(p):
	return "smtgp.fillHoles" in p and p["smtgp.fillHoles"] == "false"


def p_bench_keijzer12(p):
	return p["smtgp.pathTests"] == "data/int/keijzer12.csv"
def p_bench_koza1(p):
	return p["smtgp.pathTests"] == "data/int/koza1.csv"
def p_bench_koza1_p(p):
	return p["smtgp.pathTests"] == "data/int/koza1-p.csv"
def p_bench_koza1_d2(p):
	return p["smtgp.pathTests"] == "data/int/koza1-2d.csv"
def p_bench_koza1_p_d2(p):
	return p["smtgp.pathTests"] == "data/int/koza1-p-2d.csv"


def p_optimal(p):
	return p["result.best.isOptimal"] == "1"






plotter.set_latex(True)


dim_benchmark = Dim([Config(r"\texttt{Keijzer12}", p_bench_keijzer12, 49),
                     Config(r"\texttt{Koza1}", p_bench_koza1, 11),
                     Config(r"\texttt{Koza1-p}", p_bench_koza1_p, 11),
                     Config(r"\texttt{Koza1-2D}", p_bench_koza1_d2, 49),
                     Config(r"\texttt{Koza1-p-2D}", p_bench_koza1_p_d2, 49)])

conf_cv = Config(r"$GP$", p_cv)
conf_cv_timed = Config(r"$GP_T$", p_cv_timed)
conf_cv_5000 = Config(r"$GP_5000$", p_cv5000)

dim_fill = Dim([Config(r"$EPS$-$L", p_fill),
                Config(r"$EPS$-$B", p_notFill)])
dim_usedHoles_ho = Dim([Config( "C$", p_Cv),
                        Config( "V$", p_cV),
                        Config( "{CV}$", p_CV)])
dim_optMode = Dim([Config("optSolver", p_optSolver),
                   Config("optBisecting", p_optBisecting)])








def get_num_optimal(props):
	props2 = [p for p in props if p["result.best.isOptimal"] == "1"]
	return len(props2)
def get_stats_fitness(props):
	fits = [float(vp["result.best.eval"]) for vp in props]
	if len(fits) == 0:
		return -1.0, -1.0
	else:
		return numpy.mean(fits), numpy.std(fits)
def get_stats_duration(props):
	fits = [float(vp["result.totalTimeSystem"]) / 1000 for vp in props]
	if len(fits) == 0:
		return -1.0, -1.0
	else:
		return numpy.mean(fits), numpy.std(fits)
def get_gens_of_best(props):
	return [p["result.best.generation"] for p in props if p["result.best.isOptimal"] == "1"]
def get_sum(props, key):
	return sum([float(p[key]) for p in props])

def round_tuple_str(tup):
	return ("(%0.2f" % tup[0]) + ", " + ("%0.2f" % tup[1]) + ")"


def print_stats_filtered(filtered, show_solver_stats = True):
	"""Prints run statistics for a provided list of results."""
	print("number of results: " + str(len(filtered)))
	if len(filtered) != 0:
		num_opt = get_num_optimal(filtered)
		print("optimal gen nums: " + str(get_gens_of_best(filtered)))
		print("optimal: " + str(num_opt))
		print("success rate: " + str(round(float(num_opt) / float(len(filtered)), 3)))
		print("(avg, std) fitness of bestOfRun: " + round_tuple_str(get_stats_fitness(filtered)))
		print("(avg, std) totalTimeSystem [s]: " + round_tuple_str(get_stats_duration(filtered)))
		print("")

		if show_solver_stats:
			evalSolver = get_sum(filtered, "result.stats.evaluatedSolver")
			evalSolverUnknown = get_sum(filtered, "result.stats.evaluatedSolverUnknown")
			evalSolverTimeout = get_sum(filtered, "result.stats.evaluatedSolverTimeout")
			percentUnsuccessful = None
			if evalSolver > 0:
				percentUnsuccessful = '%.4f' % (float(evalSolverTimeout + evalSolverUnknown) / float(evalSolver))
			print("evalSolverTotal: " + str(evalSolver))
			print("evalSolverUnknown: " + str(evalSolverUnknown))
			print("evalSolverTimeout: " + str(evalSolverTimeout))
			print("ratio unsuccessful: " + str(percentUnsuccessful))


def print_table(props, dim_rows, dim_cols):
	def fun0(filtered):
		return str(len(filtered))
	textStatus = printer.latex_table(props, dim_rows, dim_cols, fun0, "Status:")
	print(textStatus)
	print("\n\n")

	def fun1(filtered):
		if len(filtered) == 0:
			return None
		num_opt = get_num_optimal(filtered)
		# return "{0}/{1}".format(str(num_opt), str(len(filtered)))
		return "{0}".format(str(num_opt))
	textNumOptimal = printer.latex_table(props, dim_rows, dim_cols, fun1, "Num optimal:")
	print(textNumOptimal)
	print("\n\n")


	def fun2(filtered):
		if len(filtered) == 0:
			return None
		avgFit = round(get_stats_fitness(filtered)[0], 2)
		return "{0}".format(str(avgFit))
	textAvgFitness = printer.latex_table(props, dim_rows, dim_cols, fun2, "Avg fitness:")
	print(textAvgFitness)
	print("\n\n")


	def fun3(filtered):
		if len(filtered) == 0:
			return None
		avg_time = round(get_stats_duration(filtered)[0], 1)
		return "{0}".format(str(avg_time))
	textAvgRuntime = printer.latex_table(props, dim_rows, dim_cols, fun3, "Avg runtime:")
	print(textAvgRuntime)
	print("\n\n")


	def fun4(filtered):
		if len(filtered) == 0:
			return None
		evalSolver = get_sum(filtered, "result.stats.evaluatedSolver")
		evalSolverUnknown = get_sum(filtered, "result.stats.evaluatedSolverUnknown")
		evalSolverTimeout = get_sum(filtered, "result.stats.evaluatedSolverTimeout")
		if evalSolver > 0:
			percentUnsuccessful = float(evalSolverTimeout + evalSolverUnknown) / float(evalSolver)
			return str(round(percentUnsuccessful,3))
		else:
			return None
	textRatioOfUnknowns = printer.latex_table(props, dim_rows, dim_cols, fun4, "Ratio of unknowns:")
	print(textRatioOfUnknowns)
	print("\n\n")

	report = reporting.ReportPDF()
	section1 = reporting.BlockSection("Experiments", [])
	subsects = [("Status (correctly finished processes)", textStatus, reporting.color_scheme_red_r),
	            ("Number of optimal solutions (max=100)", textNumOptimal, reporting.color_scheme_green),
	            ("Average fitness", textAvgFitness, reporting.color_scheme_green),
	            ("Average runtime", textAvgRuntime, reporting.color_scheme_blue),
	            ("Ratio of unknowns", textRatioOfUnknowns, reporting.color_scheme_yellow)]
	for title, table, cs in subsects:
		sub = reporting.BlockSubSection(title, [cs, reporting.BlockLatex(table + "\n")])
		section1.add(sub)
	report.add(section1)
	report.save_and_compile("eps_results.tex")



def print_stats_unsuccesful(filtered):
	evalSolver = get_sum(filtered, "result.stats.evaluatedSolver")
	evalSolverUnsat = get_sum(filtered, "result.stats.evaluatedSolverUnsat")
	evalSolverUnknown = get_sum(filtered, "result.stats.evaluatedSolverUnknown")
	evalSolverTimeout = get_sum(filtered, "result.stats.evaluatedSolverTimeout")
	percentUnsuccessful = None
	if evalSolver > 0:
		percentUnsuccessful = '%.4f' % (float(evalSolverTimeout + evalSolverUnknown + evalSolverUnsat) / float(evalSolver))
	text = "numRuns: " + str(len(filtered)) + "\n"
	text += "evalSolverTotal: " + str(evalSolver) + "\n"
	text += "evalSolverUnsat: " + str(evalSolverUnsat) + "\n"
	text += "evalSolverUnknown: " + str(evalSolverUnknown) + "\n"
	text += "evalSolverTimeout: " + str(evalSolverTimeout) + "\n"
	text += "% unsuccessful: " + str(percentUnsuccessful) + "\n"
	text += "(avg, std) fitness of bestOfRun: " + ('(%.2f, %.2f)' % get_stats_fitness(filtered)) + "\n"
	text += "(avg, std) totalTimeSystem [s]: " + ('(%.1f, %.1f)' % get_stats_duration(filtered)) + "\n"
	return text


def print_stats(props, dim):
	"""Loops over all possible configs and filters results depending on the configs predicates. For each config file standard run stats will be printed."""
	for config in dim:
		filtered = config.filter_props(props)
		str_vname = config.get_caption()
		print("(*) VARIANT: " + str_vname)
		print_stats_filtered(filtered)
		print("\n\n\n\n")


def print_opt_mode_stats(props):
	"""Collects aggregated statistics about unsuccessful solver evaluations."""
	props = [p for p in props if p_onlyWithHoles(p)]
	text = printer.text_listing(props, dim_optMode, print_stats_unsuccesful)
	print(text)


def search_differing_evals(props):
	"""Searching for runs with differing result.best.eval and result.best.evalNormally."""
	for p in props:
		eval = int(p["result.best.eval"])
		evalNormally = int(p["result.best.evalNormally"])
		if eval != evalNormally:
			sol = p["result.best"]
			sol_fill = p["result.best.holesFilled"]
			print("File: " + p["thisFileName"])
			if "smtgp.holesConsts" in p and "smtgp.holesVars" in p:
				print("variant: CV")
			elif "smtgp.holesConsts" in p and "smtgp.holesVars" not in p:
				print("variant: Cv")
			elif "smtgp.holesConsts" not in p and "smtgp.holesVars" in p:
				print("variant: cV")
			else:
				print("variant: cv")
			if p["smtgp.fillHoles"] == "true":
				print("fillHoles: 1")
			else:
				print("fillHoles: 0")
			if p_optSolver(p):
				print("optSolver")
			else:
				print("optBisecting")
			print("solution: ".ljust(26) + sol)
			print("solution filled holes: ".ljust(26) + sol_fill)
			print("eval: " + str(eval))
			print("evalNormally: " + str(evalNormally))
			print("\n\n")


def print_optimals(props):
	props = [p for p in props if p_optimal(p)]
	dim = Dim(conf_cv_timed)
	dim += dim_usedHoles_ho * dim_fill
	def print_optimal(p):
		opt = p["result.best"]
		optFilled = p["result.best.holesFilled"]
		return "Found optimal:\t" + optFilled + "\n"
	text = printer.text_listing(props, dim, print_optimal)
	print(text)
	printer.save_to_file(text, "figures/optimals.txt")


def print_optimals_per_benchmark(props):
	props = [p for p in props if p_optimal(p)]
	dim_variants = Dim([conf_cv, conf_cv_timed, conf_cv_5000]) + dim_usedHoles_ho * dim_fill
	dim = dim_benchmark * dim_variants
	def print_optimal(p):
		optFilled = p["result.best.holesFilled"]
		return optFilled + "\n"
	text = printer.text_listing(props, dim, print_optimal, is_fun_single_prop=True)
	print(text)
	printer.save_to_file(text, "figures/optimals.txt")


def draw_boxplots(props):
	dim_variants = Dim([conf_cv, conf_cv_timed, conf_cv_5000])
	dim_variants += dim_fill * dim_usedHoles_ho
	plotter.compare_fitness_on_benchmarks(props, dim_benchmark, dim_variants, use_latex=True)


def draw_fitness_progression(props, plot_individual_runs=True):
	dim_variants = Dim(conf_cv)
	dim_variants += dim_fill * dim_usedHoles_ho
	plotter.plot_fitness_progression_on_benchmarks(props, dim_benchmark, dim_variants, plot_individual_runs=plot_individual_runs)








props = [p for p in props if p_optSolver(p) or p_cv(p)]  #p_cv added because by mistake they have set 'bisecting' flag.




# dim = dim_benchmark * dim_usedHoles_ho * dim_fill
# print_stats(props, dim)




# Printing a table with results.
dim_variants = Dim([conf_cv, conf_cv_timed, conf_cv_5000])
dim_variants += dim_fill * dim_usedHoles_ho
# print_table(props, dim_variants, dim_benchmark)
print_table(props, dim_benchmark, dim_variants)




# print_opt_mode_stats(props)
# print_optimals(props)
# print_optimals_per_benchmark(props)
# search_differing_evals(props)




# draw_boxplots(props)
# draw_fitness_progression(props, plot_individual_runs=True)
# draw_fitness_progression(props, plot_individual_runs=False)
