from evoplotter import utils
from evoplotter import printer
from evoplotter import reporting
from evoplotter.dims import *
import numpy


def load_correct_props(folders, name = ""):
    props = utils.load_properties_dirs(folders, exts=[".txt"])

    print("\n*** Loading props: " + name)

    # Printing names of files which finished with error status.
    print("Files with error status:")
    props_errors = [p for p in props if "status" not in p or (p["status"] != "completed" and p["status"] != "initialized")]
    for p in props_errors:
        if "thisFileName" in p:
            print(p["thisFileName"])
        else:
            print("'thisFileName' not specified! Printing content instead: " + str(p))

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
def p_Generational(p):
    return p["searchAlgorithm"] == "Lexicase" or  p["searchAlgorithm"] == "GP"
def p_SteadyState(p):
    return p["searchAlgorithm"] == "LexicaseSteadyState" or  p["searchAlgorithm"] == "GPSteadyState"
def p_sel_lexicase(p):
    return p["searchAlgorithm"] == "LexicaseSteadyState" or p["searchAlgorithm"] == "Lexicase"
def p_sel_gp(p):
    return p["searchAlgorithm"] == "GPSteadyState" or p["searchAlgorithm"] == "GP"


d1 = "other/"
d2 = "sygus16/"
benchmarks_simple_names = {d1 + "ArithmeticSeries3.sl": "IsSeries3",
                           d1 + "ArithmeticSeries4.sl": "IsSeries4",
                           d1 + "CountPositive2.sl": "CountPos2",
                           d1 + "CountPositive3.sl": "CountPos3",
                           d1 + "Median3.sl": "Median3",
                           d1 + "Range3.sl": "Range3",
                           d1 + "SortedAscending4.sl": "IsSorted4",
                           d2 + "fg_array_search_2.sl": "Search2",
                           d2 + "fg_array_search_4.sl": "Search4",
                           d2 + "fg_array_sum_2_15.sl": "Sum2",
                           d2 + "fg_array_sum_4_15.sl": "Sum4",
                           d2 + "fg_max2.sl": "Max2",
                           d2 + "fg_max4.sl": "Max4"}


dim_method = Dim([Config("CDGP non-conservative", p_method0),
                  Config("CDGP conservative", p_method1),
                  Config("GPR", p_method2)])
dim_sa = Dim([Config("GP", p_GP),
              Config("GPSS", p_GPSteadyState),
              Config("Lex", p_Lexicase),
              Config("LexSS", p_LexicaseSteadyState)])
dim_ea_type = Dim([Config("Gener.", p_Generational),
                   Config("SteadySt.", p_SteadyState)])
dim_sel = Dim([Config("$Tour$", p_sel_gp),
               Config("$Lex$", p_sel_lexicase)])
# dim_sa = Dim([Config("$CDGP$", p_GP),
# 			    Config("$CDGP^{ss}$", p_GPSteadyState),
#               Config("$CDGP_{lex}$", p_Lexicase),
#               Config("$CDGP_{lex}^{ss}$", p_LexicaseSteadyState)])


def is_optimal_solution(e):
    return e == "-1" or e[:7] == "List(-1" or e[:9] == "Vector(-1"

def get_num_optimal(props):
    props2 = [p for p in props if ("result.best.eval" in p and is_optimal_solution(p["result.best.eval"]))] #or \
#("result.successRate" in p and p["result.successRate"] == "1.0")]
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
        return "%0.1f" % numpy.mean(vals)#, numpy.std(vals)
def get_avg_totalTests(props):
    vals = [float(p["totalTests"]) for p in props]
    if len(vals) == 0:
        return "-"  # -1.0, -1.0
    else:
        return "%0.1f" % numpy.mean(vals)  # , numpy.std(vals)
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
def get_avg_runtime_helper(vals):
    if len(vals) == 0:
        return "-1"  # -1.0, -1.0
    else:
        return "%0.1f" % numpy.mean(vals)  # , numpy.std(vals)
def get_avg_runtimeOnlySuccessful(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(p["result.totalTimeSystem"]) / 1000.0 for p in props if is_optimal_solution(p["result.best.eval"])]
        return get_avg_runtime_helper(vals)
def get_avg_runtime(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(p["result.totalTimeSystem"]) / 1000.0 for p in props]
        return get_avg_runtime_helper(vals)
def get_avg_generation(props):
    if len(props) == 0:
        return "-"
    vals = [float(p["result.best.generation"]) for p in props]
    return "%0.1f" % numpy.mean(vals)  # , numpy.std(vals)
def get_avg_generationSuccessful(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(p["result.best.generation"]) for p in props if is_optimal_solution(p["result.best.eval"])]
        if len(vals) == 0:
            return "-1"  # -1.0, -1.0
        else:
            return "%0.1f" % numpy.mean(vals)  # , numpy.std(vals)
def get_avg_runtimePerProgram(props):
    if len(props) == 0:
        return "-"  # -1.0, -1.0
    avgGen = float(get_avg_generation(props))
    avgRuntime = float(get_avg_runtime(props))
    populationSize = float(props[0]["populationSize"])
    if props[0]["searchAlgorithm"] == "GPSteadyState" or \
       props[0]["searchAlgorithm"] == "LexicaseSteadyState":
        approxNumPrograms = populationSize + avgGen
    else:
        approxNumPrograms = populationSize * avgGen
    approxTimePerProgram = avgRuntime / approxNumPrograms
    return "%0.3f" % approxTimePerProgram
def get_sum_solverRestarts(props):
    if len(props) == 0:
        return "-"
    vals = [int(p["doneSolverRetries"]) for p in props if "doneSolverRetries" in p]
    if len(vals) == 0:
        return "0"
    else:
        return str(numpy.sum(vals))


def create_section_with_results(title, desc, folders, numRuns=10, use_bench_simple_names=True):
    assert isinstance(title, str)
    assert isinstance(desc, str)
    assert isinstance(folders, list)
    print("\n*** Processing: {0}***".format(title))
    for f in folders:
        print(f)
    if folders is None or len(folders) == 0:
        return None

    props = load_correct_props(folders, name=title)

    def post(s):
        return s.replace("{ccccccccccccc}", "{rrrrrrrrrrrrr}").replace("{rrr", "{lrr").replace(r"\_{lex}", "_{lex}").replace(r"\_{7}", "_{7}").replace("other/", "").replace("sygus16/", "")


    dim_benchmarks = Dim.from_data(props, lambda p: p["benchmark"])
    if use_bench_simple_names:
        configs = [Config(benchmarks_simple_names[c.get_caption()], c.filters[0][1]) for c in dim_benchmarks.configs]
        dim_benchmarks = Dim(configs)
        dim_benchmarks.sort()
    dim_cols = dim_method * dim_ea_type * dim_sel
    # dim_cols = dim_method * dim_sa

    print("STATUS")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_num_computed, layered_headline=True))
    latex_status = printer.table_color_map(text, 0.0, numRuns/2, numRuns, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    print("SUCCESS RATES")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, fun_successRates, layered_headline=True))
    latex_successRates = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    # print("SUCCESS RATES (FULL INFO)")
    # text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, fun_successRates_full, layered_headline=True))
    # print(text + "\n\n")

    print("AVG BEST-OF-RUN FITNESS")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_avg_fitness, layered_headline=True))
    latex_avgBestOfRunFitness = printer.table_color_map(text, 0.6, 0.98, 1.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    print("AVG TOTAL TESTS")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_avg_totalTests, layered_headline=True))
    latex_avgTotalTests = printer.table_color_map(text, 0.0, 1000.0, 2000.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    print("AVG RUNTIME")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_avg_runtime, layered_headline=True))
    latex_avgRuntime = printer.table_color_map(text, 0.0, 1800.0, 3600.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    print("AVG RUNTIME (SUCCESSFUL)")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_avg_runtimeOnlySuccessful, layered_headline=True))
    latex_avgRuntimeOnlySuccessful = printer.table_color_map(text, 0.0, 1800.0, 3600.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    print("AVG RUNTIME PER PROGRAM")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_avg_runtimePerProgram, layered_headline=True))
    latex_avgRuntimePerProgram = printer.table_color_map(text, 0.01, 1.0, 2.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    print("AVG GENERATION")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_avg_generation, layered_headline=True))
    latex_avgGeneration = printer.table_color_map(text, 0.0, 50.0, 100.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    print("AVG GENERATION (SUCCESSFUL)")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_avg_generationSuccessful, layered_headline=True))
    latex_avgGenerationSuccessful = printer.table_color_map(text, 0.0, 50.0, 100.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    print("AVG SIZES")
    text = post(printer.latex_table(props, dim_benchmarks.sort(), dim_cols, get_stats_size, layered_headline=True))
    latex_sizes = printer.table_color_map(text, 0.0, 100.0, 200.0, "colorLow", "colorMedium", "colorHigh")
    # print(text + "\n\n")

    section = reporting.Section(title, [])
    subsects = [("Status (correctly finished processes)", latex_status, reversed(reporting.color_scheme_red)),
                ("Success rates", latex_successRates, reporting.color_scheme_green),
                ("Average best-of-run ratio of passed tests", latex_avgBestOfRunFitness, reporting.color_scheme_green),
                ("Average sizes of $T_C$ (total tests in run)", latex_avgTotalTests, reporting.color_scheme_blue),
                ("Average runtime [s]", latex_avgRuntime, reporting.color_scheme_violet),
                ("Average runtime (only successful) [s]", latex_avgRuntimeOnlySuccessful, reporting.color_scheme_violet),
                ("Average generation (optimal and nonoptimal bestOfRuns)", latex_avgGeneration, reporting.color_scheme_teal),
                ("Average generation (only optimal bestOfRuns)", latex_avgGenerationSuccessful, reporting.color_scheme_teal),
                ("Approximate average runtime per program [s]", latex_avgRuntimePerProgram, reporting.color_scheme_brown),
                ("Average sizes of best of runs (number of nodes)", latex_sizes, reporting.color_scheme_yellow)]
    bl_desc = reporting.BlockLatex(desc + "\n")
    section.add(bl_desc)
    for title, table, cs in subsects:
        if isinstance(cs, reporting.ColorScheme3):
            cs = cs.toBlockLatex()
        sub = reporting.Subsection(title, [cs, reporting.BlockLatex(table + "\n")])
        section.add(sub)
    section.add(reporting.BlockLatex(r"\vspace{1cm}" + "\n"))
    return section


def print_time_bounds_for_benchmarks(props):
    dim_benchmarks = Dim.from_data(props, lambda p: p["benchmark"])
    print("\nTime bounds for benchmarks:")
    dim_benchmarks.sort()
    scalaMap = "Map("
    scalaMapEntries = []
    for conf in dim_benchmarks:
        filtered = conf.filter_props(props)
        vals = [float(p["result.totalTimeSystem"]) / 1000.0 for p in filtered if is_optimal_solution(p["result.best.eval"])]
        if len(vals) == 0:
            result = "No optimal solutions!"
            suggestion = 1800
        else:
            vals = [p for p in vals if p > 0.0]
            avg = int(numpy.mean(vals))
            result = int(numpy.min(vals)), avg, int(numpy.max(vals))
            result = "{0}, {1}, {2}".format(str(result[0]), str(result[1]), str(result[2]))
            suggestion = avg if avg < 1800 else 1800
        print(conf.get_caption() + ": " + str(suggestion) + " \t\t[{0}]".format(result))
        scalaMapEntries.append('"{0}"->"{1}000"'.format(conf.get_caption(), str(suggestion)))
    scalaMap += ", ".join(scalaMapEntries) + ")"
    print("Scala Map: " + scalaMap)





folders_exp2 = ["res1_rms", "res1_first", "res1_gprlexss"]
folders_exp3 = ["res1_tournDeselection", "res1_tournDeselection_gprNew"]
folders_exp4 = ["res1_cdgpOneTest", "res1_gprManyTest"]
folders_expEvalsFINAL = ["resFinal_GenEvals", "resFinal_SSEvals", "resFinal_spare_CDGPEvals", "resFinal_myspare"]
folders_expTimedFINAL = ["resFinal_GenTimed", "resFinal_spare"]
folders_expEvalsFINAL2 = ["resFinal2_evals"]
folders_expTimedFINAL2 = ["resFinal2_timed"]


name_exp2 = "Initial experiments 2"
desc_exp2 = ""

name_exp3 = "Initial experiments 3"
desc_exp3 = r"""
Changes:
\begin{itemize}
\item LexicaseSteadyState uses Tournament ($k=7$) deselection.
\item GPR uses range [-100, 100].
\item Various improvements in the code.
\end{itemize}
"""

name_exp4 = "Initial experiments 4"
desc_exp4 = r"""
The same as experiment 3, apart from:
\begin{itemize}
\item GPR allowed to add many tests in one iteration.
\item CDGP limited to only 1 test per iteration.
\end{itemize}
"""

name_expEvalsFINAL = "Final Experiments (stop: number of iterations)"
desc_expEvalsFINAL = r"""
Important information:
\begin{itemize}
\item All configurations, unless specified otherwise, has \textbf{population size 500}, and \textbf{number of iterations 100}.
\item GPR allowed to add many tests in one iteration.
\item CDGP allowed to add many tests in one iteration.
\item GPR uses range [-100, 100] and a population size 1000.
\item LexicaseSteadyState uses Tournament ($k=7$) deselection.
\end{itemize}
"""

name_expTimedFINAL = "Final Experiments (stop: time limit)"
desc_expTimedFINAL = r"""
Important information:
\begin{itemize}
\item Experiments have time limits according to the following table:\\\\
\begin{tabular}{lcc}
\hline
Benchmark & Time limit & [min, avg, max] for optimal solutions\\
\hline
other/ArithmeticSeries3.sl & 1639 	&	[2, 1639, 71529]\\
other/CountPositive2.sl & 531 	&	[2, 531, 15863]\\
other/CountPositive3.sl & 1800 &		[6, 2034, 66430]\\
other/Median3.sl & 953 	&	[5, 953, 29819]\\
other/Range3.sl & 1800 	&	[5, 2234, 38703]\\
other/SortedAscending4.sl & 1800 	&	[5, 2835, 74128]\\
sygus16/fg\_array\_search\_2.sl & 1800 	&	[4, 1951, 51400]\\
sygus16/fg\_array\_search\_4.sl & 1800 	&	[No optimal solutions!]\\
sygus16/fg\_array\_sum\_2\_15.sl & 1800 	&	[7, 3068, 72045]\\
sygus16/fg\_array\_sum\_4\_15.sl & 1800 	&	[6069, 22675, 50260]\\
sygus16/fg\_max2.sl: & 158 	&	[1, 158, 1692]\\
sygus16/fg\_max4.sl: & 971 	&	[4, 971, 47556]\\
\hline
\end{tabular}
\end{itemize}
"""

name_expEvalsFINAL2 = "Final Experiments (stop: number of iterations), better solver interactions"
desc_expEvalsFINAL2 = r"""
This configation is the same as Final Experiments (stop: number of iterations), only with
better mechanism of finding correct outputs of test cases.
"""

name_expTimedFINAL2 = "Final Experiments (stop: time limit)"
desc_expTimedFINAL2 = r"""
This configation is the same as Final Experiments (stop: time limit), only with
better mechanism of finding correct outputs of test cases.
"""

# print_time_bounds_for_benchmarks(props_expEvalsFINAL)

# Uncomment this to print names of files with results of a certain configuration
# print("(***) props_expEvalsFINAL:")
# for p in props_expEvalsFINAL:
#     if p["method"] == "2" and p["searchAlgorithm"] == "LexicaseSteadyState" and p["benchmark"] == "sygus16/fg_array_search_4.sl":
#         print(p["thisFileName"] + ", " + p["benchmark"])







# -------- Creating nice LaTeX report of the above results --------


report = reporting.ReportPDF(geometry_params = "[paperwidth=45cm, paperheight=40cm, margin=0.3cm]")
section1 = r"""\section{Initial experiments}
\subsection{Success rates}
\definecolor{colorLow}{rgb}{1.0, 1.0, 1.0} % white
\definecolor{colorMedium}{rgb}{0.76, 0.98, 0.76} % colorHigh
\definecolor{colorHigh}{rgb}{0.66, 0.90, 0.66} % green
\begin{tabular}{lcccccc}
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
# report.add(reporting.BlockLatex(section1))

sects = [#create_section_with_results(name_exp2, desc_exp2, folders_exp2),
         #create_section_with_results(name_exp3, desc_exp3, folders_exp3),
         #create_section_with_results(name_exp4, desc_exp4, folders_exp4),
         #create_section_with_results(name_expEvalsFINAL, desc_expEvalsFINAL, folders_expEvalsFINAL, numRuns=30),
         #create_section_with_results(name_expTimedFINAL, desc_expTimedFINAL, folders_expTimedFINAL, numRuns=30),
         create_section_with_results(name_expEvalsFINAL2, desc_expEvalsFINAL2, folders_expEvalsFINAL2, numRuns=15),
         create_section_with_results(name_expTimedFINAL2, desc_expTimedFINAL2, folders_expTimedFINAL2, numRuns=15)]
for s in sects:
    if s is not None:
        report.add(s)



print("\n\nGenerating PDF report ...")
report.save_and_compile("cdgp_results.tex")
