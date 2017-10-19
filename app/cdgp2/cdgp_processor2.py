from src import utils
from src import plotter
from src import printer
from src import reporting
from src.dims import *
import numpy as np


# This processor is to be used for exp3 onward.


def load_correct_props(folders, name = ""):
    props_cdgpError = utils.load_properties_dirs(folders, exts=[".cdgp.error"])
    props0 = utils.load_properties_dirs(folders, exts=[".cdgp", ".cvc4_head_cegqi", ".cvc4_cegqi", ".eusolver"])
    all_logs = utils.load_properties_dirs(folders, exts=[".cdgp", ".txt", ".cvc4_head_cegqi", ".cvc4_cegqi", ".eusolver"])

    print("\nLoading props: " + name)

    def is_correct(p):
        return "status" in p and (p["status"] == "completed" or p["status"] == "initialized") and "benchmark" in p

    # Printing names of files which finished with error status or are incomplete.
    props_errors = [p for p in props0 if not is_correct(p)]
    if len(props_errors) > 0:
        print("Files with error status:")
    for p in props_errors:
        if "thisFileName" in p:
            print(p["thisFileName"])
        else:
            print("'thisFileName' not specified! Printing content instead: " + str(p))

    # Filtering props so only correct ones are left
    props = [p for p in props0 if is_correct(p)]
    print("Loaded: {0} correct property files, {1} incorrect; All log files: {2}".format(len(props), len(props_errors), len(all_logs)))
    print("Runs that ended with error: {0}".format(len(props_cdgpError)))
    return props


def produce_status_matrix(dim, props):
    """Generates a status data in the form of a python list. It can be
    later used to retry missing runs.
    
    :param dim: (Dimension) dimensions no which data are to be divided.
    :param props: (dict[str,str]) properties files.
    :return: (str) Python code of a list containing specified data.
    """
    text = "["
    for config in dim:
        numRuns = len(config.filter_props(props))
        text += "({0}, {1}), ".format(config.stored_values, numRuns)
    return text + "]"



def p_GP(p):
    return p["searchAlgorithm"] == "GP"
def p_GPSteadyState(p):
    return p["searchAlgorithm"] == "GPSteadyState"
def p_Lexicase(p):
    return p["searchAlgorithm"] == "Lexicase"
def p_LexicaseSteadyState(p):
    return p["searchAlgorithm"] == "LexicaseSteadyState"
def p_testsRatioSel(p):
    if p["method"] == "CDGP":
        return p["CDGPtestsRatio"]
    elif p["method"] == "GPR":
        return p["GPRtestsRatio"]
    else:
        raise Exception("Unknown method!")
def p_testsRatio(value):
    return lambda p, value=value: p_testsRatioSel(p) == value
def p_method_cdgp(p):
    return p["method"] == "CDGP"
def p_method_gpr(p):
    return p["method"] == "GPR"
def p_method_for(name):
    return lambda p, name=name: p["method"] == name
def p_Generational(p):
    return p["searchAlgorithm"] == "Lexicase" or  p["searchAlgorithm"] == "GP"
def p_SteadyState(p):
    return p["searchAlgorithm"] == "LexicaseSteadyState" or  p["searchAlgorithm"] == "GPSteadyState"
def p_sel_lexicase(p):
    return p["searchAlgorithm"] == "LexicaseSteadyState" or p["searchAlgorithm"] == "Lexicase"
def p_sel_tourn(p):
    return p["searchAlgorithm"] == "GPSteadyState" or p["searchAlgorithm"] == "GP"


d1 = "benchmarks/LIA/cdgp_paper17/other/"
d2 = "benchmarks/LIA/cdgp_paper17/"
benchmarks_simple_names = {d1 + "ArithmeticSeries3.sl": "IsSeries3",
                           d1 + "ArithmeticSeries4.sl": "IsSeries4",
                           d1 + "CountPositive2.sl": "CountPos2",
                           d1 + "CountPositive3.sl": "CountPos3",
                           d1 + "CountPositive4.sl": "CountPos4",
                           d1 + "Median3.sl": "Median3",
                           d1 + "Range3.sl": "Range3",
                           d1 + "SortedAscending4.sl": "IsSorted4",
                           d1 + "SortedAscending5.sl": "IsSorted5",
                           d2 + "fg_array_search_2.sl": "Search2",
                           d2 + "fg_array_search_3.sl": "Search3",
                           d2 + "fg_array_search_4.sl": "Search4",
                           d2 + "fg_array_sum_2_15.sl": "Sum2",
                           d2 + "fg_array_sum_3_15.sl": "Sum3",
                           d2 + "fg_array_sum_4_15.sl": "Sum4",
                           d2 + "fg_max2.sl": "Max2",
                           d2 + "fg_max4.sl": "Max4",
                           "benchmarks/NIA/rsconf.sl": "rsconf"}

dim_true = Dim(Config("All benchmarks", lambda p: True, method=None))
dim_method = Dim([
    Config("CDGP", p_method_for("CDGP"), method="CDGP"),
    Config("GPR", p_method_for("GPR"), method="GPR")
])
dim_methodCDGP = Dim([
    Config("CDGP", p_method_for("CDGP"), method="CDGP")
])
dim_methodGPR = Dim([
    Config("GPR", p_method_for("GPR"), method="GPR")
])
dim_methodFormal = Dim([
    Config("EUSOLVER", p_method_for("eusolver"), method="eusolver"),
    Config("CVC4_cegqi", p_method_for("cvc4_head_cegqi"), method="cvc4_head_cegqi"),
])
dim_sa = Dim([
    Config("Tour", p_GP, searchAlgorithm="GP"),
    Config("TourSS", p_GPSteadyState, searchAlgorithm="GPSteadyState"),
    Config("Lex", p_Lexicase, searchAlgorithm="Lexicase"),
    Config("LexSS", p_LexicaseSteadyState, searchAlgorithm="LexicaseSteadyState")
])
dim_testsRatio = Dim([
    Config("0.0", p_testsRatio("0.0"), testsRatio="0.0"),
    Config("0.25", p_testsRatio("0.25"), testsRatio="0.25"),
    Config("0.5", p_testsRatio("0.5"), testsRatio="0.5"),
    Config("0.75", p_testsRatio("0.75"), testsRatio="0.75"),
    Config("1.0", p_testsRatio("1.0"), testsRatio="1.0")
])
dim_testsRatioGPR = Dim([
    Config("0.75", p_testsRatio("0.75"), testsRatio="0.75"),
    Config("1.0", p_testsRatio("1.0"), testsRatio="1.0")
])
dim_ea_type = Dim([Config("Gener.", p_Generational),
                   Config("SteadySt.", p_SteadyState)])
dim_sel = Dim([Config("$Tour$", p_sel_tourn),
               Config("$Lex$", p_sel_lexicase)])
dim_sa_ss = Dim([
    Config("GPSS", p_GPSteadyState, searchAlgorithm="GPSteadyState"),
    Config("LexSS", p_LexicaseSteadyState, searchAlgorithm="LexicaseSteadyState")
])
# dim_sa = Dim([Config("$CDGP$", p_GP),
# 			    Config("$CDGP^{ss}$", p_GPSteadyState),
#               Config("$CDGP_{lex}$", p_Lexicase),
#               Config("$CDGP_{lex}^{ss}$", p_LexicaseSteadyState)])


def is_optimal_solution(p):
    return "result.best.isOptimal" in p and p["result.best.isOptimal"] == "true"

def get_num_optimal(props):
    props2 = [p for p in props if is_optimal_solution(p)]
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
        return "-"
    num_opt = get_num_optimal(filtered)
    sr = float(num_opt) / float(len(filtered))
    return "{0}".format("%0.2f" % sr)
def get_stats_size(props):
    vals = [float(p["result.best.size"]) for p in props]
    if len(vals) == 0:
        return "-"#-1.0, -1.0
    else:
        return "%0.1f" % np.mean(vals)#, np.std(vals)
def get_stats_maxSolverTime(props):
    if len(props) == 0 or "cdgp.solverAllTimesCountMap" not in props[0]:
        return "-"
    times = []
    for p in props:
        timesMap = p["cdgp.solverAllTimesCountMap"]
        parts = timesMap.split(", ")[-1].split(",")
        times.append(float(parts[0].replace("(", "")))
    return "%0.3f" % max(times)
def get_stats_avgSolverTime(props):
    if len(props) == 0 or "cdgp.solverAllTimesCountMap" not in props[0]:
        return "-"
    sum = 0.0
    sumWeights = 0.0
    for p in props:
        timesMap = p["cdgp.solverAllTimesCountMap"]
        pairs = timesMap.split(", ")
        if len(pairs) == 0:
            continue
        for x in pairs:
            time = float(x.split(",")[0].replace("(", ""))
            weight = float(x.split(",")[1].replace(")", ""))
            sum += time * weight
            sumWeights += weight
    if sumWeights == 0.0:
        return "%0.3f" % 0.0
    else:
        return "%0.3f" % (sum / sumWeights)
def get_avgSolverTotalCalls(props):
    if len(props) == 0 or "cdgp.solverTotalCalls" not in props[0]:
        return "-"
    vals = [float(p["cdgp.solverTotalCalls"]) for p in props]
    return "%d" % round(np.mean(vals))
def get_numSolverCallsOverXs(props):
    if len(props) == 0 or "cdgp.solverAllTimesCountMap" not in props[0]:
        return "-"
    TRESHOLD = 0.5
    sum = 0
    for p in props:
        timesMap = p["cdgp.solverAllTimesCountMap"]
        pairs = timesMap.split(", ")
        if len(pairs) == 0:
            continue
        for x in pairs:
            time = float(x.split(",")[0].replace("(", ""))
            if time > TRESHOLD:
                # print("Name of file: " + p["thisFileName"])
                weight = int(x.split(",")[1].replace(")", ""))
                sum += weight
    return sum
def get_avg_totalTests(props):
    vals = [float(p["cdgp.totalTests"]) for p in props if p_method_cdgp(p) or p_method_gpr(p)]
    if len(vals) == 0:
        return "-"  # -1.0, -1.0
    else:
        x = np.mean(vals)
        if x < 1e-5:
            x = 0.0
        return "%0.1f" % x
def get_avg_fitness(props):
    vals = []
    for p in props:
        if "result.best.passedTestsRatio" in p:
            ratio = float(p["result.best.passedTestsRatio"])
            vals.append(ratio)
        elif p["method"] == "CDGP" or p["method"] == "GPR":
            raise Exception("Information about fitness is unavailable!")
    if len(vals) == 0:
        return "-"  # -1.0, -1.0
    else:
        return "%0.2f" % np.mean(vals)  # , np.std(vals)
def get_avg_runtime_helper(vals):
    if len(vals) == 0:
        return "n/a"  # -1.0, -1.0
    else:
        return "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_runtimeOnlySuccessful(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(p["result.totalTimeSystem"]) / 1000.0 for p in props if is_optimal_solution(p)]
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
    vals = [float(p["result.best.generation"]) for p in props if p_method_cdgp(p) or p_method_gpr(p)]
    if len(vals) == 0:
        return "-"
    else:
        return "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_generationSuccessful(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(p["result.best.generation"]) for p in props if is_optimal_solution(p) and
                (p_method_cdgp(p) or p_method_gpr(p))]
        if len(vals) == 0:
            return "n/a"  # -1.0, -1.0
        else:
            return "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_runtimePerProgram(props):
    if len(props) == 0 or not (p_method_cdgp(props[0]) or p_method_gpr(props[0])):
        return "-"  # -1.0, -1.0
    avgGen = float(get_avg_generation(props))  # avg number of generations in all runs
    avgRuntime = float(get_avg_runtime(props))  # avg runtime of all runs
    populationSize = float(props[0]["populationSize"])
    if props[0]["searchAlgorithm"] == "GPSteadyState" or \
       props[0]["searchAlgorithm"] == "LexicaseSteadyState":
        approxNumPrograms = populationSize + avgGen  # in steady state we have many generations, but in each of them created is one new program
    else:
        approxNumPrograms = populationSize * avgGen
    approxTimePerProgram = avgRuntime / approxNumPrograms
    return "%0.3f" % approxTimePerProgram
def get_sum_solverRestarts(props):
    if len(props) == 0:
        return "-"
    vals = [int(p["cdgp.solverTotalRestarts"]) for p in props if "cdgp.solverTotalRestarts" in p]
    if len(vals) != len(props):
        print("WARNING: cdgp.solverTotalRestarts was not present in all files.")
    if len(vals) == 0:
        return "0"
    else:
        return str(np.sum(vals))

def print_solved_in_time(props, upper_time):
    # totalTimeSystem is in miliseconds
    solved = 0
    solvedRuns = 0
    num = 0
    for p in props:
        if p["result.best.isOptimal"] == "false":
            continue
        num += 1
        if int(p["result.totalTimeSystem"]) <= upper_time:
            solved += 1

    for p in props:
        if int(p["result.totalTimeSystem"]) <= upper_time:
            solvedRuns += 1
    print("\nRuns which ended under {0} s:  {1} / {2}  ({3} %)".format(upper_time / 1000.0, solvedRuns, len(props), solvedRuns / len(props)))
    print("Optimal solutions found under {0} s:  {1} / {2}  ({3} %)\n".format(upper_time / 1000.0, solved, num, solved / num))


def plot_figures(props):
    print_solved_in_time(props, 12 * 3600 * 1000)
    print_solved_in_time(props, 6 * 3600 * 1000)
    print_solved_in_time(props, 3 * 3600 * 1000)
    print_solved_in_time(props, 1 * 3600 * 1000)
    print_solved_in_time(props, 0.5 * 3600 * 1000)
    print_solved_in_time(props, 0.25 * 3600 * 1000)
    print_solved_in_time(props, 0.125 * 3600 * 1000)
    print_solved_in_time(props, 600 * 1000)

    # Plot chart of number of found solutions in time
    success_props = [p for p in props if p["result.best.isOptimal"] == "true"]
    getter = lambda p: float(p["result.totalTimeSystem"]) / (3600 * 1000)  # take hours as a unit
    predicate = lambda v, v_xaxis: v <= v_xaxis
    xs = np.arange(0.0, 24.0+1e-9, 0.5) # a point every 30 minutes
    xticks = np.arange(0.0, 24.0+1e-9, 1.0) # a tick every 60 minutes
    plotter.plot_ratio_meeting_predicate(success_props, getter, predicate,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method * dim_sa, # "series_dim=None" for a single line
                                         savepath="figures/ratioTime_correctVsAllCorrect.pdf",
                                         title="Ratio of found correct solutions out of all correct solutions",
                                         xlabel="Runtime [hours]")
    plotter.plot_ratio_meeting_predicate(props, getter, predicate,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method * dim_sa,
                                         savepath="figures/ratioTime_endedVsAllEnded.pdf",
                                         title="Ratio of ended runs",
                                         xlabel="Runtime [hours]")
    def get_total_evaluated(p):
        if "searchAlgorithm" not in p:
            return None
        elif p["searchAlgorithm"].endswith("SteadyState"):
            return int(p["result.best.generation"])
        else:
            return int(p["result.best.generation"]) * int(p["populationSize"])
    xticks = np.arange(0.0, 50000.01, 5000)
    xs = np.arange(0.0, 50000.01, 1250)
    plotter.plot_ratio_meeting_predicate(success_props, get_total_evaluated, predicate,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method * dim_sa,
                                         savepath="figures/ratioEvaluated_correctVsAllCorrect.pdf",
                                         title="Ratio of found correct solutions out of all found correct solutions in the given config",
                                         xlabel="Number of evaluated solutions")
    cond = lambda p: p["result.best.isOptimal"] == "true"
    plotter.plot_ratio_meeting_predicate(props, get_total_evaluated, predicate,
                                         condition=cond,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method * dim_sa,
                                         savepath="figures/ratioEvaluated_correctVsAllRuns.pdf",
                                         title="Ratio of runs which ended with correct solution out of all runs",
                                         xlabel="Number of evaluated solutions")




def create_section_with_results(title, desc, folders, numRuns=10, use_bench_simple_names=True, print_status_matrix=True):
    assert isinstance(title, str)
    assert isinstance(desc, str)
    assert isinstance(folders, list)
    print("\n*** Processing: {0}***".format(title))
    for f in folders:
        print(f)
    if folders is None or len(folders) == 0:
        return None

    props = load_correct_props(folders, name=title)


    # Create figures in the appropriate directory
    plot_figures(props)


    # Uncomment this to print names of files with results of a certain configuration
    # print("\n(** {0} **) props_meeting the property:".format(title[:15]))
    # for p in props:
    #     if float(p["cdgp.solverTimeMaxSec"]) >= 2.0:
    #         print(p["thisFileName"] + ", " + p["cdgp.solverTimeMaxSec"])


    def post(s):
        return s.replace("{ccccccccccccc}", "{rrrrrrrrrrrrr}").replace("{rrr", "{lrr").replace(r"\_{lex}", "_{lex}").replace(r"\_{", "_{")


    dim_benchmarks = Dim.from_dict(props, "benchmark")
    if print_status_matrix:
        d = dim_benchmarks * dim_methodCDGP * dim_testsRatio * dim_sa + \
            dim_benchmarks * dim_methodGPR * dim_testsRatioGPR * dim_sa
        matrix = produce_status_matrix(d, props)
        print("\n****** Status matrix:")
        print(matrix + "\n")

    if use_bench_simple_names:
        configs = [Config(benchmarks_simple_names.get(c.get_caption(), c.get_caption()), c.filters[0][1]) for c in dim_benchmarks.configs]
        dim_benchmarks = Dim(configs)
        dim_benchmarks.sort()


    # --------------------- Shared stats ---------------------

    # -------------- Dimensions -----------------
    dim_cols = dim_methodCDGP * dim_ea_type * dim_sel * dim_testsRatio +\
               dim_methodGPR * dim_ea_type * dim_sel * dim_testsRatioGPR + \
               dim_methodFormal
    dim_rows = dim_benchmarks.sort() + dim_true
    # dim_cols = dim_method * dim_sa
    # -------------------------------------------

    vb = 1  # vertical border

    print("STATUS")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_num_computed, layered_headline=True, vertical_border=vb))
    latex_status = printer.table_color_map(text, 0.0, numRuns/2, numRuns, "colorLow", "colorMedium", "colorHigh")

    print("SUCCESS RATES")
    text = post(printer.latex_table(props, dim_rows, dim_cols, fun_successRates, layered_headline=True, vertical_border=vb))
    latex_successRates = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG RUNTIME")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_runtime, layered_headline=True, vertical_border=vb))
    latex_avgRuntime = printer.table_color_map(text, 0.0, 1800.0, 3600.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG RUNTIME (SUCCESSFUL)")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_runtimeOnlySuccessful, layered_headline=True, vertical_border=vb))
    latex_avgRuntimeOnlySuccessful = printer.table_color_map(text, 0.0, 1800.0, 3600.0, "colorLow", "colorMedium",
                                                             "colorHigh")

    # print("SUCCESS RATES (FULL INFO)")
    # text = post(printer.latex_table(props, dim_rows, dim_cols, fun_successRates_full, layered_headline=True, vertical_border=vb))

    print("AVG SIZES")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_stats_size, layered_headline=True, vertical_border=vb))
    latex_sizes = printer.table_color_map(text, 0.0, 100.0, 200.0, "colorLow", "colorMedium", "colorHigh")



    # --------------------- CDGP stats ---------------------

    # -------------- Dimensions -----------------
    dim_cols = dim_methodCDGP * dim_ea_type * dim_sel * dim_testsRatio + \
               dim_methodGPR * dim_ea_type * dim_sel * dim_testsRatioGPR
    # -------------------------------------------
    print("AVG BEST-OF-RUN FITNESS")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_fitness, layered_headline=True, vertical_border=vb))
    latex_avgBestOfRunFitness = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG TOTAL TESTS")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_totalTests, layered_headline=True, vertical_border=vb))
    latex_avgTotalTests = printer.table_color_map(text, 0.0, 1000.0, 2000.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG RUNTIME PER PROGRAM")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_runtimePerProgram, layered_headline=True, vertical_border=vb))
    latex_avgRuntimePerProgram = printer.table_color_map(text, 0.01, 1.0, 2.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG GENERATION")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_generation, layered_headline=True, vertical_border=vb))
    latex_avgGeneration = printer.table_color_map(text, 0.0, 50.0, 100.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG GENERATION (SUCCESSFUL)")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_generationSuccessful, layered_headline=True, vertical_border=vb))
    latex_avgGenerationSuccessful = printer.table_color_map(text, 0.0, 50.0, 100.0, "colorLow", "colorMedium", "colorHigh")

    print("MAX SOLVER TIME")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_stats_maxSolverTime, layered_headline=True, vertical_border=vb))
    latex_maxSolverTimes = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG SOLVER TIME")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_stats_avgSolverTime, layered_headline=True, vertical_border=vb))
    latex_avgSolverTimes = printer.table_color_map(text, 0.0, 0.015, 0.03, "colorLow", "colorMedium", "colorHigh")

    print("AVG NUM SOLVER CALLS")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avgSolverTotalCalls, layered_headline=True, vertical_border=vb))
    latex_avgSolverTotalCalls = printer.table_color_map(text, 1e3, 1e5, 5e6, "colorLow", "colorMedium", "colorHigh")

    print("NUM SOLVER CALLS > 0.5s")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_numSolverCallsOverXs, layered_headline=True, vertical_border=vb))
    latex_numSolverCallsOverXs = printer.table_color_map(text, 0, 1e4, 1e6, "colorLow", "colorMedium", "colorHigh")




    subsects_main = [
        ("Status (correctly finished processes)", latex_status, reporting.color_scheme_red_r),
        ("Success rates", latex_successRates, reporting.color_scheme_green),
        ("Average runtime [s]", latex_avgRuntime, reporting.color_scheme_violet),
        ("Average runtime (only successful) [s]", latex_avgRuntimeOnlySuccessful, reporting.color_scheme_violet),
        ("Average sizes of best of runs (number of nodes)", latex_sizes, reporting.color_scheme_yellow),
    ]
    subsects_cdgp = [
        ("Average best-of-run ratio of passed tests", latex_avgBestOfRunFitness, reporting.color_scheme_green),
        ("Average sizes of $T_C$ (total tests in run)", latex_avgTotalTests, reporting.color_scheme_blue),
        ("Average generation (all)", latex_avgGeneration, reporting.color_scheme_teal),
        ("Average generation (only successful)", latex_avgGenerationSuccessful, reporting.color_scheme_teal),
        ("Approximate average runtime per program [s]", latex_avgRuntimePerProgram, reporting.color_scheme_brown),
        ("Max solver time per query [s]", latex_maxSolverTimes, reporting.color_scheme_violet),
        ("Avg solver time per query [s]", latex_avgSolverTimes, reporting.color_scheme_brown),
        ("Avg number of solver calls", latex_avgSolverTotalCalls, reporting.color_scheme_blue),
        ("Number of solver calls $>$ 0.5s", latex_numSolverCallsOverXs, reporting.color_scheme_blue),
    ]
    figures_cdgp = [
        "figures/ratioEvaluated_correctVsAllRuns.pdf",
        "figures/ratioTime_correctVsAllCorrect.pdf",
        "figures/ratioTime_endedVsAllEnded.pdf",
    ]

    def get_content_of_subsections(subsects):
        content = []
        vspace = reporting.BlockLatex(r"\vspace{0.75cm}")
        for title, table, cs in subsects:
            sub = reporting.SectionRelative(title, contents=[cs, reporting.BlockLatex(table + "\n"), vspace])
            content.append(sub)
        return content

    section = reporting.Section(title, [])
    section.add(reporting.BlockLatex(desc + "\n"))
    section.add(reporting.Subsection("Shared Statistics", get_content_of_subsections(subsects_main)))
    section.add(reporting.Subsection("CDGP Statistics", get_content_of_subsections(subsects_cdgp)))
    for f in figures_cdgp:
        section.add(reporting.FloatFigure(f))
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
        vals = [float(p["result.totalTimeSystem"]) / 1000.0 for p in filtered if is_optimal_solution(p)]
        if len(vals) == 0:
            result = "No optimal solutions!"
            suggestion = 1800
        else:
            vals = [p for p in vals if p > 0.0]
            avg = int(np.mean(vals))
            result = int(np.min(vals)), avg, int(np.max(vals))
            result = "{0}, {1}, {2}".format(str(result[0]), str(result[1]), str(result[2]))
            suggestion = avg if avg < 1800 else 1800
        print(conf.get_caption() + ": " + str(suggestion) + " \t\t[{0}]".format(result))
        scalaMapEntries.append('"{0}"->"{1}000"'.format(conf.get_caption(), str(suggestion)))
    scalaMap += ", ".join(scalaMapEntries) + ")"
    print("Scala Map: " + scalaMap)







if __name__ == "__main__":
    folders_exp3 = ["exp3", "exp3_fix1", "exp3_fix2", "exp3_fix3", 'rsconf', "exp3gpr",
                    "exp3gpr_fix1", "exp3formal"]
    name_exp3 = "Experiments for parametrized CDGP (stop: number of iterations)"
    desc_exp3 = r"""
    Important information:
    \begin{itemize}
    \item All configurations, unless specified otherwise, has \textbf{population size 500}, and \textbf{number of iterations 100}.
    \item GPR uses range [-100, 100] and a population size 1000.
    \item SteadyState configurations has population\_size * 100 (number of iterations), so that the total number
     of generated solutions is the same.
    \item SteadyState configurations use always Tournament ($k=7$) deselection. Selection may be Tournament ($k=7$) or Lexicase.
    \end{itemize}
    """


    # print_time_bounds_for_benchmarks(props_expEvalsFINAL)



    # -------- Creating nice LaTeX report of the above results --------
    report = reporting.ReportPDF(geometry_params = "[paperwidth=75cm, paperheight=40cm, margin=0.3cm]")
    sects = [
        create_section_with_results(name_exp3, desc_exp3, folders_exp3, numRuns=10),
    ]
    for s in sects:
        if s is not None:
            report.add(s)
    print("\n\nGenerating PDF report ...")
    report.save_and_compile("cdgp_results2.tex")
