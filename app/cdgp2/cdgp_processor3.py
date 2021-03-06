import re
from evoplotter import utils
from evoplotter import plotter
from evoplotter import printer
from evoplotter import reporting
from evoplotter.dims import *
import numpy as np


# This processor is to be used for exp4 onward.
CHECK_CORRECTNESS_OF_FILES = 0
OPT_SOLUTIONS_FILE_NAME = "opt_solutions"



def save_to_file(file_path, content):
    file = open(file_path, "w")
    file.write(content)
    file.close()


def print_props_filenames(props):
    for p in props:
        if "thisFileName" in p:
            print(p["thisFileName"])
        else:
            print("'thisFileName' not specified! Printing content instead: " + str(p))


def load_correct_props(folders):
    props_cdgpError = utils.load_properties_dirs(folders, exts=[".cdgp.error"])
    exts = [".cdgp", ".cvc4_head_cegqi", ".cvc4_cegqi", ".eusolver", ".cvc4", ".cvc4_head"]
    props0 = utils.load_properties_dirs(folders, exts=exts)

    def is_correct(p):
        if p["method"] in {"GPR", "CDGP"}:
            if "SLIA" not in p["benchmark"] and p["searchAlgorithm"] in {"Lexicase", "LexicaseSteadyState"} and\
               ("lex" not in p["outDir"] and p["outDir"] != 'exp4str2_fix1'):
                return False
            else:
                return "result.best.passedTestsRatio" in p and "status" in p and\
                   (p["status"] == "completed" or p["status"] == "initialized")\
                   and "benchmark" in p and p["result.best.numTests"] != "0"

        else:
            return "status" in p and (p["status"] == "completed" or p["status"] == "initialized") and "benchmark" in p



    # Filtering props so only correct ones are left
    props = [p for p in props0 if is_correct(p)]

    # Printing names of files which finished with error status or are incomplete.
    if CHECK_CORRECTNESS_OF_FILES:
        props_errors = [p for p in props0 if not is_correct(p)]
        if len(props_errors) > 0:
            print("Files with error status:")
            print_props_filenames(props_errors)
        print("Loaded: {0} correct property files, {1} incorrect; All log files: {2}".format(len(props), len(props_errors), len(props) + len(props_errors)))
    print("Runs that ended with '.cdgp.error': {0}".format(len(props_cdgpError)))
    print_props_filenames(props_cdgpError)
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


def save_opt_solutions(dim, props, exp_prefix=None):
    text = ""
    for config in dim:
        caption = config.get_caption()
        if caption.startswith("All"):
            continue # ignore All rows
        text += caption + "\n"
        pool = config.filter_props(props)
        for p in pool:
            if is_optimal_solution(p):
                solution = re.sub(' +',' ', p["result.best.smtlib"])
                text += str(solution) + "\n"
        text += "\n\n"

    if exp_prefix is None:
        fname = OPT_SOLUTIONS_FILE_NAME + ".txt"
    else:
        fname = OPT_SOLUTIONS_FILE_NAME + "_" + str(exp_prefix) + ".txt"
    save_to_file(fname, text)




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
d3 = "benchmarks/SLIA/cdgp_ecj1/"
d4 = "benchmarks/SLIA/cdgp_ecj2/"
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
                           # String benchmarks
                           d3 + "dr-name.sl": "dr-name",
                           d3 + "firstname.sl": "firstname",
                           d3 + "initials.sl": "initials",
                           d3 + "lastname.sl": "lastname",
                           d3 + "name-combine-2.sl": "name-combine-2",
                           d3 + "name-combine-3.sl": "name-combine-3",
                           d3 + "name-combine-4.sl": "name-combine-4",
                           d3 + "name-combine.sl": "name-combine",
                           d4 + "phone-1.sl": "phone-1",
                           d4 + "phone-2.sl": "phone-2",
                           d4 + "phone-3.sl": "phone-3",
                           d4 + "phone-4.sl": "phone-4",
                           d4 + "phone.sl": "phone",
                           "benchmarks/NIA/rsconf.sl": "rsconf"}

dim_true = Dim(Config("All", lambda p: True, method=None))
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
    Config("EUSolver", p_method_for("eusolver"), method="eusolver"),
    Config("CVC4_cegqi", p_method_for("cvc4_head_cegqi"), method="cvc4_head_cegqi"),
])
dim_methodFormalStrings = Dim([
    Config("CVC4 1.5", p_method_for("cvc4"), method="cvc4"),
    Config("CVC4_head", p_method_for("cvc4_head"), method="cvc4_head"),
])
dim_sa = Dim([
    Config("Tour", p_GP, searchAlgorithm="GP"),
    Config("TourSS", p_GPSteadyState, searchAlgorithm="GPSteadyState"),
    Config("Lex", p_Lexicase, searchAlgorithm="Lexicase"),
    Config("LexSS", p_LexicaseSteadyState, searchAlgorithm="LexicaseSteadyState")
])
dim_tour = Dim([
    Config("Tour", p_GP, searchAlgorithm="GP"),
    Config("TourSS", p_GPSteadyState, searchAlgorithm="GPSteadyState"),
])
dim_lexicase = Dim([
    Config("Lex", p_Lexicase, searchAlgorithm="Lexicase"),
    Config("LexSS", p_LexicaseSteadyState, searchAlgorithm="LexicaseSteadyState"),
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



def normalized_total_time(p, max_time=3600000): # by default 1 h (in ms)
    """If time was longer than max_time, then return max_time, otherwise return time."""
    if p["result.totalTimeSystem"] == "3600.0":
        v = 3600000  # convert to ms (error in logging)
    else:
        v = int(float(p["result.totalTimeSystem"]))
    return max_time if v > max_time else v

def is_optimal_solution(p):
    return "result.best.isOptimal" in p and p["result.best.isOptimal"] == "true"

def get_num_optimal(props):
    props2 = [p for p in props if is_optimal_solution(p)]
    return len(props2)

def get_num_computed(filtered):
    return len(filtered)
def fun_successRate_full(filtered):
    if len(filtered) == 0:
        return "-"
    num_opt = get_num_optimal(filtered)
    return "{0}/{1}".format(str(num_opt), str(len(filtered)))
def get_successRate(filtered):
    # if len(filtered) == 0:
    #     return -1
    num_opt = get_num_optimal(filtered)
    return float(num_opt) / float(len(filtered))
def fun_successRate(filtered):
    if len(filtered) == 0:
        return "-"
    sr = get_successRate(filtered)
    return "{0}".format("%0.2f" % round(sr, 2))
def get_stats_size(props):
    vals = [float(p["result.best.size"]) for p in props]
    if len(vals) == 0:
        return "-"#-1.0, -1.0
    else:
        return str(int(round(np.mean(vals)))) #, np.std(vals)
def get_stats_sizeOnlySuccessful(props):
    vals = [float(p["result.best.size"]) for p in props if is_optimal_solution(p)]
    if len(vals) == 0:
        return "-"#-1.0, -1.0
    else:
        return str(int(round(np.mean(vals)))) #, np.std(vals)
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
    vals = [float(p["cdgp.solverTotalCalls"]) / 1000.0 for p in props]
    return "%0.1f" % round(np.mean(vals), 1) # "%d"
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
        return str(int(round(x))) #"%0.1f" % x
def get_avg_fitness(props):
    vals = []
    for p in props:
        if "result.best.passedTestsRatio" in p:
            if int(p["result.best.numTests"]) > 0:
                ratio = float(p["result.best.passedTests"]) / float(p["result.best.numTests"])
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
        x = np.mean(vals)
        if x >= 10.0:
            return "%d" % x
        else:
            return "%0.1f" % x  # , np.std(vals)
def get_avg_runtimeOnlySuccessful(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(normalized_total_time(p)) / 1000.0 for p in props if is_optimal_solution(p)]
        return get_avg_runtime_helper(vals)
def get_avg_runtime(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(normalized_total_time(p)) / 1000.0 for p in props]
        return get_avg_runtime_helper(vals)
def get_avg_generation(props):
    if len(props) == 0:
        return "-"
    vals = [float(p["result.best.generation"]) for p in props if p_method_cdgp(p) or p_method_gpr(p)]
    if len(vals) == 0:
        return "-"
    else:
        return str(int(round(np.mean(vals)))) #"%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_generationSuccessful(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(p["result.best.generation"]) for p in props if is_optimal_solution(p) and
                (p_method_cdgp(p) or p_method_gpr(p))]
        if len(vals) == 0:
            return "n/a"  # -1.0, -1.0
        else:
            return str(int(round(np.mean(vals))))  # "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_evaluated(props):
    if len(props) == 0:
        return "-"
    vals = []
    for p in props:
        if p["searchAlgorithm"].endswith("SteadyState"):
            vals.append(float(p["result.best.generation"]))
        else:
            vals.append(float(p["result.best.generation"]) * float(p["populationSize"]))
    return str(int(round(np.mean(vals)))) #"%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_evaluatedSuccessful(props):
    if len(props) == 0:
        return "-"
    vals = []
    for p in props:
        if is_optimal_solution(p):
            if p["searchAlgorithm"].endswith("SteadyState"):
                vals.append(float(p["result.best.generation"]))
            else:
                vals.append(float(p["result.best.generation"]) * float(p["populationSize"]))
    if len(vals) == 0:
        return "n/a"  # -1.0, -1.0
    else:
        return str(int(round(np.mean(vals))))  # "%0.1f" % np.mean(vals)  # , np.std(vals)
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
    if len(props) == 0:
        return
    # totalTimeSystem is in miliseconds
    solved = 0
    solvedRuns = 0
    num = 0
    for p in props:
        if p["result.best.isOptimal"] == "false":
            continue
        num += 1
        if int(normalized_total_time(p)) <= upper_time:
            solved += 1

    for p in props:
        if int(normalized_total_time(p)) <= upper_time:
            solvedRuns += 1
    print("\nRuns which ended under {0} s:  {1} / {2}  ({3} %)".format(upper_time / 1000.0, solvedRuns, len(props), solvedRuns / len(props)))
    print("Optimal solutions found under {0} s:  {1} / {2}  ({3} %)\n".format(upper_time / 1000.0, solved, num, solved / num))


def plot_figures(props, exp_prefix):
    # We want to consider CDGP only
    props = [p for p in props if p_method_cdgp(p) or p_method_gpr(p)]
    if len(props) == 0:
        print("No props: plots were not generated.")
        return
    # print_solved_in_time(props, 12 * 3600 * 1000)
    # print_solved_in_time(props, 6 * 3600 * 1000)
    # print_solved_in_time(props, 3 * 3600 * 1000)
    # print_solved_in_time(props, 1 * 3600 * 1000)
    # print_solved_in_time(props, 0.5 * 3600 * 1000)
    # print_solved_in_time(props, 0.25 * 3600 * 1000)
    # print_solved_in_time(props, 0.125 * 3600 * 1000)
    # print_solved_in_time(props, 600 * 1000)

    # Plot chart of number of found solutions in time
    success_props = [p for p in props if is_optimal_solution(p)]
    getter = lambda p: float(normalized_total_time(p)) / (60 * 1000)  # take minutes as a unit
    predicate = lambda v, v_xaxis: v <= v_xaxis
    xs = np.arange(0.0, 60.5+1e-9, 1.0) # a point every 1.0 minutes
    xticks = np.arange(0.0, 60.0+1e-9, 5.0) # a tick every 5 minutes
    plotter.plot_ratio_meeting_predicate(success_props, getter, predicate,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method * dim_sa, # "series_dim=None" for a single line
                                         savepath="figures/{0}_ratioTime_correctVsAllCorrect.pdf".format(exp_prefix),
                                         title="Ratio of found correct solutions out of all correct solutions",
                                         xlabel="Runtime [minutes]")
    plotter.plot_ratio_meeting_predicate(props, getter, predicate,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method * dim_sa,
                                         savepath="figures/{0}_ratioTime_endedVsAllEnded.pdf".format(exp_prefix),
                                         title="Ratio of ended runs",
                                         xlabel="Runtime [minutes]")
    def get_total_evaluated(p):
        if "searchAlgorithm" not in p:
            return None
        elif p["searchAlgorithm"].endswith("SteadyState"):
            return int(p["result.best.generation"])
        else:
            return int(p["result.best.generation"]) * int(p["populationSize"])
    xs = np.arange(0.0, 500.0 * 1000.0 + 0.01, 10000)
    xticks = np.arange(0.0, 500.0 *1000.0 + 0.01, 50000)
    plotter.plot_ratio_meeting_predicate(success_props, get_total_evaluated, predicate,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method * dim_sa,
                                         savepath="figures/{0}_ratioEvaluated_correctVsAllCorrect.pdf".format(exp_prefix),
                                         title="Ratio of found correct solutions out of all found correct solutions in the given config",
                                         xlabel="Number of evaluated solutions")
    cond = lambda p: p["result.best.isOptimal"] == "true"
    plotter.plot_ratio_meeting_predicate(props, get_total_evaluated, predicate,
                                         condition=cond,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method * dim_sa,
                                         savepath="figures/{0}_ratioEvaluated_correctVsAllRuns.pdf".format(exp_prefix),
                                         title="Ratio of runs which ended with correct solution out of all runs",
                                         xlabel="Number of evaluated solutions")

    plotter.compare_avg_data_series(props, dim_methodCDGP * dim_sa, "CDGPtestsRatio",
                                    getter_y=get_successRate,
                                    is_aggr=1,
                                    savepath="figures/{0}_q_successRates.pdf".format(exp_prefix),
                                    title="Success rates",
                                    ylabel="Success rate",
                                    xlabel="q")

    plotter.compare_avg_data_series_d(success_props, dim_methodCDGP * dim_sa, "CDGPtestsRatio", "result.best.size",
                                    savepath="figures/{0}_q_sizes.pdf".format(exp_prefix),
                                    title="Size of found correct solutions",
                                    ylabel="Size",
                                    xlabel="q")

    plotter.compare_avg_data_series_d(props, dim_methodCDGP*dim_sa, "CDGPtestsRatio", "cdgp.totalTests",
                                    savepath="figures/{0}_q_tests.pdf".format(exp_prefix),
                                    title="Total number of test cases",
                                    ylabel="Total tests",
                                    xlabel="q")



def get_content_of_subsections(subsects):
    content = []
    vspace = reporting.BlockLatex(r"\vspace{0.75cm}"+"\n")
    for title, table, cs in subsects:
        sub = reporting.SectionRelative(title, contents=[cs, reporting.BlockLatex(table + "\n"), vspace])
        content.append(sub)
    return content

def post(s):
    return s.replace("{ccccccccccccc}", "{rrrrrrrrrrrrr}").replace("{rrr", "{lrr").replace(r"\_{lex}", "_{lex}").replace(r"\_{", "_{")



def create_section_and_plots(title, desc, props, subsects, figures_list):
    assert isinstance(title, str)
    assert isinstance(desc, str)
    assert isinstance(props, list)
    assert isinstance(figures_list, list)

    section = reporting.Section(title, [])
    section.add(reporting.BlockLatex(desc + "\n"))
    for s in subsects:
        section.add(s)

    # Create figures in the appropriate directory
    for f in figures_list:
        section.add(reporting.FloatFigure(f))
    section.add(reporting.BlockLatex(r"\vspace{1cm}" + "\n"))
    return section




def create_subsection_shared_stats(props, dim_rows, dim_cols, numRuns, exp_prefix):
    vb = 1  # vertical border

    print("STATUS")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_num_computed, layered_headline=True, vertical_border=vb))
    latex_status = printer.table_color_map(text, 0.0, numRuns / 2, numRuns, "colorLow", "colorMedium", "colorHigh")

    print("SUCCESS RATES")
    print(printer.text_table(props, dim_rows, dim_cols, fun_successRate, d_cols=";"))
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, fun_successRate, layered_headline=True, vertical_border=vb))
    latex_successRates = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG RUNTIME")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_avg_runtime, layered_headline=True, vertical_border=vb))
    latex_avgRuntime = printer.table_color_map(text, 0.0, 1800.0, 3600.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG RUNTIME (SUCCESSFUL)")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_runtimeOnlySuccessful, layered_headline=True,
                                    vertical_border=vb))
    latex_avgRuntimeOnlySuccessful = printer.table_color_map(text, 0.0, 1800.0, 3600.0, "colorLow", "colorMedium",
                                                             "colorHigh")

    # print("SUCCESS RATES (FULL INFO)")
    # text = post(printer.latex_table(props, dim_rows, dim_cols, fun_successRates_full, layered_headline=True, vertical_border=vb))

    # print("AVG SIZES")
    # text = post(printer.latex_table(props, dim_rows, dim_cols, get_stats_size, layered_headline=True, vertical_border=vb))
    # latex_sizes = printer.table_color_map(text, 0.0, 100.0, 200.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG SIZES (SUCCESSFUL)")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_stats_sizeOnlySuccessful, layered_headline=True,
                                    vertical_border=vb))
    latex_sizesOnlySuccessful = printer.table_color_map(text, 0.0, 100.0, 200.0, "colorLow", "colorMedium", "colorHigh")


    # Saving optimal solutions to a file
    save_opt_solutions(dim_rows * dim_cols, props, exp_prefix=exp_prefix)


    subsects_main = [
        ("Status (correctly finished processes)", latex_status, reporting.color_scheme_red_r),
        ("Success rates", latex_successRates, reporting.color_scheme_green),
        ("Average runtime [s]", latex_avgRuntime, reporting.color_scheme_violet),
        ("Average runtime (only successful) [s]", latex_avgRuntimeOnlySuccessful, reporting.color_scheme_violet),
        # ("Average sizes of best of runs (number of nodes)", latex_sizes, reporting.color_scheme_yellow),
        ("Average sizes of best of runs (number of nodes) (only successful)", latex_sizesOnlySuccessful,
         reporting.color_scheme_yellow),
    ]
    return reporting.Subsection("Shared Statistics", get_content_of_subsections(subsects_main))



def create_subsection_cdgp_specific(props, dim_rows, dim_cols, exp_prefix):
    vb = 1  # vertical border

    print("AVG BEST-OF-RUN FITNESS")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_avg_fitness, layered_headline=True, vertical_border=vb))
    latex_avgBestOfRunFitness = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG TOTAL TESTS")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_avg_totalTests, layered_headline=True, vertical_border=vb))
    latex_avgTotalTests = printer.table_color_map(text, 0.0, 1000.0, 2000.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG RUNTIME PER PROGRAM")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_runtimePerProgram, layered_headline=True,
                                    vertical_border=vb))
    latex_avgRuntimePerProgram = printer.table_color_map(text, 0.01, 1.0, 2.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG GENERATION")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_avg_generation, layered_headline=True, vertical_border=vb))
    latex_avgGeneration = printer.table_color_map(text, 0.0, 50.0, 100.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG EVALUATED SOLUTIONS")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_avg_evaluated, layered_headline=True, vertical_border=vb))
    latex_avgEvaluated = printer.table_color_map(text, 500.0, 25000.0, 100000.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG EVALUATED SOLUTIONS (SUCCESSFUL)")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_evaluatedSuccessful, layered_headline=True,
                                    vertical_border=vb))
    latex_avgEvaluatedSuccessful = printer.table_color_map(text, 500.0, 25000.0, 100000.0, "colorLow", "colorMedium",
                                                            "colorHigh")

    print("MAX SOLVER TIME")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_stats_maxSolverTime, layered_headline=True,
                                    vertical_border=vb))
    latex_maxSolverTimes = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG SOLVER TIME")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_stats_avgSolverTime, layered_headline=True,
                                    vertical_border=vb))
    latex_avgSolverTimes = printer.table_color_map(text, 0.0, 0.015, 0.03, "colorLow", "colorMedium", "colorHigh")

    print("AVG NUM SOLVER CALLS")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_avgSolverTotalCalls, layered_headline=True,
                                    vertical_border=vb))
    latex_avgSolverTotalCalls = printer.table_color_map(text, 1e1, 1e2, 1e4, "colorLow", "colorMedium", "colorHigh")

    print("NUM SOLVER CALLS > 0.5s")
    text = post(printer.latex_table(props, dim_rows, dim_cols, get_numSolverCallsOverXs, layered_headline=True,
                                    vertical_border=vb))
    latex_numSolverCallsOverXs = printer.table_color_map(text, 0, 50, 100, "colorLow", "colorMedium", "colorHigh")

    plot_figures(props, exp_prefix=exp_prefix)
    subsects_cdgp = [
        ("Average best-of-run ratio of passed tests", latex_avgBestOfRunFitness, reporting.color_scheme_green),
        ("Average sizes of $T_C$ (total tests in run)", latex_avgTotalTests, reporting.color_scheme_blue),
        ("Average generation (all)", latex_avgGeneration, reporting.color_scheme_teal),
        #("Average generation (only successful)", latex_avgGenerationSuccessful, reporting.color_scheme_teal),
        ("Average evaluated solutions", latex_avgEvaluated, reporting.color_scheme_teal),
        ("Average evaluated solutions (only successful)", latex_avgEvaluatedSuccessful, reporting.color_scheme_teal),
        ("Approximate average runtime per program [s]", latex_avgRuntimePerProgram, reporting.color_scheme_brown),
        ("Max solver time per query [s]", latex_maxSolverTimes, reporting.color_scheme_violet),
        ("Avg solver time per query [s]", latex_avgSolverTimes, reporting.color_scheme_brown),
        ("Avg number of solver calls (in thousands; 1=1000)", latex_avgSolverTotalCalls, reporting.color_scheme_blue),
        ("Number of solver calls $>$ 0.5s", latex_numSolverCallsOverXs, reporting.color_scheme_blue),
    ]
    return reporting.Subsection("CDGP Statistics", get_content_of_subsections(subsects_cdgp))



_prev_props = None
def prepare_report(sects, fname, use_bench_simple_names=True, print_status_matrix=True, reuse_props=False):
    """Creating nice LaTeX report of the results."""
    global _prev_props
    report = reporting.ReportPDF(geometry_params="[paperwidth=75cm, paperheight=40cm, margin=0.3cm]")
    latex_sects = []
    for title, desc, folders, subs, figures in sects:
        print("\nLoading props for: " + title)
        print("Scanned folders:")
        for f in folders:
            print(f)

        # Load props
        if reuse_props:
            props = _prev_props
        else:
            props = load_correct_props(folders)
            _prev_props = props
        dim_benchmarks = Dim.from_dict(props, "benchmark")

        print("\nFiltered Info:")
        for p in props:
            if p["method"] in {"GPR", "CDGP"} and p["searchAlgorithm"] in {"GP", "Lexicase"} and\
                    int(p["result.best.generation"]) >= 99990:
                print(p["thisFileName"] + "   --->  " + str(p["result.best.generation"]))
            # Print file names of certain config
            # if "fg_array_search_2" in p["benchmark"] and "searchAlgorithm" in p and\
            #    p["searchAlgorithm"] == "Lexicase" and p["method"] == "CDGP" and\
            #    p["CDGPtestsRatio"] == "0.0":
            #     print(p["thisFileName"] + "   --->  " + str(float(p["result.totalTimeSystem"]) / 1000.0))
            #     print("BEST: " + p["result.best.smtlib"])
            # Print bests
            # if p["method"] in {"GPR", "CDGP"} and is_optimal_solution(p) and\
            #    p["benchmark"] == "benchmarks/SLIA/cdgp_ecj1/initials.sl":
            #     print(p["thisFileName"] + "   --->  " + str(float(p["result.totalTimeSystem"]) / 1000.0))
            #     print("BEST: " + p["result.best.smtlib"])

        if use_bench_simple_names:
            configs = [Config(benchmarks_simple_names.get(c.get_caption(), c.get_caption()), c.filters[0][1],
                              benchmark=c.get_caption()) for c in
                       dim_benchmarks.configs]
            dim_benchmarks = Dim(configs)
            dim_benchmarks.sort()
        if print_status_matrix:
            # Corrections for whole non-formal LIA
            #d = dim_benchmarks * dim_methodCDGP * dim_testsRatio * dim_sa +\
            #    dim_benchmarks * dim_methodGPR * dim_testsRatioGPR * dim_sa

            # Corrections for Lexicase
            d = dim_benchmarks * dim_methodCDGP * dim_testsRatio * dim_lexicase +\
                dim_benchmarks * dim_methodGPR * dim_testsRatioGPR * dim_lexicase

            # Corrections for Strings
            # d = dim_benchmarks * dim_methodCDGP * dim_testsRatio * dim_sa

            matrix = produce_status_matrix(d, props)
            print("\n****** Status matrix:")
            print(matrix + "\n")


        dim_rows = dim_benchmarks.sort() + dim_true
        subsects = []
        for fun, args in subs:
            if args[0] is None:  # no dimensions for rows, take benchmarks as the default
                args[0] = dim_rows
            args2 = [props] + args
            subsects.append(fun(*args2))

        s = create_section_and_plots(title, desc, props, subsects, figures)
        latex_sects.append(s)

    for s in latex_sects:
        if s is not None:
            report.add(s)
    print("\n\nGenerating PDF report ...")
    report.save_and_compile(fname)




def reports_exp4int():
    folders = ["exp4int", "exp4int_fix1", "exp4int_fix2", "exp4int_fix3", "exp4int_fix4",
               "exp4int_fix5", "exp4int_lex", "exp4int_lex_fix1", "exp3formal"]
    #folders = ["exp4int_lex", "exp4int_lex_fix1", "exp3formal"]
    title = "Experiments for parametrized CDGP (stop: 1h)"
    desc = r""""""
    dimColsCdgp = dim_methodCDGP * dim_ea_type * dim_sel * dim_testsRatio + \
                  dim_methodGPR * dim_ea_type * dim_sel * dim_testsRatioGPR
    dimColsShared = dimColsCdgp + dim_methodFormal
    #dimColsCdgp_v2 = dim_methodCDGP * dim_ea_type * dim_sel + \
    #                 dim_methodGPR * dim_ea_type * dim_sel
    dimColsCdgp_v2 = dim_methodCDGP * dim_ea_type * dim_testsRatio + \
                     dim_methodGPR * dim_ea_type * dim_testsRatioGPR
    dimColsShared_v2 = dimColsCdgp_v2 + dim_methodFormal
    subs = [
        (create_subsection_shared_stats, [None, dimColsShared, 25, "exp4int"]),
        (create_subsection_cdgp_specific, [None, dimColsCdgp, "exp4int"]),
    ]
    subs_v2 = [
        (create_subsection_shared_stats, [None, dimColsShared_v2, 25, "exp4int"]),
        (create_subsection_cdgp_specific, [None, dimColsCdgp_v2, "exp4int"]),
    ]
    figures = [
        "figures/exp4int_ratioEvaluated_correctVsAllRuns.pdf",
        "figures/exp4int_ratioTime_correctVsAllCorrect.pdf",
        "figures/exp4int_ratioTime_endedVsAllEnded.pdf",
        "figures/exp4int_q_successRates.pdf",
        "figures/exp4int_q_sizes.pdf",
        "figures/exp4int_q_tests.pdf",
    ]
    sects = [(title, desc, folders, subs, figures)]
    sects_v2 = [(title, desc, folders, subs_v2, figures)]

    prepare_report(sects, "cdgp_exp4int.tex")
    prepare_report(sects_v2, "cdgp_exp4int_v2.tex", reuse_props=True)


def reports_exp4str():
    folders = ["exp4str", "exp4str_fix1", "exp4str_fix2", "exp4str2", "exp4str2_fix1",
               "exp4str2_fix2", "exp4str_formal_final"]
    title = "Experiments for parametrized CDGP (stop: 1h)"
    desc = r""""""
    str_dimColsCdgp = dim_methodCDGP * dim_ea_type * dim_sel * dim_testsRatio
    str_dimColsShared = str_dimColsCdgp + dim_methodFormalStrings
    str_dimColsCdgp_v2 = dim_methodCDGP * dim_ea_type * dim_testsRatio
    str_dimColsShared_v2 = str_dimColsCdgp_v2 + dim_methodFormalStrings
    subs = [
        (create_subsection_shared_stats, [None, str_dimColsShared, 25, "exp4str"]),
        (create_subsection_cdgp_specific, [None, str_dimColsCdgp, "exp4str"]),
    ]
    subs_v2 = [
        (create_subsection_shared_stats, [None, str_dimColsShared_v2, 25, "exp4str"]),
        (create_subsection_cdgp_specific, [None, str_dimColsCdgp_v2, "exp4str"]),
    ]
    figures = [
        "figures/exp4str_ratioEvaluated_correctVsAllRuns.pdf",
        "figures/exp4str_ratioTime_correctVsAllCorrect.pdf",
        "figures/exp4str_ratioTime_endedVsAllEnded.pdf",
        "figures/exp4str_q_successRates.pdf",
        "figures/exp4str_q_sizes.pdf",
        "figures/exp4str_q_tests.pdf",
    ]
    sects = [(title, desc, folders, subs, figures)]
    sects_v2 = [(title, desc, folders, subs_v2, figures)]

    prepare_report(sects, "cdgp_exp4str.tex")
    prepare_report(sects_v2, "cdgp_exp4str_v2.tex", reuse_props=True)


def reports_exp3():
    folders = ["exp3", "exp3_fix1", "exp3_fix2", "exp3_fix3", 'rsconf', "exp3gpr",
               "exp3gpr_fix1", "exp3gpr_fix2", "exp3gpr_fix3", "exp3gpr_fix4", "exp3formal"]
    title = "Experiments for parametrized CDGP (stop: number of iterations)"
    desc = r"""
    Important information:
    \begin{itemize}
    \item All configurations, unless specified otherwise, has \textbf{population size 500}, and \textbf{number of iterations 100}.
    \item GPR uses range [-100, 100] and a population size 1000.
    \item SteadyState configurations has population\_size * 100 (number of iterations), so that the total number
     of generated solutions is the same.
    \item SteadyState configurations use always Tournament ($k=7$) deselection. Selection may be Tournament ($k=7$) or Lexicase.
    \end{itemize}
    """
    dimColsCdgp = dim_methodCDGP * dim_ea_type * dim_sel * dim_testsRatio + \
                  dim_methodGPR * dim_ea_type * dim_sel * dim_testsRatioGPR
    dimColsShared = dim_methodFormal
    dimColsCdgp_v2 = dim_methodCDGP * dim_ea_type * dim_sel + \
                     dim_methodGPR * dim_ea_type * dim_sel
    dimColsShared_v2 = dimColsCdgp_v2 + dim_methodFormal
    subs = [
        (create_subsection_shared_stats, [None, dimColsShared, 10, "exp3"]),
        (create_subsection_cdgp_specific, [None, dimColsCdgp, "exp3"]),
    ]
    subs_v2 = [
        (create_subsection_shared_stats, [None, dimColsShared_v2, 10, "exp3"]),
        (create_subsection_cdgp_specific, [None, dimColsCdgp_v2, "exp3"]),
    ]
    sects = [(title, desc, folders, subs, [])]
    sects_v2 = [(title, desc, folders, subs_v2, [])]

    prepare_report(sects, "cdgp_exp3.tex")
    prepare_report(sects_v2, "cdgp_exp3_v2.tex")



if __name__ == "__main__":
    # reports_exp4int()
    reports_exp4str()