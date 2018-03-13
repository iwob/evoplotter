import os
from src import utils
from src import plotter
from src import printer
from src import reporting
from src.dims import *
import numpy as np


# This processor is to be used for exp4 onward.
CHECK_CORRECTNESS_OF_FILES = 0
STATUS_FILE_NAME = "status.txt"


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    exts = [".cdgp"]
    props0 = utils.load_properties_dirs(folders, exts=exts)

    def is_correct(p):
        return True



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
    
    :param dim: (Dimension) dimensions on which data are to be divided.
    :param props: (dict[str,str]) properties files.
    :return: (str) Python code of a list containing specified data.
    """
    text = "["
    for config in dim:
        numRuns = len(config.filter_props(props))
        text += "({0}, {1}), ".format(config.stored_values, numRuns)
    return text + "]"



def p_method_for(name):
    return lambda p, name=name: p["method"] == name
def p_generational(p):
    return p["evolutionMode"] == "generational"
def p_steadyState(p):
    return p["evolutionMode"] == "steadyState"
def p_sel_lexicase(p):
    return p["selection"] == "lexicase"
def p_sel_tourn(p):
    return p["selection"] == "tournament"
def p_testsRatio_equalTo(ratio):
    return lambda p, ratio=ratio: p["testsRatio"] == ratio




def simplify_benchmark_name(name):
    """Shortens or modifies the path of the benchmark."""
    if name.startswith("benchmarks/physics/"):
        return name.replace("benchmarks/physics/", "physics/", 1)
    else:
        return name



dim_true = Dim(Config("All", lambda p: True, method=None))
dim_methodCDGP = Dim([Config("CDGP", p_method_for("CDGP"), method="CDGP")])
dim_methodGP = Dim([Config("GP", p_method_for("GP"), method="GP")])
dim_method = dim_methodCDGP + dim_methodGP
dim_sel = Dim([#Config("$Tour$", p_sel_tourn, selection="tournament"),
               Config("$Lex$", p_sel_lexicase, selection="lexicase")])
dim_evoMode = Dim([Config("$steadyState$", p_steadyState, evolutionMode="steadyState"),])
                   #Config("$generational$", p_sel_lexicase, evolutionMode="generational")])
dim_testsRatio = Dim([Config("$0.75$", p_testsRatio_equalTo("0.75"), testsRatio="0.75"),
                      Config("$1.0$", p_testsRatio_equalTo("1.0"), testsRatio="1.0")])
# dim_sa = Dim([Config("$CDGP$", p_GP),
# 			    Config("$CDGP^{ss}$", p_steadyState),
#               Config("$CDGP_{lex}$", p_lexicase),
#               Config("$CDGP_{lex}^{ss}$", p_LexicaseSteadyState)])



def normalized_total_time(p, max_time=3600000 * 5): # by default 5 h (in ms)
    """If time was longer than max_time, then return max_time, otherwise return time."""
    if p["result.totalTimeSystem"] == "3600.0":
        v = 3600000  # convert to ms (error in logging)
    else:
        v = int(float(p["result.totalTimeSystem"]))
    return max_time if v > max_time else v

def is_optimal_solution(p):
    k = "result.best.verificationDecision"
    return p["result.best.isOptimal"] == "true" and \
           (k not in p or p[k] == "unsat")
    # return "result.best.verificationDecision" not in p or p["result.best.verificationDecision"] == "unsat"

def get_num_optimal(props):
    props2 = [p for p in props if is_optimal_solution(p)]
    return len(props2)

def get_num_propertiesMet(props):
    props2 = [p for p in props if p["result.best.verificationDecision"] == "unsat"]
    return len(props2)

def get_num_computed(filtered):
    return len(filtered)
def fun_successRate_full(filtered):
    if len(filtered) == 0:
        return "-"
    num_opt = get_num_optimal(filtered)
    return "{0}/{1}".format(str(num_opt), str(len(filtered)))
def get_successRate(filtered):
    num_opt = get_num_optimal(filtered)
    return float(num_opt) / float(len(filtered))
def fun_successRate(filtered):
    if len(filtered) == 0:
        return "-"
    sr = get_successRate(filtered)
    return "{0}".format("%0.2f" % round(sr, 2))
def fun_propertiesMet(filtered):
    if len(filtered) == 0:
        return "-"
    num_opt = get_num_propertiesMet(filtered)
    sr = float(num_opt) / float(len(filtered))
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
    if len(props) == 0 or "solver.allTimesCountMap" not in props[0]:
        return "-"
    times = []
    for p in props:
        timesMap = p["solver.allTimesCountMap"]
        parts = timesMap.split(", ")[-1].split(",")
        times.append(float(parts[0].replace("(", "")))
    return "%0.3f" % max(times)
def get_stats_avgSolverTime(props):
    if len(props) == 0 or "solver.allTimesCountMap" not in props[0] or props[0]["method"] != "CDGP":
        return "-"
    sum = 0.0
    sumWeights = 0.0
    for p in props:
        timesMap = p["solver.allTimesCountMap"]
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
    if len(props) == 0 or "solver.totalCalls" not in props[0]:
        return "-"
    vals = [float(p["solver.totalCalls"]) / 1000.0 for p in props]
    return "%0.1f" % round(np.mean(vals), 1) # "%d"
def get_numSolverCallsOverXs(props):
    if len(props) == 0 or "solver.allTimesCountMap" not in props[0]:
        return "-"
    TRESHOLD = 0.5
    sum = 0
    for p in props:
        timesMap = p["solver.allTimesCountMap"]
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
    vals = [float(p["tests.total"]) for p in props]
    if len(vals) == 0:
        return "-"  # -1.0, -1.0
    else:
        x = np.mean(vals)
        if x < 1e-5:
            x = 0.0
        return str(int(round(x))) #"%0.1f" % x
def get_avg_mse(props):
    vals = []
    for p in props:
        vals.append(float(p["result.best.mse"]))
    if len(vals) == 0:
        return "-"  # -1.0, -1.0
    else:
        return "%0.5f" % np.mean(vals)  # , np.std(vals)
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
    if len(props) > 0 and "result.totalGenerations" not in props[0]:
        return "-"
    vals = [float(p["result.totalGenerations"]) for p in props]
    if len(vals) == 0:
        return "-"
    else:
        return str(int(round(np.mean(vals)))) #"%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_generationSuccessful(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(p["result.best.generation"]) for p in props if is_optimal_solution(p)]
        if len(vals) == 0:
            return "n/a"  # -1.0, -1.0
        else:
            return str(int(round(np.mean(vals))))  # "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_evaluated(props):
    if len(props) == 0:
        return "-"
    vals = []
    for p in props:
        if p["evolutionMode"] == "steadyState":
            vals.append(float(p["result.totalGenerations"]))
        else:
            vals.append(float(p["result.totalGenerations"]) * float(p["populationSize"]))
    return str(int(round(np.mean(vals)))) #"%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_evaluatedSuccessful(props):
    if len(props) == 0:
        return "-"
    vals = []
    for p in props:
        if is_optimal_solution(p):
            if p["evolutionMode"] == "steadyState":
                vals.append(float(p["result.totalGenerations"]))
            else:
                vals.append(float(p["result.totalGenerations"]) * float(p["populationSize"]))
    if len(vals) == 0:
        return "n/a"  # -1.0, -1.0
    else:
        return str(int(round(np.mean(vals))))  # "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_runtimePerProgram(props):
    if len(props) == 0:
        return "-"  # -1.0, -1.0
    sAvgGen = get_avg_generation(props)
    if sAvgGen == "-" or sAvgGen is None:
        return "-"
    avgGen = float(sAvgGen)  # avg number of generations in all runs
    avgRuntime = float(get_avg_runtime(props))  # avg runtime of all runs
    populationSize = float(props[0]["populationSize"])
    if props[0]["evolutionMode"] == "steadyState":
        approxNumPrograms = populationSize + avgGen  # in steady state we have many generations, but in each of them created is one new program
    else:
        approxNumPrograms = populationSize * avgGen
    approxTimePerProgram = avgRuntime / approxNumPrograms
    return "%0.3f" % approxTimePerProgram
def get_sum_solverRestarts(props):
    if len(props) == 0:
        return "-"
    vals = [int(p["solver.totalRestarts"]) for p in props if "solver.totalRestarts" in p]
    if len(vals) != len(props):
        print("WARNING: solver.totalRestarts was not present in all files.")
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
    props = [p for p in props]
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
    xs = np.arange(0.0, 5.0 * 60.5 + 1e-9, 5.0) # a point every 5.0 minutes
    xticks = np.arange(0.0, 5.0 * 60.0 + 1e-9, 15.0) # a tick every 15 minutes
    plotter.plot_ratio_meeting_predicate(success_props, getter, predicate,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method, # "series_dim=None" for a single line
                                         savepath="figures/{0}_ratioTime_correctVsAllCorrect.pdf".format(exp_prefix),
                                         title="Ratio of found correct solutions out of all correct solutions",
                                         xlabel="Runtime [minutes]")
    plotter.plot_ratio_meeting_predicate(props, getter, predicate,
                                         xs=xs, xticks=xticks, show_plot=0,
                                         series_dim=dim_method,
                                         savepath="figures/{0}_ratioTime_endedVsAllEnded.pdf".format(exp_prefix),
                                         title="Ratio of ended runs",
                                         xlabel="Runtime [minutes]")
    # def get_total_evaluated(p):
    #     if "evolutionMode" not in p:
    #         return None
    #     elif p["evolutionMode"] == "steadyState":
    #         return int(p["result.totalGenerations"])
    #     else:
    #         return int(p["result.totalGenerations"]) * int(p["populationSize"])
    # xs = np.arange(0.0, 500.0 * 1000.0 + 0.01, 10000)
    # xticks = np.arange(0.0, 500.0 *1000.0 + 0.01, 50000)
    # plotter.plot_ratio_meeting_predicate(success_props, get_total_evaluated, predicate,
    #                                      xs=xs, xticks=xticks, show_plot=0,
    #                                      series_dim=dim_method,
    #                                      savepath="figures/{0}_ratioEvaluated_correctVsAllCorrect.pdf".format(exp_prefix),
    #                                      title="Ratio of found correct solutions out of all found correct solutions in the given config",
    #                                      xlabel="Number of evaluated solutions")
    # cond = lambda p: p["result.best.isOptimal"] == "true"
    # plotter.plot_ratio_meeting_predicate(props, get_total_evaluated, predicate,
    #                                      condition=cond,
    #                                      xs=xs, xticks=xticks, show_plot=0,
    #                                      series_dim=dim_method,
    #                                      savepath="figures/{0}_ratioEvaluated_correctVsAllRuns.pdf".format(exp_prefix),
    #                                      title="Ratio of runs which ended with correct solution out of all runs",
    #                                      xlabel="Number of evaluated solutions")

    # plotter.compare_avg_data_series(props, dim_methodCDGP, "CDGPtestsRatio",
    #                                 getter_y=get_successRate,
    #                                 is_aggr=1,
    #                                 savepath="figures/{0}_q_successRates.pdf".format(exp_prefix),
    #                                 title="Success rates",
    #                                 ylabel="Success rate",
    #                                 xlabel="q")

    # plotter.compare_avg_data_series_d(success_props, dim_methodCDGP, "CDGPtestsRatio", "result.best.size",
    #                                 savepath="figures/{0}_q_sizes.pdf".format(exp_prefix),
    #                                 title="Size of found correct solutions",
    #                                 ylabel="Size",
    #                                 xlabel="q")
    #
    # plotter.compare_avg_data_series_d(props, dim_methodCDGP, "CDGPtestsRatio", "tests.total",
    #                                 savepath="figures/{0}_q_tests.pdf".format(exp_prefix),
    #                                 title="Total number of test cases",
    #                                 ylabel="Total tests",
    #                                 xlabel="q")



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




def create_subsection_shared_stats(props, dim_rows, dim_cols, numRuns):
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

    print("FINAL PROPERTIES")
    print(printer.text_table(props, dim_rows, dim_cols, fun_propertiesMet, d_cols=";"))
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, fun_propertiesMet, layered_headline=True, vertical_border=vb))
    latex_propertiesMet = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

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

    subsects_main = [
        ("Status (correctly finished runs)", latex_status, reporting.color_scheme_red_r),
        ("Success rates (mse below thresh (1.0e-25) + properties met)", latex_successRates, reporting.color_scheme_green),
        ("Success rates (properties met)", latex_propertiesMet, reporting.color_scheme_green),
        ("Average runtime [s]", latex_avgRuntime, reporting.color_scheme_violet),
        ("Average runtime (only successful) [s]", latex_avgRuntimeOnlySuccessful, reporting.color_scheme_violet),
        # ("Average sizes of best of runs (number of nodes)", latex_sizes, reporting.color_scheme_yellow),
        ("Average sizes of best of runs (number of nodes) (only successful)", latex_sizesOnlySuccessful,
         reporting.color_scheme_yellow),
    ]
    return reporting.Subsection("Shared Statistics", get_content_of_subsections(subsects_main))



def create_subsection_cdgp_specific(props, dim_rows, dim_cols, exp_prefix):
    vb = 1  # vertical border

    print("AVG BEST-OF-RUN FITNESS (MSE)")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_avg_mse, layered_headline=True, vertical_border=vb))
    latex_avgBestOfRunFitness = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG TOTAL TESTS")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_avg_totalTests, layered_headline=True, vertical_border=vb))
    latex_avgTotalTests = printer.table_color_map(text, 0.0, 1000.0, 2000.0, "colorLow", "colorMedium", "colorHigh")

    # print("AVG RUNTIME PER PROGRAM")
    # text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_runtimePerProgram, layered_headline=True,
    #                                 vertical_border=vb))
    # latex_avgRuntimePerProgram = printer.table_color_map(text, 0.01, 1.0, 2.0, "colorLow", "colorMedium", "colorHigh")

    print("AVG GENERATION")
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, get_avg_generation, layered_headline=True, vertical_border=vb))
    latex_avgGeneration = printer.table_color_map(text, 0.0, 50.0, 100.0, "colorLow", "colorMedium", "colorHigh")

    # print("AVG EVALUATED SOLUTIONS")
    # text = post(
    #     printer.latex_table(props, dim_rows, dim_cols, get_avg_evaluated, layered_headline=True, vertical_border=vb))
    # latex_avgEvaluated = printer.table_color_map(text, 500.0, 25000.0, 100000.0, "colorLow", "colorMedium", "colorHigh")

    # print("AVG EVALUATED SOLUTIONS (SUCCESSFUL)")
    # text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_evaluatedSuccessful, layered_headline=True,
    #                                 vertical_border=vb))
    # latex_avgEvaluatedSuccessful = printer.table_color_map(text, 500.0, 25000.0, 100000.0, "colorLow", "colorMedium",
    #                                                         "colorHigh")

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
        ("Average best-of-run MSE", latex_avgBestOfRunFitness, reporting.color_scheme_green),
        ("Average sizes of $T_C$ (total tests in run)", latex_avgTotalTests, reporting.color_scheme_blue),
        ("Average generation (all)", latex_avgGeneration, reporting.color_scheme_teal),
        #("Average generation (only successful)", latex_avgGenerationSuccessful, reporting.color_scheme_teal),
        # ("Average evaluated solutions", latex_avgEvaluated, reporting.color_scheme_teal),
        # ("Average evaluated solutions (only successful)", latex_avgEvaluatedSuccessful, reporting.color_scheme_teal),
        # ("Approximate average runtime per program [s]", latex_avgRuntimePerProgram, reporting.color_scheme_brown),
        ("Max solver time per query [s]", latex_maxSolverTimes, reporting.color_scheme_violet),
        ("Avg solver time per query [s]", latex_avgSolverTimes, reporting.color_scheme_brown),
        ("Avg number of solver calls (in thousands; 1=1000)", latex_avgSolverTotalCalls, reporting.color_scheme_blue),
        ("Number of solver calls $>$ 0.5s", latex_numSolverCallsOverXs, reporting.color_scheme_blue),
    ]
    return reporting.Subsection("CDGP Statistics", get_content_of_subsections(subsects_cdgp))



_prev_props = None
def prepare_report(sects, fname, use_bench_simple_names=True, print_status_matrix=True, reuse_props=False,
                   paperwidth=75, include_all_row=True):
    """Creating nice LaTeX report of the results."""
    global _prev_props
    report = reporting.ReportPDF(geometry_params="[paperwidth={0}cm, paperheight=40cm, margin=0.3cm]".format(paperwidth))
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
            if p["method"] in {"CDGP"} and p["benchmark"].endswith("resistance_par2_25.sl"):
                # print(p["thisFileName"] + "   --->  " + "{0}, best={1}".format(p["result.best"], p["result.best.mse"]))
                print("isOptimal: {0};  finalVer={3};  mse={1};  program={2}".format(is_optimal_solution(p), p["result.best.mse"], p["result.best"], p["result.best.verificationDecision"]))
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
            configs = [Config(simplify_benchmark_name(c.get_caption()), c.filters[0][1],
                              benchmark=c.get_caption()) for c in
                       dim_benchmarks.configs]
            dim_benchmarks = Dim(configs)
            dim_benchmarks.sort()
        if print_status_matrix:
            d = dim_benchmarks * dim_methodGP  * dim_sel * dim_evoMode +\
                dim_benchmarks * dim_methodCDGP * dim_sel * dim_evoMode * dim_testsRatio

            matrix = produce_status_matrix(d, props)
            print("\n****** Status matrix:")
            print(matrix + "\n")
            print("Saving status matrix to file: {0}".format(STATUS_FILE_NAME))
            save_to_file(STATUS_FILE_NAME, matrix)


        dim_rows = dim_benchmarks.sort()
        if include_all_row:
            dim_rows += dim_true
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




def reports_e1():
    folders = ["e1_0.75", "e1_0.75_10", "e1_1.0"]
    #folders = ["exp4int_lex", "exp4int_lex_fix1", "exp3formal"]
    title = "Experiments for regression CDGP (stop: 5h)"
    desc = r""""""
    dimColsCdgp = dim_methodCDGP * dim_testsRatio + dim_methodGP
    dimColsShared = dimColsCdgp
    # dimColsCdgp_v2 = dim_method
    # dimColsShared_v2 = dimColsCdgp_v2
    subs = [
        (create_subsection_shared_stats, [None, dimColsShared, 25]),
        (create_subsection_cdgp_specific, [None, dimColsCdgp, "e0"]),
    ]
    # subs_v2 = [
    #     (create_subsection_shared_stats, [None, dimColsShared_v2, 25]),
    #     (create_subsection_cdgp_specific, [None, dimColsCdgp_v2, "exp4int"]),
    # ]
    figures = [
        # "figures/e0_ratioEvaluated_correctVsAllRuns.pdf",
        "figures/e0_ratioTime_correctVsAllCorrect.pdf",
        "figures/e0_ratioTime_endedVsAllEnded.pdf",
        # "figures/e0_q_successRates.pdf",
        # "figures/e0_q_sizes.pdf",
        # "figures/e0_q_tests.pdf",
    ]
    sects = [(title, desc, folders, subs, figures)]
    # sects_v2 = [(title, desc, folders, subs_v2, figures)]

    prepare_report(sects, "cdgp_e1.tex", paperwidth=30, include_all_row=False)
    # prepare_report(sects_v2, "cdgp_exp4int_v2.tex", reuse_props=True)





if __name__ == "__main__":
    ensure_dir("figures/")

    reports_e1()