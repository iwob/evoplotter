import os
import shutil
from src import utils
from src.dims import *
import numpy as np


CHECK_CORRECTNESS_OF_FILES = 1
STATUS_FILE_NAME = "results/status.txt"
OPT_SOLUTIONS_FILE_NAME = "opt_solutions.txt"


def ensure_dir(file_path):
    assert file_path[-1] == "/"  # directory path must end with "/", otherwise it's not recognized
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_clear_dir(file_path):
    assert file_path[-1] == "/"  # directory path must end with "/", otherwise it's not recognized
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        shutil.rmtree(directory, ignore_errors=False)
    os.makedirs(directory)

def save_to_file(file_path, content):
    file = open(file_path, "w")
    file.write(content)
    file.close()


def delete_logs(props, pred, verbose=True, simulate=False):
    for p in props:
        if "evoplotter.file" in p and pred(p):
            path = p["evoplotter.file"]
            if not simulate:
                os.remove(path)
            if verbose:
                print("File removed: {0}".format(path))


def print_props_filenames(props):
    for p in props:
        if "thisFileName" in p:
            print(p["thisFileName"])
        else:
            print("'thisFileName' not specified! Printing content instead: " + str(p))


def create_errors_listing(error_props, filename):
    f = open("results/listings/{0}".format(filename), "w")
    print("Creating log of errors ({0})...".format(filename))
    for i, p in enumerate(error_props):
        if i > 0:
            f.write("\n" + ("-" * 50) + "\n")
        for k in sorted(p.keys()):
            v = p[k]
            f.write("{0} = {1}\n".format(k, v))
    f.close()


def create_errors_solver_listing(error_props, filename):
    f = open("results/listings/{0}".format(filename), "w")
    print("Creating log of errors ({0})...".format(filename))
    for i, p in enumerate(error_props):
        if i > 0:
            f.write("\n" + ("-" * 50) + "\n\n")

        # read the whole original file, because multiline error messages are not preserved in dicts
        with open(p["evoplotter.file"], 'r') as content_file:
            content = content_file.read()
            f.write(content)
    f.close()


def load_correct_props(folders):
    props_cdgpError = utils.load_properties_dirs(folders, exts=[".cdgp.error"], add_file_path=True)
    exts = [".cdgp"]
    props0 = utils.load_properties_dirs(folders, exts=exts, add_file_path=True)

    def is_correct(p):
        return "result.best.verificationDecision" in p

    # Filtering props so only correct ones are left
    props = [p for p in props0 if is_correct(p)]

    # print("Filtered (props):")
    # for p in props:
    #     if "resistance_par3_c1_10" in p["benchmark"] and p["method"] == "CDGP":
    #         print(p["evoplotter.file"])
    # print("Filtered (props_cdgpError):")
    # for p in props_cdgpError:
    #     if "resistance_par3_c1_10" in p["benchmark"] and p["method"] == "CDGP":
    #         print(p["evoplotter.file"])

    # Clear log file
    # print("[del] props")
    # fun = lambda p: p["method"] == "CDGP" and p["partialConstraintsInFitness"] == "true"
    # delete_logs(props, fun, simulate=True)
    # print("[del] props_cdgpError")
    # delete_logs(props_cdgpError, fun, simulate=True)


    create_errors_solver_listing(props_cdgpError, "errors_solver.txt")

    # Printing names of files which finished with error status or are incomplete.
    if CHECK_CORRECTNESS_OF_FILES:
        props_errors = [p for p in props0 if not is_correct(p)]
        create_errors_listing(props_errors, "errors_run.txt")
        if len(props_errors) > 0:
            print("Files with error status:")
            print_props_filenames(props_errors)
        print("Loaded: {0} correct property files, {1} incorrect; All log files: {2}".format(len(props), len(props_errors), len(props) + len
                                                                                                 (props_errors)))
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



def save_listings(props, dim_rows, dim_cols):
    """Saves listings of various useful info to separate text files."""
    assert isinstance(dim_rows, Dim)
    assert isinstance(dim_cols, Dim)
    ensure_dir("results/listings/errors/")

    # Saving optimal verified solutions
    for dr in dim_rows:
        bench = dr.get_caption()
        bench = bench[:bench.rfind(".")] if "." in bench else bench
        f = open("results/listings/verified_{0}.txt".format(bench), "w")
        f_errors = open("results/listings/errors/verified_{0}.txt".format(bench), "w")

        props_bench = dr.filter_props(props)
        for dc in dim_cols:
            f.write("{0}\n".format(dc.get_caption()))
            f_errors.write("{0}\n".format(dc.get_caption())) # TODO: finish
            props_final = [p for p in dc.filter_props(props_bench) if is_verified_solution(p)]

            for p in props_final:
                fname = p["thisFileName"].replace("/home/ibladek/workspace/GECCO19/gecco19/", "")
                best = p["result.best"]
                fit = float(p["result.best.mse"])
                if fit >= 1e-15:
                    f.write("{0}\t\t\t(FILE: {1}) (MSE: {2})\n".format(best, fname, fit))
                else:
                    f.write("{0}\t\t\t(FILE: {1})\n".format(best, fname))

            f.write("\n\n")
        f.close()
        f_errors.close()



def normalized_total_time(p, max_time=3600000):
    """If time was longer than max_time, then return max_time, otherwise return time."""
    if "cdgp.wasTimeout" in p and p["cdgp.wasTimeout"] == "true":
        v = 3600000
    else:
        v = int(float(p["result.totalTimeSystem"]))
    return max_time if v > max_time else v

def is_verified_solution(p):
    k = "result.best.verificationDecision"
    return p["result.best.isOptimal"] == "true" and p[k] == "unsat"

def is_approximated_solution(p):
    """Checks if the MSE was below the threshold."""
    tr = float(p["optThreshold"])
    # TODO: finish
    k = "result.best.verificationDecision"
    return p["result.best.isOptimal"] == "true" and p[k] == "unsat"

def get_num_optimal(props):
    props2 = [p for p in props if is_verified_solution(p)]
    return len(props2)
def get_num_optimalOnlyMse(props):
    # "cdgp.optThreshold" in p and
    props2 = [p for p in props if float(p["result.best.mse"]) <= float(p["cdgp.optThreshold"])]
    return len(props2)

def get_num_allPropertiesMet(props):
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
def fun_successRateMseOnly(filtered):
    if len(filtered) == 0:
        return "-"
    n = get_num_optimalOnlyMse(filtered)
    if n == 0:
        return "-"
    else:
        sr = n / float(len(filtered))
        return "{0}".format("%0.2f" % round(sr, 2))
def fun_successRate(filtered):
    if len(filtered) == 0:
        return "-"
    sr = get_successRate(filtered)
    return "{0}".format("%0.2f" % round(sr, 2))
def fun_allPropertiesMet(filtered):
    if len(filtered) == 0:
        return "-"
    num_opt = get_num_allPropertiesMet(filtered)
    sr = float(num_opt) / float(len(filtered))
    return "{0}".format("%0.2f" % round(sr, 2))
def get_stats_size(props):
    vals = [float(p["result.best.size"]) for p in props]
    if len(vals) == 0:
        return "-"#-1.0, -1.0
    else:
        return str(int(round(np.mean(vals)))) #, np.std(vals)
def get_stats_sizeOnlySuccessful(props):
    vals = [float(p["result.best.size"]) for p in props if is_verified_solution(p)]
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
        vals = [float(normalized_total_time(p)) / 1000.0 for p in props if is_verified_solution(p)]
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
        vals = [float(p["result.best.generation"]) for p in props if is_verified_solution(p)]
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
        if is_verified_solution(p):
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