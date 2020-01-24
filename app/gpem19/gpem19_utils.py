from src import utils
from src.dims import *
from src import printer
import numpy as np


CHECK_CORRECTNESS_OF_FILES = 1
STATUS_FILE_NAME = "results/status.txt"
OPT_SOLUTIONS_FILE_NAME = "opt_solutions.txt"



class TableGenerator:
    """Generates table from data."""
    def __init__(self, f_cell, dim_rows, dim_cols, headerRowNames, title="", color_scheme=None,
                 table_postprocessor=None, vertical_border=1, table_variants=None,
                 default_color_thresholds=None, layered_headline=True, color_value_extractor=None,
                 only_nonempty_rows=True, **kwargs):
        self.f_cell = f_cell
        self.dim_rows = dim_rows
        self.dim_cols = dim_cols
        self.title = title
        self.color_scheme = color_scheme
        self.table_postprocessor = table_postprocessor
        self.vertical_border = vertical_border
        self.headerRowNames = headerRowNames
        # create a table for each variant and put them next to each other
        self.table_variants = table_variants if table_variants is not None else [lambda p: True]
        self.default_color_thresholds = default_color_thresholds
        self.layered_headline = layered_headline
        self.color_value_extractor = color_value_extractor
        self.only_nonempty_rows = only_nonempty_rows
        self.init_kwargs = kwargs.copy()


    def apply(self, props, new_color_thresholds=None):
        text = ""
        for variant in self.table_variants:  # each variant is some predicate on data
            props_variant = [p for p in props if variant(p)]
            if self.only_nonempty_rows:
                dim_rows_variant = Dim([c for c in self.dim_rows.configs if len(c.filter_props(props_variant)) > 0])
            else:
                dim_rows_variant = self.dim_rows

            txt = printer.latex_table(props_variant, dim_rows_variant, self.dim_cols, self.f_cell,
                                      layered_headline=self.layered_headline, vertical_border=self.vertical_border,
                                      headerRowNames=self.headerRowNames, **self.init_kwargs)
            txt = self.table_postprocessor(txt)
            ct = new_color_thresholds if new_color_thresholds is not None else self.default_color_thresholds
            if self.color_scheme is not None and ct is not None:
                cv0, cv1, cv2 = ct
                txt = printer.table_color_map(txt, cv0, cv1, cv2, "colorLow", "colorMedium", "colorHigh", funValueExtractor=self.color_value_extractor)

            text += r"\noindent"
            text += txt
        return text



class Experiment:
    def __init__(self):
        self.tables = []
        self.listings = []



def print_props_filenames(props):
    for p in props:
        if "evoplotter.file" in p:
            print(p["evoplotter.file"])
        elif "thisFileName" in p:
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


def create_errors_solver_listing(error_props, filename, pred=None):
    if pred is None:
        pred = lambda x: True
    f = open("results/listings/{0}".format(filename), "w")
    print("Creating log of errors ({0})...".format(filename))
    for i, p in enumerate(error_props):
        if not pred(p):  # ignore properties with certain features, e.g., types of errors
            continue

        if i > 0:
            f.write("\n" + ("-" * 50) + "\n\n")

        # read the whole original file, because multiline error messages are not preserved in dicts
        with open(p["evoplotter.file"], 'r') as content_file:
            content = content_file.read()
            f.write(content)
    f.close()

def load_correct_props(folders, exts=None):
    # if exts is None:
    #     exts = [".cdgp"]
    props_cdgpError = utils.load_properties_dirs(folders, exts=[".cdgp.error"], add_file_path=True)
    props0 = utils.load_properties_dirs(folders, exts=exts, ignoreExts=[".txt", ".error"], add_file_path=True)

    # Delete wrong files
    # utils.deleteFilesByPredicate(props0, lambda p: len(p["maxGenerations"]) > 7, simulate=False)
    # utils.deleteFilesByPredicate(props_cdgpError, lambda p: len(p["maxGenerations"]) > 7, simulate=False)

    def is_correct(p):
        return p["method"] not in {"CDGP", "CDGPprops", "GP"} or "result.best.verificationDecision" in p

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
    # Terminating program due to time limit may trigger solver manager in CDGP, which shows as solver error.
    # This type of error can be recognized by the "No line found" message, so we create here a file without those false positives.
    def predSolverIssue(p):
        return "terminatingException" not in p or \
               ("No line found" not in p["terminatingException"] and "model is not available" not in p["terminatingException"])
    create_errors_solver_listing(props_cdgpError, "errors_solver_issues.txt", predSolverIssue)


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
    utils.ensure_dir("results/listings/errors/")

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
            props_final = [p for p in dc.filter_props(props_bench) if is_optimal_solution(p)]

            for p in props_final:
                fname = p["thisFileName"].replace("/home/ibladek/workspace/GECCO19/gecco19/", "")
                best = p["result.best"]
                fit = float(p["result.best.trainMSE"])
                if fit >= 1e-15:
                    f.write("{0}\t\t\t(FILE: {1}) (MSE: {2})\n".format(best, fname, fit))
                else:
                    f.write("{0}\t\t\t(FILE: {1})\n".format(best, fname))

            f.write("\n\n")
        f.close()
        f_errors.close()



def normalized_total_time(p, max_time=3600000):
    """If time was longer than max_time, then return max_time, otherwise return time. Time is counted in miliseconds."""
    if "cdgp.wasTimeout" in p and p["cdgp.wasTimeout"] == "true":
        v = 3600000
    else:
        v = int(float(p["result.totalTimeSystem"]))
    return max_time if v > max_time else v

def isOptimalVerification(p):
    if "result.best.correctVerification" not in p:
        return False
    else:
        return p["result.best.correctVerification"] == "true"
def isOptimalTests(p):
    # p["result.best.correctTests"] == "true" is computed wrongly for CDGPprops
    value = float(p["result.best.trainMSE"])
    return value < float(p["cdgp.optThresholdMSE"])
def is_optimal_solution(p):
    if p["method"] in {"CDGP", "CDGPprops", "GP"}:
        return isOptimalVerification(p) and isOptimalTests(p)
    else:
        return False




def get_num_optimal(props):
    props2 = [p for p in props if is_optimal_solution(p)]
    return len(props2)
def get_num_optimalOnlyMse(props):
    # "cdgp.optThreshold" in p and
    for p in props:
        if "optThreshold" not in p:
            print(str(p))
    # Sometimes it is 'optThreshold', and sometimes 'cdgp.optThreshold'...
    # props2 = [p for p in props if float(p["result.best.trainMSE"]) <= float(p["optThreshold"])]
    num = 0
    for p in props:
        if "optThreshold" in p:
            tr = p["optThreshold"]
        elif "optThreshold" in p:
            tr = p["cdgp.optThreshold"]
        else:
            raise Exception("No optThreshold in log file")
        if float(p["result.best.trainMSE"]) <= tr:
            num += 1
    return num

def scNotClearTrailingZeros(a):
    tab = a.split('E')
    if "-" in tab[1]:
        r = a.split('E-')[1]
        return tab[0].rstrip('0').rstrip('.') + 'E-' + r[:-1].lstrip('0') + r[-1]
    else:
        r = a.split('E')[1]
        return tab[0].rstrip('0').rstrip('.') + 'E' + r[:-1].lstrip('0') + r[-1]
def scientificNotationLatex(x):
    s = "%0.1E" % x
    if "E" not in s:
        s += "E0"
    s = "$" + s.replace("E+", "E")
    s = scNotClearTrailingZeros(s)
    s = s.replace("E", "\cdot 10^{") + "}$"
    return s

def get_num_allPropertiesMet(props):
    props2 = [p for p in props if "result.best.correctVerification" in p and p["result.best.correctVerification"] == "true"]
    return len(props2)
def get_num_trainMseBelowThresh(props):
    # "result.best.correctTests" cannot be trusted, results were wrong
    # props2 = [p for p in props if p["result.best.correctTests"] == "true"]
    numSucc = 0
    for p in props:
        value = float(p["result.best.trainMSE"])
        if value < float(p["cdgp.optThresholdMSE"]):
            numSucc += 1
    return numSucc

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
    if len(filtered) == 0 or "result.best.correctVerification" not in filtered[0]:
        return "-"
    num_opt = get_num_allPropertiesMet(filtered)
    sr = float(num_opt) / float(len(filtered))
    return "{0}".format("%0.2f" % round(sr, 2))
def fun_trainMseBelowThresh(filtered):
    if len(filtered) == 0:
        return "-"
    num_opt = get_num_trainMseBelowThresh(filtered)
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
        return "-"
    else:
        x = np.mean(vals)
        if x < 1e-5:
            x = 0.0
        return str(int(round(x))) #"%0.1f" % x
mse_dformat = "%0.4f"
def get_avg_trainMSE(props):
    vals = []
    for p in props:
        vals.append(float(p["result.best.trainMSE"]))
    if len(vals) == 0:
        return "-"
    else:
        return mse_dformat % np.mean(vals)  # , np.std(vals)
def get_median_trainMSE(props):
    vals = []
    for p in props:
        vals.append(float(p["result.best.trainMSE"]))
    if len(vals) == 0:
        return "-"
    else:
        # return mse_dformat % np.median(vals)  # , np.std(vals)
        return scientificNotationLatex(np.median(vals))
def get_median_mseOptThresh(props):
    vals = []
    for p in props:
        vals.append(float(p["cdgp.optThresholdMSE"]))
    if len(vals) == 0:
        return "-"
    else:
        # return mse_dformat % np.median(vals)  # , np.std(vals)
        return scientificNotationLatex(np.median(vals))
def get_avg_testMSE(props):
    vals = []
    for p in props:
        vals.append(float(p["result.best.testMSE"]))
    if len(vals) == 0:
        return "-"
    else:
        return mse_dformat % np.mean(vals)  # , np.std(vals)
def get_median_testMSE(props):
    if len(props) == 0:
        return "-"
    else:
        median = np.median([float(p["result.best.testMSE"]) for p in props])
        return scientificNotationLatex(median)  # , np.std(vals)
def get_median_testMSE_noScNot(props):
    vals = [float(p["result.best.testMSE"]) for p in props]
    if len(vals) == 0:
        return "-"
    else:
        return mse_dformat % np.median(vals)  # , np.std(vals)
def get_median_testMSEsuccessRateForThresh(props):
    if len(props) == 0:
        return "-"
    else:
        numSucc = 0.0
        for p in props:
            value = float(p["result.best.testMSE"])
            if value < float(p["cdgp.optThresholdMSE"]):
                numSucc += 1.0
        return "%0.2f" % round(numSucc / len(props), 2)
def get_median_testMSEsuccessRateForThresh_onlyVerified(props):
    if len(props) == 0:
        return "-"
    else:
        numSucc = 0.0
        for p in props:
            value = float(p["result.best.testMSE"])
            if is_optimal_solution(p) and value < float(p["cdgp.optThresholdMSE"]):
                numSucc += 1.0
        return "%0.2f" % round(numSucc / len(props), 2)
def get_avg_doneAlgRestarts(props):
    vals = []
    for p in props:
        if "cdgp.doneAlgRestarts" in p:
            vals.append(float(p["cdgp.doneAlgRestarts"]))
    if len(vals) == 0:
        return "-"
    else:
        return "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_runtime_helper(vals):
    if len(vals) == 0:
        return "n/a"
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
        vals = [float(normalized_total_time(p, max_time=1800000)) / 1000.0 for p in props if is_optimal_solution(p)]
        return get_avg_runtime_helper(vals)
def get_avg_runtime(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(normalized_total_time(p, max_time=1800000)) / 1000.0 for p in props]
        return get_avg_runtime_helper(vals)
def get_avg_generation(props):
    if len(props) == 0:
        return "-"
    else:
        vals = [float(p["result.totalGenerations"]) for p in props if "result.totalGenerations" in p]
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
            return "n/a"
        else:
            return str(int(round(np.mean(vals))))  # "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_evaluated(props):
    if len(props) == 0:
        return "-"
    vals = []
    for p in props:
        if "evolutionMode" in p:
            if p["evolutionMode"] == "steadyState":
                vals.append(float(p["result.totalGenerations"]))
            else:
                vals.append(float(p["result.totalGenerations"]) * float(p["populationSize"]))
    if len(vals) == 0:
        return "-"
    else:
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
        return "n/a"
    else:
        return str(int(round(np.mean(vals))))  # "%0.1f" % np.mean(vals)  # , np.std(vals)
def get_avg_runtimePerProgram(props):
    if len(props) == 0:
        return "-"
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
def get_freqCounterexamples(props):
    if len(props) == 0:
        return "-"
    import re
    counterex = {}
    for p in props:
        # e.g. "ArrayBuffer((Map(m1 -> 0.0, m2 -> 0.0, r -> 0.0),None), (Map(m1 -> 1.0, m2 -> 2.0, r -> 1.0),None)
        s = p["tests.collected"]
        cxs = re.findall("\(Map\((?:[^,]+ -> [^,]+(?:, )?)+\),None\)", s)  # ^ makes to match all charecters other than ','
        # cxs = <class 'list'>: ['(Map(m1 -> 0.0, m2 -> 0.0, r -> 0.0),None)', '(Map(m1 -> 1.0, m2 -> 2.0, r -> 1.0),None)']
        for cx in cxs:\
            # e.g. cx = (Map(m1 -> 0.0, m2 -> 0.0, r -> 0.0),None)
            if "None" in cx: # we are interested only in the noncomplete tests
                cx = cx[len("(Map("):-len("),None)")]
                # cx = 'm1 -> 0.0, m2 -> 0.0, r -> 0.0'
                fargs = re.findall("[^,]+ -> [^,]+(?:, )?", cx)  # findall returns non-overlapping matches in string
                fargs = [a.replace(", ", "") for a in fargs]
                arg_names = [a.split(" -> ")[0]  for a in fargs]
                fargs.sort(key=lambda a: a.split(" -> ")[0])
                sargs = ",".join(fargs).replace(" -> ", "=")
                if sargs in counterex:
                    counterex[sargs] += 1
                else:
                    counterex[sargs] = 1
    counterex_items = list(counterex.items())
    if len(counterex_items) == 0:
        return "n/a"
    else:
        counterex_items.sort(key=lambda x: (x[1], x[0]), reverse=True)
        NUM_SHOWN = 5
        # For some strange reason makecell doesn't work, even when it is a suggested answer (https://tex.stackexchange.com/questions/2441/how-to-add-a-forced-line-break-inside-a-table-cell)
        # return "\\makecell{" + "{0}  ({1})\\\\{2}  ({3})".format(counterex_items[0][0], counterex_items[0][1],
        #                                        counterex_items[1][0], counterex_items[1][1]) + "}"
        res = "\\pbox[l][" + str(17 * min(NUM_SHOWN, len(counterex_items))) + "pt][b]{20cm}{"
        for i in range(NUM_SHOWN):
            if i >= len(counterex_items):
                break
            if i > 0:
                res += "\\\\"
            # res += "{0}  ({1})".format(counterex_items[i][0], counterex_items[i][1])  # absolute value
            percent = round(100.0*float(counterex_items[i][1]) / len(props), 1)
            color = printer.getLatexColorCode(percent, [0., 50., 100.], ["darkred!50!white", "orange", "darkgreen"])
            res += "{0}  ({1})".format(counterex_items[i][0], r"\textbf{\textcolor{" + color + "}{" + str(percent) + "\%}}")  # percentage of runs
        res += "}"
        return res

def get_rankingOfBestSolvers(dim_ranking, NUM_SHOWN=5):
    def lambda_to_return(props):
        if len(props) == 0:
            return "-"
        valuesList = []
        for config in dim_ranking:
            name = config.get_caption()
            props2 = config.filter_props(props)
            if len(props2) > 0:
                v = np.median([float(p["result.best.testMSE"]) for p in props2])
                valuesList.append((name, float(v)))

        valuesList.sort(key=lambda x: (x[1], x[0]), reverse=False)
        # For some strange reason makecell doesn't work, even when it is a suggested answer (https://tex.stackexchange.com/questions/2441/how-to-add-a-forced-line-break-inside-a-table-cell)
        # return "\\makecell{" + "{0}  ({1})\\\\{2}  ({3})".format(counterex_items[0][0], counterex_items[0][1],
        #                                        counterex_items[1][0], counterex_items[1][1]) + "}"
        res = "\\pbox[l][" + str(17 * min(NUM_SHOWN, len(valuesList))) + "pt][c]{15cm}{"
        for i in range(NUM_SHOWN):
            if i >= len(valuesList):
                break
            if i > 0:
                res += "\\\\"
            value = float(valuesList[i][1])
            valueSc = scientificNotationLatex(value)
            nameFormatter = lambda x: r"\textcolor{darkblue}{" + str(x) + "}"  if "CDGP" in str(x) else x
            color = printer.getLatexColorCode(value, [valuesList[0][1], (valuesList[-1][1]+valuesList[0][1])/2.0, valuesList[-1][1]],
                                              ["darkgreen", "orange", "darkred!50!white"])
            res += "{0}  ({1})".format(nameFormatter(valuesList[i][0]), r"\textbf{\textcolor{" + color + "}{" + str(valueSc) + "}}")  # percentage of runs
        res += "}"
        return res
    return lambda_to_return


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
        if int(normalized_total_time(p, max_time=1800000)) <= upper_time:
            solved += 1

    for p in props:
        if int(normalized_total_time(p, max_time=1800000)) <= upper_time:
            solvedRuns += 1
    print("\nRuns which ended under {0} s:  {1} / {2}  ({3} %)".format(upper_time / 1000.0, solvedRuns, len(props), solvedRuns / len(props)))
    print("Optimal solutions found under {0} s:  {1} / {2}  ({3} %)\n".format(upper_time / 1000.0, solved, num, solved / num))


def print_status_matrix(props, dim_rows, dim_cols):
    d = dim_rows * dim_cols
    matrix = produce_status_matrix(d, props)
    print("\n****** Status matrix:")
    print(matrix + "\n")
    print("Saving status matrix to file: {0}".format(STATUS_FILE_NAME))
    utils.save_to_file(STATUS_FILE_NAME, matrix)