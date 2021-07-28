import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app.phd.cdsr.utils import *
import evoplotter
from evoplotter import utils
from evoplotter import plotter
from evoplotter import printer
from evoplotter import reporting
from evoplotter.dims import *
from evoplotter import templates
from evoplotter.templates import *



def simplify_benchmark_name(name):
    """Shortens or modifies the path of the benchmark in order to make the table more readable."""
    i = name.rfind("/")
    name = name.replace("resistance_par", "res")
    name = name if i == -1 else name[i+1:]
    name = name[:name.rfind(".")]  # cut off '.sl'
    name = name[:name.rfind("_")]  # cut off number of tests
    # return name.replace("resistance_par", "res").replace("gravity", "gr")
    return name


def benchmark_get_num_tests(name):
    i_dot = name.rfind(".")
    i_us = name.rfind("_")
    return name[i_us+1:i_dot]


def benchmark_shorter(name):
    name = simplify_benchmark_name(name)
    i_us = name.rfind("_")
    x = name[:i_us]
    return x


def sort_benchmark_dim(d):
    assert isinstance(d, Dim)
    def s(config1):
        return config1.get_caption()  # sort benchmarks by their names
    return Dim(sorted(d.configs, key=s))


def get_benchmarks_from_props(props, simplify_names=False, ignoreNumTests=False):
    if ignoreNumTests:
        dim_benchmarks = Dim.from_dict_postprocess(props, "benchmark", fun=benchmark_shorter)
    else:
        dim_benchmarks = Dim.from_dict(props, "benchmark")
        if simplify_names:
            configs = [Config(simplify_benchmark_name(c.get_caption()), c.filters[0][1],
                              benchmark=c.get_caption()) for c in dim_benchmarks.configs]
            dim_benchmarks = Dim(configs)
        dim_benchmarks.sort()
    return sort_benchmark_dim(dim_benchmarks)


def standardize_benchmark_names(props):
    for p in props:
        p["benchmark"] = simplify_benchmark_name(p["benchmark"])



def p_method_for(name):
    return lambda p, name=name: p["method"] == name
def p_matches_dict(p, d):
    for k, v in d.items():
        if k not in p:
            return False  # throw here an exception if you know that p should always contain k
        if p[k] != v:
            return False
    return True
def p_dict_matcher(d):
    assert isinstance(d, dict)
    d = d.copy()
    return lambda p, d=d: p_matches_dict(p, d)
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
def p_true(p):
    return True



dim_true = Dim(Config("ALL", lambda p: True, method=None))
# dim_methodCDGP = Dim([Config("CDGP", p_method_for("CDGP"), method="CDGP")])
# dim_methodGP = Dim([Config("GP", p_method_for("GP"), method="GP")])
dim_methodCDGP = Dim([
    Config("CDGP", p_dict_matcher({"method": "CDGP"}), method="CDGP"),
    # Config("$CDGP_{props}$", p_dict_matcher({"method": "CDGPprops"}), method="CDGPprops"),
])
dim_methodCDGPprops = Dim([
    Config("$CDGP_{props}$", p_dict_matcher({"method": "CDGPprops"}), method="CDGPprops"),
])
dim_methodGP = Dim([
    Config("$GP$", p_dict_matcher({"method": "GP", "populationSize": "500"}), method="GP500"),
    # Config("$GP_{1000}$", p_dict_matcher({"method": "GP", "populationSize": "1000"}), method="GP1000"),
    # Config("$GP_{5000}$", p_dict_matcher({"method": "GP", "populationSize": "5000"}), method="GP5000"),
])
scikit_algs = sorted(["AdaBoostRegressor", "XGBoost", "SGDRegressor", "RandomForestRegressor", "MLPRegressor",
        "LinearSVR", "LinearRegression", "LassoLars", "KernelRidge", "GradientBoostingRegressor"]) # , "GSGP"  - not used due to issues
dim_methodScikit = Dim([Config(sa.replace("Regressor","").replace("Regression",""), p_dict_matcher({"method": sa}), method=sa) for sa in scikit_algs])
dim_method = dim_methodScikit + dim_methodCDGP + dim_methodCDGPprops #+ dim_methodGP
dim_sel = Dim([Config("$Tour$", p_sel_tourn, selection="tournament"),
               Config("$Lex$", p_sel_lexicase, selection="lexicase")])
# dim_evoMode = Dim([Config("$steadyState$", p_steadyState, evolutionMode="steadyState"),
#                    Config("$generational$", p_generational, evolutionMode="generational")])
dim_evoMode = Dim([Config("$generational$", p_generational, evolutionMode="generational")])
dim_testsRatio = Dim([
    Config("$0.75$", p_testsRatio_equalTo("0.75"), testsRatio="0.75"),
    Config("$1.0$", p_testsRatio_equalTo("1.0"), testsRatio="1.0"),
])
dim_optThreshold = Dim([
    Config("$0.01$", p_dict_matcher({"optThresholdC": "0.01"}), optThreshold="0.01"),
    # Config("$0.1$", p_dict_matcher({"optThresholdC": "0.1"}), optThreshold="0.1"),
])
dim_numGensBeforeRestart = Dim([
    # Config("$250$k", p_dict_matcher({"maxGenerations": "250000"}), optThreshold="250000"),
    Config("$50$k", p_dict_matcher({"maxGenerations": "50000"}), optThreshold="50000"),
    # Config("$25$k", p_dict_matcher({"maxGenerations": "25000"}), optThreshold="25000"),
])
dim_benchmarkNumTests = Dim([
    Config("$10$ tests", p_dict_matcher({"sizeTrainSet": "10"}), benchmarkNumTests="10"),
    Config("$100$ tests", p_dict_matcher({"sizeTrainSet": "100"}), benchmarkNumTests="100"),
])
dim_weight = Dim([
    Config("$1$", p_dict_matcher({"partialConstraintsWeight": "1"}), benchmarkNumTests="1"),
    Config("$5$", p_dict_matcher({"partialConstraintsWeight": "5"}), benchmarkNumTests="5"),
])
# variants_benchmarkNumTests = [p_dict_matcher({"sizeTrainSet": "10"}), p_dict_matcher({"sizeTrainSet": "100"})]




def get_content_of_subsections(subsects):
    content = []
    vspace = reporting.BlockLatex(r"\vspace{0.75cm}"+"\n")
    for title, table, cs in subsects:
        if isinstance(cs, reporting.ColorScheme3):
            cs = cs.toBlockLatex()
        sub = reporting.SectionRelative(title, contents=[cs, reporting.BlockLatex(table + "\n"), vspace])
        content.append(sub)
    return content

def post(s):
    s = s.replace("{ccccccccccccc}", "{rrrrrrrrrrrrr}").replace("{rrr", "{lrr")\
         .replace(r"\_{lex}", "_{lex}").replace(r"\_{", "_{").replace("resistance_par", "res")
    return s


def create_section(title, desc, props, subsects, figures_list, exp_prefix):
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


def create_single_table_bundle(props, dim_rows, dim_cols, cellLambda, headerRowNames, cv0, cv1, cv2, vb=1,
                               tableVariants=None, onlyNonemptyRows=True, tablePostprocessor=post,
                               printTextTable=False, middle_col_align="c"):
    if tableVariants is None:
        tableVariants = [p_true]
    assert isinstance(tableVariants, list)

    text = ""
    for variant in tableVariants:  # each variant is some predicate on data
        props_variant = [p for p in props if variant(p)]
        if onlyNonemptyRows:
            dim_rows_variant = Dim([c for c in dim_rows.configs if len(c.filter_props(props_variant)) > 0])
        else:
            dim_rows_variant = dim_rows

        tableText = tablePostprocessor(
            printer.latex_table(props_variant, dim_rows_variant, dim_cols, cellLambda, layered_headline=True,
                                vertical_border=vb, headerRowNames=headerRowNames, middle_col_align=middle_col_align))

        if printTextTable:
            print("VARIANT: " + str(variant))
            print(printer.text_table(props, dim_rows_variant, dim_cols, cellLambda, d_cols=";"))

        text += r"\noindent"
        text += printer.table_color_map(tableText, cv0, cv1, cv2, "colorLow", "colorMedium", "colorHigh")
        # text += "\n"

    return text


def getAndSaveTextForTableVariants(table, props):
    """Produces a list of LaTeX codes for each table variant. If outputFiles are specified,
    then the outputs will be saved there."""
    tsv = table.apply_listed(props)
    if table.outputFiles is not None:
        assert len(tsv) == len(table.outputFiles), "Number of output files must be the same as the number of table variants."
        for tsv_txt, path in zip(tsv, table.outputFiles):
            os.makedirs(os.path.dirname(path), exist_ok=True)  # automatically create directories
            f = open(path, "w")
            f.write(tsv_txt)
            f.close()
    return tsv


def createRelativeSectionForTable(tableGen, props):
    cs = tableGen.color_scheme
    if isinstance(cs, reporting.ColorScheme3):
        cs = cs.toBlockLatex()
    table_tex = " ".join(getAndSaveTextForTableVariants(tableGen, props))
    vspace = reporting.BlockLatex(r"\vspace{0.75cm}" + "\n")
    sub = reporting.SectionRelative(tableGen.title, contents=[cs, reporting.BlockLatex(table_tex + "\n"), vspace])
    return sub


def createSubsectionWithTables(title, tables, props):
    subsects_main = []
    for t in tables:
        tsv = getAndSaveTextForTableVariants(t, props)
        tup = (t.title, r"\noindent " + " ".join(tsv), t.color_scheme)
        subsects_main.append(tup)
    return reporting.Subsection(title, get_content_of_subsections(subsects_main))


def _get_median_testMSE(props):
    if len(props) == 0:
        return None
    else:
        return np.median([get_testMSE_bestOnValidSet_p(p) for p in props])
def _getAvgSatisfiedRatios(props):
    if len(props) == 0:
        return None
    elif any(["result.best.verificator.ratios" not in p for p in props]):
        return "n/a"
    sumSat = 0
    numProps = None
    for p in props:
        satVector = p["result.best.verificator.ratios"].split(",")
        satVector = [float(s) for s in satVector]
        sumSat += sum(satVector)
        numProps = len(satVector)
    avgSat = float(sumSat) / len(props)
    avgSat = avgSat / numProps
    return avgSat


def create_subsection_figures_analysis(props, dim_cols, dim_benchmarks, path, dir_path):
    """Creates a section filled with figures placed under appropriate paths."""
    if path[-1] != "/":
        path += "/"

    section = reporting.Section("Figures", [])

    # 2-dimensional scatter plot, a different one per bechmark: percent of satisfied properties vs error on test set
    for b_config in dim_benchmarks:
        props_bench = b_config.filter_props(props)
        b = b_config.get_caption(sep="_")
        fig_path = path + b + ".pdf"

        names, xs, ys = [], [], []
        min_sat_ratio = None
        for conf in dim_cols:
            props_method = conf.filter_props(props_bench)
            if len(props_method) == 0:
                continue

            name = conf.get_caption(sep="_")

            x = math.log10(_get_median_testMSE(props_method))
            y = _getAvgSatisfiedRatios(props_method)  #use 'getAvgSatStochasticVerificator' for number of satisfied properties
            if min_sat_ratio is None or y < min_sat_ratio:
                min_sat_ratio = y  # for the x axis

            if x is None or y is None:
                print("Data could not be plotted: {0}".format(name))
            else:
                names.append(name)
                xs.append(x)
                ys.append(y)


        # Idea: draw a cloud of all CDGP configs and where they landed


        # Plotting in matplotlib
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 7))  #figsize=(12, 7)
        plt.title(str(b))
        plt.margins(0.01)  # Pad margins so that markers don't get clipped by the axes
        plt.xlabel('Median MSE on test set (log10 scale)')
        plt.ylabel('Average ratio of satisfied properties')
        min_sat_ratio_norm = round_decimals_down(min_sat_ratio, 1)
        plt.yticks(np.arange(min_sat_ratio_norm, 1.01, 0.1))
        plt.ylim((min_sat_ratio_norm, 1.0))
        plt.grid('on')
        plt.scatter(xs, ys)
        for i, txt in enumerate(names):
            ax.annotate(txt, (xs[i], ys[i]))


        # Drawing small points for CDGP configs
        # TODO

        plt.savefig(dir_path + fig_path)
        # plt.show()
        plt.clf()

        # Plotting in seaborn
        import seaborn as sns
        # sns.set_theme(style="darkgrid")
        # sns.relplot(data=)

        section.add(reporting.BlockLatex(r"\includegraphics{" + fig_path + r"}\\"))

    return section




def create_subsection_figures_analysis_seaborn(dataFrame, path):
    """Creates a section filled with figures placed under appropriate paths."""
    if path[-1] != "/":
        path += "/"

    section = reporting.Section("Figures", [])

    # "evoplotter.file",
    # "benchmark",
    # "method",
    # "selection",
    # "testsRatio",
    # "partialConstraintsWeight",
    # "result.best",
    # "result.best.smtlib",
    # "result.best.size",
    # "result.best.correctVerification",
    # "result.best.passedConstraints",
    # "result.validation.best.passedConstraints",
    # "result.best.verificator.decisions",
    # "result.best.verificator.ratios",
    # (getAvgSatRatioStochasticVerifier_p, "result.best.verificator.satRatio"),
    # "result.best.trainMSE",
    # "result.best.validMSE",
    # "result.best.testMSE",
    # (get_trainMSE_bestOnValidSet_p, "result.validation.best.trainMSE"),
    # (get_validMSE_bestOnValidSet_p, "result.validation.best.validMSE"),
    # (get_testMSE_bestOnValidSet_p, "result.validation.best.testMSE"),
    import seaborn as sns
    # g = sns.FacetGrid(dataFrame, row="benchmark")
    # g.map(sns.scatterplot, x="result.best.testMSE", y="result.best.verificator.satRatio", hue="method")
    x = math.log10(dataFrame["result.best.testMSE"])
    sns_plot = sns.relplot(data=dataFrame, x=x, y="result.best.verificator.satRatio", hue=dataFrame["method"].tolist())
    fig_path = path + "scatterplot" + ".pdf"
    sns_plot.savefig(fig_path)
    section.add(reporting.BlockLatex(r"\includegraphics{" + fig_path + r"}\\"))


    # 2-dimensional scatter plot, a different one per bechmark: percent of satisfied properties vs error on test set
    # for b_config in dim_benchmarks:
    #     props_bench = b_config.filter_props(props)
    #     b = b_config.get_caption(sep="_")
    #     fig_path = path + b + ".pdf"
    #
    #     names, xs, ys = [], [], []
    #     min_sat_ratio = None
    #     for conf in dim_cols:
    #         props_method = conf.filter_props(props_bench)
    #         if len(props_method) == 0:
    #             continue
    #
    #         name = conf.get_caption(sep="_")
    #
    #         x = math.log10(_get_median_testMSE(props_method))
    #         y = _getAvgSatisfiedRatios(props_method)  #use 'getAvgSatStochasticVerificator' for number of satisfied properties
    #         if min_sat_ratio is None or y < min_sat_ratio:
    #             min_sat_ratio = y  # for the x axis
    #
    #         if x is None or y is None:
    #             print("Data could not be plotted: {0}".format(name))
    #         else:
    #             names.append(name)
    #             xs.append(x)
    #             ys.append(y)
    #
    #
    #     # Idea: draw a cloud of all CDGP configs and where they landed
    #
    #
    #     # Plotting in matplotlib
    #     plt.clf()
    #     fig, ax = plt.subplots(figsize=(12, 7))  #figsize=(12, 7)
    #     plt.title(str(b))
    #     plt.margins(0.01)  # Pad margins so that markers don't get clipped by the axes
    #     plt.xlabel('Median MSE on test set (log10 scale)')
    #     plt.ylabel('Average ratio of satisfied properties')
    #     min_sat_ratio_norm = round_decimals_down(min_sat_ratio, 1)
    #     plt.yticks(np.arange(min_sat_ratio_norm, 1.01, 0.1))
    #     plt.ylim((min_sat_ratio_norm, 1.0))
    #     plt.grid('on')
    #     plt.scatter(xs, ys)
    #     for i, txt in enumerate(names):
    #         ax.annotate(txt, (xs[i], ys[i]))
    #
    #
    #     # Drawing small points for CDGP configs
    #     # TODO
    #
    #     plt.savefig("reports/" + fig_path)
    #     # plt.show()
    #     plt.clf()

    # Plotting in seaborn
    import seaborn as sns
    # sns.set_theme(style="darkgrid")
    # sns.relplot(data=)

    # section.add(reporting.BlockLatex(r"\includegraphics{"  + fig_path + r"}\\"))

    return section


def scNotValueExtractor(s):
    if s == "-" or "10^{" not in s:
        return s
    else:
        numbers = re.findall("[-]?[0-9]+[.]?[0-9]*", s)
        if len(numbers) != 3:
            return s  # not a scientific notation
        mantissa = float(numbers[0])
        base = float(numbers[1])
        exp = float(numbers[2])
        return mantissa * base ** exp


def scNotLog10ValueExtractor(s):
    base = 10.0
    v = scNotValueExtractor(s)
    if isinstance(v, str): # scientific notation was not interpreted correctly
        return s
    else:
        return math.log(v, base)


def scNotExponentExtractor(s):
    if s == "-" or "10^{" not in s:
        return s
    else:
        r = s.split("10^{")
        return r[1][:-2]


def create_subsection_figures(props, dim_rows, dim_cols, exp_prefix, dir_path):
    if len(props) == 0:
        print("No props: plots were not generated.")
        return

    section = reporting.Section("Figures", [])

    getter_mse = lambda p: float(p["result.best.trainMSE"])
    predicate = lambda v, v_xaxis: v <= v_xaxis
    N = 50  # number of points per plot line
    r = (0.0, 1e0)
    xs = np.linspace(r[0], r[1], N)
    xticks = np.arange(r[0], r[1], r[1] / 10)
    savepath = "reports/figures/ratioMSE.pdf"
    plotter.plot_ratio_meeting_predicate(props, getter_mse, predicate, xs=xs, xticks=xticks,
                                         show_plot=False,
                                         title="Ratio of solutions with MSE under the certain level",
                                         xlabel="MSE",
                                         series_dim=dim_method,
                                         xlogscale=False,
                                         savepath=savepath)
    section.add(reporting.FloatFigure(savepath.replace("reports/", "")))


    # Illustration of individual runs and their errors on training and validation sets
    savepath = "reports/figures/progressionGrid.pdf"
    dim_rows = get_benchmarks_from_props(props, ignoreNumTests=True)
    dim_cols = (dim_methodGP * dim_all + dim_methodCDGP * dim_all + dim_methodCDGPprops * dim_weight) * \
               dim_benchmarkNumTests  # * dim_optThreshold
    plotter.plot_value_progression_grid_simple(props, dim_rows, dim_cols, ["cdgp.logTrainSet", "cdgp.logValidSet"], ["train", "valid"],
                                               plot_individual_runs=True,
                                               savepath=savepath)
    section.add(reporting.FloatFigure(savepath.replace("reports/", "")))
    return section


def cellShading(a, b, c):
    assert c >= b >= a
    return printer.CellShading(a, b, c, "colorLow", "colorMedium", "colorHigh")

def cellScNotShading(a, b, c):
    assert c >= b >= a
    return printer.CellShading(a, b, c, "colorLow", "colorMedium", "colorHigh", valueExtractor=scNotValueExtractor)


rBoldWhen1 = printer.LatexTextbf(lambda v, b: v == "1.00")
thesis_color_scheme = reporting.ColorScheme3(["1.0, 1.0, 1.0", "0.9, 0.9, 0.9", "0.75, 0.75, 0.75"], ["white", "gray", "gray"])



def create_subsection_shared_status(props, title, dim_rows, dim_cols, numRuns, headerRowNames):
    vb = 1  # vertical border
    variants = None  # variants_benchmarkNumTests
    dim_rows_v2 = get_benchmarks_from_props(props, ignoreNumTests=True)

    color_scheme_red_status = reporting.ColorScheme3(["1.0, 1.0, 1.0", "0.92, 0.3, 0.3", "0.8, 0, 0"],
                                    ["white", "light red", "red"])

    tables = [
        TableGenerator(get_num_computed, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Status (correctly finished runs)",
                       color_scheme=reversed(color_scheme_red_status),
                       default_color_thresholds=(0.8 * numRuns, 0.9 * numRuns, numRuns),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       )
    ]

    subsects_main = []
    for t in tables:
        tup = (t.title, t.apply(props), t.color_scheme)
        subsects_main.append(tup)

    return reporting.Subsection(title, get_content_of_subsections(subsects_main))



def create_subsection_shared_stats(props, title, dim_rows, dim_cols, numRuns, headerRowNames):
    vb = 1  # vertical border
    variants = None  # variants_benchmarkNumTests
    dim_rows_v2 = get_benchmarks_from_props(props, ignoreNumTests=True)
    # dim_rows_v2 += dim_true  #TODO: within dict

    print("\nFiles with a conflict between SMT and stochastic verificators:")
    for p in props:
        if "CDGP" in p["method"]:
            if not p_allPropertiesMet_verificator(p) and p_allPropertiesMet_smt(p) and "nguyen1" in p["benchmark"]:
                print(p["evoplotter.file"])

    tables = [
        # TableGenerator(fun_successRate, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Success rates (properties met + mse below thresh)",
        #                color_scheme=reporting.color_scheme_darkgreen,
        #                default_color_thresholds=(0.0, 0.5, 1.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        # TableGenerator(get_median_testMSEsuccessRateForThresh_onlyVerified, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Test set: success rate of accepted solutions with MSE below optThreshold  (i.e., no overfitting)",
        #                color_scheme=reporting.color_scheme_red2white2darkgreen,
        #                default_color_thresholds=(0.0, 0.5, 1.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        # TableGenerator(fun_trainMseBelowThresh, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Training set: success rates (mse below thresh)",
        #                color_scheme=reporting.color_scheme_green,
        #                default_color_thresholds=(0.0, 0.5, 1.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        # TableGenerator(get_median_mseOptThresh, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Training set: optThreshold for MSE  (median)",
        #                color_scheme=reporting.color_scheme_violet,
        #                default_color_thresholds=(-10.0, 0.0, 10.0),
        #                color_value_extractor=scNotColorValueExtractor,
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        TableGenerator(get_median_trainMSE, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Training set: MSE  (median); bestOfRun CDGP",
                       color_scheme=reversed(reporting.color_scheme_gray_dark),
                       cellRenderers=[
                           printer.LatexTextbfMinInRow(valueExtractor=scNotLog10ValueExtractor, isBoldMathMode=True),
                           printer.CellShadingRow("colorLow", "colorMedium", "colorHigh", valueExtractor=scNotLog10ValueExtractor)],
                       vertical_border=vb, table_postprocessor=post, variants=variants, middle_col_align="l",
                       addRowWithRanks=True, ranksHigherValuesBetter=False, valueExtractor=scNotValueExtractor
                       ),
        # TableGenerator(get_median_testMSE, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Test set: MSE  (median); bestOfRun CDGP",
        #                color_scheme=reversed(reporting.color_scheme_gray_dark),
        #                cellRenderers=[
        #                    printer.LatexTextbfMinInRow(valueExtractor=scNotLog10ValueExtractor, isBoldMathMode=True),
        #                    printer.CellShadingRow("colorLow", "colorMedium", "colorHigh",
        #                                           valueExtractor=scNotLog10ValueExtractor)],
        #                vertical_border=vb, table_postprocessor=post, variants=variants,
        #                ),
        TableGenerator(get_median_testMSE_bestOnValidCDGP, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Test set: MSE  (median); bestOnValidSet CDGP",
                       color_scheme=reversed(reporting.color_scheme_gray_dark),
                       cellRenderers=[
                           printer.LatexTextbfMinInRow(valueExtractor=scNotLog10ValueExtractor, isBoldMathMode=True),
                           printer.CellShadingRow("colorLow", "colorMedium", "colorHigh",
                                                  valueExtractor=scNotLog10ValueExtractor)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       addRowWithRanks=True, ranksHigherValuesBetter=False, valueExtractor=scNotValueExtractor
                       ),
        TableGenerator(get_averageAlgorithmRanksCDSR(dim_cols[:-1], dim_rows[:-1], ONLY_VISIBLE_SOLS=True, NUM_SHOWN=100),
                       Dim(dim_cols[-1]), Dim(dim_rows[-1]),
                       headerRowNames=headerRowNames,
                       title="Average ranks of the solvers (median MSE on test set)",
                       color_scheme=reporting.color_scheme_violet,
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_rankingOfBestSolversCDSR(dim_cols[:-1], ONLY_VISIBLE_SOLS=True, NUM_SHOWN=100),
                       Dim(dim_cols[-1]), dim_rows,
                       headerRowNames=headerRowNames,
                       title="Best solvers for the given benchmark (median MSE on test set)",
                       color_scheme=reporting.color_scheme_violet,
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_allPropertiesMet, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Success rates (properties met) -- verified formally by SMT solver",
                       color_scheme=reporting.color_scheme_green,
                       # default_color_thresholds=(0.0, 0.5, 1.0),
                       cellRenderers=[
                           printer.LatexTextbfMaxInRow(),
                           cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(fun_allPropertiesMet_verificator, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Success rates (properties met) -- verified by stochastic verificator",
                       color_scheme=reporting.color_scheme_green,
                       # default_color_thresholds=(0.0, 0.5, 1.0),
                       cellRenderers=[
                           printer.LatexTextbfMaxInRow(),
                           cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(getAvgSatisfiedProps1, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average ratio of satisfied properties -- verified by SMT solver (CDSR configs) and stochastic verificator (scikit baselines)",
                       color_scheme=reporting.color_scheme_green,
                       # default_color_thresholds=(0.0, 0.5, 1.0),
                       cellRenderers=[
                           printer.LatexTextbfMaxInRow(),
                           cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(get_avg_runtime, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average runtime [s]",
                       color_scheme=reporting.color_scheme_violet,
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(get_avg_runtimeOnlySuccessful, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Average runtime (only successful) [s]",
        #                color_scheme=reporting.color_scheme_violet,
        #                default_color_thresholds=(0.0, 900.0, 1800.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
    ]

    subsects_main = []
    for t in tables:
        tup = (t.title, t.apply(props), t.color_scheme)
        subsects_main.append(tup)

    return reporting.Subsection(title, get_content_of_subsections(subsects_main))



def create_subsection_ea_stats(props, title, dim_rows, dim_cols, headerRowNames):
    vb = 1  # vertical border
    variants = None  # variants_benchmarkNumTests
    dim_rows_v2 = get_benchmarks_from_props(props, ignoreNumTests=True)
    # dim_rows_v2 += dim_true  # TODO: within dict

    tables = [
        TableGenerator(get_rankingOfBestSolutionsCDSR(ONLY_VISIBLE_SOLS=True, NUM_SHOWN=15, showSimplified=True),
                       Dim(dim_cols.configs[:-1]), Dim(dim_rows.configs[:-1]),
                       headerRowNames=headerRowNames,
                       title="The best solutions found for each benchmark and their sizes. Format: solution (MSE on test set) (size)",
                       color_scheme=reporting.color_scheme_violet, middle_col_align="l",
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_stats_size, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average sizes of best of runs (number of nodes)",
                       color_scheme=reporting.color_scheme_yellow,
                       default_color_thresholds=(0.0, 100.0, 200.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(get_validation_testSet_ratio, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="A ratio of bestOfRun / bestOfValidation for MSE on test set.",
        #                color_scheme=reporting.color_scheme_green,
        #                default_color_thresholds=(0.0, 100.0, 200.0),
        #                vertical_border=vb, table_postprocessor=post, variants=variants,
        #                ),
        # TableGenerator(get_stats_sizeOnlySuccessful, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Average sizes of best of runs (number of nodes) (only successful)",
        #                color_scheme=reporting.color_scheme_yellow,
        #                default_color_thresholds=(0.0, 100.0, 200.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        TableGenerator(get_avg_generation, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average generation (all)",
                       color_scheme=reporting.color_scheme_teal,
                       default_color_thresholds=(0.0, 5000.0, 10000.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_avg_evaluated, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average number of evaluated solutions",
                       color_scheme=reporting.color_scheme_brown,
                       default_color_thresholds=(0.0, 5000.0, 10000.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(get_avg_evaluatedSuccessful, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Average number of evaluated solutions",
        #                color_scheme=reporting.color_scheme_brown,
        #                default_color_thresholds=(500.0, 25000.0, 100000.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        # TableGenerator(get_avg_doneAlgRestarts, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Number of algorithm restarts  (avg)",
        #                color_scheme=reporting.color_scheme_gray_light,
        #                default_color_thresholds=(0.0, 1e2, 1e4),
        #                vertical_border=vb, table_postprocessor=post, variants=variants,
        #                ),
    ]

    subsects_main = []
    for t in tables:
        tup = (t.title, t.apply(props), t.color_scheme)
        subsects_main.append(tup)

    return reporting.Subsection(title, get_content_of_subsections(subsects_main))



def create_subsection_cdgp_specific(props, title, dim_rows, dim_cols, headerRowNames):
    vb = 1  # vertical border
    # variants = variants_benchmarkNumTests
    variants = None

    print("AVG TOTAL TESTS")
    latex_avgTotalTests = create_single_table_bundle(props, dim_rows, dim_cols, get_avg_totalTests, headerRowNames,
                                                     cv0=0.0, cv1=1000.0, cv2=2000.0, tableVariants=variants)

    print("MAX SOLVER TIME")
    latex_maxSolverTimes = create_single_table_bundle(props, dim_rows, dim_cols, get_stats_maxSolverTime, headerRowNames,
                                                     cv0=0.0, cv1=5.0, cv2=10.0, tableVariants=variants)

    print("AVG SOLVER TIME")
    latex_avgSolverTimes = create_single_table_bundle(props, dim_rows, dim_cols, get_stats_avgSolverTime, headerRowNames,
                                                      cv0=0.0, cv1=0.015, cv2=0.03, tableVariants=variants)

    print("AVG NUM SOLVER CALLS")
    latex_avgSolverTotalCalls = create_single_table_bundle(props, dim_rows, dim_cols, get_avgSolverTotalCalls, headerRowNames,
                                                      cv0=1e1, cv1=1e2, cv2=1e4, tableVariants=variants)

    print("NUM SOLVER CALLS > 0.5s")
    latex_numSolverCallsOverXs = create_single_table_bundle(props, dim_rows, dim_cols, get_numSolverCallsOverXs, headerRowNames,
                                                           cv0=0, cv1=50, cv2=100, tableVariants=variants)

    print("MOST FREQUENTLY FOUND COUNTEREXAMPLE")
    latex_freqCounterexamples = create_single_table_bundle(props, dim_rows, dim_cols, get_freqCounterexamples, headerRowNames,
                                                           cv0=0, cv1=50, cv2=100, tableVariants=variants, middle_col_align="l")

    subsects_cdgp = [
        ("Average sizes of $T_C$ (total tests in run)", latex_avgTotalTests, reporting.color_scheme_blue),
        ("Max solver time per query [s]", latex_maxSolverTimes, reporting.color_scheme_violet),
        ("Avg solver time per query [s]", latex_avgSolverTimes, reporting.color_scheme_brown),
        ("Avg number of solver calls (in thousands; 1=1000)", latex_avgSolverTotalCalls, reporting.color_scheme_blue),
        ("Number of solver calls $>$ 0.5s", latex_numSolverCallsOverXs, reporting.color_scheme_blue),
        ("The most frequently found counterexamples for each benchmark and configuration", latex_freqCounterexamples, reporting.color_scheme_violet),
    ]
    return reporting.Subsection(title, get_content_of_subsections(subsects_cdgp))




def create_subsection_custom_tables(props, title, dimens, exp_variant, dir_path, variants=None):
    assert isinstance(dimens, dict)
    assert exp_variant == "noNoise" or exp_variant == "withNoise"
    vb = 1  # vertical border

    dim_cdsr_methods = dimens["method_CDGP"] * dimens["selection"] +\
                       dimens["method_CDGPprops"] * dimens["selection"] * dimens["weight"]

    dim_cdsr_methods_full = dimens["method_CDGP"] * dimens["selection"] * dimens["testsRatio"] + \
                            dimens["method_CDGPprops"] * dimens["selection"] * dimens["testsRatio"] * dimens["weight"]

    def postprocessorMse(s):
        s = post(s)

        s = s.replace(r"10^{-", r"10^{\unaryminus ")

        s = s.replace("AdaBoost", r"\makecell[tc]{Ada-\\Boost}")
        s = s.replace("GradientBoosting", r"\makecell[tc]{Gradient-\\Boosting}")
        s = s.replace("KernelRidge", r"\makecell[tc]{Kernel-\\Ridge}")
        s = s.replace("LassoLars", r"\makecell[tc]{Lasso-\\Lars}")
        s = s.replace("LinearSVR", r"\makecell[tc]{Linear-\\SVR}")
        s = s.replace("RandomForest", r"\makecell[tc]{Random-\\Forest}")
        s = s.replace("XGBoost", r"\makecell[tc]{XG-\\Boost}")
        return s

    # shTc = cellShading(0.0, 5000.0, 10000.0) if EXP_TYPE == "LIA" else cellShading(0.0, 250.0, 500.0)
    tables = [
        # scikit configs
        TableGenerator(get_median_testMSE,
                       dimens["benchmark"],
                       dimens["method_scikit"],
                       title="Test set: MSE  (median); bestOfRun CDGP", headerRowNames=[],
                       color_scheme=reversed(thesis_color_scheme),
                       cellRenderers=[printer.LatexTextbfMinInRow(valueExtractor=scNotLog10ValueExtractor, isBoldMathMode=True),
                                      printer.CellShadingRow("colorLow", "colorMedium", "colorHigh", valueExtractor=scNotLog10ValueExtractor)],
                       vertical_border=vb, table_postprocessor=postprocessorMse, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/scikit/scikit_testMSE_{}.tex".format(exp_variant)],
                       middle_col_align="l", addRowWithRanks=True, ranksHigherValuesBetter=False,
                       valueExtractor=scNotValueExtractor
                       ),
        TableGenerator(fun_allPropertiesMet_verificator,
                       dimens["benchmark"],
                       dimens["method_scikit"],
                       title="Success rate in terms of all properties met (stochastic verifier)", headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[rBoldWhen1, cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/scikit/scikit_succRate_{}.tex".format(exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(getAvgSatisfiedProps1,
                       dimens["benchmark"],
                       dimens["method_scikit"],
                       title="Average ratio of satisfied properties", headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/scikit/scikit_satConstrRatio_{}.tex".format(exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(getAvgSatisfiedProps1,
                       dimens["benchmark"],
                       dimens["method_scikit"],
                       title="Average ratio of satisfied properties", headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), printer.CellShadingRow()],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/scikit/_scikit_satConstrRatio_{}_rowShading.tex".format(exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        FriedmannTestPython(dimens["benchmark"],
                            dimens["method_scikit"],
                            getAvgSatisfiedProps1, p_treshold=0.05,
                            title="Friedman test for average ratio of satisfied properties",
                            pathFriedmanViz="tables/custom/scikit/friedman_scikit_avgSatConstr.gv",
                            workingDir=dir_path, higherValuesBetter=True),
        FriedmannTestPython(dimens["benchmark"],
                            dimens["method_scikit"],
                            get_median_testMSE_noScNotation, p_treshold=0.05,
                            title="Friedman test for median MSE on test set",
                            pathFriedmanViz="tables/custom/scikit/friedman_scikit_testMSE.gv",
                            workingDir=dir_path, higherValuesBetter=False),
        # CDSR configs
        TableGenerator(get_median_testMSE,
                       dimens["benchmark"],
                       dim_cdsr_methods,
                       title="Test set: MSE  (median); bestOfRun CDGP", headerRowNames=[],
                       color_scheme=reversed(thesis_color_scheme),
                       cellRenderers=[
                           printer.LatexTextbfMinInRow(valueExtractor=scNotLog10ValueExtractor, isBoldMathMode=True),
                           printer.CellShadingRow("colorLow", "colorMedium", "colorHigh",
                                                  valueExtractor=scNotLog10ValueExtractor)],
                       vertical_border=vb, table_postprocessor=postprocessorMse, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/cdsr/cdsr_testMSE_{}.tex".format(exp_variant)],
                       middle_col_align="l", addRowWithRanks=True, ranksHigherValuesBetter=False,
                       valueExtractor=scNotValueExtractor
                       ),
        TableGenerator(fun_allPropertiesMet_verificator,
                       dimens["benchmark"],
                       dim_cdsr_methods,
                       title="Success rate in terms of all properties met (stochastic verifier)", headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/cdsr/cdsr_succRate_{}.tex".format(exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(getAvgSatisfiedProps1,
                       dimens["benchmark"],
                       dim_cdsr_methods,
                       title="Average ratio of satisfied properties", headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/cdsr/cdsr_satConstrRatio_{}.tex".format(exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        FriedmannTestPython(dimens["benchmark"],
                            dim_cdsr_methods,
                            getAvgSatisfiedProps1, p_treshold=0.05,
                            title="Friedman test for average ratio of satisfied properties",
                            pathFriedmanViz="tables/custom/cdsr/friedman_cdsr_avgSatConstr.gv",
                            workingDir=dir_path, higherValuesBetter=True),
        FriedmannTestPython(dimens["benchmark"],
                            dim_cdsr_methods,
                            get_median_testMSE_noScNotation, p_treshold=0.05,
                            title="Friedman test for median MSE on test set",
                            pathFriedmanViz="tables/custom/cdsr/friedman_cdsr_testMSE.gv",
                            workingDir=dir_path, higherValuesBetter=False),
        # CDSR configs (full dimensions)
        TableGenerator(get_median_testMSE,
                       dimens["benchmark"],
                       dim_cdsr_methods_full,
                       title="Test set: MSE  (median); bestOfRun CDGP", headerRowNames=[],
                       color_scheme=reversed(thesis_color_scheme),
                       cellRenderers=[
                           printer.LatexTextbfMinInRow(valueExtractor=scNotLog10ValueExtractor, isBoldMathMode=True),
                           printer.CellShadingRow("colorLow", "colorMedium", "colorHigh",
                                                  valueExtractor=scNotLog10ValueExtractor)],
                       vertical_border=vb, table_postprocessor=postprocessorMse, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/cdsrFull/cdsrFull_testMSE_{}.tex".format(exp_variant)],
                       middle_col_align="l", addRowWithRanks=True, ranksHigherValuesBetter=False,
                       valueExtractor=scNotValueExtractor
                       ),
        TableGenerator(fun_allPropertiesMet_verificator,
                       dimens["benchmark"],
                       dim_cdsr_methods_full,
                       title="Success rate in terms of all properties met (stochastic verifier)", headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/cdsrFull/cdsrFull_succRate_{}.tex".format(exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(getAvgSatisfiedProps1,
                       dimens["benchmark"],
                       dim_cdsr_methods_full,
                       title="Average ratio of satisfied properties", headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")],  #cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/cdsrFull/cdsrFull_satConstrRatio_{}.tex".format(exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        FriedmannTestPython(dimens["benchmark"],
                            dim_cdsr_methods_full,
                            getAvgSatisfiedProps1, p_treshold=0.05,
                            title="Friedman test for average ratio of satisfied properties",
                            pathFriedmanViz="tables/custom/cdsrFull/friedman_cdsrFull_avgSatConstr.gv",
                            workingDir=dir_path, higherValuesBetter=True),
        FriedmannTestPython(dimens["benchmark"],
                            dim_cdsr_methods_full,
                            get_median_testMSE_noScNotation, p_treshold=0.05,
                            title="Friedman test for median MSE on test set",
                            pathFriedmanViz="tables/custom/cdsrFull/friedman_cdsrFull_testMSE.gv",
                            workingDir=dir_path, higherValuesBetter=False),
        # TableGenerator(get_avg_totalTests,
        #                dim_rows, dim_cols,
        #                headerRowNames=[],
        #                title="Average sizes of $T_C$ (total tests in run)",
        #                color_scheme=reporting.color_scheme_blue, middle_col_align="l",
        #                cellRenderers=[shTc],
        #                vertical_border=vb, table_postprocessor=post, variants=variants,
        #                outputFiles=[
        #                    results_dir + "/tables/custom/cdgp_Tc_rowsAsTestsRatio.tex"]
        #                ),
        # TableGenerator(get_avg_runtime, dim_rows, dim_cols, headerRowNames=[],
        #                title="Average runtime [s]",
        #                color_scheme=reporting.color_scheme_violet,
        #                cellRenderers=[cellShading(0.0, 900.0, 1800.0)],
        #                vertical_border=vb, table_postprocessor=post, variants=variants,
        #                outputFiles=[results_dir + "/tables/custom/cdgp_runtime_rowsAsTestsRatio.tex"]
        #                ),
        # TableGenerator(get_avg_runtimeOnlySuccessful, dim_rows, dim_cols, headerRowNames=[],
        #                title="Average runtime (only successful) [s]",
        #                color_scheme=reporting.color_scheme_violet,
        #                cellRenderers=[cellShading(0.0, 900.0, 1800.0)],
        #                vertical_border=vb, table_postprocessor=post, variants=variants,
        #                outputFiles=[results_dir + "/tables/custom/cdgp_runtime_rowsAsTestsRatio_successful.tex"]
        #                ),
    ]

    return createSubsectionWithTables(title, tables, props)



def create_subsection_individual_constraints(props, title, dimens, exp_variant, dir_path, variants=None):
    assert isinstance(dimens, dict)
    assert exp_variant == "noNoise" or exp_variant == "withNoise"
    vb = 1  # vertical border

    dim_cdsr_methods = dimens["method_CDGP"] * dimens["selection"] + \
                       dimens["method_CDGPprops"] * dimens["selection"] * dimens["weight"]

    dim_cdsr_methods_full = dimens["method_CDGP"] * dimens["selection"] * dimens["testsRatio"] + \
                            dimens["method_CDGPprops"] * dimens["selection"] * dimens["testsRatio"] * dimens["weight"]

    c_arg_symmetry = "brown!30"
    c_value_bound = "blue!20"
    c_value_bound2 = "blue!35"
    c_monotonicity = "green!20"
    c_equality = "red!20"

    def postprocessor(s):
        s = post(s)

        s = s.replace("AdaBoost", r"\makecell[tc]{Ada-\\Boost}")
        s = s.replace("GradientBoosting", r"\makecell[tc]{Gradient-\\Boosting}")
        s = s.replace("KernelRidge", r"\makecell[tc]{Kernel-\\Ridge}")
        s = s.replace("LassoLars", r"\makecell[tc]{Lasso-\\Lars}")
        s = s.replace("LinearSVR", r"\makecell[tc]{Linear-\\SVR}")
        s = s.replace("RandomForest", r"\makecell[tc]{Random-\\Forest}")
        s = s.replace("XGBoost", r"\makecell[tc]{XG-\\Boost}")

        col_arg_symmetry = r"\cellcolor{" + c_arg_symmetry + "}"
        col_value_bound = r"\cellcolor{" + c_value_bound + "}"
        col_value_bound2 = r"\cellcolor{" + c_value_bound2 + "}"
        col_monotonicity = r"\cellcolor{" + c_monotonicity + "}"
        col_equality = r"\cellcolor{" + c_equality + "}"

        s = s.replace("gravity-0", "{}gravity-0".format(col_arg_symmetry))
        s = s.replace("keijzer14-3", "{}keijzer14-3".format(col_arg_symmetry))
        s = s.replace("pagie1-2", "{}pagie1-2".format(col_arg_symmetry))
        s = s.replace("res2-0", "{}res2-0".format(col_arg_symmetry))
        s = s.replace("res3-0", "{}res3-0".format(col_arg_symmetry))
        s = s.replace("res3-1", "{}res3-1".format(col_arg_symmetry))
        s = s.replace("res3-2", "{}res3-2".format(col_arg_symmetry))

        s = s.replace("gravity-1", "{}gravity-1".format(col_value_bound))
        s = s.replace("keijzer14-0", "{}keijzer14-0".format(col_value_bound))
        s = s.replace("keijzer14-1", "{}keijzer14-1".format(col_value_bound))
        s = s.replace("keijzer5-1", "{}keijzer5-1".format(col_value_bound))
        s = s.replace("keijzer5-2", "{}keijzer5-2".format(col_value_bound))
        s = s.replace("keijzer15-1", "{}keijzer15-1".format(col_value_bound))
        s = s.replace("keijzer15-2", "{}keijzer15-2".format(col_value_bound))
        s = s.replace("nguyen1-0", "{}nguyen1-0".format(col_value_bound))
        s = s.replace("nguyen1-1", "{}nguyen1-1".format(col_value_bound))
        s = s.replace("nguyen3-0", "{}nguyen3-0".format(col_value_bound))
        s = s.replace("nguyen3-1", "{}nguyen3-1".format(col_value_bound))
        s = s.replace("nguyen4-0", "{}nguyen4-0".format(col_value_bound))
        s = s.replace("nguyen4-1", "{}nguyen4-1".format(col_value_bound))
        s = s.replace("pagie1-0", "{}pagie1-0".format(col_value_bound))
        s = s.replace("pagie1-1", "{}pagie1-1".format(col_value_bound))
        s = s.replace("res2-2", "{}res2-2".format(col_value_bound))
        s = s.replace("res3-3", "{}res3-3".format(col_value_bound))
        s = s.replace("res3-4", "{}res3-4".format(col_value_bound))

        s = s.replace("res2-1", "{}res2-1".format(col_value_bound2))
        s = s.replace("res3-3", "{}res3-3".format(col_value_bound2))
        s = s.replace("nguyen4-2", "{}nguyen4-2".format(col_value_bound2))
        s = s.replace("nguyen3-2", "{}nguyen3-2".format(col_value_bound2))
        s = s.replace("nguyen1-2", "{}nguyen1-2".format(col_value_bound2))
        s = s.replace("keijzer14-2", "{}keijzer14-2".format(col_value_bound2))
        s = s.replace("keijzer12-0", "{}keijzer12-0".format(col_value_bound2))
        s = s.replace("keijzer12-1", "{}keijzer12-1".format(col_value_bound2))

        s = s.replace("gravity-2", "{}gravity-2".format(col_monotonicity))
        s = s.replace("gravity-3", "{}gravity-3".format(col_monotonicity))
        s = s.replace("keijzer12-3", "{}keijzer12-3".format(col_monotonicity))
        s = s.replace("keijzer12-4", "{}keijzer12-4".format(col_monotonicity))
        s = s.replace("keijzer12-5", "{}keijzer12-5".format(col_monotonicity))

        s = s.replace("keijzer5-0", "{}keijzer5-0".format(col_equality))
        s = s.replace("keijzer12-2", "{}keijzer12-2".format(col_equality))
        s = s.replace("keijzer15-0", "{}keijzer15-0".format(col_equality))
        return s

    # we need to create a situation where we can use individual constraints as a row dimension
    propsConstr = []
    for p in props:
        if "result.best.verificator.decisions" in p:
            satVector = p["result.best.verificator.decisions"].split(",")
            satVector = [int(s) for s in satVector]
            for i, satOutcome in enumerate(satVector):
                new_p = p.copy()
                new_p["propsConstr.constraint"] = p["benchmark"] + "-" + str(i)
                new_p["propsConstr.satOutcome"] = satOutcome
                new_p["propsConstr.method"] = p["method"]
                propsConstr.append(new_p)
    dimConstr = Dim.from_dict(propsConstr, "propsConstr.constraint").sort()

    def funSatOutcome(props):
        props2 = [float(p["propsConstr.satOutcome"]) for p in props if "propsConstr.satOutcome" in p]
        if len(props2) == 0:
            return "-"
        else:
            # return "{}".format("%0.2f" % (sum(props2) / len(props2)))
            return "{:.2f}".format(sum(props2) / len(props2))

    tables = [
        TableGenerator(funSatOutcome,
                       dimConstr,
                       dimens["method_scikit"],
                       title="Average ratio of satisfied individual properties. \colorbox{{{}}}{{Symmetry w.r.t.\ arguments}}, \colorbox{{{}}}{{constant output bound}}, \colorbox{{{}}}{{variable output bound}}, \colorbox{{{}}}{{monotonicity}}, \colorbox{{{}}}{{equality}}.".format(c_arg_symmetry, c_value_bound, c_value_bound2, c_monotonicity, c_equality), headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=postprocessor, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/scikit/scikit_satIndividualConstrRatio_{}.tex".format(
                           exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(funSatOutcome,
                       dimConstr,
                       dim_cdsr_methods,
                       title="Average ratio of satisfied individual properties. \colorbox{{{}}}{{Symmetry w.r.t.\ arguments}}, \colorbox{{{}}}{{constant output bound}}, \colorbox{{{}}}{{variable output bound}}, \colorbox{{{}}}{{monotonicity}}, \colorbox{{{}}}{{equality}}.".format(c_arg_symmetry, c_value_bound, c_value_bound2, c_monotonicity, c_equality), headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=postprocessor, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/cdsr/cdsr_satIndividualConstrRatio_{}.tex".format(
                           exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(funSatOutcome,
                       dimConstr,
                       dim_cdsr_methods_full,
                       title="Average ratio of satisfied individual properties. \colorbox{{{}}}{{Symmetry w.r.t.\ arguments}}, \colorbox{{{}}}{{constant output bound}}, \colorbox{{{}}}{{variable output bound}}, \colorbox{{{}}}{{monotonicity}}, \colorbox{{{}}}{{equality}}.".format(c_arg_symmetry, c_value_bound, c_value_bound2, c_monotonicity, c_equality), headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=postprocessor, variants=variants,
                       outputFiles=[dir_path + "/tables/custom/cdsrFull/cdsrFull_satIndividualConstrRatio_{}.tex".format(
                           exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
        TableGenerator(funSatOutcome,
                       dimConstr,
                       dimens["method_scikit"] + dim_cdsr_methods_full,
                       title="Average ratio of satisfied individual properties. \colorbox{{{}}}{{Symmetry w.r.t.\ arguments}}, \colorbox{{{}}}{{constant output bound}}, \colorbox{{{}}}{{variable output bound}}, \colorbox{{{}}}{{monotonicity}}, \colorbox{{{}}}{{equality}}.".format(
                           c_arg_symmetry, c_value_bound, c_value_bound2, c_monotonicity, c_equality),
                       headerRowNames=[],
                       color_scheme=thesis_color_scheme,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=postprocessor, variants=variants,
                       outputFiles=[
                           dir_path + "/tables/custom/all/all_satIndividualConstrRatio_{}.tex".format(
                               exp_variant)],
                       addRowWithRanks=True, ranksHigherValuesBetter=True
                       ),
    ]
    content = [createRelativeSectionForTable(t, propsConstr) for t in tables]
    return reporting.Subsection("Custom tables – individual constraints", content)


def convertPropsToDataFrame(props):
    attributes = [
        "evoplotter.file",
        "benchmark",
        "method",
        "selection",
        "testsRatio",
        "partialConstraintsWeight",
        "result.best",
        "result.best.smtlib",
        "result.best.size",
        "result.best.correctVerification",
        "result.best.passedConstraints",
        "result.validation.best.passedConstraints",
        "result.best.verificator.decisions",
        "result.best.verificator.ratios",
        (getAvgSatRatioStochasticVerifier_p, "result.best.verificator.satConstrRatio"),
        "result.best.trainMSE",
        "result.best.validMSE",
        "result.best.testMSE",
        (get_trainMSE_bestOnValidSet_p, "result.validation.best.trainMSE"),
        (get_validMSE_bestOnValidSet_p, "result.validation.best.validMSE"),
        (get_testMSE_bestOnValidSet_p, "result.validation.best.testMSE"),
    ]
    lambdas, key_names = [], []
    for a in attributes:
        if isinstance(a, tuple):
            lambdas.append(a[0])
            key_names.append(a[1])
        else:  # vanilla dictionary key, create lambda for it
            lambdas.append(lambda p, k=a: p[k] if k in p else None)
            key_names.append(a)

    frame = evoplotter.utils.props_to_DataFrame(props, lambdas, key_names)
    return frame


def saveLogsAsCsv(props, dim_rows, dim_cols, dir_path, path="data.csv", frame=None):
    if frame is None:
        frame = convertPropsToDataFrame(props)
    frame.to_csv("{}csv_data/{}".format(dir_path, path), sep=";")

    # utils.ensure_dir("{}csv_data/by_benchmarks/".format(dir_path))
    # for config_b in dim_rows:
    #     csv_path = "{}csv_data/by_benchmarks/{}.csv".format(dir_path, config_b.get_caption())
    #     props_b = config_b.filter_props(props)
    #     frame_b = evoplotter.utils.props_to_DataFrame(props_b, lambdas, key_names)
    #     frame_b.to_csv(csv_path, sep=";")

    utils.ensure_dir("{}csv_data/by_benchmarks/".format(dir_path))
    for b in frame["benchmark"].unique():
        csv_path = "{}csv_data/by_benchmarks/{}.csv".format(dir_path, b)
        frame_b = frame.loc[frame['benchmark'] == b]
        frame_b.to_csv(csv_path, sep=";")

    return frame



def reports_universal(folders, dir_path="reports/", exp_variant=""):
    if dir_path[-1] != "/":
        dir_path += "/"
    utils.ensure_clear_dir("{}".format(dir_path))
    utils.ensure_dir("{}figures/".format(dir_path))
    utils.ensure_dir("{}csv_data/".format(dir_path))
    utils.ensure_dir("{}listings/".format(dir_path))
    # utils.ensure_dir("{}tables/".format(dir_path))
    utils.ensure_dir("{}listings/errors/".format(dir_path))

    title = "Experiments for regression CDGP and  baseline regressors from Scikit."
    desc = r"""
\parbox{30cm}{
Training set: 300\\
Validation set (GP/CDGP only): 75\\
Test set: 125\\

Sets were shuffled randomly from the 500 cases present in each generated benchmark.
}

"""

    desc += "\n\\bigskip\\noindent Folders with data: " + r"\lstinline{" + str(folders) + "}\n"
    props = load_correct_props(folders, dir_path)

    # manually correct CDSR logs so that always "result.best.testMSE" = "result.validation.best.testMSE"
    for p in props:
        if "result.validation.best.testMSE" in p:
            p["result.best"] = p["result.validation.best"]
            p["result.best.smtlib"] = p["result.validation.best.smtlib"]
            p["result.best.correctTests"] = p["result.validation.best.correctTests"]
            p["result.best.correctVerification"] = p["result.validation.best.correctVerification"]
            p["result.best.height"] = p["result.validation.best.height"]
            p["result.best.mse"] = p["result.validation.best.mse"]
            p["result.best.passedConstraints"] = p["result.validation.best.passedConstraints"]
            p["result.best.size"] = p["result.validation.best.size"]
            p["result.best.testEval"] = p["result.validation.best.testEval"]
            p["result.best.testMSE"] = p["result.validation.best.testMSE"]
            p["result.best.trainEval"] = p["result.validation.best.trainEval"]
            p["result.best.trainMSE"] = p["result.validation.best.trainMSE"]
            p["result.best.validEval"] = p["result.validation.best.validEval"]
            p["result.best.validMSE"] = p["result.validation.best.validMSE"]
            p["result.best.verificationDecision"] = p["result.validation.best.verificationDecision"]
            p["result.best.verificationModel"] = p["result.validation.best.verificationModel"]
            p["result.bestOrig"] = p["result.validation.bestOrig"]
            p["result.bestOrig.height"] = p["result.validation.bestOrig.height"]
            p["result.bestOrig.size"] = p["result.validation.bestOrig.size"]
            p["result.bestOrig.smtlib"] = p["result.validation.bestOrig.smtlib"]
            p["result.best.verificationModel"] = p["result.validation.best.verificationModel"]
            p["result.best.verificationModel"] = p["result.validation.best.verificationModel"]

    standardize_benchmark_names(props)
    dim_benchmarks = get_benchmarks_from_props(props)
    dim_rows_all = dim_benchmarks.copy()
    dim_rows_all += dim_benchmarks.dim_true_within("ALL")

    dim_cols_scikit = dim_methodScikit

    dim_cols_cdgp = dim_methodCDGP * dim_sel * dim_testsRatio + dim_methodCDGPprops * dim_sel * dim_testsRatio * dim_weight
    dim_cols_ea = dim_cols_cdgp
    dim_cols = dim_methodScikit + dim_cols_ea

    dim_cols_all = dim_cols.copy()
    dim_cols_all += dim_cols.dim_true_within()
    dim_cols_ea += dim_cols_ea.dim_true_within()
    dim_cols_cdgp += dim_cols_cdgp.dim_true_within()


    dataFrame = convertPropsToDataFrame(props)
    saveLogsAsCsv(props, dim_benchmarks, dim_cols, dir_path=dir_path, frame=dataFrame)

    # utils.reorganizeExperimentFiles(props, dim_benchmarks * dim_cols, "results_thesis_pop500_final/{}/".format(exp_variant), maxRuns=50)
    # utils.reorganizeExperimentFiles(props, dim_benchmarks * dim_cols, "results_thesis_pop1k_final/{}/".format(exp_variant), maxRuns=50)

    dimensions_dict = {"benchmark": dim_benchmarks,
                       "testsRatio": dim_testsRatio,
                       "testsRatio_0.75": Dim(dim_testsRatio[0]),
                       "testsRatio_1.0": Dim(dim_testsRatio[1]),
                       "selection": dim_sel,
                       "selection_tour": Dim(dim_sel[0]),
                       "selection_lex": Dim(dim_sel[1]),
                       "method": dim_methodScikit + dim_methodCDGP + dim_methodCDGPprops,
                       "method_CDGP": dim_methodCDGP,
                       "method_CDGPprops": dim_methodCDGPprops,
                       "method_scikit": dim_methodScikit,
                       "weight": dim_weight}

    headerRowNames = ["method"]
    subs = [
        (create_subsection_shared_status, [props, "Shared Statistics", dim_rows_all, dim_cols_all, 50, headerRowNames]),
        (create_subsection_shared_stats, [props, "Shared Statistics", dim_rows_all, dim_cols_all, 50, headerRowNames]),
        (create_subsection_ea_stats, [props, "EA/CDGP Statistics", dim_rows_all, dim_cols_ea, headerRowNames]),
        (create_subsection_cdgp_specific, [props, "CDGP Statistics", dim_rows_all, dim_cols_cdgp, headerRowNames]),
        (create_subsection_custom_tables, [props, "Custom tables", dimensions_dict, exp_variant, dir_path, None]),
        (create_subsection_individual_constraints, [props, "Custom tables -- individual constraints", dimensions_dict, exp_variant, dir_path, None]),
        # (create_subsection_figures_analysis, [props, dim_cols, dim_rows, "figures/"]),
        # (create_subsection_figures_analysis, [dataFrame, "figures/"]),
        # (create_subsection_figures, [dim_rows, dim_cols, exp_prefix]),
    ]
    sects = [(title, desc, subs, [])]


    save_listings(props, dim_benchmarks, dim_cols, dir_path=dir_path)
    user_declarations = r"""\definecolor{darkred}{rgb}{0.56, 0.05, 0.0}
\definecolor{darkgreen}{rgb}{0.0, 0.5, 0.0}
\definecolor{darkblue}{rgb}{0.0, 0.0, 0.55}
\definecolor{darkorange}{rgb}{0.93, 0.53, 0.18}

\usepackage{listings}
\lstset{
basicstyle=\small\ttfamily,
columns=flexible,
breaklines=true
}

\usepackage{makecell} % introduces two very useful commands, \thead and \makecell, useful

\newcommand{\unaryminus}{\scalebox{0.4}[1.0]{\( - \)}} % for shorter minus sign, see: https://tex.stackexchange.com/questions/6058/making-a-shorter-minus
"""
    templates.prepare_report(sects, "cdsr_{}.tex".format(exp_variant), dir_path=dir_path, paperwidth=190, user_declarations=user_declarations)



def reports_noNoise_pop500():
    folders = ["results_thesis_pop500_final/noNoise/"]
    reports_universal(folders=folders, dir_path="reports_pop500/noNoise/", exp_variant="noNoise")

def reports_withNoise_pop500():
    folders = ["results_thesis_pop500_final/withNoise/"]
    reports_universal(folders=folders, dir_path="reports_pop500/withNoise/", exp_variant="withNoise")


def reports_noNoise_pop1k():
    folders = ["results_thesis_pop1k_final/noNoise/"]
    reports_universal(folders=folders, dir_path="reports_pop1k/noNoise/", exp_variant="noNoise")

def reports_withNoise_pop1k():
    folders = ["results_thesis_pop1k_final/withNoise/"]
    reports_universal(folders=folders, dir_path="reports_pop1k/withNoise/", exp_variant="withNoise")


if __name__ == "__main__":
    # utils.ensure_clear_dir("reports/")
    # reports_noNoise_pop500()
    # reports_withNoise_pop500()
    reports_noNoise_pop1k()
    reports_withNoise_pop1k()
