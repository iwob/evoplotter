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
scikit_algs = ["AdaBoostRegressor", "XGBoost", "SGDRegressor", "RandomForestRegressor", "MLPRegressor",
        "LinearSVR", "LinearRegression", "LassoLars", "KernelRidge", "GradientBoostingRegressor"] # , "GSGP"  - not used due to issues
dim_methodScikit = Dim([Config(sa.replace("Regressor","").replace("Regression",""), p_dict_matcher({"method": sa}), method=sa) for sa in scikit_algs])
dim_method = dim_methodScikit + dim_methodCDGP + dim_methodCDGPprops + dim_methodGP
dim_sel = Dim([Config("$Tour$", p_sel_tourn, selection="tournament"),
               Config("$Lex$", p_sel_lexicase, selection="lexicase")])
# dim_evoMode = Dim([Config("$steadyState$", p_steadyState, evolutionMode="steadyState"),
#                    Config("$generational$", p_generational, evolutionMode="generational")])
dim_evoMode = Dim([Config("$steadyState$", p_steadyState, evolutionMode="steadyState")])
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
         .replace(r"\_{lex}", "_{lex}").replace(r"\_{", "_{").replace("resistance_par", "res")\
         .replace("gravity", "gr")
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


def getAndSaveTextForTableVariants(table, props, outputFiles=None):
    """Produces a list of LaTeX codes for each table variant. If outputFiles are specified,
    then the outputs will be saved there."""
    tsv = table.apply_listed(props)
    if outputFiles is not None:
        assert len(tsv) == len(outputFiles), "Number of output files must be the same as the number of table variants."
        for tsv_txt, path in zip(tsv, outputFiles):
            os.makedirs(os.path.dirname(path), exist_ok=True)  # automatically create directories
            f = open(path, "w")
            f.write(tsv_txt)
            f.close()
    return tsv


def createSubsectionWithTables(title, tables, props):
    subsects_main = []
    for t in tables:
        tsv = getAndSaveTextForTableVariants(t, props, t.outputFiles)
        tup = (t.title, r"\noindent " + " ".join(tsv), t.color_scheme)
        subsects_main.append(tup)
    return reporting.Subsection(title, get_content_of_subsections(subsects_main))


def create_subsection_aggregation_tests(props, dim_rows, dim_cols, headerRowNames):
    vb = 1  # vertical border
    variants = None
    dim_rows_v2 = get_benchmarks_from_props(props, ignoreNumTests=True)
    dim_rows_v2 += dim_true

    # By default: dim_cols = (dim_methodGP * dim_empty + dim_methodCDGP * dim_testsRatio) * dim_optThreshold

    tables = [
        # TableGenerator(fun_successRateMseOnly, dim_rows_v2,
        #                (dim_methodGP + dim_methodCDGP),
        #                headerRowNames=[""],
        #                title="Success rates (mse below thresh)",
        #                color_scheme=reporting.color_scheme_violet,
        #                default_color_thresholds=(0.0, 0.5, 1.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),


        TableGenerator(fun_successRate, dim_rows_v2,
                       dim_benchmarkNumTests * dim_method,
                       headerRowNames=["", ""],
                       title="Success rates (mse below thresh + properties met)",
                       color_scheme=reporting.color_scheme_darkgreen,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_successRate, dim_rows_v2,
                       dim_method * dim_benchmarkNumTests,
                       headerRowNames=["", ""],
                       title="Success rates (mse below thresh + properties met)",
                       color_scheme=reporting.color_scheme_darkgreen,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_successRate, dim_rows_v2,
                       dim_optThreshold * dim_method,
                       headerRowNames=["tolerance", ""],
                       title="Success rates (mse below thresh + properties met)",
                       color_scheme=reporting.color_scheme_darkgreen,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_successRate, dim_rows_v2,
                       dim_method,
                       headerRowNames=[""],
                       title="Success rates (mse below thresh + properties met)",
                       color_scheme=reporting.color_scheme_darkgreen,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),

        TableGenerator(fun_allPropertiesMet_verificator, dim_rows_v2,
                       dim_benchmarkNumTests * dim_method,
                       headerRowNames=["", ""],
                       title="Success rates (properties met) -- verified by stochastic verificator",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_allPropertiesMet, dim_rows_v2,
                       dim_benchmarkNumTests * dim_method,
                       headerRowNames=["", ""],
                       title="Success rates (properties met) -- verified formally by SMT solver",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_allPropertiesMet, dim_rows_v2,
                       dim_method * dim_benchmarkNumTests,
                       headerRowNames=["", ""],
                       title="Success rates (properties met)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(fun_allPropertiesMet, dim_rows_v2,
        #                dim_benchmarkNumTests * dim_optThreshold * dim_method,
        #                headerRowNames=["", "tolerance", ""],
        #                title="Success rates (properties met)",
        #                color_scheme=reporting.color_scheme_green,
        #                default_color_thresholds=(0.0, 0.5, 1.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        TableGenerator(fun_allPropertiesMet, dim_rows_v2,
                       dim_optThreshold * dim_method,
                       headerRowNames=["tolerance", ""],
                       title="Success rates (properties met)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_allPropertiesMet, dim_rows_v2,
                       dim_method,
                       headerRowNames=[""],
                       title="Success rates (properties met)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
    ]

    subsects_main = []
    for t in tables:
        tup = (t.title, t.apply(props), t.color_scheme)
        subsects_main.append(tup)

    return reporting.Subsection("Tests of different data aggregation in tables", get_content_of_subsections(subsects_main))



def _get_median_testMSE(props):
    if len(props) == 0:
        return None
    else:
        return np.median([get_testMSE_bestOnValidSet_p(p) for p in props])
def _getAvgSatisfiedRatios(props):
    if len(props) == 0:
        return None
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


def create_subsection_figures_analysis(props, dim_cols, dim_benchmarks, path):
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

        plt.savefig("reports/" + fig_path)
        # plt.show()
        plt.clf()

        # Plotting in seaborn
        import seaborn as sns
        # sns.set_theme(style="darkgrid")
        # sns.relplot(data=)

        section.add(reporting.BlockLatex(r"\includegraphics{"  + fig_path + r"}\\"))

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




def create_subsection_figures(props, dim_rows, dim_cols, exp_prefix):
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



def create_subsection_scikit(props, title, dim_rows, dim_cols, numRuns, headerRowNames):
    vb = 1  # vertical border
    variants = None  # variants_benchmarkNumTests

    tables = [
        TableGenerator(getAvgSatisfiedPropsForScikit1,
                       dim_rows, dim_cols,
                       title="Average ratio of satisfied properties",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(getAvgSatisfiedPropsForScikit2,
                       dim_rows, dim_cols,
                       title="Average number of satisfied properties",
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_allPropertiesMet, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Success rates (ratio of runs which satisfied all properties)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(getAvgSatisfiedRatios,
                       dim_rows, dim_cols,
                       color_scheme=reporting.color_scheme_violet,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       title="Average ratio of satisfied properties (how close approximation was to meeting all properties)",
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
    ]

    return createSubsectionWithTables(title, tables, props)


def create_subsection_shared_stats(props, title, dim_rows, dim_cols, numRuns, headerRowNames):
    vb = 1  # vertical border
    variants = None  # variants_benchmarkNumTests
    dim_rows_v2 = get_benchmarks_from_props(props, ignoreNumTests=True)
    # dim_rows_v2 += dim_true  #TODO: within dict

    # ----------------------------------------------------
    # Cleaning experiment here, because dimension can be easily constructed.
    # dim_rows_v3 = get_benchmarks_from_props(props, simple_names=True, ignoreNumTests=True)
    # utils.reorganizeExperimentFiles(props, dim_rows_v3 * dim_benchmarkNumTests * dim_cols, target_dir="./exp3_final/", maxRuns=numRuns)
    # utils.deleteFilesByPredicate(props, lambda p: len(p["maxGenerations"]) > 7, simulate=False)
    # ----------------------------------------------------

    def scNotColorValueExtractor(s):
        if s == "-" or "10^{" not in s:
            return s
        else:
            r = s.split("10^{")
            return r[1][:-2]

    print("\nFiles with a conflict between SMT and stochastic verificators:")
    for p in props:
        if "CDGP" in p["method"]:
            if not p_allPropertiesMet_verificator(p) and p_allPropertiesMet_smt(p) and "nguyen1" in p["benchmark"]:
                print(p["evoplotter.file"])

    tables = [
        TableGenerator(get_num_computed, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Status (correctly finished runs)",
                       color_scheme=reversed(reporting.color_scheme_red),
                       default_color_thresholds=(0.0, numRuns/2, numRuns),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
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
                       title="Training set: MSE  (median)",
                       color_scheme=reporting.color_scheme_gray_dark,
                       default_color_thresholds=(-10.0, 0.0, 10.0),
                       color_value_extractor=scNotColorValueExtractor,
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(get_avg_trainMSE, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Training set: MSE  (avg)",
        #                color_scheme=reporting.color_scheme_green,
        #                default_color_thresholds=(0.0, 1e2, 1e4),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        TableGenerator(get_median_testMSE, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Test set: MSE  (median); bestOfRun CDGP",
                       color_scheme=reporting.color_scheme_gray_dark,
                       default_color_thresholds=(-10.0, 0.0, 10.0),
                       color_value_extractor=scNotColorValueExtractor,
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_median_testMSE_bestOnValidCDGP, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Test set: MSE  (median); bestOnValidSet CDGP",
                       color_scheme=reporting.color_scheme_gray_dark,
                       default_color_thresholds=(-10.0, 0.0, 10.0),
                       color_value_extractor=scNotColorValueExtractor,
                       vertical_border=vb, table_postprocessor=post, variants=variants,
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
        # TableGenerator(get_median_testMSE_noScNot, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Test set: MSE  (median) (noScNot)",
        #                color_scheme=reporting.color_scheme_green,
        #                default_color_thresholds=(0.0, 1e2, 1e4),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        # TableGenerator(get_avg_testMSE, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Test set: MSE  (avg)",
        #                color_scheme=reporting.color_scheme_green,
        #                default_color_thresholds=(0.0, 1e2, 1e4),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        TableGenerator(fun_allPropertiesMet, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Success rates (properties met) -- verified formally by SMT solver",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_allPropertiesMet_verificator, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Success rates (properties met) -- verified by stochastic verificator",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(getAvgSatisfiedPropsForScikit1, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average ratio of satisfied properties -- verified by SMT solver (CDSR configs) and stochastic verificator (scikit baselines)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
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
        TableGenerator(get_rankingOfBestSolutionsCDSR(ONLY_VISIBLE_SOLS=True, NUM_SHOWN=15),
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
        TableGenerator(get_validation_testSet_ratio, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="A ratio of bestOfRun / bestOfValidation for MSE on test set.",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 100.0, 200.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
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
        TableGenerator(get_avg_doneAlgRestarts, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Number of algorithm restarts  (avg)",
                       color_scheme=reporting.color_scheme_gray_light,
                       default_color_thresholds=(0.0, 1e2, 1e4),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
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

    # props = [p for p in props if p["method"] in {"CDGP", "CDGPprops"}]

    print("AVG TOTAL TESTS")
    latex_avgTotalTests = create_single_table_bundle(props, dim_rows, dim_cols, get_avg_totalTests, headerRowNames,
                                                     cv0=0.0, cv1=1000.0, cv2=2000.0, tableVariants=variants)

    # print("AVG RUNTIME PER PROGRAM")
    # text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_runtimePerProgram, layered_headline=True,
    #                                 vertical_border=vb, headerRowNames=headerRowNames))
    # latex_avgRuntimePerProgram = printer.table_color_map(text, 0.01, 1.0, 2.0, "colorLow", "colorMedium", "colorHigh")

    # print("AVG GENERATION")
    # latex_avgGeneration = create_single_table_bundle(props, dim_rows, dim_cols, get_avg_generation, headerRowNames,
    #                                                  cv0=0.0, cv1=100.0, cv2=200.0, tableVariants=variants)
    # text = post(
    #     printer.latex_table(props, dim_rows, dim_cols, get_avg_generation, layered_headline=True, vertical_border=vb, headerRowNames=headerRowNames))
    # latex_avgGeneration = printer.table_color_map(text, 0.0, 50.0, 100.0, "colorLow", "colorMedium", "colorHigh")

    # print("AVG EVALUATED SOLUTIONS")
    # text = post(
    #     printer.latex_table(props, dim_rows, dim_cols, get_avg_evaluated, layered_headline=True, vertical_border=vb, headerRowNames=headerRowNames))
    # latex_avgEvaluated = printer.table_color_map(text, 500.0, 25000.0, 100000.0, "colorLow", "colorMedium", "colorHigh")

    # print("AVG EVALUATED SOLUTIONS (SUCCESSFUL)")
    # text = post(printer.latex_table(props, dim_rows, dim_cols, get_avg_evaluatedSuccessful, layered_headline=True,
    #                                 vertical_border=vb, headerRowNames=headerRowNames))
    # latex_avgEvaluatedSuccessful = printer.table_color_map(text, 500.0, 25000.0, 100000.0, "colorLow", "colorMedium",
    #                                                         "colorHigh")

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
        # ("Average generation (all)", latex_avgGeneration, reporting.color_scheme_teal),
        #("Average generation (only successful)", latex_avgGenerationSuccessful, reporting.color_scheme_teal),
        # ("Average evaluated solutions", latex_avgEvaluated, reporting.color_scheme_brown),
        # ("Average evaluated solutions (only successful)", latex_avgEvaluatedSuccessful, reporting.color_scheme_teal),
        # ("Approximate average runtime per program [s]", latex_avgRuntimePerProgram, reporting.color_scheme_brown),
        ("Max solver time per query [s]", latex_maxSolverTimes, reporting.color_scheme_violet),
        ("Avg solver time per query [s]", latex_avgSolverTimes, reporting.color_scheme_brown),
        ("Avg number of solver calls (in thousands; 1=1000)", latex_avgSolverTotalCalls, reporting.color_scheme_blue),
        ("Number of solver calls $>$ 0.5s", latex_numSolverCallsOverXs, reporting.color_scheme_blue),
        ("The most frequently found counterexamples for each benchmark and configuration", latex_freqCounterexamples, reporting.color_scheme_violet),
    ]
    return reporting.Subsection(title, get_content_of_subsections(subsects_cdgp))


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
        (getAvgSatRatioStochasticVerifier_p, "result.best.verificator.satRatio"),
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


def saveLogsAsCsv(props, dim_rows, dim_cols, path="data.csv", frame=None):
    if frame is None:
        frame = convertPropsToDataFrame(props)
    frame.to_csv("reports/csv_data/{}".format(path), sep=";")

    # utils.ensure_dir("reports/csv_data/by_benchmarks/")
    # for config_b in dim_rows:
    #     csv_path = "reports/csv_data/by_benchmarks/{}.csv".format(config_b.get_caption())
    #     props_b = config_b.filter_props(props)
    #     frame_b = evoplotter.utils.props_to_DataFrame(props_b, lambdas, key_names)
    #     frame_b.to_csv(csv_path, sep=";")

    utils.ensure_dir("reports/csv_data/by_benchmarks/")
    for b in frame["benchmark"].unique():
        csv_path = "reports/csv_data/by_benchmarks/{}.csv".format(b)
        frame_b = frame.loc[frame['benchmark'] == b]
        frame_b.to_csv(csv_path, sep=";")

    return frame





def reports_withNoise():
    title = "Experiments for regression CDGP and  baseline regressors from Scikit. A01 - with noise."
    desc = r"""
\parbox{30cm}{
Training set: 300\\
Validation set (GP/CDGP only): 75\\
Test set: 125\\

Sets were shuffled randomly from the 500 cases present in each generated benchmark.
In this experiment, A01, noise is already present in the benchmark and is generated from the normal distribution
with mean at the original value and standard deviation at 1% of the value.
}

\begin{lstlisting}[breaklines]
# shared_dims={'benchmark': ['benchmarks/gpem/withNoise/keijzer5_500.sl', 'benchmarks/gpem/withNoise/nguyen4_500.sl', 'benchmarks/gpem/withNoise/resistance_par2_500.sl', 'benchmarks/gpem/withNoise/pagie1_500.sl', 'benchmarks/gpem/withNoise/keijzer15_500.sl', 'benchmarks/gpem/withNoise/nguyen3_500.sl', 'benchmarks/gpem/withNoise/gravity_500.sl', 'benchmarks/gpem/withNoise/keijzer12_500.sl', 'benchmarks/gpem/withNoise/nguyen1_500.sl', 'benchmarks/gpem/withNoise/keijzer14_500.sl', 'benchmarks/gpem/withNoise/resistance_par3_500.sl'], 'selection': ['lexicase'], 'evolutionMode': ['steadyState'], 'populationSize': [500], 'optThreshold': [0.0], 'sizeTrainSet': [300], 'maxRestarts': [1], 'maxGenerations': [200]}
# dims_cdgp = {'method': ['CDGP'], 'testsRatio': [1.0], 'testsTypesForRatio': ['i'], 'benchmark': ['benchmarks/gpem/withNoise/keijzer5_500.sl', 'benchmarks/gpem/withNoise/nguyen4_500.sl', 'benchmarks/gpem/withNoise/resistance_par2_500.sl', 'benchmarks/gpem/withNoise/pagie1_500.sl', 'benchmarks/gpem/withNoise/keijzer15_500.sl', 'benchmarks/gpem/withNoise/nguyen3_500.sl', 'benchmarks/gpem/withNoise/gravity_500.sl', 'benchmarks/gpem/withNoise/keijzer12_500.sl', 'benchmarks/gpem/withNoise/nguyen1_500.sl', 'benchmarks/gpem/withNoise/keijzer14_500.sl', 'benchmarks/gpem/withNoise/resistance_par3_500.sl'], 'selection': ['lexicase'], 'evolutionMode': ['steadyState'], 'populationSize': [500], 'optThreshold': [0.0], 'sizeTrainSet': [300], 'maxRestarts': [1], 'maxGenerations': [200]}
# dims_cdgp = {'method': ['CDGPprops'], 'testsRatio': [1.0], 'testsTypesForRatio': ['i'], 'partialConstraintsWeight': [1, 5], 'benchmark': ['benchmarks/gpem/withNoise/keijzer5_500.sl', 'benchmarks/gpem/withNoise/nguyen4_500.sl', 'benchmarks/gpem/withNoise/resistance_par2_500.sl', 'benchmarks/gpem/withNoise/pagie1_500.sl', 'benchmarks/gpem/withNoise/keijzer15_500.sl', 'benchmarks/gpem/withNoise/nguyen3_500.sl', 'benchmarks/gpem/withNoise/gravity_500.sl', 'benchmarks/gpem/withNoise/keijzer12_500.sl', 'benchmarks/gpem/withNoise/nguyen1_500.sl', 'benchmarks/gpem/withNoise/keijzer14_500.sl', 'benchmarks/gpem/withNoise/resistance_par3_500.sl'], 'selection': ['lexicase'], 'evolutionMode': ['steadyState'], 'populationSize': [500], 'optThreshold': [0.0], 'sizeTrainSet': [300], 'maxRestarts': [1], 'maxGenerations': [200]}
# dims_gp = {'method': ['GP'], 'benchmark': ['benchmarks/gpem/withNoise/keijzer5_500.sl', 'benchmarks/gpem/withNoise/nguyen4_500.sl', 'benchmarks/gpem/withNoise/resistance_par2_500.sl', 'benchmarks/gpem/withNoise/pagie1_500.sl', 'benchmarks/gpem/withNoise/keijzer15_500.sl', 'benchmarks/gpem/withNoise/nguyen3_500.sl', 'benchmarks/gpem/withNoise/gravity_500.sl', 'benchmarks/gpem/withNoise/keijzer12_500.sl', 'benchmarks/gpem/withNoise/nguyen1_500.sl', 'benchmarks/gpem/withNoise/keijzer14_500.sl', 'benchmarks/gpem/withNoise/resistance_par3_500.sl'], 'selection': ['lexicase'], 'evolutionMode': ['steadyState'], 'populationSize': [500], 'optThreshold': [0.0], 'sizeTrainSet': [300], 'maxRestarts': [1], 'maxGenerations': [200]}
# 
# opt={'seed': '$RANDOM', 'maxTime': 1800000, 'tournamentSize': 7, 'tournamentDeselectSize': 7, 'populationSize': 500, 'initMaxTreeDepth': 4, 'maxSubtreeDepth': 4, 'maxTreeDepth': 12, 'stoppingDepthRatio': 0.8, 'operatorProbs': '0.5,0.5', 'deleteOutputFile': 'true', 'parEval': 'false', 'maxNewTestsPerIter': 10, 'silent': 'true', 'solverPath': "'solver/z3'", 'solverType': 'z3', 'maxSolverRestarts': 2, 'regression': 'true', 'saveTests': 'true', 'outDir': 'phd_A01', 'solverTimeout': 3000, 'notes': "'gpem19_A01'", 'noiseDeltaX': 0.0, 'noiseDeltaY': 0.0, 'sizeValidationSet': 75, 'sizeTestSet': 125, 'notImprovedWindow': 1000, 'reportFreq': 200}
\end{lstlisting}

NOTE: for steady state, maxGenerations is multiplied by populationSize. 
"""

    # folders = ["exp3", "regression_results_withNoise"]  # "regression_results_noNoise"
    folders = ["CDSR_logs/phd_A01", "CDSR_logs/phd_A01_run2", "results_scikit_withNoise"]  # "regression_results_noNoise"
    desc += "\n\\bigskip\\noindent Folders with data: " + r"\lstinline{" + str(folders) + "}\n"
    props = load_correct_props(folders)
    standardize_benchmark_names(props)
    dim_rows = get_benchmarks_from_props(props)
    dim_rows += dim_rows.dim_true_within("ALL")

    dim_cols_scikit = dim_methodScikit

    dim_cols = dim_methodScikit + dim_methodGP + dim_methodCDGP + dim_methodCDGPprops*dim_weight
    dim_cols += dim_cols.dim_true_within()

    dim_cols_ea = dim_methodGP + dim_methodCDGP + dim_methodCDGPprops * dim_weight
    dim_cols_ea += dim_cols_ea.dim_true_within()

    dim_cols_cdgp = dim_methodCDGP + dim_methodCDGPprops*dim_weight
    dim_cols_cdgp += dim_cols_cdgp.dim_true_within()

    headerRowNames = ["method", "weight"]
    subs = [
        (create_subsection_shared_stats, [props, "Shared Statistics", dim_rows, dim_cols, 25, headerRowNames]),
        (create_subsection_scikit, [props, "Scikit Baselines Statistics", dim_rows, dim_cols_scikit, 25, headerRowNames]),
        (create_subsection_ea_stats, [props, "EA/CDGP Statistics", dim_rows, dim_cols_ea, headerRowNames]),
        (create_subsection_cdgp_specific, [props, "CDGP Statistics", dim_rows, dim_cols_cdgp, headerRowNames]),
        # (create_subsection_aggregation_tests, [dim_rows, dim_cols, headerRowNames]),
        # (create_subsection_figures, [dim_rows, dim_cols, exp_prefix]),
    ]
    sects = [(title, desc, subs, [])]


    save_listings(props, dim_rows, dim_cols)
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
"""
    templates.prepare_report(sects, "cdsr_withNoise.tex", dir_path="reports/", paperwidth=190, user_declarations=user_declarations)






def reports_noNoise():
    title = "Experiments for regression CDGP and  baseline regressors from Scikit."
    desc = r"""
\parbox{30cm}{
Training set: 300\\
Validation set (GP/CDGP only): 75\\
Test set: 125\\

Sets were shuffled randomly from the 500 cases present in each generated benchmark.
}

"""

    folders = ["results_thesis/noNoise/", "results_scikit_noNoise"]
    desc += "\n\\bigskip\\noindent Folders with data: " + r"\lstinline{" + str(folders) + "}\n"
    props = load_correct_props(folders)
    standardize_benchmark_names(props)
    dim_rows = get_benchmarks_from_props(props)
    dim_rows_all = dim_rows.copy()
    dim_rows_all += dim_rows.dim_true_within("ALL")

    dim_cols_scikit = dim_methodScikit

    dim_cols_cdgp = dim_methodCDGP * dim_sel * dim_testsRatio + dim_methodCDGPprops * dim_sel  * dim_testsRatio * dim_weight
    dim_cols_ea = dim_methodGP + dim_cols_cdgp
    dim_cols = dim_methodScikit + dim_cols_ea

    dim_cols_all = dim_cols.copy()
    dim_cols_all += dim_cols.dim_true_within()
    dim_cols_ea += dim_cols_ea.dim_true_within()
    dim_cols_cdgp += dim_cols_cdgp.dim_true_within()


    dataFrame = convertPropsToDataFrame(props)
    saveLogsAsCsv(props, dim_rows, dim_cols, frame=dataFrame)

    headerRowNames = ["method", "weight"]
    subs = [
        # (create_subsection_shared_stats, [props, "Shared Statistics", dim_rows_all, dim_cols_all, 25, headerRowNames]),
        # (create_subsection_scikit, [props, "Scikit Baselines Statistics", dim_rows_all, dim_cols_scikit, 25, headerRowNames]),
        # (create_subsection_ea_stats, [props, "EA/CDGP Statistics", dim_rows_all, dim_cols_ea, headerRowNames]),
        # (create_subsection_cdgp_specific, [props, "CDGP Statistics", dim_rows_all, dim_cols_cdgp, headerRowNames]),
        # (create_subsection_figures_analysis, [props, dim_cols, dim_rows, "figures/"]),
        (create_subsection_figures_analysis_seaborn, [dataFrame, "figures/"]),
        # (create_subsection_aggregation_tests, [dim_rows, dim_cols, headerRowNames]),
        # (create_subsection_figures, [dim_rows, dim_cols, exp_prefix]),
    ]
    sects = [(title, desc, subs, [])]


    save_listings(props, dim_rows, dim_cols)
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
"""
    templates.prepare_report(sects, "cdsr_noNoise.tex", dir_path="reports/", paperwidth=190, user_declarations=user_declarations)




if __name__ == "__main__":
    utils.ensure_clear_dir("reports/")
    utils.ensure_dir("reports/figures/")
    utils.ensure_dir("reports/csv_data/")
    utils.ensure_dir("reports/listings/")
    # utils.ensure_dir("reports/tables/")
    utils.ensure_dir("reports/listings/errors/")

    # reports_withNoise()
    reports_noNoise()
