import os
from app.gpem19.gpem19_utils import *
from src import utils
from src import plotter
from src import printer
from src import reporting
from src.dims import *



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
        "LinearSVR", "LinearRegression", "LassoLars", "KernelRidge", "GradientBoostingRegressor"]
dim_methodScikit = Dim([Config(sa.replace("Regressor","").replace("Regression",""), p_dict_matcher({"method": sa}), method=sa) for sa in scikit_algs])
dim_method = dim_methodScikit + dim_methodCDGP + dim_methodCDGPprops + dim_methodGP
dim_sel = Dim([#Config("$Tour$", p_sel_tourn, selection="tournament"),
               Config("$Lex$", p_sel_lexicase, selection="lexicase")])
# dim_evoMode = Dim([Config("$steadyState$", p_steadyState, evolutionMode="steadyState"),
#                    Config("$generational$", p_generational, evolutionMode="generational")])
dim_evoMode = Dim([Config("$steadyState$", p_steadyState, evolutionMode="steadyState")])
dim_testsRatio = Dim([
    # Config("$0.8$", p_testsRatio_equalTo("0.8"), testsRatio="0.8"),
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
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        TableGenerator(fun_successRate, dim_rows_v2,
                       dim_method * dim_benchmarkNumTests,
                       headerRowNames=["", ""],
                       title="Success rates (mse below thresh + properties met)",
                       color_scheme=reporting.color_scheme_darkgreen,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        TableGenerator(fun_successRate, dim_rows_v2,
                       dim_optThreshold * dim_method,
                       headerRowNames=["tolerance", ""],
                       title="Success rates (mse below thresh + properties met)",
                       color_scheme=reporting.color_scheme_darkgreen,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        TableGenerator(fun_successRate, dim_rows_v2,
                       dim_method,
                       headerRowNames=[""],
                       title="Success rates (mse below thresh + properties met)",
                       color_scheme=reporting.color_scheme_darkgreen,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),


        TableGenerator(fun_allPropertiesMet, dim_rows_v2,
                       dim_benchmarkNumTests * dim_method,
                       headerRowNames=["", ""],
                       title="Success rates (properties met)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        TableGenerator(fun_allPropertiesMet, dim_rows_v2,
                       dim_method * dim_benchmarkNumTests,
                       headerRowNames=["", ""],
                       title="Success rates (properties met)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
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
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        TableGenerator(fun_allPropertiesMet, dim_rows_v2,
                       dim_method,
                       headerRowNames=[""],
                       title="Success rates (properties met)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
    ]

    subsects_main = []
    for t in tables:
        tup = (t.title, t.apply(props), t.color_scheme)
        subsects_main.append(tup)

    return reporting.Subsection("Tests of different data aggregation in tables", get_content_of_subsections(subsects_main))



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
    savepath = "results/figures/ratioMSE.pdf"
    plotter.plot_ratio_meeting_predicate(props, getter_mse, predicate, xs=xs, xticks=xticks,
                                         show_plot=False,
                                         title="Ratio of solutions with MSE under the certain level",
                                         xlabel="MSE",
                                         series_dim=dim_method,
                                         xlogscale=False,
                                         savepath=savepath)
    section.add(reporting.FloatFigure(savepath.replace("results/", "")))


    # Illustration of individual runs and their errors on training and validation sets
    savepath = "results/figures/progressionGrid.pdf"
    dim_rows = get_benchmarks_from_props(props, ignoreNumTests=True)
    dim_cols = (dim_methodGP * dim_empty + dim_methodCDGP * dim_empty + dim_methodCDGPprops * dim_weight) * \
               dim_benchmarkNumTests  # * dim_optThreshold
    plotter.plot_value_progression_grid_simple(props, dim_rows, dim_cols, ["cdgp.logTrainSet", "cdgp.logValidSet"], ["train", "valid"],
                                               plot_individual_runs=True,
                                               savepath=savepath)
    section.add(reporting.FloatFigure(savepath.replace("results/", "")))
    return section



def create_subsection_shared_stats(props, dim_rows, dim_cols, numRuns, headerRowNames):
    vb = 1  # vertical border
    variants = None  # variants_benchmarkNumTests
    dim_rows_v2 = get_benchmarks_from_props(props, ignoreNumTests=True)
    dim_rows_v2 += dim_true

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

    tables = [
        TableGenerator(get_num_computed, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Status (correctly finished runs)",
                       color_scheme=reversed(reporting.color_scheme_red),
                       default_color_thresholds=(0.0, numRuns/2, numRuns),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
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
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        # TableGenerator(get_avg_trainMSE, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Training set: MSE  (avg)",
        #                color_scheme=reporting.color_scheme_green,
        #                default_color_thresholds=(0.0, 1e2, 1e4),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        TableGenerator(get_median_testMSE, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Test set: MSE  (median)",
                       color_scheme=reporting.color_scheme_gray_dark,
                       default_color_thresholds=(-10.0, 0.0, 10.0),
                       color_value_extractor=scNotColorValueExtractor,
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
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
                       title="Success rates (properties met)",
                       color_scheme=reporting.color_scheme_green,
                       default_color_thresholds=(0.0, 0.5, 1.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        TableGenerator(get_avg_runtime, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average runtime [s]",
                       color_scheme=reporting.color_scheme_violet,
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        # TableGenerator(get_avg_runtimeOnlySuccessful, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Average runtime (only successful) [s]",
        #                color_scheme=reporting.color_scheme_violet,
        #                default_color_thresholds=(0.0, 900.0, 1800.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        TableGenerator(get_avg_doneAlgRestarts, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Number of algorithm restarts  (avg)",
                       color_scheme=reporting.color_scheme_gray_light,
                       default_color_thresholds=(0.0, 1e2, 1e4),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        # TableGenerator(get_stats_size, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Average sizes of best of runs (number of nodes)",
        #                color_scheme=reporting.color_scheme_yellow,
        #                default_color_thresholds=(0.0, 100.0, 200.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
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
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        TableGenerator(get_avg_evaluated, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average number of evaluated solutions",
                       color_scheme=reporting.color_scheme_brown,
                       default_color_thresholds=(0.0, 5000.0, 10000.0),
                       vertical_border=vb, table_postprocessor=post, table_variants=variants,
                       ),
        # TableGenerator(get_avg_evaluatedSuccessful, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Average number of evaluated solutions",
        #                color_scheme=reporting.color_scheme_brown,
        #                default_color_thresholds=(500.0, 25000.0, 100000.0),
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
    ]

    subsects_main = []
    for t in tables:
        tup = (t.title, t.apply(props), t.color_scheme)
        subsects_main.append(tup)

    return reporting.Subsection("Shared Statistics", get_content_of_subsections(subsects_main))



def create_subsection_cdgp_specific(props, dim_rows, dim_cols, headerRowNames):
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
                                                     cv0=0.0, cv1=0.5, cv2=1.0, tableVariants=variants)

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
    return reporting.Subsection("CDGP Statistics", get_content_of_subsections(subsects_cdgp))



_prev_props = None
def prepare_report(sects, filename, exp_prefix, print_status_matrix=True, reuse_props=False,
                   paperwidth=75, include_all_row=True, dim_cols_listings=None):
    """Creating nice LaTeX report of the results."""
    global _prev_props  # used in case reuse_props was set to True
    user_declarations = """\definecolor{darkred}{rgb}{0.56, 0.05, 0.0}
\definecolor{darkgreen}{rgb}{0.0, 0.5, 0.0}
\definecolor{darkblue}{rgb}{0.0, 0.0, 0.55}
\definecolor{darkorange}{rgb}{0.93, 0.53, 0.18}"""
    report = reporting.ReportPDF(geometry_params="[paperwidth={0}cm, paperheight=40cm, margin=0.3cm]".format(paperwidth),
                                 packages=["pbox", "makecell"], user_declarations=user_declarations)
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

        standardize_benchmark_names(props)

        print("\nFiltered Info:")
        for p in props:
            # if "nguyen4" in p["benchmark"]:
            #     print("file: {0}".format(p["thisFileName"]))
            if float(p["result.best.testMSE"]) > 2432971527315918274461803655258245399.0:
                print("file: {0}".format(p["thisFileName"]))


        # Automatically detect benchmarks used
        dim_benchmarks = get_benchmarks_from_props(props)


        if print_status_matrix:
            d = dim_benchmarks * (dim_methodGP * dim_sel * dim_evoMode + (dim_methodCDGP + dim_methodCDGPprops) * dim_sel * dim_evoMode * dim_testsRatio)

            matrix = produce_status_matrix(d, props)
            print("\n****** Status matrix:")
            print(matrix + "\n")
            print("Saving status matrix to file: {0}".format(STATUS_FILE_NAME))
            utils.save_to_file(STATUS_FILE_NAME, matrix)


        dim_rows = dim_benchmarks #.sort()
        if include_all_row:
            dim_rows += dim_true

        if dim_cols_listings is not None:
            save_listings(props, dim_rows, dim_cols_listings)


        subsects = []
        for fun, args in subs:
            if args[0] is None:  # no dimensions for rows, take benchmarks as the default
                args[0] = dim_rows
            args2 = [props] + args
            subsects.append(fun(*args2))

        s = create_section(title, desc, props, subsects, figures, exp_prefix)
        latex_sects.append(s)

    for s in latex_sects:
        if s is not None:
            report.add(s)
    print("\n\nGenerating PDF report ...")
    cwd = os.getcwd()
    os.chdir("results/")
    report.save_and_compile(filename)
    os.chdir(cwd)




def prepare_report_for_dims(props, dim_rows, dim_cols, sects, fname, exp_prefix,
                            print_status_matrix=True, paperwidth=75, include_all_row=True,
                            dim_cols_listings=None):
    """Creating a LaTeX report of the results, where in each table data are presented along the
    sam dimensions."""
    report = reporting.ReportPDF(geometry_params="[paperwidth={0}cm, paperheight=40cm, margin=0.3cm]".format(paperwidth))

    # dim_rows = dim_rows.sort()
    if include_all_row:
        dim_rows += dim_true

    latex_sects = []
    for title, desc, folders, subs, figures in sects:
        if print_status_matrix:
            d = dim_rows * dim_cols

            matrix = produce_status_matrix(d, props)
            print("\n****** Status matrix:")
            print(matrix + "\n")
            print("Saving status matrix to file: {0}".format(STATUS_FILE_NAME))
            utils.save_to_file(STATUS_FILE_NAME, matrix)


        if dim_cols_listings is not None:
            save_listings(props, dim_rows, dim_cols_listings)


        subsects = []
        for fun, args in subs:
            if args[0] is None:  # no dimensions for rows, take benchmarks as the default
                args[0] = dim_rows
            args2 = [props] + args
            subsects.append(fun(*args2))

        s = create_section(title, desc, props, subsects, figures, exp_prefix)
        latex_sects.append(s)

    for s in latex_sects:
        if s is not None:
            report.add(s)
    print("\n\nGenerating PDF report ...")
    cwd = os.getcwd()
    os.chdir("results/")
    report.save_and_compile(fname)
    os.chdir(cwd)




def reports_expR01():
    exp_prefix = "e3"
    # folders = ["exp2_noRestarts"]  # for the exp2 experiment
    folders = ["exp3", "regression_results"]
    title = "Experiments for regression CDGP and  baseline regressors from Scikit. A01 - no noise."
    desc = r""""""
    dim_cols = dim_methodScikit + (dim_methodGP*dim_empty + dim_methodCDGP*dim_empty + dim_methodCDGPprops*dim_weight) *\
               dim_benchmarkNumTests + dim_true # * dim_optThreshold
    headerRowNames = ["method", "weight", "tests"]
    subs = [
        (create_subsection_shared_stats, [None, dim_cols, 25, headerRowNames]),
        (create_subsection_cdgp_specific, [None, dim_cols, headerRowNames]),
        # (create_subsection_aggregation_tests, [None, dim_cols, headerRowNames]),
        # (create_subsection_figures, [None, dim_cols, exp_prefix]),
    ]
    sects = [(title, desc, folders, subs, [])]

    prepare_report(sects, "cdgp_expA01.tex", exp_prefix, paperwidth=100, include_all_row=True, dim_cols_listings=dim_cols)



if __name__ == "__main__":
    utils.ensure_clear_dir("results/")
    utils.ensure_dir("results/figures/")
    utils.ensure_dir("results/listings/")
    # utils.ensure_dir("results/tables/")
    utils.ensure_dir("results/listings/errors/")

    reports_expR01()
