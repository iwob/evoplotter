import os
import math
from app.phd.cdgp.phd_utils import *
from evoplotter import utils
from evoplotter import plotter
from evoplotter import printer
from evoplotter import templates
from evoplotter.templates import *



def simplify_benchmark_name(name):
    """Shortens or modifies the path of the benchmark in order to make the table more readable."""
    i = name.rfind("/")
    name = name if i == -1 else name[i + 1:]
    name = name.replace("ArithmeticSeries3", "IsSeries3")\
               .replace("ArithmeticSeries4", "IsSeries4")\
               .replace("CountPositive2", "CountPos2")\
               .replace("CountPositive3", "CountPos3")\
               .replace("CountPositive4", "CountPos4")\
               .replace("SortedAscending4", "IsSorted4")\
               .replace("SortedAscending5", "IsSorted5")\
               .replace("fg_array_search_2", "Search2")\
               .replace("fg_array_search_3", "Search3")\
               .replace("fg_array_search_4", "Search4")\
               .replace("fg_array_sum_2_15", "Sum2")\
               .replace("fg_array_sum_3_15", "Sum3")\
               .replace("fg_array_sum_4_15", "Sum4")\
               .replace("fg_max4", "Max4")\
               .replace("name-combine", "combine")
    name = name[:name.rfind(".")]  # cut off '.sl'
    return name


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


def get_benchmarks_from_props(props, simplify_name_lambda=None, ignoreNumTests=False):
    if ignoreNumTests:
        dim_benchmarks = Dim.from_dict_postprocess(props, "benchmark", fun=benchmark_shorter)
    else:
        dim_benchmarks = Dim.from_dict(props, "benchmark")
        if simplify_name_lambda is not None:
            configs = [Config(simplify_name_lambda(c.get_caption()), c.filters[0][1],
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
])
dim_methodCDGPprops = Dim([
    Config("$CDGP_{props}$", p_dict_matcher({"method": "CDGPprops"}), method="CDGPprops"),
])
dim_methodGPR = Dim([
    Config("$GPR$", p_dict_matcher({"method": "GPR"}), method="GPR"),
])
dim_operatorProbs = Dim([
    Config("$m0.25,c0.75$", p_dict_matcher({"operatorProbs": "0.25,0.75"}), method="0.25,0.75"),
    Config("$m0.5,c0.5$", p_dict_matcher({"operatorProbs": "0.5,0.5"}), method="0.5,0.5"),
    Config("$m0.75,c0.25$", p_dict_matcher({"operatorProbs": "0.75,0.25"}), method="0.75,0.25"),
    Config("$m1.0,c0.0$", p_dict_matcher({"operatorProbs": "1.0,0.0"}), method="1.0,0.0"),
])
baseline_algs = ["CVC4", "EUSolver"]
dim_methodBaseline = Dim([Config(a, p_dict_matcher({"method": a}), method=a) for a in baseline_algs])
dim_method = dim_methodBaseline + dim_methodGPR + dim_methodCDGP + dim_methodCDGPprops
dim_sel = Dim([
    Config("$Tour$", p_sel_tourn, selection="tournament"),
    Config("$Lex$", p_sel_lexicase, selection="lexicase"),
])
# dim_evoMode = Dim([Config("$steadyState$", p_steadyState, evolutionMode="steadyState"),
#                    Config("$generational$", p_generational, evolutionMode="generational")])
dim_evoMode = Dim([
    Config("$generational$", p_dict_matcher({"evolutionMode": "generational"}), evolutionMode="generational"),
    Config("$steadyState$", p_dict_matcher({"evolutionMode": "steadyState"}), evolutionMode="steadyState"),
])
dim_testsRatio = Dim([
    Config("$0.0$", p_testsRatio_equalTo("0.0"), testsRatio="0.0"),
    Config("$0.25$", p_testsRatio_equalTo("0.25"), testsRatio="0.25"),
    Config("$0.5$", p_testsRatio_equalTo("0.5"), testsRatio="0.5"),
    Config("$0.75$", p_testsRatio_equalTo("0.75"), testsRatio="0.75"),
    Config("$1.0$", p_testsRatio_equalTo("1.0"), testsRatio="1.0"),
])
dim_weight = Dim([
    Config("$1$", p_dict_matcher({"partialConstraintsWeight": "1"}), benchmarkNumTests="1"),
    # Config("$5$", p_dict_matcher({"partialConstraintsWeight": "5"}), benchmarkNumTests="5"),
])




def get_content_of_subsections(subsects):
    content = []
    vspace = reporting.BlockLatex(r"\vspace{0.75cm}"+"\n")
    for title, table, cs in subsects:
        if isinstance(cs, reporting.ColorScheme3):
            cs = cs.toBlockLatex()
        sub = reporting.SectionRelative(title, contents=[cs, table, vspace])
        content.append(sub)
    return content


def post(s):
    s = s.replace("{ccccccccccccc}", "{rrrrrrrrrrrrr}").replace("{rrr", "{lrr")\
         .replace(r"\_{lex}", "_{lex}").replace(r"\_{", "_{").replace("resistance_par", "res")\
         .replace("gravity", "gr")
    return s

def postRemoveZeros(s):
    s = post(s)
    s = s.replace("}0.00", "}\\phantom{0.00}").replace("{0.00}", "{\\phantom{0.00}}")
    return s



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


def cellShading(a, b, c):
    assert c >= b >= a
    return printer.CellShading(a, b, c, "colorLow", "colorMedium", "colorHigh")

rBoldWhen1 = printer.LatexTextbf(lambda v, b: v == "1.00")


def create_subsection_shared_stats(props, title, dim_rows, dim_cols, numRuns, headerRowNames, results_dir, variants=None):
    vb = 1  # vertical border
    variants_ids = ["all"] if variants is None else [v.get_caption() for v in variants]

    def scNotColorValueExtractor(s):
        if s == "-" or "10^{" not in s:
            return s
        else:
            r = s.split("10^{")
            return r[1][:-2]

    status_color_scheme = reporting.ColorScheme3(["1.0, 1.0, 1.0", "0.65, 0.0, 0.0", "0.8, 0, 0"],
                                                 ["white", "light red", "red"])

    tables = [
        TableGenerator(get_num_computed, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Status (correctly finished runs)",
                       color_scheme=reversed(status_color_scheme),
                       cellRenderers=[cellShading(0.0, 0.8*numRuns, numRuns)],
                       vertical_border=vb, table_postprocessor=post, variants=variants
                       ),
        TableGenerator(fun_successRate, Dim(dim_rows[:-1]), Dim(dim_cols[:-1]), headerRowNames=headerRowNames,
                       title="Success rates (properties met)",
                       color_scheme=reporting.color_scheme_darkgreen,
                       cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=postRemoveZeros, variants=variants,
                       addRowWithRanks=True, addRowWithMeans=True,
                       outputFiles=[results_dir + "/tables/cdgp_succRate_{0}.tex".format(utils.normalize_name(vid)) for vid in variants_ids]
                       ),
        TableGenerator(get_averageAlgorithmRanksCDGP(dim_cols[:-1], dim_rows[:-1], ONLY_VISIBLE_SOLS=True, NUM_SHOWN=100),
                       Dim(dim_rows[-1]), Dim(dim_cols[-1]),
                       headerRowNames=headerRowNames,
                       title="Average ranks of the solvers (success rate)",
                       color_scheme=reporting.color_scheme_violet,
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(
        #     get_averageAlgorithmRanksCDGP(dim_operatorProbs, dim_rows[:-1], ONLY_VISIBLE_SOLS=True, NUM_SHOWN=100),
        #     Dim(dim_cols[-1]), dim_methodGPR + dim_methodCDGP,
        #     headerRowNames=headerRowNames,
        #     title="Average ranks of the solvers (success rate)",
        #     color_scheme=reporting.color_scheme_violet,
        #     default_color_thresholds=(0.0, 900.0, 1800.0),
        #     vertical_border=vb, table_postprocessor=post, variants=variants,
        # ),
        # TableGenerator(
        #     get_averageAlgorithmRanksCDGP(dim_operatorProbs, dim_rows[:-1], ONLY_VISIBLE_SOLS=True, NUM_SHOWN=100),
        #     Dim(dim_rows[-1]), dim_cols,
        #     headerRowNames=headerRowNames,
        #     title="Average ranks of the solvers (success rate) MODIFIED",
        #     color_scheme=reporting.color_scheme_violet,
        #     default_color_thresholds=(0.0, 900.0, 1800.0),
        #     vertical_border=vb, table_postprocessor=post, variants=variants,
        # ),
        TableGenerator(get_rankingOfBestSolversCDGP(dim_cols[:-1], ONLY_VISIBLE_SOLS=True, NUM_SHOWN=100),
                       Dim(dim_cols[-1]), dim_rows,
                       headerRowNames=headerRowNames,
                       title="Best solvers for the given benchmark (success rate)",
                       color_scheme=reporting.color_scheme_violet,
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_avg_runtime, dim_rows, Dim(dim_cols[:-1]), headerRowNames=headerRowNames,
                       title="Average runtime [s]",
                       color_scheme=reporting.color_scheme_violet,
                       cellRenderers=[cellShading(0.0, 900.0, 1800.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/cdgp_avgRuntime_{0}.tex".format(utils.normalize_name(vid)) for vid in variants_ids]
                       ),
        TableGenerator(get_avg_runtimeOnlySuccessful, dim_rows, Dim(dim_cols[:-1]), headerRowNames=headerRowNames,
                       title="Average runtime (only successful) [s]",
                       color_scheme=reporting.color_scheme_violet,
                       cellRenderers=[cellShading(0.0, 900.0, 1800.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(get_avg_runtimeOnlyUnsuccessful, dim_rows, Dim(dim_cols[:-1]), headerRowNames=headerRowNames,
        #                title="Average runtime (only unsuccessful) [s]",
        #                color_scheme=reporting.color_scheme_violet,
        #                cellRenderers=[cellShading(0.0, 900.0, 1800.0)],
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
    ]

    return createSubsectionWithTables(title, tables, props)




def create_subsection_ea_stats(props, title, dim_rows, dim_cols, headerRowNames, results_dir, variants=None):
    vb = 1  # vertical border

    tables = [
        TableGenerator(get_rankingOfBestSolutionsCDGP(ONLY_VISIBLE_SOLS=True, NUM_SHOWN=15),
                       Dim(dim_cols.configs[:-1]), Dim(dim_rows.configs[:-1]),
                       headerRowNames=headerRowNames,
                       title="The best solutions (simplified) found for each benchmark and their sizes. Format: solution (isCorrect?) (size)",
                       color_scheme=reporting.color_scheme_violet, middle_col_align="l",
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_size, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average sizes of best of runs (number of nodes)",
                       color_scheme=reporting.color_scheme_yellow,
                       cellRenderers=[cellShading(0.0, 100.0, 200.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(fun_sizeOnlySuccessful, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average sizes of best of runs (number of nodes) (only successful)",
                       color_scheme=reporting.color_scheme_yellow,
                       cellRenderers=[cellShading(0.0, 100.0, 200.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_avg_generation, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average generation (all)",
                       color_scheme=reporting.color_scheme_teal,
                       cellRenderers=[cellShading(0.0, 5000.0, 10000.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_avg_evaluated, dim_rows, dim_cols, headerRowNames=headerRowNames,
                       title="Average number of evaluated solutions",
                       color_scheme=reporting.color_scheme_brown,
                       cellRenderers=[cellShading(0.0, 5000.0, 10000.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(get_avg_evaluatedSuccessful, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Average number of evaluated solutions",
        #                color_scheme=reporting.color_scheme_brown,
        #                cellRenderers = [cellShading(500.0, 25000.0, 100000.0)],
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
        # TableGenerator(get_avg_doneAlgRestarts, dim_rows, dim_cols, headerRowNames=headerRowNames,
        #                title="Number of algorithm restarts  (avg)",
        #                color_scheme=reporting.color_scheme_gray_light,
        #                cellRenderers = [cellShading(0.0, 1e2, 1e4)],
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
    ]

    return createSubsectionWithTables(title, tables, props)



def create_subsection_cdgp_specific(props, title, dim_rows, dim_cols, headerRowNames, results_dir, variants=None):
    vb = 1  # vertical border
    variants_ids = ["all"] if variants is None else [v.get_caption() for v in variants]

    tables = [
        TableGenerator(get_avg_totalTests,
                       dim_rows, Dim(dim_cols[:-1]),
                       headerRowNames=headerRowNames,
                       title="Average sizes of $T_C$ (total tests in run)",
                       color_scheme=reporting.color_scheme_blue, middle_col_align="l",
                       cellRenderers=[cellShading(0.0, 1000.0, 2000.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/cdgp_Tc_{0}.tex".format(utils.normalize_name(vid)) for vid in variants_ids]
                       ),
        TableGenerator(get_stats_maxSolverTime,
                       dim_rows, dim_cols,
                       headerRowNames=headerRowNames,
                       title="Max solver time per query [s]",
                       color_scheme=reporting.color_scheme_violet, middle_col_align="l",
                       cellRenderers=[cellShading(0.0, 5.0, 10.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_stats_avgSolverTime,
                       dim_rows, dim_cols,
                       headerRowNames=headerRowNames,
                       title="Avg solver time per query [s]",
                       color_scheme=reporting.color_scheme_brown, middle_col_align="l",
                       cellRenderers=[cellShading(0.0, 5.0, 10.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_avgSolverTotalCalls,
                       dim_rows, dim_cols,
                       headerRowNames=headerRowNames,
                       title="Avg number of solver calls (in thousands; 1=1000)",
                       color_scheme=reporting.color_scheme_blue, middle_col_align="l",
                       cellRenderers=[cellShading(0.0, 5.0, 10.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        TableGenerator(get_numSolverCallsOverXs,
                       dim_rows, dim_cols,
                       headerRowNames=headerRowNames,
                       title="Number of solver calls $>$ 0.5s",
                       color_scheme=reporting.color_scheme_blue, middle_col_align="l",
                       cellRenderers=[cellShading(0.0, 10.0, 20.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(get_freqCounterexamples,
        #                Dim(dim_rows.configs[:-1]), Dim(dim_cols.configs[:-1]),
        #                headerRowNames=headerRowNames,
        #                title="The most frequently found counterexamples for each benchmark and configuration",
        #                color_scheme=reporting.color_scheme_blue, middle_col_align="l",
        #                cellRenderers=[cellShading(0.0, 5.0, 10.0)],
        #                vertical_border=vb, table_postprocessor=post, table_variants=variants,
        #                ),
     ]

    return createSubsectionWithTables(title, tables, props)


def log10ValueExtractor(s):
    base = 10.0
    if not utils.isfloat(s):
        return s
    else:
        return math.log(float(s), base)


def create_subsection_custom_tables(props, title, EXP_TYPE, dimens, results_dir, variants=None):
    assert isinstance(dimens, dict)
    assert EXP_TYPE == "SLIA" or EXP_TYPE == "LIA"
    vb = 1  # vertical border

    dim_rows = reversed(dimens["testsRatio"])
    dim_cols = dimens["method"] * dimens["evoMode"] * dimens["selection"]
    tables = [
        TableGenerator(fun_successRate,
                       dim_rows,
                       dimens["method"] * dimens["evoMode"] * dimens["selection"],
                       title="Success rates", headerRowNames=[],
                       color_scheme=reporting.color_scheme_darkgreen,
                       cellRenderers=[printer.LatexTextbfMaxInTable(), printer.CellShadingTable("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/custom/cdgp_succRate_rowsAsTestsRatio.tex"]
                       ),
        TableGenerator(fun_successRate,
                       dim_rows,
                       dimens["method"] * dimens["evoMode"],
                       title="Success rates", headerRowNames=[],
                       color_scheme=reporting.color_scheme_darkgreen,
                       cellRenderers=[printer.LatexTextbfMaxInTable(), printer.CellShadingTable("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/custom/cdgp_succRate_rowsAsTestsRatio_colsAsEvoMode.tex"]
                       ),
        TableGenerator(fun_successRate,
                       dim_rows,
                       dimens["method"] * dimens["selection"],
                       title="Success rates", headerRowNames=[],
                       color_scheme=reporting.color_scheme_darkgreen,
                       cellRenderers=[printer.LatexTextbfMaxInTable(), printer.CellShadingTable("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/custom/cdgp_succRate_rowsAsTestsRatio_colsAsSelection.tex"]
                       ),
        TableGenerator(get_avg_totalTests,
                       dim_rows, dim_cols,
                       headerRowNames=[],
                       title="Average sizes of $T_C$ (total tests in run)",
                       color_scheme=reporting.color_scheme_blue, middle_col_align="l",
                       cellRenderers=[printer.LatexTextbfMaxInTable(), printer.CellShadingTable("colorLow", "colorMedium", "colorHigh", valueExtractor=log10ValueExtractor)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[
                           results_dir + "/tables/custom/cdgp_Tc_rowsAsTestsRatio.tex"]
                       ),
        TableGenerator(get_avg_runtime, dim_rows, dim_cols, headerRowNames=[],
                       title="Average runtime [s]",
                       color_scheme=reporting.color_scheme_violet,
                       cellRenderers=[printer.LatexTextbfMinInTable(), printer.CellShadingTable("colorHigh", "colorMedium", "colorLow", valueExtractor=log10ValueExtractor)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/custom/cdgp_runtime_rowsAsTestsRatio.tex"]
                       ),
        TableGenerator(get_avg_runtimeOnlySuccessful, dim_rows, dim_cols, headerRowNames=[],
                       title="Average runtime (only successful) [s]",
                       color_scheme=reporting.color_scheme_violet,
                       cellRenderers=[printer.LatexTextbfMinInTable(), printer.CellShadingTable("colorHigh", "colorMedium", "colorLow")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/custom/cdgp_runtime_rowsAsTestsRatio_successful.tex"]
                       ),
        FriedmannTestPython(dimens["benchmark"],
                            dimens["method"] * dimens["evoMode"] * dimens["selection"] * dimens["testsRatio"],
                            get_successRate, p_treshold=0.05,
                            title="Friedman test for success rates (all variants)",
                            pathFriedmanViz="tables/custom/friedman_all.gv",
                            workingDir=results_dir),
        # FriedmannTestPython(dimensions_dict["benchmark"],
        #                     dimensions_dict["method"] * Dim(dimensions_dict["evoMode"][0]) * dimensions_dict["selection"] * dimensions_dict["testsRatio"],
        #                     get_successRate, p_treshold=0.05,
        #                     title="Friedman test for success rates (generational variants only)",
        #                     pathFriedmanViz="tables/custom/friedman_generational.gv",
        #                     workingDir=results_dir),
        # FriedmannTestPython(dimensions_dict["benchmark"],
        #                     dimensions_dict["method"] * Dim(dimensions_dict["evoMode"][1]) * dimensions_dict["selection"] * dimensions_dict["testsRatio"],
        #                     get_successRate, p_treshold=0.05,
        #                     title="Friedman test for success rates (steadyState variants only)",
        #                     pathFriedmanViz="tables/custom/friedman_steadyState.gv",
        #                     workingDir=results_dir),
        TableGenerator(fun_successRate,
                       dimens["benchmark"],
                       dimens["method"] * dimens["testsRatio"],
                       title="Success rates", headerRowNames=[],
                       color_scheme=reporting.color_scheme_darkgreen,
                       cellRenderers=[rBoldWhen1, cellShading(0.0, 0.5, 1.0)],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/custom/cdgp_succRate_colsAsTestsRatio.tex"]
                       ),
        FriedmannTestPython(dimens["benchmark"],
                            dimens["method"] * dimens["testsRatio"],
                            get_successRate, p_treshold=0.05,
                            title="Friedman test for success rates (testsRatio)",
                            pathFriedmanViz= "tables/custom/friedman_testsRatio.gv",
                            workingDir=results_dir),
    ]

    if EXP_TYPE == "LIA":  # LIA specific tables
        tables.extend([
            FriedmannTestPython(dimens["benchmark"],
                                dimens["method_CDGP"] * dimens["evoMode"] * dimens["selection"] * dimens["testsRatio"],
                                get_successRate, p_treshold=0.05,
                                title="Friedman test for success rates (CDGP variants)",
                                pathFriedmanViz="tables/custom/friedman_cdgp.gv",
                                workingDir=results_dir),
            FriedmannTestPython(dimens["benchmark"],
                                dimens["method_GPR"] * dimens["evoMode"] * dimens["selection"] * dimens["testsRatio"],
                                get_successRate, p_treshold=0.05,
                                title="Friedman test for success rates (GPR variants)",
                                pathFriedmanViz="tables/custom/friedman_gpr.gv",
                                workingDir=results_dir),
            TableGenerator(fun_successRate,
                           dimens["benchmark"],
                           dimens["method_CDGP"] * dimens["testsRatio"],
                           title="Success rates", headerRowNames=[],
                           color_scheme=reporting.color_scheme_darkgreen,
                           cellRenderers=[rBoldWhen1, cellShading(0.0, 0.5, 1.0)],
                           vertical_border=vb, table_postprocessor=post, variants=variants,
                           outputFiles=[results_dir + "/tables/custom/cdgp_succRate_colsAsTestsRatio2.tex"]
                           ),
            FriedmannTestPython(dimens["benchmark"],
                                dimens["method_CDGP"] * dimens["testsRatio"],
                                get_successRate, p_treshold=0.05,
                                title="Friedman test for success rates (testsRatio)",
                                pathFriedmanViz="tables/custom/friedman_testsRatio2.gv",
                                workingDir=results_dir),
            TableGenerator(fun_successRate,
                           dimens["benchmark"],
                           dimens["method_CDGP"] * dimens["evoMode_g"] * dimens["selection_lex"] * dimens["testsRatio_0.25"] +
                           dimens["method_CDGP"] * dimens["evoMode_s"] * dimens["selection_tour"] * dimens["testsRatio_1.0"] +
                           dimens["method_GPR"] * dimens["evoMode_g"] * dimens["selection_lex"] * dimens["testsRatio_0.25"],
                           title="Success rates", headerRowNames=[],
                           color_scheme=reporting.color_scheme_darkgreen,
                           cellRenderers=[rBoldWhen1, cellShading(0.0, 0.5, 1.0)],
                           vertical_border=vb, table_postprocessor=post, variants=variants
                           ),
            WilcoxonSignedRankTest(dimens["benchmark"],
                                   dimens["method_CDGP"] * dimens["evoMode_g"] * dimens["selection_lex"] * dimens["testsRatio_0.25"] +
                                   dimens["method_GPR"] * dimens["evoMode_g"] * dimens["selection_lex"] * dimens["testsRatio_0.25"],
                                   get_successRate, p_treshold = 0.05,
                                   title="Wilcoxon signed-rank test for success rates (best CDGP vs best GPR)",),
            WilcoxonSignedRankTest(dimens["benchmark"],
                                   dimens["method_CDGP"] * dimens["evoMode_s"] * dimens["selection_tour"] * dimens["testsRatio_1.0"] +
                                   dimens["method_GPR"] * dimens["evoMode_g"] * dimens["selection_lex"] * dimens["testsRatio_0.25"],
                                   get_successRate, p_treshold=0.05,
                                   title="Wilcoxon signed-rank test for success rates (worst CDGP vs best GPR)", )
        ])

    return createSubsectionWithTables(title, tables, props)




def create_subsection_formal(props_lia, props_slia, title, EXP_TYPE, dimens, results_dir, variants=None):
    assert EXP_TYPE == "SLIA" or EXP_TYPE == "LIA"
    vb = 1  # vertical border

    props = []
    props.extend(props_lia)
    props.extend(props_slia)

    dim_cols = dimens["method"] +\
               dimens["method_CDGP"] * dimens["evoMode_g"] * dimens["selection_lex"] * dimens["testsRatio_0.25"] +\
               dimens["method_CDGP"] * dimens["evoMode_s"] * dimens["selection_tour"] * dimens["testsRatio_1.0"] +\
               dimens["method_GPR"] * dimens["evoMode_g"] * dimens["selection_lex"] * dimens["testsRatio_0.25"]

    tables = [
        TableGenerator(get_num_computed,
                       dimens["benchmark"], dim_cols,
                       title="Status (correctly finished runs)",
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       ),
        # TableGenerator(fun_successRate,
        #                dimens["benchmark_lia"], dim_cols,
        #                title="Success rates", headerRowNames=[],
        #                color_scheme=reporting.color_scheme_darkgreen,
        #                cellRenderers=[rBoldWhen1, cellShading(0.0, 0.5, 1.0)],
        #                vertical_border=vb, table_postprocessor=post, variants=variants,
        #                outputFiles=[results_dir + "/tables/formal/succRate_lia.tex"]
        #                ),
        # TableGenerator(fun_successRate,
        #                dimens["benchmark_slia"], dim_cols,
        #                title="Success rates",
        #                color_scheme=reporting.color_scheme_darkgreen,
        #                cellRenderers=[rBoldWhen1, cellShading(0.0, 0.5, 1.0)],
        #                vertical_border=vb, table_postprocessor=post, variants=variants,
        #                outputFiles=[results_dir + "/tables/formal/succRate_slia.tex"]
        #                ),
        TableGenerator(fun_successRate,
                       dimens["benchmark"], dim_cols,
                       title="Success rates",
                       color_scheme=reporting.color_scheme_darkgreen,
                       # cellRenderers=[printer.LatexTextbfMaxInRow(), cellShading(0.0, 0.5, 1.0)],
                       cellRenderers=[printer.LatexTextbfMaxInRow(), printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/formal/succRate_all.tex"]
                       ),
        TableGenerator(get_avg_runtime,
                       dimens["benchmark"], dim_cols,
                       title="Average runtime [s]",
                       color_scheme=reporting.color_scheme_violet,
                       cellRenderers=[printer.LatexTextbfMinInRow(), printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       addRowWithRanks=True, ranksHigherValuesBetter=False
                       #outputFiles=[results_dir + "/tables/formal/runtime_all.tex"]
                       ),
        TableGenerator(get_avg_runtimeOnlySuccessful,
                       dimens["benchmark"], dim_cols,
                       title="Average runtime (only successful) [s]",
                       color_scheme=reporting.color_scheme_violet,
                       cellRenderers=[printer.LatexTextbfMinInRow(), printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/formal/runtime_successful.tex"]
                       ),
        TableGenerator(fun_sizeOnlySuccessful_original,
                       dimens["benchmark"], dim_cols,
                       title="Average sizes of best of runs (number of nodes) (only successful) (NOT SIMPLIFIED)",
                       color_scheme=reporting.color_scheme_yellow,
                       cellRenderers=[printer.LatexTextbfMinInRow(), printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants
                       ),
        TableGenerator(fun_sizeOnlySuccessful_simplified,
                       dimens["benchmark"], dim_cols,
                       title="Average sizes of best of runs (number of nodes) (only successful) (SIMPLIFIED)",
                       color_scheme=reporting.color_scheme_yellow,
                       cellRenderers=[printer.LatexTextbfMinInRow(), printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants
                       ),
        TableGenerator(fun_sizeOnlySuccessful_chooseBest,
                       dimens["benchmark"], dim_cols,
                       title="Average sizes of best of runs (number of nodes) (only successful) (CHOOSE BEST)",
                       color_scheme=reporting.color_scheme_yellow,
                       cellRenderers=[printer.LatexTextbfMinInRow(), printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")],
                       vertical_border=vb, table_postprocessor=post, variants=variants,
                       outputFiles=[results_dir + "/tables/formal/size_chooseBest.tex"]
                       ),
        TableGenerator(get_rankingOfBestSolutionsCDGP(ONLY_VISIBLE_SOLS=True, NUM_SHOWN=15, STR_LEN_LIMIT=75, key_solution="result.best.smtlib"),
                       dimens["benchmark"], dim_cols,
                       title="The best solutions (simplified) found for each benchmark and their sizes. Format: solution (isCorrect?) (size)",
                       color_scheme=reporting.color_scheme_violet, middle_col_align="l",
                       default_color_thresholds=(0.0, 900.0, 1800.0),
                       vertical_border=vb, table_postprocessor=post, variants=variants
                       ),
    ]

    return createSubsectionWithTables(title, tables, props)



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



def reports_e0_paramTests():

    name = "e0_paramTests"
    results_dir = "results_{0}".format(name)
    ensure_result_dir(results_dir)
    title = "Experiments for CDGP: checking the impact of different evolution parameters"
    desc = r"""
\parbox{30cm}{
Rerun of the CDGP experiments series for my PhD thesis.
}

\begin{lstlisting}[breaklines]

\end{lstlisting}

NOTE: for steady state, maxGenerations is multiplied by populationSize. 
"""

    # folders = ["phd_cdgp_e0_paramTests_01", "phd_cdgp_e0_paramTests_02"]
    folders = ["phd_cdgp_e0_paramTests", "phd_cdgp_e0_paramTests_fix1"]
    desc += "\n\\bigskip\\noindent Folders with data: " + r"\lstinline{" + str(folders) + "}\n"
    props = load_correct_props(folders, results_dir)
    standardize_benchmark_names(props)
    dim_rows = get_benchmarks_from_props(props)  #, simplify_name_lambda=simplify_benchmark_name)
    dim_rows += dim_rows.dim_true_within("ALL")

    dim_cols_cdgp = (dim_methodCDGP + dim_methodCDGPprops * dim_weight) * dim_testsRatio
    dim_cols_ea = (dim_methodGPR + dim_cols_cdgp) * dim_operatorProbs
    dim_cols = dim_methodBaseline + dim_cols_ea


    dim_cols += dim_cols.dim_true_within()
    dim_cols_ea += dim_cols_ea.dim_true_within()
    dim_cols_cdgp += dim_cols_cdgp.dim_true_within()

    headerRowNames = ["method"]
    subs = [
        (create_subsection_shared_stats, [props, "Shared Statistics", dim_rows, dim_cols, 25, headerRowNames, results_dir]),
        (create_subsection_ea_stats, [props, "EA/CDGP Statistics", dim_rows, dim_cols_ea, headerRowNames, results_dir]),
        (create_subsection_cdgp_specific, [props, "CDGP Statistics", dim_rows, dim_cols_cdgp, headerRowNames, results_dir]),
        # (create_subsection_aggregation_tests, [dim_rows, dim_cols, headerRowNames]),
        # (create_subsection_figures, [dim_rows, dim_cols, exp_prefix]),
    ]
    sects = [(title, desc, subs, [])]


    save_listings(props, dim_rows, dim_cols, results_dir=results_dir)
    templates.prepare_report(sects, "cdgp_{0}.tex".format(name), dir_path=results_dir, paperwidth=190, user_declarations=user_declarations)




def reports_e0_lia():

    name = "e0_lia"
    results_dir = "reports/results_{0}".format(name)
    ensure_result_dir(results_dir)
    title = "Final CDGP experiment for the LIA logic"
    desc = r"""
\parbox{30cm}{
Rerun of the CDGP experiments series for my PhD thesis.
}

\begin{lstlisting}[breaklines]

\end{lstlisting}

NOTE: for steady state, maxGenerations is multiplied by populationSize. 
"""

    # folders = ["phd_cdgp_e0_paramTests_01", "phd_cdgp_e0_paramTests_02"]
    folders = ["LIA"]
    desc += "\n\\bigskip\\noindent Folders with data: " + r"\lstinline{" + str(folders) + "}\n"
    props = load_correct_props(folders, results_dir)
    standardize_benchmark_names(props)
    dim_benchmarks = get_benchmarks_from_props(props)  #, simplify_name_lambda=simplify_benchmark_name)
    dim_rows = dim_benchmarks + dim_benchmarks.dim_true_within("ALL")

    dimensions_dict = {"benchmark": dim_benchmarks,
                       "testsRatio": dim_testsRatio,
                       "testsRatio_0.0": Dim(dim_testsRatio[0]),
                       "testsRatio_0.25": Dim(dim_testsRatio[1]),
                       "testsRatio_0.5": Dim(dim_testsRatio[2]),
                       "testsRatio_0.75": Dim(dim_testsRatio[3]),
                       "testsRatio_1.0": Dim(dim_testsRatio[4]),
                       "evoMode": dim_evoMode,
                       "evoMode_g": Dim(dim_evoMode[0]),
                       "evoMode_s": Dim(dim_evoMode[1]),
                       "selection": dim_sel,
                       "selection_tour": Dim(dim_sel[0]),
                       "selection_lex": Dim(dim_sel[1]),
                       "method": dim_methodCDGP + dim_methodGPR,
                       "method_CDGP": dim_methodCDGP,
                       "method_GPR": dim_methodGPR,}


    # ----- One big table -----
    # dim_cols_cdgp = dim_methodCDGP * dim_evoMode * dim_sel * dim_testsRatio + dim_methodGPR * dim_evoMode * dim_sel * dim_testsRatio
    # dim_cols_ea = dim_cols_cdgp
    # dim_cols = dim_cols_ea
    # variants = None
    # -------------------------

    # ----- Several tables -----
    # dim_cols_cdgp = dim_methodCDGP * dim_sel * dim_testsRatio + dim_methodGPR * dim_sel * dim_testsRatio
    # dim_cols_ea = dim_cols_cdgp
    # dim_cols = dim_cols_ea
    # variants = dim_evoMode.configs
    # -------------------------


    # ----- Several tables (GPR-divided and evoMode-divided) -----
    # dim_cols_cdgp = dim_sel * dim_testsRatio
    # dim_cols_ea = dim_cols_cdgp
    # dim_cols = dim_cols_ea
    # variants = (dim_evoMode * (dim_methodCDGP + dim_methodGPR)).configs
    # -------------------------


    # ----- Several tables (GPR-divided) -----
    dim_cols_cdgp = dim_evoMode * dim_sel * dim_testsRatio
    dim_cols_ea = dim_cols_cdgp
    dim_cols = dim_cols_ea
    variants = (dim_methodCDGP + dim_methodGPR).configs
    # -------------------------

    dim_cols += dim_cols.dim_true_within()
    dim_cols_ea += dim_cols_ea.dim_true_within()
    dim_cols_cdgp += dim_cols_cdgp.dim_true_within()

    headerRowNames = [""]
    subs = [
        (create_subsection_shared_stats, [props, "Shared Statistics", dim_rows, dim_cols, 50, headerRowNames, results_dir, variants]),
        (create_subsection_ea_stats, [props, "EA/CDGP Statistics", dim_rows, dim_cols_ea, headerRowNames, results_dir, variants]),
        (create_subsection_cdgp_specific, [props, "CDGP Statistics", dim_rows, dim_cols_cdgp, headerRowNames, results_dir, variants]),
        (create_subsection_custom_tables, [props, "Custom tables", "LIA", dimensions_dict, results_dir, None])
        # (create_subsection_aggregation_tests, [dim_rows, dim_cols, headerRowNames]),
        # (create_subsection_figures, [dim_rows, dim_cols, exp_prefix]),
    ]
    sects = [(title, desc, subs, [])]


    save_listings(props, dim_rows, dim_cols, results_dir=results_dir)
    templates.prepare_report(sects, "cdgp_{0}.tex".format(name), dir_path=results_dir, paperwidth=190, user_declarations=user_declarations)




def reports_e0_slia():

    name = "e0_slia"
    results_dir = "reports/results_{0}".format(name)
    ensure_result_dir(results_dir)
    title = "Final CDGP experiment for the SLIA logic"
    desc = r"""
\parbox{30cm}{
Rerun of the CDGP experiments series for my PhD thesis.
}

\begin{lstlisting}[breaklines]

\end{lstlisting}

NOTE: for steady state, maxGenerations is multiplied by populationSize. 
"""

    # folders = ["phd_cdgp_e0_paramTests_01", "phd_cdgp_e0_paramTests_02"]
    folders = ["SLIA"]
    desc += "\n\\bigskip\\noindent Folders with data: " + r"\lstinline{" + str(folders) + "}\n"
    props = load_correct_props(folders, results_dir)
    standardize_benchmark_names(props)
    dim_benchmarks = get_benchmarks_from_props(props)  #, simplify_name_lambda=simplify_benchmark_name)
    dim_rows = dim_benchmarks + dim_benchmarks.dim_true_within("ALL")

    dimensions_dict = {"benchmark": dim_benchmarks,
                       "testsRatio": dim_testsRatio,
                       "testsRatio_0.0": Dim(dim_testsRatio[0]),
                       "testsRatio_0.25": Dim(dim_testsRatio[1]),
                       "testsRatio_0.5": Dim(dim_testsRatio[2]),
                       "testsRatio_0.75": Dim(dim_testsRatio[3]),
                       "testsRatio_1.0": Dim(dim_testsRatio[4]),
                       "evoMode": dim_evoMode,
                       "evoMode_g": Dim(dim_evoMode[0]),
                       "evoMode_s": Dim(dim_evoMode[1]),
                       "selection": dim_sel,
                       "selection_tour": Dim(dim_sel[0]),
                       "selection_lex": Dim(dim_sel[1]),
                       "method": dim_methodCDGP,
                       "method_CDGP": dim_methodCDGP}


    # ----- One big table -----
    dim_cols_cdgp = dim_methodCDGP * dim_evoMode * dim_sel * dim_testsRatio
    dim_cols_ea = dim_cols_cdgp
    dim_cols = dim_cols_ea
    variants = None
    # headerRowNames = ["method", "scheme", "selection", r"$\alpha$"]
    headerRowNames = []
    # -------------------------

    # ----- Several tables -----
    # dim_cols_cdgp = dim_methodCDGP * dim_sel * dim_testsRatio
    # dim_cols_ea = dim_cols_cdgp
    # dim_cols = dim_cols_ea
    # variants = dim_evoMode.configs
    # headerRowNames = ["", "method", "selection", r"$\alpha$"]
    # -------------------------


    # Adding "all" dimension
    dim_cols += dim_cols.dim_true_within()
    dim_cols_ea += dim_cols_ea.dim_true_within()
    dim_cols_cdgp += dim_cols_cdgp.dim_true_within()


    subs = [
        (create_subsection_shared_stats, [props, "Shared Statistics", dim_rows, dim_cols, 50, headerRowNames, results_dir, variants]),
        (create_subsection_ea_stats, [props, "EA/CDGP Statistics", dim_rows, dim_cols_ea, headerRowNames, results_dir, variants]),
        (create_subsection_cdgp_specific, [props, "CDGP Statistics", dim_rows, dim_cols_cdgp, headerRowNames, results_dir, variants]),
        (create_subsection_custom_tables, [props, "Custom tables", "SLIA", dimensions_dict, results_dir, None]),
        # (create_subsection_aggregation_tests, [dim_rows, dim_cols, headerRowNames]),
        # (create_subsection_figures, [dim_rows, dim_cols, exp_prefix]),
    ]
    sects = [(title, desc, subs, [])]


    save_listings(props, dim_rows, dim_cols, results_dir=results_dir)
    templates.prepare_report(sects, "cdgp_{0}.tex".format(name), dir_path=results_dir, paperwidth=190, user_declarations=user_declarations)




def reports_e0_formal():
    name = "e0_formal"
    results_dir = "reports/results_{0}".format(name)
    ensure_result_dir(results_dir)
    title = "Comparison of CDGP with formal approaches to formal synthesis"
    desc = r"""
\parbox{30cm}{
Rerun of the CDGP experiments series for my PhD thesis.
}

\begin{lstlisting}[breaklines]

\end{lstlisting}
"""
    dim_methodEUSolver = Dim([Config("EUSolver", p_dict_matcher({"method": "eusolver"}), method="eusolver")])
    dim_methodCVC4 = Dim([Config("CVC4", p_dict_matcher({"method": "cvc4_1.8"}), method="cvc4_1.8")])

    # dim_bestCDGP_lia = Dim()

    folders_lia = ["FORMAL_SIMPLIFIED/data_formal_lia"]
    folders_lia_mine = ["LIA"]
    folders_slia = ["FORMAL_SIMPLIFIED/data_formal_slia"]
    folders_slia_mine = ["SLIA"]

    desc += "\n\\bigskip\\noindent Folders with data:\\\\"
    desc += r"LIA: \lstinline{" + str(folders_lia) + "}\\\\\n"
    desc += r"SLIA: \lstinline{" + str(folders_slia) + "}\\\\\n"

    props_lia = load_correct_props(folders_lia, results_dir)
    standardize_benchmark_names(props_lia)
    dim_benchmarks_lia = get_benchmarks_from_props(props_lia)

    props_slia = load_correct_props(folders_slia, results_dir)
    standardize_benchmark_names(props_slia)
    dim_benchmarks_slia = get_benchmarks_from_props(props_slia)

    dimensions_dict = {"benchmark": dim_benchmarks_lia + dim_benchmarks_slia,
                       "benchmark_lia": dim_benchmarks_lia,
                       "benchmark_slia": dim_benchmarks_slia,
                       "testsRatio": dim_testsRatio,
                       "testsRatio_0.0": Dim(dim_testsRatio[0]),
                       "testsRatio_0.25": Dim(dim_testsRatio[1]),
                       "testsRatio_0.5": Dim(dim_testsRatio[2]),
                       "testsRatio_0.75": Dim(dim_testsRatio[3]),
                       "testsRatio_1.0": Dim(dim_testsRatio[4]),
                       "evoMode": dim_evoMode,
                       "evoMode_g": Dim(dim_evoMode[0]),
                       "evoMode_s": Dim(dim_evoMode[1]),
                       "selection": dim_sel,
                       "selection_tour": Dim(dim_sel[0]),
                       "selection_lex": Dim(dim_sel[1]),
                       "method": dim_methodEUSolver + dim_methodCVC4,
                       "method_eusolver": dim_methodEUSolver,
                       "method_cvc4_1.8": dim_methodCVC4,
                       "method_CDGP": dim_methodCDGP,
                       "method_GPR": dim_methodGPR}

    dim_methodsMine = dimensions_dict["method_CDGP"] * dimensions_dict["evoMode_g"] * dimensions_dict["selection_lex"] * dimensions_dict["testsRatio_0.25"] + \
                      dimensions_dict["method_CDGP"] * dimensions_dict["evoMode_s"] * dimensions_dict["selection_tour"] * dimensions_dict["testsRatio_1.0"] + \
                      dimensions_dict["method_GPR"] * dimensions_dict["evoMode_g"] * dimensions_dict["selection_lex"] * dimensions_dict["testsRatio_0.25"]

    props_lia_mine = load_correct_props_simple(folders_lia_mine, dim_filter=dim_methodsMine)
    props_lia_mine = dim_methodsMine.filter_props(props_lia_mine)
    standardize_benchmark_names(props_lia_mine)

    props_slia_mine = load_correct_props_simple(folders_slia_mine, dim_filter=dim_methodsMine)
    props_slia_mine = dim_methodsMine.filter_props(props_slia_mine)
    standardize_benchmark_names(props_slia_mine)

    props_lia.extend(props_lia_mine)
    props_slia.extend(props_slia_mine)

    subs = [
        (create_subsection_formal, [props_lia, props_slia, "Custom tables", "SLIA", dimensions_dict, results_dir, None]),
    ]
    sects = [(title, desc, subs, [])]
    templates.prepare_report(sects, "cdgp_{0}.tex".format(name), dir_path=results_dir, paperwidth=190, user_declarations=user_declarations)





if __name__ == "__main__":
    # reports_e0_paramTests()
    reports_e0_lia()
    reports_e0_slia()
    # reports_e0_formal()
