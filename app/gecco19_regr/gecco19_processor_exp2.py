from src import utils
from src import plotter
from src import printer
from src import reporting
from src.dims import *
from gecco19_utils import *


def simplify_benchmark_name(name):
    """Shortens or modifies the path of the benchmark in order to make the table more readable."""
    i = name.rfind("/")
    name = name if i == -1 else name[i+1:]
    return name.replace("_3", "_03").replace("_5", "_05")



def p_method_for(name):
    return lambda p, name=name: p["method"] == name
def p_matches_dict(p, d):
    for k, v in d.items():
        if p[k] != v:
            return False
    return True
def p_method_for_dict(d):
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



dim_true = Dim(Config("All", lambda p: True, method=None))
# dim_methodCDGP = Dim([Config("CDGP", p_method_for("CDGP"), method="CDGP")])
# dim_methodGP = Dim([Config("GP", p_method_for("GP"), method="GP")])
dim_methodCDGP = Dim([Config("CDGP", p_method_for_dict({"method":"CDGP","partialConstraintsInFitness":"false"}), method="CDGP"),
                      Config("$CDGP_{props}$", p_method_for_dict({"method":"CDGP","partialConstraintsInFitness":"true"}), method="CDGPprops"),])
dim_methodGP = Dim([Config("$GP_{500}$", p_method_for_dict({"method":"GP","populationSize":"500"}), method="GP500"),
                    Config("$GP_{1000}$", p_method_for_dict({"method": "GP", "populationSize": "1000"}), method="GP1000"),
                    Config("$GP_{5000}$", p_method_for_dict({"method": "GP", "populationSize": "5000"}), method="GP5000"),])
dim_method = dim_methodCDGP + dim_methodGP
dim_sel = Dim([#Config("$Tour$", p_sel_tourn, selection="tournament"),
               Config("$Lex$", p_sel_lexicase, selection="lexicase")])
# dim_evoMode = Dim([Config("$steadyState$", p_steadyState, evolutionMode="steadyState"),
#                    Config("$generational$", p_generational, evolutionMode="generational")])
dim_evoMode = Dim([Config("$generational$", p_generational, evolutionMode="generational")])
# dim_testsRatio = Dim([Config("$0.8$", p_testsRatio_equalTo("0.8"), testsRatio="0.8"),
#                       Config("$1.0$", p_testsRatio_equalTo("1.0"), testsRatio="1.0")])
dim_testsRatio = Dim([Config("$1.0$", p_testsRatio_equalTo("1.0"), testsRatio="1.0")])
# dim_sa = Dim([Config("$CDGP$", p_GP),
# 			    Config("$CDGP^{ss}$", p_steadyState),
#               Config("$CDGP_{lex}$", p_lexicase),
#               Config("$CDGP_{lex}^{ss}$", p_LexicaseSteadyState)])





def plot_figures(props, exp_prefix):
    # We want to consider CDGP only
    props = [p for p in props]
    if len(props) == 0:
        print("No props: plots were not generated.")
        return

    getter_mse = lambda p: float(p["result.best.mse"])
    predicate = lambda v, v_xaxis: v <= v_xaxis
    N = 50  # number of points per plot line
    r = (0.0, 1e0)
    xs = np.linspace(r[0], r[1], N)
    xticks = np.arange(r[0], r[1], r[1] / 10)
    plotter.plot_ratio_meeting_predicate(props, getter_mse, predicate, xs=xs, xticks=xticks,
                                         title="Ratio of solutions with MSE under the certain level",
                                         xlabel="MSE",
                                         series_dim=dim_method,
                                         xlogscale=False,
                                         savepath="results/figures/ratioMSE.pdf".format(exp_prefix))


    # print_solved_in_time(props, 12 * 3600 * 1000)
    # print_solved_in_time(props, 6 * 3600 * 1000)
    # print_solved_in_time(props, 3 * 3600 * 1000)
    # print_solved_in_time(props, 1 * 3600 * 1000)
    # print_solved_in_time(props, 0.5 * 3600 * 1000)
    # print_solved_in_time(props, 0.25 * 3600 * 1000)
    # print_solved_in_time(props, 0.125 * 3600 * 1000)
    # print_solved_in_time(props, 600 * 1000)

    # Plot chart of number of found solutions in time
    # success_props = [p for p in props if is_optimal_solution(p)]
    # getter = lambda p: float(normalized_total_time(p)) / (60 * 1000)  # take minutes as a unit
    # predicate = lambda v, v_xaxis: v <= v_xaxis
    # xs = np.arange(0.0, 5.0 * 60.5 + 1e-9, 5.0) # a point every 5.0 minutes
    # xticks = np.arange(0.0, 5.0 * 60.0 + 1e-9, 15.0) # a tick every 15 minutes
    # plotter.plot_ratio_meeting_predicate(success_props, getter, predicate,
    #                                      xs=xs, xticks=xticks, show_plot=0,
    #                                      series_dim=dim_method, # "series_dim=None" for a single line
    #                                      savepath="figures/{0}_ratioTime_correctVsAllCorrect.pdf".format(exp_prefix),
    #                                      title="Ratio of found correct solutions out of all correct solutions",
    #                                      xlabel="Runtime [minutes]")
    # plotter.plot_ratio_meeting_predicate(props, getter, predicate,
    #                                      xs=xs, xticks=xticks, show_plot=0,
    #                                      series_dim=dim_method,
    #                                      savepath="figures/{0}_ratioTime_endedVsAllEnded.pdf".format(exp_prefix),
    #                                      title="Ratio of ended runs",
    #                                      xlabel="Runtime [minutes]")


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
        if isinstance(cs, reporting.ColorScheme3):
            cs = cs.toBlockLatex()
        sub = reporting.SectionRelative(title, contents=[cs, reporting.BlockLatex(table + "\n"), vspace])
        content.append(sub)
    return content

def post(s):
    return s.replace("{ccccccccccccc}", "{rrrrrrrrrrrrr}").replace("{rrr", "{lrr").replace(r"\_{lex}", "_{lex}").replace(r"\_{", "_{")



def create_section_and_plots(title, desc, props, subsects, figures_list, exp_prefix):
    assert isinstance(title, str)
    assert isinstance(desc, str)
    assert isinstance(props, list)
    assert isinstance(figures_list, list)

    plot_figures(props, exp_prefix=exp_prefix)

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

    print("SUCCESS RATES (mse below thresh (1.0e-25))")
    print(printer.text_table(props, dim_rows, dim_cols, fun_successRateMseOnly, d_cols=";"))
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, fun_successRateMseOnly, layered_headline=True, vertical_border=vb))
    latex_successRatesMseOnly = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("SUCCESS RATES (mse below thresh (1.0e-25) + properties met)")
    print(printer.text_table(props, dim_rows, dim_cols, fun_successRate, d_cols=";"))
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, fun_successRate, layered_headline=True, vertical_border=vb))
    latex_successRates = printer.table_color_map(text, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")

    print("SUCCESS RATES (properties met)")
    print(printer.text_table(props, dim_rows, dim_cols, fun_allPropertiesMet, d_cols=";"))
    text = post(
        printer.latex_table(props, dim_rows, dim_cols, fun_allPropertiesMet, layered_headline=True, vertical_border=vb))
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
        ("Status (correctly finished runs)", latex_status, reversed(reporting.color_scheme_red)),
        ("Success rates (mse below thresh (1.0e-25))", latex_successRatesMseOnly, reporting.color_scheme_teal),
        ("Success rates (mse below thresh (1.0e-25) + properties met)", latex_successRates, reporting.color_scheme_green),
        ("Success rates (properties met)", latex_propertiesMet, reporting.color_scheme_green),
        ("Average runtime [s]", latex_avgRuntime, reporting.color_scheme_violet),
        ("Average runtime (only successful) [s]", latex_avgRuntimeOnlySuccessful, reporting.color_scheme_violet),
        # ("Average sizes of best of runs (number of nodes)", latex_sizes, reporting.color_scheme_yellow),
        ("Average sizes of best of runs (number of nodes) (only successful)", latex_sizesOnlySuccessful,
         reporting.color_scheme_yellow),
    ]
    return reporting.Subsection("Shared Statistics", get_content_of_subsections(subsects_main))



def create_subsection_cdgp_specific(props, dim_rows, dim_cols):
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



def get_benchmarks_from_props(props, simple_names=True):
    dim_benchmarks = Dim.from_dict(props, "benchmark")
    if simple_names:
        configs = [Config(simplify_benchmark_name(c.get_caption()), c.filters[0][1],
                          benchmark=c.get_caption()) for c in
                   dim_benchmarks.configs]
        dim_benchmarks = Dim(configs)
        dim_benchmarks.sort()
    return dim_benchmarks


_prev_props = None
def prepare_report(sects, fname, exp_prefix, simple_bench_names=True, print_status_matrix=True, reuse_props=False,
                   paperwidth=75, include_all_row=True, dim_cols_listings=None):
    """Creating nice LaTeX report of the results."""
    global _prev_props  # used in case reuse_props was set to True
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


        print("\nFiltered Info:")
        for p in props:
            if "result.best.verificationDecision" not in p:
                print("file: {0}".format(p["thisFileName"]))


        # Automatically detect benchmarks used
        dim_benchmarks = get_benchmarks_from_props(props, simple_names=simple_bench_names)


        if print_status_matrix:
            d = dim_benchmarks * dim_methodGP * dim_sel * dim_evoMode +\
                dim_benchmarks * dim_methodCDGP * dim_sel * dim_evoMode * dim_testsRatio

            matrix = produce_status_matrix(d, props)
            print("\n****** Status matrix:")
            print(matrix + "\n")
            print("Saving status matrix to file: {0}".format(STATUS_FILE_NAME))
            save_to_file(STATUS_FILE_NAME, matrix)


        dim_rows = dim_benchmarks.sort()
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

        s = create_section_and_plots(title, desc, props, subsects, figures, exp_prefix)
        latex_sects.append(s)

    for s in latex_sects:
        if s is not None:
            report.add(s)
    print("\n\nGenerating PDF report ...")
    cwd = os.getcwd()
    os.chdir("results/")
    report.save_and_compile(fname)
    os.chdir(cwd)



def reports_exp2():
    folders = ["exp2_physics3_run1", "exp2_physics3_run2", "exp2_physics3_partial", "exp2_physics5"]
    title = "Experiments for regression CDGP (stop: 0.5h)"
    desc = r""""""
    dimColsCdgp = dim_methodCDGP * dim_evoMode * dim_testsRatio + dim_methodGP * dim_evoMode
    dimColsShared = dimColsCdgp
    subs = [
        (create_subsection_shared_stats, [None, dimColsShared, 20]),
        (create_subsection_cdgp_specific, [None, dimColsCdgp]),
    ]
    figures = [
        "figures/ratioMSE.pdf"
        # "figures/e0_ratioEvaluated_correctVsAllRuns.pdf",
        # "figures/e0_ratioTime_correctVsAllCorrect.pdf",
        # "figures/e0_ratioTime_endedVsAllEnded.pdf"
    ]
    sects = [(title, desc, folders, subs, figures)]

    prepare_report(sects, "cdgp_r_exp2.tex", "e2", paperwidth=40, include_all_row=True, dim_cols_listings=dimColsShared)



if __name__ == "__main__":
    ensure_clear_dir("results/")
    ensure_dir("results/figures/")
    ensure_dir("results/listings/")
    ensure_dir("results/listings/errors/")

    reports_exp2()