import os
import numpy as np
from . import reporting
from . import printer
from .dims import *
import pandas as pd


def create_section(title, desc, subsects, figures_list):
    assert isinstance(title, str)
    assert isinstance(desc, str)
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


def prepare_report(sects, filename, dir_path="results/", paperwidth=75, user_declarations=""):
    """Creates a LaTeX report of the results, where properties are shared for all subsections and
    dimension for rows (e.g. benchmarks) is also the same for each subsection.

    :param sects: ((title, desc, subsections, figures)), where figures are paths to images. Subsections
    are specified as pairs (function, arguments), where function is supposed to return reporting.Subsection.
    :param filename: (str) name of the LaTeX and PDF files to be generated.
    :param dir_path: (str) path to a folder in which PDF with filename name will be created.
    :param paperwidth: (float) width of the page.
    :param user_declarations: (str) user LaTeX code to be placed in the preamble of the document.
    :return: None
    """
    report = reporting.ReportPDF(geometry_params="[paperwidth={0}cm, paperheight=40cm, margin=0.3cm]".format(paperwidth),
                                 packages=["pbox", "makecell"], user_declarations=user_declarations)
    latex_sects = []
    for title, desc, subsections, figures in sects:
        # subsects = []
        # for fun, args in subsections:
        #     subsects.append(fun(*args))
        subsects = [fun(*args) for fun, args in subsections]

        s = create_section(title, desc, subsects, figures)
        latex_sects.append(s)

    for s in latex_sects:
        if s is not None:
            report.add(s)
    print("\n\nGenerating PDF report ...")
    cwd = os.getcwd()
    os.chdir(dir_path)
    report.save_and_compile(filename)
    os.chdir(cwd)


class TableGenerator:
    """Generates table from data. kwargs will be propagated to the table printing."""
    def __init__(self, f_cell, dim_rows, dim_cols, headerRowNames=None, title="", color_scheme=None,
                 table_postprocessor=None, cellRenderers=None, vertical_border=1, variants=None,
                 default_color_thresholds=None, layered_headline=True, color_value_extractor=None,
                 only_nonempty_rows=True, outputFiles=None, **kwargs):
        assert outputFiles is None or isinstance(outputFiles, list), "outputFiles must be either None, or a list of configs"
        self.f_cell = f_cell
        self.dim_rows = dim_rows
        self.dim_cols = dim_cols
        self.title = title
        if color_scheme is None:
            color_scheme = reporting.color_scheme_gray_dark
        self.color_scheme = color_scheme
        self.table_postprocessor = table_postprocessor
        self.cellRenderers = cellRenderers
        self.vertical_border = vertical_border
        if headerRowNames is None:
            headerRowNames = [""]
        self.headerRowNames = headerRowNames
        # create a table for each variant and put them next to each other
        self.variants = variants
        self.default_color_thresholds = default_color_thresholds
        self.layered_headline = layered_headline
        self.color_value_extractor = color_value_extractor
        self.only_nonempty_rows = only_nonempty_rows
        self.outputFiles = outputFiles
        self.init_kwargs = kwargs.copy()

    def __call__(self, props):
        return self.apply(props)

    def apply(self, props):
        """Returns a content of the subsection as a LaTeX formatted string."""
        tables = self.apply_listed(props)
        text = ""
        for t in tables:
            text += r"\noindent"
            text += t
        return text


    def apply_listed(self, props):
        """The same as apply, but returned is a list of tables."""
        if self.variants is None:
            dummy_variant = lambda p: True
            return [self.getTableText(props, dummy_variant)]
        else:
            return [self.getTableText(props, variant) for variant in self.variants]


    def generateTable(self, props, dim_rows=None, dim_cols=None):
        """Generates a table object for the given props and dimensions."""
        if dim_rows is None:
            dim_rows = self.dim_rows
        if dim_cols is None:
            dim_cols = self.dim_cols
        cells = printer.generateTableCells(props, dim_rows, dim_cols, self.f_cell)
        table = printer.Table(cells, dimRows=dim_rows, dimCols=dim_cols,
                              cellRenderers=self.cellRenderers,
                              layeredHeadline=self.layered_headline,
                              verticalBorder=self.vertical_border,
                              headerRowNames=self.headerRowNames)
        return table


    def generateTableForVariant(self, props, variant):
        """Generates a table for a given variant using dimensions defined for that table."""
        if isinstance(variant, ConfigList):
            # if a variant is specified as a ConfigList then it has a caption, and it will be put in the table
            # header so that it is clear for which variant the table was produced.
            dim_cols_to_be_used = Dim([variant]) * self.dim_cols
        else:
            # print("Warning: variant caption will not be generated.")
            dim_cols_to_be_used = self.dim_cols

        props_variant = [p for p in props if variant(p)]
        if self.only_nonempty_rows:
            dim_rows_variant = Dim([c for c in self.dim_rows.configs if len(c.filter_props(props_variant)) > 0])
        else:
            dim_rows_variant = self.dim_rows
        return self.generateTable(props_variant, dim_rows_variant, dim_cols_to_be_used)


    def getTableText(self, props, variant):
        table = self.generateTableForVariant(props, variant)
        txt = table.render()

        if self.color_scheme is not None and self.default_color_thresholds is not None:
            cv0, cv1, cv2 = self.default_color_thresholds
            txt = printer.table_color_map(txt, cv0, cv1, cv2, "colorLow", "colorMedium", "colorHigh",
                                          funValueExtractor=self.color_value_extractor)

        txt = self.table_postprocessor(txt)
        return txt



class FriedmannTestKK:
    def __init__(self, dimRows, dimCols, fun, title="", color_scheme="", showRanks=True, showRawOutput=False):
        self.dimRows = dimRows
        self.dimCols = dimCols
        self.fun = fun
        self.title = title
        self.color_scheme = color_scheme
        self.showRanks = showRanks
        self.showRawOutput = showRawOutput

    def getFriedmanData(self, props):
        """Runs R script to obtain FriedmanData."""
        tableContent = printer.generateTableContent(props, dimRows=self.dimRows, dimCols=self.dimCols, fun=self.fun)
        table = printer.Table(tableContent)
        from src.stats import friedman
        return friedman.runFriedmanKK(table)

    def getSignificantPairsTable(self, friedmanData):
        text = r"\begin{tabular}{lcl}" + "\n"
        pairs = friedmanData.getSignificantPairs()
        for L, R in pairs:
            text += "{0} & $>$ & {1} \\\\\n".format(L, R)
        text += r"\end{tabular}" + "\n"
        return text

    def apply(self, props, **kwargs):
        """Returns a content of the subsection as a LaTeX formatted string."""
        friedmanData = self.getFriedmanData(props)

        text = r"\textbf{p-value:} " + str(friedmanData.p_value) + "\n\n"

        text += r"\noindent \textbf{Significant pairs:} "
        if friedmanData.cmp_matrix is not None:
            text += r"\\" + "\n"
            text += self.getSignificantPairsTable(friedmanData) + r"\\\\"
            # text += r"\vspace{0.5cm}" + "\n"
            posthoc_method = friedmanData.cmp_method if friedmanData.cmp_method is not None else "not specified"
            text += r"\noindent \textbf{Post-hoc method:} " + posthoc_method + "\n\n"
        else:
            text += "None"  + "\n\n"

        if self.showRanks and friedmanData.ranks is not None:
            text += r"\vspace{0.5cm}" + "\n"
            text += r"\noindent \textbf{Ranks:}" + "\n\n" + r"\medskip" + "\n"
            text += friedmanData.ranks.to_latex(index=False, escape=False) + r"\\\\"

        if self.showRawOutput:
            text += r"\vspace{0.5cm}" + "\n"
            text += r"\noindent \textbf{Raw output from R script:}\\" + "\n"
            text += reporting.BlockEnvironment("verbatim", [friedmanData.output]).getText({}) + r"\\\\"
        return text

    def apply_listed(self, props, **kwargs):
        return [self.apply(props, **kwargs)]



class FriedmannTestPython:
    def __init__(self, dimRows, dimCols, fun, title="", p_treshold=0.05, color_scheme="", showRanks=True, variants=None,
                 higherValuesBetter=True, printOnlyOverDiagonal=True, pathFriedmanViz=None, workingDir=None):
        self.dimRows = dimRows
        self.dimCols = dimCols
        self.fun = fun
        self.title = title
        self.p_treshold = p_treshold
        self.color_scheme = color_scheme
        self.showRanks = showRanks
        self.variants = variants
        self.variants_to_be_used = self.variants if self.variants is not None else [lambda p: True]
        self.outputFiles = None
        self.higherValuesBetter = higherValuesBetter
        self.printOnlyOverDiagonal = printOnlyOverDiagonal
        self.pathFriedmanViz = pathFriedmanViz
        if workingDir is not None and workingDir.endswith("/"):
            workingDir = workingDir[:-1]
        self.workingDir = workingDir


    def getFriedmanData(self, props):
        """Runs R script to obtain FriedmanData."""
        tableContent = printer.generateTableContent(props, dimRows=self.dimRows, dimCols=self.dimCols, fun=self.fun)
        table = printer.Table(tableContent)
        from src.stats import friedman
        return friedman.runFriedmanPython(table)

    def getGraphvizFigure(self, pairs, collapse_nodes=True):
        """Returns a text of the Graphviz file illustrating the results of the test.
        :param friedmanData: results of the Friedman test.
        :param pairs: (tuple[int, int]) a pairs (L, R), where L is the index of the significantly better configuration than R.
        :param collapse_nodes: (bool) if true, then configurations dominated by all Ls will be put into a single block.
        :return: (str) text of GraphViz script.
        """
        def getName(i):
            return self.dimCols[i].get_caption()

        dominance_dict = {}
        for L, R in pairs:
            if L not in dominance_dict:
                dominance_dict[L] = [R]
            else:
                dominance_dict[L].append(R)

        def get_text_edges():
            def text_edges_v3_helper(dominance_dict_L_to_R):
                if len(dominance_dict_L_to_R) == 0:
                    return []
                all_L = set()
                all_R = set()
                for L in dominance_dict:
                    for R in dominance_dict[L]:
                        all_L.add(L)
                        all_R.add(R)

                R_dominated_all = []
                R_not_dominated_all = []
                for R in all_R:
                    if all([R in dominance_dict_L_to_R[L] for L in dominance_dict_L_to_R]):
                        R_dominated_all.append(R)
                    else:
                        R_not_dominated_all.append(R)
                assert len(R_not_dominated_all) + len(R_dominated_all) == len(all_R)

                Ls_dominating_only_this = [L for L in dominance_dict_L_to_R if set(dominance_dict_L_to_R[L]) == set(R_dominated_all)]

                new_dominance_dict_L_to_R = dominance_dict_L_to_R.copy()
                for L in Ls_dominating_only_this:
                    del new_dominance_dict_L_to_R[L]
                for L in new_dominance_dict_L_to_R:
                    for R in R_dominated_all:
                        if R in new_dominance_dict_L_to_R[L]:
                            new_dominance_dict_L_to_R[L].remove(R)

                level_tuple = "\\n".join([getName(L) for L in Ls_dominating_only_this]), "\\n".join([getName(R) for R in R_dominated_all])
                result = [level_tuple]
                result.extend(text_edges_v3_helper(new_dominance_dict_L_to_R))
                return result

            levels = text_edges_v3_helper(dominance_dict)
            text_edges = ""
            prev_levels = []
            for i, level in enumerate(levels):
                nodeL, nodeR = level
                if i == 0:
                    text_edges += '\t"{0}" [fontcolor=red]\n'.format(nodeR)
                if i == len(levels)-1:
                    text_edges += '\t"{0}" [fontcolor=blue]\n'.format(nodeL)
                text_edges += '\t"{0}" -> "{1}"\n'.format(nodeR, nodeL)
                for pL, pR in prev_levels:
                    text_edges += '\t"{0}" -> "{1}"\n'.format(pR, nodeL)
                prev_levels.append((nodeL, nodeR))
            return text_edges


        text_edges= get_text_edges()

        text  = "digraph {\n"
        # text += "\tlayout=\"circo\"\n"
        text += "\trankdir=TB\n"  #LR
        text += "\tsplines=\"line\"\n"
        text += "\tK=1.0\n"  # can be used to push nodes apart
        text += "\tnode [shape=plaintext]\n"
        text += text_edges
        text += "}"
        return text

    def getName(self, i):
        """Returns a name of the config with index i."""
        return self.dimCols[i].get_caption()

    def getRank(self, i, avgRanks):
        """Returns a rank of the config with index i."""
        name = self.getName(i)
        for n, rank in avgRanks:
            if n == name:
                return rank
        return None

    def getSortedSignificantPairs(self, friedmanData, avgRanks):
        """Returns a sorted list of (L,R) pairs, where L is significantly better than R."""
        getRank = lambda i: self.getRank(i, avgRanks)
        pairs = friedmanData.getSignificantPairs()
        pairs = [(L, R) for (L, R) in pairs if L < R]  # remove redundant pairs (cmp_matrix is symmetric)
        pairs = list(map(lambda p: (p[1], p[0]) if getRank(p[0]) > getRank(p[1]) else (p[0], p[1]),
                         pairs))  # make the elements with lower rank go first
        # pairs = [(R, L) for (L, R) in pairs if getRank(L) > getRank(R)]  # make the elements with lower rank go first
        pairs.sort(key=lambda x: (getRank(x[0]), getRank(x[1])))  # start with the lowest ranks
        return pairs


    def getSignificantPairsTable(self, friedmanData, avgRanks, pairs=None):
        getName = lambda i: self.getName(i)
        getRank = lambda i: self.getRank(i, avgRanks)
        if pairs is None:
            pairs = self.getSortedSignificantPairs(friedmanData, avgRanks)

        past_L = None
        text = r"\begin{tabular}{lllclll|l}" + "\n"
        for L, R in pairs:
            L_caption = getName(L)
            L_rank = round(getRank(L), 2)
            if past_L is None or past_L != L_caption:
                past_L = L_caption
                L_caption = r"\textcolor{blue}{" + L_caption + "}"
            R_caption = getName(R)
            R_rank = round(getRank(R), 2)
            sign = ">" if L_rank < R_rank else "<"
            text += "{0}~~&({1})&~(\\textbf{{rank}}={2}) & ${7}$ & {3}~~&({4})&~(\\textbf{{rank}}={5})  & \\textbf{{p-value}}={6} \\\\\n".\
                format(L, L_caption, L_rank, R, R_caption, R_rank, friedmanData.cmp_matrix.iat[L, R], sign)
        text += r"\end{tabular}" + "\n"
        return text


    def apply(self, props, **kwargs):
        """Returns a content of the subsection as a LaTeX formatted string."""
        friedmanData = self.getFriedmanData(props)
        avgRanks = getSortedAveragedRanks(props, self.dimRows, self.dimCols, self.fun, higherValuesBetter=self.higherValuesBetter)

        def entry_formatter_lambda(allSolutions, entryIndex):
            entry = allSolutions[entryIndex]
            value = round(float(entry[1]), 2)
            nameFormatter = lambda x: r"\textcolor{darkblue}{" + str(x) + "}"  # if "CDGP" in str(x) else x
            color = printer.getLatexColorCode(value,
                                              [allSolutions[0][1], (allSolutions[-1][1] + allSolutions[0][1]) / 2.0,
                                               allSolutions[-1][1]],
                                              ["darkgreen", "orange", "darkred!50!white"])
            return "{0}  ({1})".format(nameFormatter(entry[0]),
                                       r"\textbf{\textcolor{" + color + "}{" + str(value) + "}}")

        rankingFun = rankingFunctionGenerator(lambda props: avgRanks, entry_formatter_lambda, ONLY_VISIBLE_SOLS=True,
                                              NUM_SHOWN=100, usePbox=False)
        avgRanksText = rankingFun(props)

        text = r"\noindent \textbf{Average ranks:}\\" + "\n" + avgRanksText + "\\\\\n\n"

        text += r"\noindent \textbf{p-value:} " + str(friedmanData.p_value) + "\n\n"

        text += r"\noindent \textbf{Significant pairs:} "
        if friedmanData.cmp_matrix is not None:
            pairs = self.getSortedSignificantPairs(friedmanData, avgRanks)
            text += r"\\" + "\n"
            def fBold(v):
                if pd.isnull(v):
                    return ""
                else:
                    return r"\textbf{" + ("%1.3f" % v) + "}" if v < self.p_treshold else "%1.3f" % v
            cmp_matrix = friedmanData.cmp_matrix.copy()
            # Print only elements over the diagonal
            if self.printOnlyOverDiagonal:
                for i in range(cmp_matrix.shape[0]):
                    for j in range(cmp_matrix.shape[1]):
                        if i >= j:
                            cmp_matrix.at[i, j] = None
            text += cmp_matrix.to_latex(escape=False, formatters=[fBold] * friedmanData.cmp_matrix.shape[1], na_rep="")
            text += "\n\n" + r"\vspace{0.5cm}" + "\n"
            text += r"\noindent " + self.getSignificantPairsTable(friedmanData, avgRanks, pairs=pairs) + r"\\\\"
            if self.pathFriedmanViz is not None:
                assert self.workingDir is not None, "In order to save a graphviz file a working directory needs to be specified"
                utils.save_to_file(self.workingDir + "/" + self.pathFriedmanViz, self.getGraphvizFigure(pairs, collapse_nodes=True))
                utils.compile_graphviz_file_to_pdf(self.workingDir + "/" + self.pathFriedmanViz)
                pdf_name = self.pathFriedmanViz[:self.pathFriedmanViz.rfind(".")] + ".pdf"
                text += r"\includegraphics{" + pdf_name + "}"
            text += r"\\" + "\n"
            posthoc_method = friedmanData.cmp_method if friedmanData.cmp_method is not None else "not specified"
            text += r"\noindent \textbf{Post-hoc method:} " + posthoc_method + "\n\n"
        else:
            text += "None"  + "\n\n"

        if self.showRanks and friedmanData.ranks is not None:
            text += r"\vspace{0.5cm}" + "\n"
            text += r"\noindent \textbf{Ranks:}" + "\n\n" + r"\medskip" + "\n"
            text += friedmanData.ranks.to_latex(index=False, escape=False) + r"\\\\"

        return text

    def apply_listed(self, props, **kwargs):
        return [self.apply(props, **kwargs)]



def rankingFunctionGenerator(sorted_list_lambda, entry_formatter_lambda, ONLY_VISIBLE_SOLS=True, NUM_SHOWN=15, usePbox=True):
    """Returns a function for producing a ranking of data produced from props and sorted on some property.

    :param sorted_list_lambda: a function which takes as arguments: props. It returns a sorted list of elements, usually tuples.
    :param entry_formatter_lambda: a function which takes as arguments: allSolutions, entryIndex. allSolutions is a list
     returned by sorted_list_lambda. entryIndex is the index of the currently processed element in allSolutions. entry_formatter_lambda
     should return a string ready to be placed on a given line in the ranking.
    """
    def fun(props):
        """Returns a ranking of shortest solutions (both their size and error are printed)."""
        if len(props) == 0:
            return "-"

        solutions = sorted_list_lambda(props)
        if solutions is None:
            return "n/a"

        if ONLY_VISIBLE_SOLS:
            # drop solutions which won't be shown so that color scale is adjusted to what is visible
            solutions = solutions[:min(NUM_SHOWN, len(solutions))]


        # For some strange reason makecell doesn't work, even when it is a suggested answer (https://tex.stackexchange.com/questions/2441/how-to-add-a-forced-line-break-inside-a-table-cell)
        # return "\\makecell{" + "{0}  ({1})\\\\{2}  ({3})".format(counterex_items[0][0], counterex_items[0][1],
        #                                        counterex_items[1][0], counterex_items[1][1]) + "}"
        if usePbox:
            res = r"\pbox[l][" + str(15 * min(NUM_SHOWN, len(solutions))) + r"pt][c]{15cm}{\footnotesize "  #scriptsize, footnotesize
        else:
            res = r"{\footnotesize "
        for i in range(NUM_SHOWN):
            if i >= len(solutions):
                break
            if i > 0:
                res += "\\\\ \\ "
            res += entry_formatter_lambda(solutions, i)
        res += "}"
        return res
    return fun


def getSortedAveragedRanks(props, dim_ranks_trials, dim_ranking, value_getter, higherValuesBetter=True):
    """Produces a list of config names and their average ranks.

    :param props: (list[dict]) list of dictionaries.
    :param dim_ranks_trials: (Dim) a set of "tests" on which ranks will be computed and then averaged (e.g., benchmarks).
    :param dim_ranking: (Dim) a ranking will be created for configs in this dimension (e.g., method x selection x testsRatio).
    :param value_getter: (lambda) a function from a list of dictionaries to the value.
    :param higherValuesBetter: (bool) if True, then higher values are treated as better.
    :return: (list[(str,float)]) a list of tuples, containing config name and its average rank.
    """
    assert isinstance(dim_ranking, Dim)
    assert isinstance(dim_ranks_trials, Dim)
    allRanks = {}  # for each config name contains a list of its ranks
    for config_trial in dim_ranks_trials:
        props_trial = config_trial.filter_props(props)

        valuesList = []
        for config in dim_ranking:
            name = config.get_caption()
            if name not in allRanks:
                allRanks[name] = []
            props2 = config.filter_props(props_trial)
            if len(props2) > 0:
                valuesList.append((name, value_getter(props2)))

        valuesList.sort(key=lambda x: (x[1], x[0]), reverse=higherValuesBetter) # True

        # "If there are tied values, assign to each tied value the average of
        #  the ranks that would have been assigned without ties."
        import scipy.stats as ss
        # In[19]: ss.rankdata([3, 1, 4, 15, 92])
        # Out[19]: array([2., 1., 3., 4., 5.])
        #
        # In[20]: ss.rankdata([1, 2, 3, 3, 3, 4, 5])
        # Out[20]: array([1., 2., 4., 4., 4., 6., 7.])
        if higherValuesBetter:
            ranks = ss.rankdata([-x[1] for x in valuesList])
        else:
            ranks = ss.rankdata([x[1] for x in valuesList])
        for (r, (name, value)) in zip(ranks, valuesList):
            allRanks[name].append(r)

        # The code below incorrectly handles ties
        # for i, (name, value) in enumerate(valuesList):
        #     allRanks[name].append(i + 1)  # 'i' is incremented so that the first element has rank 1

    # Remove from allRanks all algorithms with empty list of ranks
    allRanks = {k:allRanks[k] for k in allRanks if len(allRanks[k]) > 0}

    # Here we should have a dictionary containing lists of ranks
    valuesList = [(name, np.mean(ranks)) for (name, ranks) in allRanks.items()]
    valuesList.sort(key=lambda x: (x[1], x[0]), reverse=False)
    return valuesList