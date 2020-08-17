import os
from . import reporting
from . import printer
from .dims import *


def create_section(title, desc, props, subsects, figures_list):
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


def prepare_report(props, sects, filename, dir_path="results/", paperwidth=75, user_declarations=""):
    """Creates a LaTeX report of the results, where properties are shared for all subsections and
    dimension for rows (e.g. benchmarks) is also the same for each subsection.

    :param props: (list[dict]) list of properties dicts to be processed.
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
        subsects = []
        for fun, args in subsections:
            args2 = [props] + args
            subsects.append(fun(*args2))

        s = create_section(title, desc, props, subsects, figures)
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
    def __init__(self, f_cell, dim_rows, dim_cols, headerRowNames, title="", color_scheme=None,
                 table_postprocessor=None, vertical_border=1, table_variants=None,
                 default_color_thresholds=None, layered_headline=True, color_value_extractor=None,
                 only_nonempty_rows=True, outputFiles=None, **kwargs):
        assert outputFiles is None or isinstance(outputFiles, list), "outputFiles must be either None, or a list of configs"
        self.f_cell = f_cell
        self.dim_rows = dim_rows
        self.dim_cols = dim_cols
        self.title = title
        self.color_scheme = color_scheme
        self.table_postprocessor = table_postprocessor
        self.vertical_border = vertical_border
        self.headerRowNames = headerRowNames
        # create a table for each variant and put them next to each other
        self.table_variants = table_variants
        self.default_color_thresholds = default_color_thresholds
        self.layered_headline = layered_headline
        self.color_value_extractor = color_value_extractor
        self.only_nonempty_rows = only_nonempty_rows
        self.outputFiles = outputFiles
        self.init_kwargs = kwargs.copy()

    def __call__(self, props, new_color_thresholds=None):
        return self.apply(props, new_color_thresholds)

    def apply(self, props, new_color_thresholds=None):
        """Returns a content of the subsection as a LaTeX formatted string."""
        tables = self.apply_listed(props, new_color_thresholds)
        text = ""
        for t in tables:
            text += r"\noindent"
            text += t
        return text


    def apply_listed(self, props, new_color_thresholds=None):
        """The same as apply, but returned is a list of tables."""
        tables = []
        variants_to_be_used = self.table_variants if self.table_variants is not None else [lambda p: True]
        for variant in variants_to_be_used:  # each variant is some predicate on data
            if isinstance(variant, ConfigList):
                dim_cols_to_be_used = Dim([(variant.get_caption(), lambda p: True)]) * self.dim_cols
            else:
                dim_cols_to_be_used = self.dim_cols

            props_variant = [p for p in props if variant(p)]
            if self.only_nonempty_rows:
                dim_rows_variant = Dim([c for c in self.dim_rows.configs if len(c.filter_props(props_variant)) > 0])
            else:
                dim_rows_variant = self.dim_rows

            txt = printer.latex_table(props_variant, dim_rows_variant, dim_cols_to_be_used, self.f_cell,
                                      layered_headline=self.layered_headline, vertical_border=self.vertical_border,
                                      headerRowNames=self.headerRowNames, **self.init_kwargs)
            txt = self.table_postprocessor(txt)
            ct = new_color_thresholds if new_color_thresholds is not None else self.default_color_thresholds
            if self.color_scheme is not None and ct is not None:
                cv0, cv1, cv2 = ct
                txt = printer.table_color_map(txt, cv0, cv1, cv2, "colorLow", "colorMedium", "colorHigh", funValueExtractor=self.color_value_extractor)
            tables.append(txt)
        return tables




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
            if friedmanData.cmp_method is not None:
                text += r"\noindent \textbf{Post-hoc method:} " + str(friedmanData.cmp_method) + "\n\n"
            else:
                text += r"\noindent \textbf{Post-hoc method:} " + "not specified" + "\n\n"
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



def rankingFunctionGenerator(sorted_list_lambda, entry_formatter_lambda, ONLY_VISIBLE_SOLS=True, NUM_SHOWN=15):
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
        res = r"\pbox[l][" + str(15 * min(NUM_SHOWN, len(solutions))) + r"pt][c]{15cm}{\footnotesize "  #scriptsize, footnotesize
        for i in range(NUM_SHOWN):
            if i >= len(solutions):
                break
            if i > 0:
                res += "\\\\ \\ "
            res += entry_formatter_lambda(solutions, i)
        res += "}"
        return res
    return fun