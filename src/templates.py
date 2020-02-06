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


def prepare_report(props, sects, filename, paperwidth=75, user_declarations=""):
    """Creates a LaTeX report of the results, where properties are shared for all subsections and
    dimension for rows (e.g. benchmarks) is also the same for each subsection.

    :param props: (list[dict]) list of properties dicts to be processed.
    :param sects: ((title, desc, subsections, figures)), where figures are paths to images. Subsections
    are specified as pairs (function, arguments), where function is supposed to return reporting.Subsection.
    :param filename: (str) name of the LaTeX and PDF files to be generated.
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
    os.chdir("results/")
    report.save_and_compile(filename)
    os.chdir(cwd)


class TableGenerator:
    """Generates table from data. kwargs will be propagated to the table printing."""
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

    def __call__(self, props, new_color_thresholds=None):
        return self.apply(props, new_color_thresholds)

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