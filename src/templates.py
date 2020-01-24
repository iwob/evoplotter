import os
from . import reporting


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
