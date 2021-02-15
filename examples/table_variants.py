from evoplotter import printer
from evoplotter import reporting
from evoplotter.dims import *


tableBody = r"""
    P0 (3)    &   0.70 &   0.54 &    0.34 &   0.42 &   0.88 &  0.94 &  1.00 &  1.00 &  0.58 &  0.66 &     0.40 &  0.58 &  0.72 &  0.82 &  1.00 &  1.00 &    0.00 &  0.00 &  0.72 \\
    P1 (3)    &   0.18 &   0.16 &    0.26 &   0.24 &   0.24 &  0.20 &  0.54 &  0.58 &  0.16 &  0.08 &     0.20 &  0.12 &  0.60 &  0.44 &  0.96 &  0.96 &    0.00 &  0.00 &  0.37 \\
    P2 (2)    &   1.00 &   1.00 &    1.00 &   1.00 &   1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &     1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &    0.00 &  0.00 &  1.00 \\
    P3 (4)    &   0.14 &   0.16 &    0.12 &   0.12 &   0.46 &  0.48 &  1.00 &  0.96 &  0.52 &  0.62 &     0.28 &  0.54 &  0.82 &  0.76 &  1.00 &  1.00 &    0.00 &  0.00 &  0.56 \\
    P4 (5)    &   0.14 &   0.06 &    0.02 &   0.08 &   0.02 &  0.02 &  0.00 &  0.00 &  0.52 &  0.56 &     0.38 &  0.44 &  0.38 &  0.18 &  0.14 &  0.14 &    0.00 &  0.00 &  0.19 \\
    P5 (2)    &   1.00 &   1.00 &    1.00 &   1.00 &   1.00 &  1.00 &  1.00 &  1.00 &  0.98 &  0.92 &     1.00 &  0.98 &  1.00 &  0.98 &  1.00 &  1.00 &    0.00 &  0.00 &  0.99 \\
    P6 (4)    &   0.08 &   0.08 &    0.06 &   0.14 &   0.02 &  0.14 &  0.04 &  0.04 &  0.40 &  0.60 &     0.82 &  0.68 &  0.68 &  0.74 &  0.78 &  0.80 &    0.00 &  0.00 &  0.38 \\
    P7 (3)    &   0.16 &   0.08 &    0.34 &   0.16 &   0.34 &  0.44 &  0.56 &  0.58 &  1.00 &  1.00 &     1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &    0.00 &  0.00 &  0.67 \\
    P8 (4)    &   0.18 &   0.36 &    0.10 &   0.12 &   0.14 &  0.18 &  0.28 &  0.32 &  0.36 &  0.46 &     0.26 &  0.30 &  0.50 &  0.34 &  0.82 &  0.76 &    0.00 &  0.00 &  0.34 \\
    mean      &   0.40 &   0.38 &    0.36 &   0.36 &   0.46 &  0.49 &  0.60 &  0.61 &  0.61 &  0.66 &     0.59 &  0.63 &  0.74 &  0.70 &  0.86 &  0.85 &    0.00 &  0.00 &   - \\
    rank      &  10.72 &  10.94 &   12.00 &  11.33 &  10.67 &  9.61 &  8.28 &  8.06 &  8.50 &  8.22 &     8.17 &  8.78 &  5.39 &  7.00 &  4.17 &  4.17 &    0.00 &  0.00 &   - \\
    """


def generateTableText(verticalBorder, horizontalBorder, useBooktabs):
    print("Generating a table for table_variants_text.tex ..")
    dimCx = Dim([("0.0", None), ("0.5", None)])
    dimSel = Dim([("T", None), ("L", None)])
    dimMethod = Dim([("U", None), ("P", None), ("S", None), ("IS", None)])
    dimSingleCol = Dim([("C", None), ("D", None)])
    main = dimSel * dimMethod * dimCx + dimSingleCol * dim_all * dim_all
    dimCols = main + Dim([("mean", None)])
    cells, rows_names = printer.latexToArrayRowNames(tableBody)  # maybe switch for pandas as a primary representation?
    dimRows = Dim.from_names(rows_names)

    rBold = printer.LatexTextbf(lambda v, b: v == "1.00")
    rShading = printer.CellShading(0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    table = printer.Table(cells, dimRows=dimRows, dimCols=dimCols, cellRenderers=[rBold, rShading], verticalBorder=verticalBorder,
                          horizontalBorder=horizontalBorder, useBooktabs=useBooktabs, headerRowNames=["method", "", "cx"])
    return table


def generateTableTextRemovedCols(verticalBorder, horizontalBorder, useBooktabs):
    print("Generating a table for table_variants_text_rc.tex ..")
    dimCx = Dim([("0.0", None), ("0.5", None)])
    dimSel = Dim([("T", None), ("L", None)])
    dimMethod = Dim([("U", None), ("P", None), ("S", None), ("IS", None)])
    dimSingleCol = Dim([("C", None), ("D", None)])
    main = dimSel * dimMethod * dimCx + dimSingleCol
    dimCols = main + Dim([("mean", None)])
    cells, rows_names = printer.latexToArrayRowNames(tableBody)  # maybe switch for pandas as a primary representation?
    dimRows = Dim.from_names(rows_names)

    rBold = printer.LatexTextbf(lambda v, b: v == "1.00")
    rShading = printer.CellShading(0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    table = printer.Table(cells, dimRows=dimRows, dimCols=dimCols, cellRenderers=[rBold, rShading], verticalBorder=verticalBorder,
                          horizontalBorder=horizontalBorder, useBooktabs=useBooktabs, headerRowNames=["method", "", "cx"])

    table.leaveColumns([0, 2, 6, 8, 10, 14])  # leaving out S and C
    return table



data = [
    {"A": 0, "B": 1, "C": 0, "D": 4, "E": 0, "F": 0, "value": 2},
    {"A": 1, "B": 1, "C": 0, "D": 4, "E": 0, "F": 0, "value": 4},
    {"A": 0, "B": 0, "C": 0, "D": 4, "E": 0, "F": 0, "value": 0},
    {"A": 0, "B": 0, "C": 0, "D": 4, "E": 0, "F": 0, "value": 3},
    {"A": 0, "B": 0, "C": 1, "D": 4, "E": 0, "F": 0, "value": 1},
    {"A": 0, "B": 0, "C": 1, "D": 4, "E": 0, "F": 0, "value": 9},
    {"A": 0, "B": 1, "C": 1, "D": 4, "E": 0, "F": 1, "value": 1},
    {"A": 0, "B": 1, "C": 1, "D": 4, "E": 0, "F": 1, "value": 3},
    {"A": 0, "B": 1, "C": 0, "D": 4, "E": 0, "F": 1, "value": 0},
    {"A": 0, "B": 1, "C": 0, "D": 4, "E": 0, "F": 1, "value": 4},
    {"A": 1, "B": 1, "C": 0, "D": 4, "E": 0, "F": 2, "value": 5},
    {"A": 1, "B": 1, "C": 1, "D": 4, "E": 0, "F": 2, "value": 7},
    {"A": 1, "B": 0, "C": 1, "D": 4, "E": 0, "F": 2, "value": 1},
    {"A": 1, "B": 0, "C": 1, "D": 4, "E": 0, "F": 2, "value": 2},
    {"A": 1, "B": 0, "C": 0, "D": 4, "E": 0, "F": 3, "value": 2},
    {"A": 1, "B": 0, "C": 0, "D": 4, "E": 0, "F": 3, "value": 4},
    {"A": 1, "B": 0, "C": 0, "D": 4, "E": 0, "F": 3, "value": 2},
    {"A": 1, "B": 0, "C": 0, "D": 4, "E": 0, "F": 3, "value": 4},
]


def generateTableData(verticalBorder, horizontalBorder, useBooktabs):
    print("Generating a table for table_variants_data.tex ..")

    dimCols = Dim.from_dict(data, "A", nameFun=lambda v: "A={0}".format(v)) * Dim.from_dict(data, "B", nameFun=lambda v: "B={0}".format(v))
    dimRows = Dim.from_dict(data, "F", nameFun=lambda v: "F={0}".format(v))

    cells = printer.generateTableCells(data, dimRows=dimRows, dimCols=dimCols, fun=lambda props: sum([p["value"] for p in props]))

    rShading = printer.CellShading(0.0, 5.0, 10.0, "colorLow", "colorMedium", "colorHigh")
    table = printer.Table(cells, dimCols=dimCols, dimRows=dimRows, cellRenderers=[rShading], verticalBorder=verticalBorder,
                          horizontalBorder=horizontalBorder, useBooktabs=useBooktabs, headerRowNames=["A-value"])
    return table



def generateReport(tableGenerator):
    report = reporting.ReportPDF()
    report.add(reporting.color_scheme_gray_light.toBlockLatex())

    verticalBorder_list = [0, 1, 2]
    horizontalBorder_list = [0, 1, 2]
    useBooktabs_list = [False, True]

    for ub in useBooktabs_list:
        sec1 = reporting.SectionRelative("useBooktabs={0}".format(ub))
        report.add(sec1)
        # report.add(reporting.BlockLatex(r"\bigskip"))
        for hb in horizontalBorder_list:
            sec2 = reporting.SectionRelative("horizontalBorder={0}".format(hb))
            sec1.add(sec2)
            report.add(reporting.BlockLatex(r"\bigskip\bigskip"))
            for vb in verticalBorder_list:
                subsec = reporting.SectionRelative("verticalBorder={2}".format(ub, hb, vb))
                subsec.add(tableGenerator(verticalBorder=vb, horizontalBorder=hb, useBooktabs=ub))
                subsec.add(reporting.BlockLatex(r"\bigskip"))
                sec2.add(subsec)
    return report



report = generateReport(generateTableText)
report.save_and_compile("table_variants_text.tex")

report = generateReport(generateTableTextRemovedCols)
report.save_and_compile("table_variants_text_rc.tex")

report = generateReport(generateTableData)
report.save_and_compile("table_variants_data.tex")