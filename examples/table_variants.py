from src import printer
from src import reporting
from src.dims import Dim
from src.dims import Config


tableBody = r"""
    P0 (3)    &   0.70 &   0.54 &    0.34 &   0.42 &   0.88 &  0.94 &  1.00 &  1.00 &  0.58 &  0.66 &     0.40 &  0.58 &  0.72 &  0.82 &  1.00 &  1.00 &  0.72 \\
    P1 (3)    &   0.18 &   0.16 &    0.26 &   0.24 &   0.24 &  0.20 &  0.54 &  0.58 &  0.16 &  0.08 &     0.20 &  0.12 &  0.60 &  0.44 &  0.96 &  0.96 &  0.37 \\
    P2 (2)    &   1.00 &   1.00 &    1.00 &   1.00 &   1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &     1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 \\
    P3 (4)    &   0.14 &   0.16 &    0.12 &   0.12 &   0.46 &  0.48 &  1.00 &  0.96 &  0.52 &  0.62 &     0.28 &  0.54 &  0.82 &  0.76 &  1.00 &  1.00 &  0.56 \\
    P4 (5)    &   0.14 &   0.06 &    0.02 &   0.08 &   0.02 &  0.02 &  0.00 &  0.00 &  0.52 &  0.56 &     0.38 &  0.44 &  0.38 &  0.18 &  0.14 &  0.14 &  0.19 \\
    P5 (2)    &   1.00 &   1.00 &    1.00 &   1.00 &   1.00 &  1.00 &  1.00 &  1.00 &  0.98 &  0.92 &     1.00 &  0.98 &  1.00 &  0.98 &  1.00 &  1.00 &  0.99 \\
    P6 (4)    &   0.08 &   0.08 &    0.06 &   0.14 &   0.02 &  0.14 &  0.04 &  0.04 &  0.40 &  0.60 &     0.82 &  0.68 &  0.68 &  0.74 &  0.78 &  0.80 &  0.38 \\
    P7 (3)    &   0.16 &   0.08 &    0.34 &   0.16 &   0.34 &  0.44 &  0.56 &  0.58 &  1.00 &  1.00 &     1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  0.67 \\
    P8 (4)    &   0.18 &   0.36 &    0.10 &   0.12 &   0.14 &  0.18 &  0.28 &  0.32 &  0.36 &  0.46 &     0.26 &  0.30 &  0.50 &  0.34 &  0.82 &  0.76 &  0.34 \\
    mean   &   0.40 &   0.38 &    0.36 &   0.36 &   0.46 &  0.49 &  0.60 &  0.61 &  0.61 &  0.66 &     0.59 &  0.63 &  0.74 &  0.70 &  0.86 &  0.85 &   - \\
    rank   &  10.72 &  10.94 &   12.00 &  11.33 &  10.67 &  9.61 &  8.28 &  8.06 &  8.50 &  8.22 &     8.17 &  8.78 &  5.39 &  7.00 &  4.17 &  4.17 &   - \\
    """


def generateTable(verticalBorder, horizontalBorder, useBooktabs):
    print("Generating a table for table_variants.tex ..")
    dimCx = Dim([("0.0", None), ("0.5", None)])
    dimSel = Dim([("T", None), ("L", None)])
    dimMethod = Dim([("U", None), ("P", None), ("S", None), ("IS", None)])
    main = dimSel * dimMethod * dimCx
    dimCols = Dim([Config([("method", None), ("", None), ("cx", None)])]) + main + Dim([("mean", None)])

    rBold = printer.LatexTextbf(lambda v, b: v == "1.00")
    rShading = printer.CellShading(0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    table = printer.Table(tableBody, dimCols=dimCols, cellRenderers=[rBold, rShading], verticalBorder=verticalBorder,
                          horizontalBorder=horizontalBorder, useBooktabs=useBooktabs)
    return table


def generateTableRemovedCols(verticalBorder, horizontalBorder, useBooktabs):
    print("Generating a table for table_variants_removedCols.tex ..")
    dimCx = Dim([("0.0", None), ("0.5", None)])
    dimSel = Dim([("T", None), ("L", None)])
    dimMethod = Dim([("U", None), ("P", None), ("S", None), ("IS", None)])
    main = dimSel * dimMethod * dimCx
    dimCols = Dim([Config([("method", None), ("", None), ("cx", None)])]) + main + Dim([("mean", None)])

    rBold = printer.LatexTextbf(lambda v, b: v == "1.00")
    rShading = printer.CellShading(0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    table = printer.Table(tableBody, dimCols=dimCols, cellRenderers=[rBold, rShading], verticalBorder=verticalBorder,
                          horizontalBorder=horizontalBorder, useBooktabs=useBooktabs)

    table.leaveColumns([0, 1, 3, 7, 9, 11, 15])  # leaving out S and C
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



report = generateReport(generateTable)
report.save_and_compile("table_variants.tex")

report = generateReport(generateTableRemovedCols)
report.save_and_compile("table_variants_removedCols.tex")
