from src import utils
from src import plotter
from src import printer
from src import reporting
from src.dims import *



def produceHeader():
    def fun(p):
        return False

    # method	gp	gp	gpprior	gpprior	gps	gps	gpsi	gpsi	lex	lex	lexprior	lexprior
    # 	lexs	lexs	lexsi	lexsi	means
    dim_method = Dim([
        Config("gp", None),
        Config("gpprior", None),
        Config("gps", None),
        Config("gpsi", None),
        Config("lex", None),
        Config("lexprior", None),
        Config("lexs", None),
        Config("lexsi", None),
    ])
    dim_cx = Dim([
        Config("0.0", None),
        Config("0.5", None)
    ])
    dim_means = Dim([Config("means", None)])
    d = dim_method * dim_cx + dim_means
    text = printer.latex_table([], dim_cx, d, fun, layered_headline=True, vertical_border=0)
    print("\nHEADER:")
    print(text)

    print("\nHEADER Linear:")
    print(printer.text_table_header(d))




def processTable(table, title):
    print(title)
    colored = printer.table_color_map(table, 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    print(reporting.color_scheme_green)
    print(colored)


def processTableNewInterface_SvsSI(tableBody, title):
    print(title)
    rBold = printer.LatexTextbf(lambda v, b: v == "1.00")
    rShading = printer.CellShading(0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    table = printer.Table(printer.latexToArray(tableBody), cellRenderers=[rBold, rShading])
    table.leaveColumns([0, 5, 6, 7, 8, 13, 14, 15, 16]) # comparing only SI with S

    colored = printer.table_color_map(str(table), 0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    #print(reporting.color_scheme_green)
    print(colored)
    print(table.renderCsv())


def processTableNewInterface_withoutSC(tableBody, title):
    dimCx = Dim([("0.0", None), ("0.5", None)])
    dimSel = Dim([("T", None), ("L", None)])
    dimMethod = Dim([("U", None), ("P", None), ("S", None), ("IS", None)])
    main = dimSel * dimMethod * dimCx
    dimCols = Dim([("method",None)]) + main + Dim([("mean",None)])
    print("numConfigs: " + str(len(dimCols.configs)))
    print("asd: "+title)
    rBold = printer.LatexTextbf(lambda v, b: v == "1.00")
    rShading = printer.CellShading(0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")
    table = printer.Table(printer.latexToArray(tableBody), dimCols=dimCols, cellRenderers=[rBold, rShading])
    table.leaveColumns([0, 1, 3, 7, 9, 11, 15]) # leaving out S and C
    print(table.render())
    print(table.renderCsv())


def computeAvg(table):
    suma = 0.0
    n = 0.0
    for r in table.rows:
        for c in r:
            n += 1.0
            suma += float(c)
    return suma / n

def printTablesAverages(tableBodySmall, tableBodyLarge, title=""):
    print(title)
    tableSmall = printer.Table(printer.latexToArray(tableBodySmall))
    tableLarge = printer.Table(printer.latexToArray(tableBodyLarge))

    tableSmall.rows = tableSmall.rows[:-2]
    tableLarge.rows = tableLarge.rows[:-2]

    tableSmall.removeColumns([0, 17])
    tableLarge.removeColumns([0, 17])

    print("Avg success rate tableSmall: {0}".format(computeAvg(tableSmall)))
    print("Avg success rate tableLarge: {0}".format(computeAvg(tableLarge)))


def printTablesAveragesCvsN(tableBody, title="", doPrint=True):
    print(title)
    tableC = printer.Table(printer.latexToArray(tableBody))
    tableN = printer.Table(printer.latexToArray(tableBody))

    tableC.rows = tableC.rows[:-2]
    tableN.rows = tableN.rows[:-2]

    tableC.removeColumns([0, 2, 4, 6, 8, 10, 12, 14, 16, 17])
    tableN.removeColumns([0, 1, 3, 5, 7, 9, 11, 13, 15, 17])

    avgC = computeAvg(tableC)
    avgN = computeAvg(tableN)
    if doPrint:
        print("Avg success rate C: {0}".format(avgC))
        print("Avg success rate N: {0}".format(avgN))
    return avgC, avgN


def printTablesAveragesSvsIS(tableBody, title="", doPrint=True):
    print(title)
    tableS = printer.Table(printer.latexToArray(tableBody))
    tableIS = printer.Table(printer.latexToArray(tableBody))

    tableS.rows = tableS.rows[:-2]
    tableIS.rows = tableIS.rows[:-2]

    tableS.leaveColumns([5, 6, 13, 14])
    tableIS.leaveColumns([7, 8, 15, 16])

    avgS = computeAvg(tableS)
    avgIS = computeAvg(tableIS)
    if doPrint:
        print("Avg success rate S: {0}".format(avgS))
        print("Avg success rate IS: {0}".format(avgIS))
    return avgS, avgIS


def printTablesAveragesCvsN_merge(tableSmall, tableLarge, title=""):
    avgSmallC, avgSmallN = printTablesAveragesCvsN(tableSmall, doPrint=False)
    avgLargeC, avgLargeN = printTablesAveragesCvsN(tableLarge, doPrint=False)
    print(title)
    print("Avg success rate C: {0}".format((avgSmallC + avgLargeC) / 2.0))
    print("Avg success rate N: {0}".format((avgSmallN + avgLargeN) / 2.0))


def printTablesAveragesSvsIS_merge(tableSmall, tableLarge, title=""):
    avgSmallS, avgSmallIS = printTablesAveragesSvsIS(tableSmall, doPrint=False)
    avgLargeS, avgLargeIS = printTablesAveragesSvsIS(tableLarge, doPrint=False)
    print(title)
    print("Avg success rate S: {0}".format((avgSmallS + avgLargeS) / 2.0))
    print("Avg success rate IS: {0}".format((avgSmallIS + avgLargeIS) / 2.0))



tableSmall = r"""
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

tableLarge = r"""
P0 (3)    &   0.70 &   0.54 &    0.34 &   0.38 &   0.82 &  0.78 &  1.00 &  1.00 &  0.58 &  0.66 &     0.54 &  0.58 &  0.64 &  0.68 &  1.00 &  1.00 &  0.70 \\
P1 (3)    &   0.18 &   0.16 &    0.20 &   0.20 &   0.18 &  0.24 &  0.58 &  0.62 &  0.16 &  0.08 &     0.16 &  0.16 &  0.48 &  0.32 &  0.98 &  0.88 &  0.35 \\
P2 (2)    &   1.00 &   1.00 &    1.00 &   1.00 &   1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &     1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 \\
P3 (4)    &   0.14 &   0.16 &    0.10 &   0.10 &   0.12 &  0.28 &  0.68 &  0.74 &  0.52 &  0.62 &     0.46 &  0.52 &  0.60 &  0.74 &  0.98 &  0.94 &  0.48 \\
P4 (5)    &   0.14 &   0.06 &    0.02 &   0.00 &   0.02 &  0.02 &  0.00 &  0.00 &  0.52 &  0.56 &     0.52 &  0.50 &  0.22 &  0.32 &  0.32 &  0.08 &  0.21 \\
P5 (2)    &   1.00 &   1.00 &    1.00 &   1.00 &   1.00 &  1.00 &  1.00 &  1.00 &  0.98 &  0.92 &     0.98 &  0.96 &  1.00 &  1.00 &  1.00 &  1.00 &  0.99 \\
P6 (4)    &   0.08 &   0.08 &    0.00 &   0.04 &   0.08 &  0.10 &  0.12 &  0.12 &  0.40 &  0.60 &     0.64 &  0.70 &  0.64 &  0.70 &  0.72 &  0.74 &  0.36 \\
P7 (3)    &   0.16 &   0.08 &    0.28 &   0.24 &   0.48 &  0.42 &  0.78 &  0.90 &  1.00 &  1.00 &     0.98 &  1.00 &  1.00 &  1.00 &  1.00 &  1.00 &  0.71 \\
P8 (4)    &   0.18 &   0.36 &    0.16 &   0.08 &   0.18 &  0.24 &  0.42 &  0.42 &  0.36 &  0.46 &     0.32 &  0.34 &  0.28 &  0.32 &  0.84 &  0.78 &  0.36 \\
mean   &   0.40 &   0.38 &    0.34 &   0.34 &   0.43 &  0.45 &  0.62 &  0.64 &  0.61 &  0.66 &     0.62 &  0.64 &  0.65 &  0.68 &  0.87 &  0.82 &   - \\
rank   &  10.56 &  11.06 &   12.33 &  12.67 &  10.44 &  9.56 &  7.28 &  6.89 &  8.50 &  7.83 &     9.39 &  8.50 &  7.17 &  6.11 &  3.56 &  4.17 &   - \\
"""


processTableNewInterface_SvsSI(tableSmall, "PROCESSED (NEW) tableSmall:\n")
processTableNewInterface_SvsSI(tableLarge, "PROCESSED (NEW) tableLarge:\n")


processTableNewInterface_withoutSC(tableSmall, "PROCESSED (NEW) withoutSC tableSmall:\n")
processTableNewInterface_withoutSC(tableLarge, "PROCESSED (NEW) withoutSC tableLarge:\n")

printTablesAverages(tableSmall, tableLarge)

printTablesAveragesCvsN_merge(tableSmall, tableLarge, title="printTablesAveragesCvsN")
printTablesAveragesSvsIS_merge(tableSmall, tableLarge, title="printTablesAveragesSvsIS")

# printTablesAveragesCvsN(tableSmall, title="printTablesAveragesCvsN: tableSmall")
# printTablesAveragesCvsN(tableLarge, title="printTablesAveragesCvsN: tableLarge")

# processTable(tableSmall, "PROCESSED tableSmall:\n")
# processTable(tableLarge, "PROCESSED tableLarge:\n")




produceHeader()






def generateRanksVertical(label, data):
    text = r"\begin{tabular}{lr}" + "\n"
    text += r"\hline" + "\n"
    text += "{0} & {1}\\\\".format("method", label) + "\n"
    text += r"\hline" + "\n"
    values = sorted(zip(data.keys(), data.values()), key=lambda tup: tup[1])
    # print(values)
    for k, v in values:
        val = "{0:.3f}".format(v)
        text += "{0} & {1}\\\\".format(k, val) + "\n"
    text += r"\hline" + "\n"
    text += r"\end{tabular}"
    # print("\n\nFRIEDMAN for: " + label)
    print(text)
    print(r"%")

def generateRanksHorizontal(label, data):
    text = "\n" + label + "\n"
    text += r"\begin{tabular}{" + "l" + ("r" * len(data))  + "}\n"
    text += r"\hline" + "\n"
    values = sorted(zip(data.keys(), data.values()), key=lambda tup: tup[1])
    # print(values)
    rowMethods = "method"
    rowRanks = "rank"
    for k, v in values:
        val = "{0:.2f}".format(round(v, 2))
        rowMethods += " & {0}".format(k)
        rowRanks += " & {0}".format(val)
    rowMethods += r"\\" + "\n"
    rowRanks += r"\\" + "\n"
    text += rowMethods
    text += rowRanks
    # print("\n\nFRIEDMAN for: " + label)
    text += r"%\hline" + "\n"
    text += r"\end{tabular}"
    print(text)
    print(r"\smallskip")


def generate(label, data):
    # generateRanksVertical(label, data)
    generateRanksHorizontal(label, data)


ranksLarge0 = {"\lexsi": 2.056, "\lexs": 3.611, "\gpsi":3.722 , "\lex":4.444,
               "\lexprior": 4.833, "\gps":5.444, "\gp":5.500, "\gpprior":6.389}

ranksLarge5 = {"\lexsi": 2.222, "\lexs": 3.500, "\gpsi":3.833 , "\lex":4.333,
               "\lexprior": 4.556, "\gps":5.111, "\gp":5.833, "\gpprior":6.611}

ranksSmall0 = {"\lexsi": 2.500, "\lexs": 3.056, "\gpsi":4.278 , "\lex":4.556,
               "\lexprior": 4.278, "\gps":5.500, "\gp":5.667, "\gpprior":6.167}

ranksSmall5 = {"\lexsi": 2.167, "\lexs": 3.611, "\gpsi":4.333 , "\lex":4.333,
               "\lexprior": 4.722, "\gps":5.222, "\gp":5.722, "\gpprior":5.889}


print("\n\n")
generate("small$_{0.0}$", ranksSmall0)
generate("small$_{0.5}$", ranksSmall5)
generate("large$_{0.0}$", ranksLarge0)
generate("large$_{0.5}$", ranksLarge5)







