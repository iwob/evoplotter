from src import utils
from src import plotter
from src import printer
from src import reporting
from src.dims import *


tableRaw = r"""
P0 & 0.5 & 0.4 & 1.0 & 0.1 \\
P1 & 0.2 & 0.4 & 0.5 & 0.9 \\
P2 & 0.8 & 0.1 & 0.8 & 0.3 \\
P3 & 0.9 & 0.0 & 0.2 & 0.3 \\
"""

table = printer.Table(printer.latexToArray(tableRaw))
table.removeColumn(1)
print(table.renderLatex())
print(table.renderCsv())

table = printer.Table(printer.latexToArray(tableRaw),
                      dimCols=Dim([("A", None), ("B", None)]) * Dim([("c", None), ("d", None)]),
                      dimRows=Dim(["P0", "P1", "P2", "P3"]))
table.removeColumn(1)
print(table.renderLatex())
print(table.renderCsv())

table.insertColumn(1, ["0.0", "0.0", "0.0", "0.0"])
print(table.renderLatex())
print(table.renderCsv())


tableRaw2 = r"""
0.5 & 0.4 & 1.0 & 0.1 \\
0.2 & 0.4 & 0.5 & 0.9 \\
0.8 & 0.1 & 0.8 & 0.3 \\
0.9 & 0.0 & 0.2 & 0.3 \\
"""

table = printer.Table(printer.latexToArray(tableRaw2),
                      dimCols=Dim([("A", None), ("B", None)]) * Dim([("c", None), ("d", None)]),
                      dimRows=Dim(["P0", "P1", "P2", "P3"]))
print(table.renderLatex())
print(table.renderCsv())
