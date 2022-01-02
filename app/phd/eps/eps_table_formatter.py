import os
import math
from app.phd.cdgp.phd_utils import *
from evoplotter import utils
from evoplotter import plotter
from evoplotter import printer
from evoplotter import templates
from evoplotter.templates import *








def testRegex():
    def process(text):
        return re.sub(r"\\[^{}\\]+\{([^{}]+)\}", r"\g<1>", text)
    def process2(text):
        text = re.sub(r"\\[^{}\\]+\{.*\}\s*([0-9.-]+|\\[^{}\\]+\{([^{}]+)\})", r"\g<1>", text)  # handle case when \command{}VALUE or \command{} \command2{VALUE}
        return re.sub(r"\\[^{}\\]+\{([0-9.-]+)\}", r"\g<1>", text)  # handle case when VALUE or \command2{VALUE}

    print(process(r"-2.00"))
    print(process(r"\textbf{0.00}"))
    print(process(r"\cellcolor{colorLow!2.1799999999999953!colorMedium}-978.2"))
    print(process(r"\cellcolor{colorLow!2.1799999999999953!colorMedium}\textbf{978.2}"))

    print()
    print(process2(r"-2.00"))
    print(process2(r"\textbf{0.00}"))
    print(process2(r"\cellcolor{colorLow!2.1799999999999953!colorMedium}-978.2"))
    print(process2(r"\cellcolor{colorLow!2.1799999999999953!colorMedium}\textbf{978.2}"))
    print(process2(r"\cellcolor{colorLow!2.1799999999999953!colorMedium} \textbf{978.2}"))
    print(process2(r"\benchNameEps{Koza1-p-2D}"))



def log10ValueExtractor(s):
    base = 10.0
    if not utils.isfloat(s):
        return s
    else:
        return math.log(float(s), base)


if __name__ == "__main__":
    # testRegex()

    text_runtime = r"""
    \benchNameEps{Keijzer12} & \cellcolor{colorLow!98.52!colorMedium}\textbf{14.8} & \cellcolor{colorHigh!100.0!colorMedium}11330.7 & \cellcolor{colorLow!50.7!colorMedium}493.0 & & \cellcolor{colorLow!22.770000000000003!colorMedium}772.3 & \cellcolor{colorLow!51.2!colorMedium}488.0 & \cellcolor{colorHigh!6.428888888888888!colorMedium}1578.6 & \cellcolor{colorHigh!100.0!colorMedium}15439.8 & \cellcolor{colorHigh!100.0!colorMedium}21172.6 & \cellcolor{colorHigh!100.0!colorMedium}28354.0\\
    \benchNameEps{Koza1} & \cellcolor{colorLow!99.52!colorMedium}\textbf{4.8} & \cellcolor{colorLow!70.9!colorMedium}291.0 & \cellcolor{colorLow!95.37!colorMedium}46.3 & & \cellcolor{colorLow!30.020000000000003!colorMedium}699.8 & - & \cellcolor{colorLow!19.860000000000003!colorMedium}801.4 & \cellcolor{colorLow!34.8!colorMedium}652.0 & - & \cellcolor{colorLow!30.420000000000005!colorMedium}695.8\\
    \benchNameEps{Koza1-p} & \cellcolor{colorLow!99.55!colorMedium}\textbf{4.5} & \cellcolor{colorLow!3.710000000000002!colorMedium}962.9 & \cellcolor{colorLow!65.6!colorMedium}344.0 & & \cellcolor{colorLow!10.770000000000003!colorMedium}892.3 & - & \cellcolor{colorLow!2.8399999999999976!colorMedium}971.6 & \cellcolor{colorLow!2.1799999999999953!colorMedium}978.2 & - & \cellcolor{colorLow!1.8!colorMedium}982.0\\
    \benchNameEps{Koza1-2D} & \cellcolor{colorLow!98.4!colorMedium}\textbf{16.0} & \cellcolor{colorHigh!73.7311111111111!colorMedium}7635.8 & \cellcolor{colorLow!56.81!colorMedium}431.9 & & \cellcolor{colorLow!20.689999999999998!colorMedium}793.1 & \cellcolor{colorLow!52.129999999999995!colorMedium}478.7 & \cellcolor{colorHigh!8.784444444444443!colorMedium}1790.6 & \cellcolor{colorHigh!89.74333333333334!colorMedium}9076.9 & \cellcolor{colorHigh!100.0!colorMedium}16280.5 & \cellcolor{colorHigh!100.0!colorMedium}23033.8\\
    \benchNameEps{Koza1-p-2D} & \cellcolor{colorLow!98.46!colorMedium}\textbf{15.4} & \cellcolor{colorHigh!91.17888888888889!colorMedium}9206.1 & \cellcolor{colorLow!48.41!colorMedium}515.9 & & \cellcolor{colorLow!24.960000000000004!colorMedium}750.4 & \cellcolor{colorLow!48.87!colorMedium}511.3 & \cellcolor{colorHigh!8.063333333333333!colorMedium}1725.7 & \cellcolor{colorHigh!100.0!colorMedium}11986.4 & \cellcolor{colorHigh!100.0!colorMedium}12390.8 & \cellcolor{colorHigh!100.0!colorMedium}27875.4\\
"""

    text_fitness = r"""
    \benchNameEps{Keijzer12} & \cellcolor{colorLow!36.6!colorMedium}15.85 & \cellcolor{colorLow!7.920000000000003!colorMedium}23.02 & \cellcolor{colorHigh!0.23999999999999488!colorMedium}25.06 & & \cellcolor{colorLow!4.319999999999993!colorMedium}23.92 & \cellcolor{colorLow!27.799999999999997!colorMedium}18.05 & \cellcolor{colorHigh!11.079999999999998!colorMedium}27.77 & \cellcolor{colorHigh!56.19999999999999!colorMedium}\textbf{39.05} & \cellcolor{colorLow!18.200000000000003!colorMedium}20.45 & \cellcolor{colorLow!30.120000000000005!colorMedium}17.47\\
    \benchNameEps{Koza1} & \cellcolor{colorLow!76.44!colorMedium}5.89 & \cellcolor{colorLow!61.04!colorMedium}9.74 & \cellcolor{colorLow!56.52!colorMedium}10.87 & & \cellcolor{colorLow!60.28!colorMedium}9.93 & - & \cellcolor{colorLow!60.68!colorMedium}9.83 & \cellcolor{colorLow!56.0!colorMedium}\textbf{11.00} & - & \cellcolor{colorLow!56.0!colorMedium}\textbf{11.00}\\
    \benchNameEps{Koza1-p} & \cellcolor{colorLow!89.64!colorMedium}2.59 & \cellcolor{colorLow!82.2!colorMedium}4.45 & \cellcolor{colorLow!84.08!colorMedium}3.98 & & \cellcolor{colorLow!63.8!colorMedium}9.05 & - & \cellcolor{colorLow!64.88!colorMedium}8.78 & \cellcolor{colorLow!56.0!colorMedium}\textbf{11.00} & - & \cellcolor{colorLow!56.0!colorMedium}\textbf{11.00}\\
    \benchNameEps{Koza1-2D} & \cellcolor{colorLow!33.84!colorMedium}16.54 & \cellcolor{colorHigh!18.92!colorMedium}29.73 & \cellcolor{colorHigh!32.72!colorMedium}33.18 & & \cellcolor{colorLow!6.439999999999998!colorMedium}23.39 & \cellcolor{colorLow!22.120000000000005!colorMedium}19.47 & \cellcolor{colorHigh!25.159999999999997!colorMedium}31.29 & \cellcolor{colorHigh!81.68!colorMedium}\textbf{45.42} & \cellcolor{colorHigh!9.439999999999998!colorMedium}27.36 & \cellcolor{colorLow!5.200000000000002!colorMedium}23.70\\
    \benchNameEps{Koza1-p-2D} & \cellcolor{colorLow!62.84!colorMedium}9.29 & \cellcolor{colorLow!31.28!colorMedium}17.18 & \cellcolor{colorLow!41.6!colorMedium}14.60 & & \cellcolor{colorLow!9.599999999999994!colorMedium}22.60 & \cellcolor{colorLow!57.36!colorMedium}10.66 & \cellcolor{colorHigh!17.879999999999995!colorMedium}29.47 & \cellcolor{colorHigh!84.91999999999999!colorMedium}\textbf{46.23} & \cellcolor{colorLow!49.76!colorMedium}12.56 & \cellcolor{colorLow!38.36!colorMedium}15.41\\    
"""

    text_successRate = r"""
	\benchNameEps{Keijzer12} & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!90.0!colorMedium}0.05 & & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!98.0!colorMedium}0.01 & \cellcolor{colorLow!22.0!colorMedium}\textbf{0.39} & \cellcolor{colorLow!98.0!colorMedium}0.01 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
	\benchNameEps{Koza1} & \cellcolor{colorLow!62.0!colorMedium}0.19 & \cellcolor{colorHigh!36.0!colorMedium}0.68 & \cellcolor{colorHigh!92.0!colorMedium}0.96 & & \cellcolor{colorLow!34.0!colorMedium}0.33 & - & \cellcolor{colorLow!36.0!colorMedium}0.32 & \cellcolor{colorHigh!100.0!colorMedium}\textbf{1.00} & - & \cellcolor{colorHigh!100.0!colorMedium}\textbf{1.00}\\
	\benchNameEps{Koza1-p} & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & & \cellcolor{colorLow!90.0!colorMedium}0.05 & - & \cellcolor{colorLow!94.0!colorMedium}0.03 & \cellcolor{colorHigh!100.0!colorMedium}\textbf{1.00} & - & \cellcolor{colorHigh!100.0!colorMedium}\textbf{1.00}\\
	\benchNameEps{Koza1-2D} & \cellcolor{colorLow!98.0!colorMedium}0.01 & \cellcolor{colorLow!76.0!colorMedium}0.12 & \cellcolor{colorLow!60.0!colorMedium}0.20 & & \cellcolor{colorLow!96.0!colorMedium}0.02 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!78.0!colorMedium}0.11 & \cellcolor{colorHigh!60.0!colorMedium}\textbf{0.80} & \cellcolor{colorLow!58.0!colorMedium}0.21 & \cellcolor{colorLow!54.0!colorMedium}0.23\\
	\benchNameEps{Koza1-p-2D} & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!98.0!colorMedium}0.01 & \cellcolor{colorHigh!50.0!colorMedium}\textbf{0.75} & \cellcolor{colorLow!100.0!colorMedium}0.00 & \cellcolor{colorLow!100.0!colorMedium}0.00\\
"""


    cells_runtime = printer.latexToArray(text_runtime, removeRenderings=True)
    table_runtime = printer.Table(cells_runtime, cellRenderers=[printer.LatexTextbfMinInRow(),
                                                                printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")])  #, valueExtractor=log10ValueExtractor
    print("RUNTIME")
    print(table_runtime.renderTableBody())


    cells_fitness = printer.latexToArray(text_fitness, removeRenderings=True)
    table_fitness = printer.Table(cells_fitness, cellRenderers=[printer.LatexTextbfMaxInRow(),
                                                                printer.CellShadingRow("colorLow", "colorMedium", "colorHigh")])  # , valueExtractor=log10ValueExtractor
    print("FITNESS")
    print(table_fitness.renderTableBody())


    cells_successRate = printer.latexToArray(text_successRate, removeRenderings=True)
    table_successRate = printer.Table(cells_successRate, cellRenderers=[printer.LatexTextbfMaxInRow(),
                                                                printer.CellShading(0.0, 0.5, 1.0, "colorLow", "colorMedium", "colorHigh")])  # , valueExtractor=log10ValueExtractor
    print("SUCCESS RATE")
    print(table_successRate.renderTableBody())