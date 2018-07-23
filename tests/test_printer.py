import unittest
from src.dims import *
from src import printer

class TestsPrinter(unittest.TestCase):
    x = "x"
    r = "r"
    c = "c"
    data = [{r: 0, c: 0, x: 1},
            {r: 0, c: 0, x: 2},
            {r: 0, c: 0, x: 3},
            {r: 0, c: 1, x: 4},
            {r: 0, c: 1, x: 5},
            {r: 0, c: 1, x: 6},
            {r: 0, c: 2, x: 7},
            {r: 0, c: 2, x: 8},
            {r: 0, c: 2, x: 9},
            {r: 1, c: 0, x: 11},
            {r: 1, c: 0, x: 12},
            {r: 1, c: 0, x: 13},
            {r: 1, c: 1, x: 14},
            {r: 1, c: 1, x: 15},
            {r: 1, c: 1, x: 16},
            {r: 1, c: 2, x: 17},
            {r: 1, c: 2, x: 18},
            {r: 1, c: 2, x: 19}]
    dim_rows = Dim([Config("r0", lambda d: d["r"] == 0),
                    Config("r1", lambda d: d["r"] == 1)])
    dim_cols = Dim([Config("c0", lambda d: d["c"] == 0),
                    Config("c1", lambda d: d["c"] == 1),
                    Config("c2", lambda d: d["c"] == 2)])
    dim_z = Dim([Config("z0", lambda d: d["x"] < 5 ),
                 Config("z1", lambda d: d["x"] >= 5)])

    def test_text_table_multichar_seps(self):
        text = printer.text_table(self.data, self.dim_rows, self.dim_cols, lambda ds: sum([d["x"] for d in ds]), d_cols=" && ", d_rows=";\n")
        self.assertEqual(" && c0 && c1 && c2;\n" + "r0 && 6 && 15 && 24;\n" + "r1 && 36 && 45 && 54;\n", text)


    def test_text_table(self):
        text = printer.text_table(self.data, self.dim_rows, self.dim_cols, lambda ds: sum([d["x"] for d in ds]))
        self.assertEqual("\tc0\tc1\tc2\n" + "r0\t6\t15\t24\n"  + "r1\t36\t45\t54\n", text)

        text = printer.text_table(self.data, self.dim_cols, self.dim_rows, lambda ds: sum([d["x"] for d in ds]))
        self.assertEqual("\tr0\tr1\n" + "c0\t6\t36\n" + "c1\t15\t45\n" + "c2\t24\t54\n", text)


    def test_latex_table_vb0(self):
        text = printer.latex_table(self.data, self.dim_rows, self.dim_cols, lambda ds: sum([d["x"] for d in ds]),
                                   vertical_border=0)
        text = self.clear_multicols(text)
        self.assertEqual(r"\begin{tabular}{lccc}" + "\n"
                         r"\hline" + "\n" +
                         r" & c0 & c1 & c2\\" + "\n" +
                         r"\hline" + "\n" +
                         r"r0 & 6 & 15 & 24\\"+"\n" + r"r1 & 36 & 45 & 54\\" + "\n" +
                         r"\hline" + "\n" +
                         r"\end{tabular}" + "\n", text)

    def test_latex_table_vb1(self):
        text = printer.latex_table(self.data, self.dim_rows, self.dim_cols, lambda ds: sum([d["x"] for d in ds]),
                                   vertical_border=1, first_col_align="c")
        text = self.clear_multicols(text)
        self.assertEqual(r"\begin{tabular}{|c|ccc|}" + "\n"
                         r"\hline" + "\n" +
                         r" & c0 & c1 & c2\\" + "\n" +
                         r"\hline" + "\n" +
                         r"r0 & 6 & 15 & 24\\"+"\n" + r"r1 & 36 & 45 & 54\\" + "\n" +
                         r"\hline" + "\n" +
                         r"\end{tabular}" + "\n", text)

    def test_latex_table_vb2(self):
        text = printer.latex_table(self.data, self.dim_rows, self.dim_cols, lambda ds: sum([d["x"] for d in ds]),
                                   vertical_border=2, first_col_align="r")
        text = self.clear_multicols(text)
        self.assertEqual(r"\begin{tabular}{|r|c|c|c|}" + "\n"
                         r"\hline" + "\n" +
                         r" & c0 & c1 & c2\\" + "\n" +
                         r"\hline" + "\n" +
                         r"r0 & 6 & 15 & 24\\"+"\n" + r"r1 & 36 & 45 & 54\\" + "\n" +
                         r"\hline" + "\n" +
                         r"\end{tabular}" + "\n", text)

    def clear_multicols(self, s):
        """Replaces \multicolumn markers and places only the content."""
        return s.replace("\multicolumn{1}{c}", "").replace("{c0}", "c0").replace("{c1}", "c1").replace("{c2}", "c2")

    def test_latex_table_header_multilayered_1(self):
        text = printer.latex_table_header_multilayered(self.dim_cols)
        text = self.clear_multicols(text)
        self.assertEqual(r" & c0 & c1 & c2\\" + "\n" + r"\hline" + "\n", text)


    def test_latex_table_header_multilayered_2(self):
        dim = self.dim_rows * self.dim_cols
        text = printer.latex_table_header_multilayered(dim)
        text = self.clear_multicols(text)
        self.assertEqual(r" & \multicolumn{3}{c}{r0} & \multicolumn{3}{c}{r1}\\" + "\n" +
                         r" & c0 & c1 & c2 & c0 & c1 & c2\\" + "\n" + r"\hline" + "\n", text)

        dim = self.dim_rows * self.dim_cols
        dim = Dim(dim.configs[:-1])
        text = printer.latex_table_header_multilayered(dim)
        text = self.clear_multicols(text)
        self.assertEqual(r" & \multicolumn{3}{c}{r0} & \multicolumn{2}{c}{r1}\\" + "\n" +
                         r" & c0 & c1 & c2 & c0 & c1\\" + "\n" + r"\hline" + "\n", text)


    def test_latex_table_header_multilayered_3(self):
        dim = self.dim_z * self.dim_rows * self.dim_cols
        text = printer.latex_table_header_multilayered(dim)
        text = self.clear_multicols(text)
        self.assertEqual(r" & \multicolumn{6}{c}{z0} & \multicolumn{6}{c}{z1}\\" + "\n" +
                         r" & \multicolumn{3}{c}{r0} & \multicolumn{3}{c}{r1} & \multicolumn{3}{c}{r0} & \multicolumn{3}{c}{r1}\\" + "\n" +
                         r" & c0 & c1 & c2 & c0 & c1 & c2 & c0 & c1 & c2 & c0 & c1 & c2\\" + "\n" + r"\hline" + "\n", text)

        dim = self.dim_z * self.dim_rows * self.dim_cols
        dim = Dim(dim.configs[:-1])
        text = printer.latex_table_header_multilayered(dim)
        text = self.clear_multicols(text)
        self.assertEqual(r" & \multicolumn{6}{c}{z0} & \multicolumn{5}{c}{z1}\\" + "\n" +
                         r" & \multicolumn{3}{c}{r0} & \multicolumn{3}{c}{r1} & \multicolumn{3}{c}{r0} & \multicolumn{2}{c}{r1}\\" + "\n" +
                         r" & c0 & c1 & c2 & c0 & c1 & c2 & c0 & c1 & c2 & c0 & c1\\" + "\n" + r"\hline" + "\n", text)


    def test_text_listing(self):
        dim = self.dim_rows * self.dim_cols
        text = printer.text_listing(self.data, dim, lambda ds: sum([d["x"] for d in ds]), d_configs="\n\n")
        self.assertEqual("(*) CONFIG: r0_c0\n6\n\n" + "(*) CONFIG: r0_c1\n15\n\n" + "(*) CONFIG: r0_c2\n24\n\n" +
                         "(*) CONFIG: r1_c0\n36\n\n" + "(*) CONFIG: r1_c1\n45\n\n" + "(*) CONFIG: r1_c2\n54\n\n", text)


    def test_decorate(self):
        text = r"""0 & 5 & 10\\
10 & 5 & 0\\
"""
        res = printer.decorate_table(text, lambda x: "#{0}#".format(x), d_cols=" & ", d_rows="\\\\\n")
        self.assertEqual("#0# & #5# & #10#\\\\\n#10# & #5# & #0#\\\\\n", res)


    def test_table_color_map(self):
        text = r"""0 & 5 & 10\\
20 & -5 & 0\\
"""
        MinNumber = 0
        MaxNumber = 10
        MidNumber = 5  # MaxNumber / 2
        MinColor = "green"
        MidColor = "yellow"
        MaxColor = "red"
        text = printer.table_color_map(text, MinNumber, MidNumber, MaxNumber, MinColor, MidColor, MaxColor)
        self.assertEqual("\cellcolor{green!100.0!yellow}0 & \cellcolor{green!0.0!yellow}5 & \cellcolor{red!100.0!yellow}10\\\\\n"+
                         "\cellcolor{red!100.0!yellow}20 & \cellcolor{green!100.0!yellow}-5 & \cellcolor{green!100.0!yellow}0\\\\\n", text)
