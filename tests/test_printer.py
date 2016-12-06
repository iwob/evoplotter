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

	def test_text_table_multichar_seps(self):
		text = printer.text_table(self.data, self.dim_rows, self.dim_cols, lambda ds: sum([d["x"] for d in ds]), d_cols=" && ", d_rows=";\n")
		self.assertEquals(" && c0 && c1 && c2;\n" + "r0 && 6 && 15 && 24;\n" + "r1 && 36 && 45 && 54;\n", text)


	def test_latex_table(self):
		text = printer.latex_table(self.data, self.dim_rows, self.dim_cols, lambda ds: sum([d["x"] for d in ds]))
		self.assertEquals(r" & c0 & c1 & c2\\" + "\n" + r"r0 & 6 & 15 & 24\\"+"\n" + r"r1 & 36 & 45 & 54\\"+"\n", text)


	def test_text_table(self):
		text = printer.text_table(self.data, self.dim_rows, self.dim_cols, lambda ds: sum([d["x"] for d in ds]))
		self.assertEquals("\tc0\tc1\tc2\n" + "r0\t6\t15\t24\n"  + "r1\t36\t45\t54\n", text)

		text = printer.text_table(self.data, self.dim_cols, self.dim_rows, lambda ds: sum([d["x"] for d in ds]))
		self.assertEquals("\tr0\tr1\n" + "c0\t6\t36\n" + "c1\t15\t45\n" + "c2\t24\t54\n", text)


	def test_text_listing(self):
		dim = self.dim_rows * self.dim_cols
		text = printer.text_listing(self.data, dim, lambda ds: sum([d["x"] for d in ds]), d_configs="\n\n")
		self.assertEquals("(*) CONFIG: r0_c0\n6\n\n" + "(*) CONFIG: r0_c1\n15\n\n" + "(*) CONFIG: r0_c2\n24\n\n" +
		                  "(*) CONFIG: r1_c0\n36\n\n" + "(*) CONFIG: r1_c1\n45\n\n" + "(*) CONFIG: r1_c2\n54\n\n", text)