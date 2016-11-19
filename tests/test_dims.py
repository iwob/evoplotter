import unittest
from src.dims import *

class TestsDims(unittest.TestCase):

	def test_Dim_mult0(self):
		dim1 = Dim([Config("A1", lambda p: True),
		            Config("A2", lambda p: False)])
		dim2 = Dim([])
		prod = dim1 * dim2
		self.assertEquals(2, len(prod))
		self.assertEquals(Config, type(prod[0]))
		self.assertEquals("A1", prod[0].get_caption(sep="_"))
		self.assertEquals("A2", prod[1].get_caption(sep="_"))

	def test_Dim_mult1(self):
		dim1 = Dim([Config("A1", lambda p: True),
		            Config("A2", lambda p: False)])
		dim2 = Dim([Config("B1", lambda p: True)])
		prod = dim1 * dim2
		self.assertEquals(2, len(prod))
		self.assertEquals(Config, type(prod[0]))
		self.assertEquals("A1_B1", prod[0].get_caption(sep="_"))
		self.assertEquals("A2_B1", prod[1].get_caption(sep="_"))

	def test_Dim_mult2(self):
		dim1 = Dim([Config("A1", lambda p: True),
		            Config("A2", lambda p: False)])
		dim2 = Dim([Config("B1", lambda p: True),
		            Config("B2", lambda p: False)])
		prod = dim1 * dim2
		self.assertEquals(4, len(prod))
		self.assertEquals(Config, type(prod[0]))
		self.assertEquals("A1_B1", prod[0].get_caption(sep="_"))
		self.assertEquals("A1_B2", prod[1].get_caption(sep="_"))
		self.assertEquals("A2_B1", prod[2].get_caption(sep="_"))
		self.assertEquals("A2_B2", prod[3].get_caption(sep="_"))

	def test_Dim_mult_three(self):
		dim1 = Dim([Config("A1", lambda p: True),
		            Config("A2", lambda p: False)])
		dim2 = Dim([Config("B1", lambda p: True),
		            Config("B2", lambda p: False)])
		dim3 = Dim([Config("C1", lambda p: True),
		            Config("C2", lambda p: False)])
		prod = dim1 * dim2 * dim3
		self.assertEquals(8, len(prod))
		self.assertEquals(Config, type(prod[0]))
		self.assertEquals("A1_B1_C1", prod[0].get_caption(sep="_"))
		self.assertEquals("A1_B1_C2", prod[1].get_caption(sep="_"))
		self.assertEquals("A1_B2_C1", prod[2].get_caption(sep="_"))
		self.assertEquals("A1_B2_C2", prod[3].get_caption(sep="_"))
		self.assertEquals("A2_B1_C1", prod[4].get_caption(sep="_"))
		self.assertEquals("A2_B1_C2", prod[5].get_caption(sep="_"))
		self.assertEquals("A2_B2_C1", prod[6].get_caption(sep="_"))
		self.assertEquals("A2_B2_C2", prod[7].get_caption(sep="_"))
