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

    def test_Dim_get_unique_values(self):
        props = [{'a': '0', 'b': '25'},
                 {'a': '4', 'b': '35'},
                 {'a': '4', 'b': '45', 'c':'34'}]
        self.assertEquals({'0', '4'}, get_unique_values(props, lambda x: x['a']))
        self.assertEquals({'25', '35', '45'}, get_unique_values(props, lambda x: x['b']))

    def test_Dim_from_data(self):
        props = [{'a': '1', 'b': '25'},
                 {'a': '4', 'b': '35'},
                 {'a': '4', 'b': '45', 'c':'34'}]
        dim = Dim.from_data(props, lambda p: p['a'])
        self.assertEquals(2, len(dim.configs))
        vals = {c.filters[0][0] for c in dim.configs}
        self.assertEquals({'1', '4'}, vals)
        if dim.configs[0].filters[0][0] == '4':
            conf4 = dim.configs[0]
            conf1 = dim.configs[1]
        else:
            conf4 = dim.configs[1]
            conf1 = dim.configs[0]

        # print("conf1")
        # for i in ['0', '1', '2', '3', '4', '5', '6']:
        # 	print(i + ": " + str(conf1.filters[0][1]({'a': i, 'b': '25'})))
        # print("conf4")
        # for i in ['0', '1', '2', '3', '4', '5', '6']:
        # 	print(i + ": " + str(conf4.filters[0][1]({'a': i, 'b': '25'})))

        self.assertEquals(False, conf4.filters[0][1]({'a':'1', 'b':'25'}))
        self.assertEquals(True, conf4.filters[0][1]({'a':'4', 'b':'25'}))
        self.assertEquals(True, conf1.filters[0][1]({'a':'1', 'b':'25'}))
        self.assertEquals(False, conf1.filters[0][1]({'a':'4', 'b':'25'}))