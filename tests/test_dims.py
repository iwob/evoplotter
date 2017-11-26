import unittest
from src.dims import *

class TestsDims(unittest.TestCase):

    def test_Dim_mult0(self):
        dim1 = Dim([Config("A1", lambda p: True, valueA="1"),
                    Config("A2", lambda p: False, valueA="2")])
        dim2 = Dim([])
        prod = dim1 * dim2
        self.assertEqual(2, len(prod))
        self.assertEqual(Config, type(prod[0]))
        self.assertEqual("A1", prod[0].get_caption(sep="_"))
        self.assertEqual("1", prod[0].stored_values["valueA"])
        self.assertEqual("A2", prod[1].get_caption(sep="_"))
        self.assertEqual("2", prod[1].stored_values["valueA"])


    def test_Dim_mult1(self):
        dim1 = Dim([Config("A1", lambda p: True),
                    Config("A2", lambda p: False)])
        dim2 = Dim([Config("B1", lambda p: True)])
        prod = dim1 * dim2
        self.assertEqual(2, len(prod))
        self.assertEqual(Config, type(prod[0]))
        self.assertEqual("A1_B1", prod[0].get_caption(sep="_"))
        self.assertEqual("A2_B1", prod[1].get_caption(sep="_"))


    def test_Dim_mult2(self):
        dim1 = Dim([Config("A1", lambda p: True, valueA="1"),
                    Config("A2", lambda p: False, valueA="2")])
        dim2 = Dim([Config("B1", lambda p: True, valueB="1"),
                    Config("B2", lambda p: False, valueB="2")])
        prod = dim1 * dim2
        self.assertEqual(4, len(prod))
        self.assertEqual(Config, type(prod[0]))
        self.assertEqual("A1_B1", prod[0].get_caption(sep="_"))
        self.assertEqual("1", prod[0].stored_values["valueA"])
        self.assertEqual("1", prod[0].stored_values["valueB"])
        self.assertEqual("A1_B2", prod[1].get_caption(sep="_"))
        self.assertEqual("1", prod[1].stored_values["valueA"])
        self.assertEqual("2", prod[1].stored_values["valueB"])
        self.assertEqual("A2_B1", prod[2].get_caption(sep="_"))
        self.assertEqual("2", prod[2].stored_values["valueA"])
        self.assertEqual("1", prod[2].stored_values["valueB"])
        self.assertEqual("A2_B2", prod[3].get_caption(sep="_"))
        self.assertEqual("2", prod[3].stored_values["valueA"])
        self.assertEqual("2", prod[3].stored_values["valueB"])


    def test_Dim_mult_three(self):
        dim1 = Dim([Config("A1", lambda p: True),
                    Config("A2", lambda p: False)])
        dim2 = Dim([Config("B1", lambda p: True),
                    Config("B2", lambda p: False)])
        dim3 = Dim([Config("C1", lambda p: True),
                    Config("C2", lambda p: False)])
        prod = dim1 * dim2 * dim3
        self.assertEqual(8, len(prod))
        self.assertEqual(Config, type(prod[0]))
        self.assertEqual("A1_B1_C1", prod[0].get_caption(sep="_"))
        self.assertEqual("A1_B1_C2", prod[1].get_caption(sep="_"))
        self.assertEqual("A1_B2_C1", prod[2].get_caption(sep="_"))
        self.assertEqual("A1_B2_C2", prod[3].get_caption(sep="_"))
        self.assertEqual("A2_B1_C1", prod[4].get_caption(sep="_"))
        self.assertEqual("A2_B1_C2", prod[5].get_caption(sep="_"))
        self.assertEqual("A2_B2_C1", prod[6].get_caption(sep="_"))
        self.assertEqual("A2_B2_C2", prod[7].get_caption(sep="_"))


    def test_Dim_get_unique_values(self):
        props = [{'a': '0', 'b': '25'},
                 {'a': '4', 'b': '35'},
                 {'a': '4', 'b': '45', 'c':'34'}]
        self.assertEqual({'0', '4'}, utils.get_unique_values(props, lambda x: x['a']))
        self.assertEqual({'25', '35', '45'}, utils.get_unique_values(props, lambda x: x['b']))


    def test_Dim_from_data(self):
        props = [{'a': '1', 'b': '25'},
                 {'a': '4', 'b': '35'},
                 {'a': '4', 'b': '45', 'c':'34'}]
        dim = Dim.from_data(props, lambda p: p['a'])
        self.assertEqual(2, len(dim.configs))
        vals = {c.filters[0][0] for c in dim.configs}
        self.assertEqual({'1', '4'}, vals)
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

        self.assertEqual(False, conf4.filters[0][1]({'a':'1', 'b':'25'}))
        self.assertEqual(True, conf4.filters[0][1]({'a':'4', 'b':'25'}))
        self.assertEqual(True, conf1.filters[0][1]({'a':'1', 'b':'25'}))
        self.assertEqual(False, conf1.filters[0][1]({'a':'4', 'b':'25'}))


    def test_Dim_from_dict(self):
        props = [{'a': '1', 'b': '25'},
                 {'a': '4', 'b': '35'},
                 {'a': '4', 'b': '45', 'c':'34'}]
        dim = Dim.from_dict(props, 'a')
        self.assertEqual(2, len(dim.configs))
        vals = {c.filters[0][0] for c in dim.configs}
        self.assertEqual({'1', '4'}, vals)
        if dim.configs[0].filters[0][0] == '4':
            conf4 = dim.configs[0]
            conf1 = dim.configs[1]
        else:
            conf4 = dim.configs[1]
            conf1 = dim.configs[0]

        self.assertEqual('1', conf1.stored_values['a'])
        self.assertEqual('4', conf4.stored_values['a'])
        self.assertEqual(False, conf4.filters[0][1]({'a':'1', 'b':'25'}))
        self.assertEqual(True, conf4.filters[0][1]({'a':'4', 'b':'25'}))
        self.assertEqual(True, conf1.filters[0][1]({'a':'1', 'b':'25'}))
        self.assertEqual(False, conf1.filters[0][1]({'a':'4', 'b':'25'}))