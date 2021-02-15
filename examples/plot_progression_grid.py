from evoplotter import plotter
from evoplotter.dims import *


# Data
props = [
    {"dimX": "A", "dimY": "C", "trainSet": "9,9,7,6,6,3,3,2,2,1", "validSet": "10,10,7,6,6,3,6,6,7,9"},
    {"dimX": "A", "dimY": "C", "trainSet": "8,8,7,5,5,3,3,2,2,1", "validSet": "9,9,7,6,6,3,5,5,7,8"},
    {"dimX": "A", "dimY": "C", "trainSet": "7,7,7,4,4,3,3,2,2,1", "validSet": "8,8,7,6,6,3,4,4,7,7"},

    {"dimX": "A", "dimY": "D", "trainSet": "9,9,7,6,6,3,3,2,2,1", "validSet": "9,9,7,6,6,3,6,6,7,9"},
    {"dimX": "A", "dimY": "D", "trainSet": "8,8,7,5,5,3,3,2,2,1", "validSet": "9,9,7,6,6,3,5,5,7,8"},
    {"dimX": "A", "dimY": "D", "trainSet": "7,7,7,4,4,3,3,2,2,1", "validSet": "9,9,7,6,6,3,4,4,7,7"},

    {"dimX": "B", "dimY": "C", "trainSet": "9,9,7,6,6,3,3", "validSet": "9,9,7,6,6,3,6"},
    {"dimX": "B", "dimY": "C", "trainSet": "8,8,7,5,5,3,3,2,2,1", "validSet": "9,9,7,6,6,3,5"},
    {"dimX": "B", "dimY": "C", "trainSet": "7,7,7,4,4,3,3,2,2,1", "validSet": "9,9,7,6,6,3,4"},

    {"dimX": "B", "dimY": "D", "trainSet": "9,9,7,6,6,3,3,2,2,1", "validSet": "9,9,7,6,6,3,6,6,7,9"},
    {"dimX": "B", "dimY": "D", "trainSet": "8,8,7,5,5,3,3,2,2,1", "validSet": "9,9,7,6,6,3,5,5,7,8"},
    {"dimX": "B", "dimY": "D", "trainSet": "7,7,7,4,4,3,3,2,2,1", "validSet": "9,9,7,6,6,3,4,4,7,7"},
    {"dimX": "B", "dimY": "D", "trainSet": "5,5,4,4,4,3,3,2,2,1", "validSet": "7,6,7,6,6,3,4,4,7,7"},
    {"dimX": "B", "dimY": "D", "trainSet": "7,7,7,4,4,3,3,2,2,1", "validSet": "8,5,7,6,6,3,4,4,7,7"},
]



dim_rows = Dim.from_dict_value_match("dimX", ["A", "B"])
dim_cols = Dim.from_dict_value_match("dimY", ["C", "D"])

plotter.plot_value_progression_grid_simple(props, dim_rows, dim_cols, key_values=["trainSet", "validSet"],
                                           title="Example progression grid", savepath="progression_grid_i.pdf",
                                           plot_individual_runs=True, opacity=0.5, show_plot=True, plot_grid=False)

plotter.plot_value_progression_grid_simple(props, dim_rows, dim_cols, key_values=["trainSet", "validSet"],
                                           title="Example progression grid", savepath="progression_grid_e.pdf",
                                           plot_individual_runs=False, opacity=0.5, show_plot=True, markevery=1,
                                           errorevery=1, plot_grid=False)