# evoplotter
A simple Python library for creating plots and tables for evolutionary algorithms (EA).

Experiment's data are internally processed as dictionaries. Each evolution run is assumed to be described in a separate file/dict. If the data is stored in external .properties files, then `utils` module can be used to load all the data into a list of dictionaries.

Data is logically grouped and processed by dimensions (`Dim` class) defined by a user. Each dimension consists of several configurations (`Config` class), where configuration (config for short) is basically a list of filtering functions. These functions are used on the data so only the data in the intersection of configs will remain. On this remaining data user's value function is called.

Dimensions may be merged by a Cartesian product (`*` operator) or a single config may be added to dimension (`+` operator).


In the app directory you may find some examples of how library is to be used.


Dependencies
============
- numpy
- matplotlib



Example plots
============

![My image](http://www.cs.put.poznan.pl/ibladek/github/evoplotter/plot1.jpg)
![My image](http://www.cs.put.poznan.pl/ibladek/github/evoplotter/plot2.jpg)
