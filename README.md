# `bdbcontrib`

A set of utilities for bayesdb. 

**Requires**

- Matplotlib
- Seaborn
- Pandas
- numpy

## Installing
Clone the repo and add it to your `PYTHONPATH`

## Use


### Shell utilities
To make ensure that the contrib is automatically loaded by the shell, you will need to `.hook` `contrib.py`. To do so add the following to your `~/.bayesliterc` file

    .hook /absolute/path/to/contrib.py

Otherwise you can load then on startup

    $ bayeslite -f path/to/contrib.py

or hook them from within the shell

    bayeslite> .hook path/to/contrib.py

#### `.zmatrix`

    .zmatrix <pairwise query> [-f|--filename path/to/file.png]

To draw a z-matrix:

    bayeslite> .zmatrix ESTIMATE PAIRWISE <something> FROM <generator>

To save a z-matrix to a file

    bayeslite> .zmatrix ESTIMATE PAIRWISE <something> FROM <generator> -f my_zmatrix.png
    bayeslite> .zmatrix ESTIMATE PAIRWISE <something> FROM <generator> --filename my_zmatrix.png


#### .pairplot
Draws or saves a pariplot of an arbitrary BQL query

    .pairplot <query> [-f|--filename path/to/file.png]

To draw 

    bayeslite> .pairplot SELECT col1, col2, col3 FROM table

To save as file

    bayeslite> .pairplot SELECT col1, col2, col3 FROM table -f my_pairplot.png
    bayeslite> .pairplot SELECT col1, col2, col3 FROM table --filename my_pairplot.png


#### .ccstate
Draws a crosscat state

    .ccstate <generator> <modelno> [filename.png]
