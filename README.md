# bdbcontrib

A set of utilities for bayesdb. 


## Installing
**Requires**

- Matplotlib
- Seaborn
- Pandas
- numpy

Clone the repo and add it to your `PYTHONPATH`

## Install 
Clone the repo and add the `bdbdcontrib` path to your `PYTHONPATH`

**Linux**
```
$ git clone https://github.com/mit-probabilistic-computing-project/bdbcontrib.git
$ cd bdbcontrib
$ echo export PYTHONPATH=\$PYTHONPATH:`pwd` >> ~/.bashrc
```

**OSX**
```
$ git clone https://github.com/mit-probabilistic-computing-project/bdbcontrib.git
$ cd bdbcontrib
$ echo export PYTHONPATH=\$PYTHONPATH:`pwd` >> ~/.bash_profile
```

**Automatically hooking the contrib**

To ensure that the contrib is automatically loaded by the shell, add the following to your `~/.bayesliterc` file

    .hook /absolute/path/to/contrib.py


## Use

### Shell utilities

#### .zmatrix

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
