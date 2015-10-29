# bdbcontrib

A set of utilities for bayesdb.

## Installing

The software in bdbcontrib requires:

- [bayeslite >=0.1.2](http://probcomp.csail.mit.edu/bayesdb/)
- [ipython notebook >=3](http://ipython.org/notebook.html)
- [markdown2](https://pypi.python.org/pypi/markdown2)
- [matplotlib](http://matplotlib.org/)
- [numpy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/)
- [seaborn >=6](http://stanford.edu/~mwaskom/software/seaborn/)
- [tornado >=4](http://www.tornadoweb.org/en/stable/)

The tests require:

- [mock](https://pypi.python.org/pypi/mock)
- [pillow](https://python-pillow.github.io/)
- [pytest](http://pytest.org/)

The documentation requires:

- [numpydoc](https://pypi.python.org/pypi/numpydoc)
- [sphinx](sphinx-doc.org)

Individual parts of bdbcontrib may have slimmer dependencies, if you
want to pull them out for more limited purposes.

## Test

To run the paltry automatic tests:

```
$ ./check.sh
```

## Install

To install system-wide, or into the current virtual environment:

```
$ python setup.py build
$ python setup.py install
```

## Documentation

The python documentation is built using [sphinx](http://sphinx-doc.org/) and
[numpydoc](https://pypi.python.org/pypi/numpydoc).

```
$ make doc
```

## Use

If you want the Python API, `import bdbcontrib.plot_utils`, &c.

If you are using the bayeslite shell, load the bayeslite shell
commands with:

```
.hook /path/to/bdbcontrib/hooks/hook_plotting.py
```
