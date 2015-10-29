# bdbcontrib

A set of utilities for bayesdb.

## Installing

**Requires**

- Matplotlib
- Seaborn == 0.5.1
- Pandas
- numpy
- markdown2 (for `.readtohtml`)
- sphinx (for documentation)
- numpydoc (for documentation)

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
