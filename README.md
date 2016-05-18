# bdbcontrib

A set of utilities and a front end for BayesDB.

## Installing

Software requirements are detailed in setup.py.
Please see http://probcomp.csail.mit.edu/bayesdb/install.html for installation.

## Test

Please run local tests before sending a pull request:

```
$ ./check.sh
```

That does not run the complete test suite, only the smoke tests, but
is usually good enough. For the full suite:

```
$ ./check.sh tests shell/tests
```

## Install

To install system-wide, or into the current virtual environment:

```
$ python setup.py build
$ python setup.py install
```

## Documentation

```
from bdbcontrib import Population
help(Population)
foo = Population(...)
foo.help("plot")
```

## Contributing

This repository is currently using "Light Review" from
http://tinyurl.com/probcomp-review-standards

Our compatibility aim is to work on probcomp machines and members'
laptops, and to provide scripts and instructions that make it not too
hard to re-create our environments elsewhere. Polished packaging,
broad installability without much work, etc. are anti-goals, because
they take attention away from our research mission: focusing on
probabilistic computing. If you want to help make this software work
in an environment where it does not yet, that effort is welcome in the
http://github.com/probcomp/packaging/ repository.
