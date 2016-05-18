# bdbcontrib

A set of utilities and a front end for BayesDB.

## Installing

Software requirements are detailed in setup.py.
Please see http://probcomp.csail.mit.edu/bayesdb/install.html for installation.

## Expectations

Users and contributors should expect **rapidly and dramatically
shifting code and behavior** at this time.

We would like all users to enable session capture so that we can see
how users behave with the software and improve it on that
basis. Please read examples/Index.ipynb for details. You **must
choose** either to opt in or to opt out.

**THIS SOFTWARE SHOULD NOT BE EXPECTED TO TREAT YOUR DATA SECURELY.**

## Contributing

This repository is currently using "Light Review" from
http://tinyurl.com/probcomp-review-standards

Our compatibility aim is to work on probcomp machines and members'
laptops, and to provide scripts and instructions that make it not too
hard to re-create our environments elsewhere. Pulls for polished
packaging, broad installability, etc. are not appropriate
contributions at this time.

Please run local tests before sending a pull request:

```
$ ./check.sh
```

That does not run the complete test suite, only the smoke tests, but
is usually good enough. For the full suite:

```
$ ./check.sh tests examples/tests
```

## Documentation

```
from bdbcontrib import Population
help(Population)
foo = Population(...)
foo.help("plot")
```
