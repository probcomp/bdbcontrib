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

    .zmatrix <pairwise query> [options]

**Options:**
- `-f, --filename <str>`: save as filename.

Example:

    bayeslite> .zmatrix ESTIMATE PAIRWISE DEPENDENCE PROBABILITY FROM satellites_cc;

![zmatrix](doc/zmatrix.png)

#### .pairplot
Draws or saves a pariplot of an arbitrary BQL query

    .pairplot <query> [options]

**Options:**
- `-g, --generator <str>`: the generator name (decreases guesswork with respect to which columns are
    which data types)
- `-f, --filename <str>`: save as filename.
- `-s, --shortnames`: use columns shornames (requires codebook) on axis labels.

Example:

    bayeslite> CREATE TEMP TABLE predprob_life AS ESTIMATE Name, Expected_Lifetime, PREDICTIVE PROBABILITY OF Expected_Lifetime AS p_lifetime, Class_of_Orbit FROM satellites_cc;
    bayeslite> .pairplot SELECT Expected_Lifetime, class_of_orbit, p_lifetime FROM predprob_life

![.pairplot](doc/pairplot.png)

#### .ccstate
Draws a crosscat state

    .ccstate <generator> <modelno> [filename.png]

Example:

    bayeslite> .ccstate dha_cc 0

![.ccstate rendering of dha](doc/ccstate_1.png)

#### .hist
Draws a histogram for a one or two-column query. The second column in a
two-column query is assumed to be a dummy variable use to divide the data
into categories.

    .hist <query> [options]

**Options:**
- `--normed`: normalize the histogram
- `-b, --bins <int>`: the number of bins
- `-f, --filename <str>`: save as filename.

Example (one-column query):

    bayeslite> .hist SELECT MDCR_SPND_OUTP FROM dha; --normed --bins 31

![histogram](doc/hist1.png)

Example (two-column query):

    bayeslite> .hist SELECT dry_mass_kg, Class_of_Orbit FROM satellites; -b 35 --normed

![histogram](doc/hist2.png)

#### .chainplot
Plots various model diagnostics as a function of iterations. To use `.chainplot`, `ANALYZE` must
be run with `CHEKPOINT`.

    .chainplot <logscore|num_views|column_crp_alpha> <generator> [filename]

Example:

    bayeslite> ANALYZE satellites_cc FOR 50 ITERATIONS CHECKPOINT 2 ITERATION WAIT;
    bayeslite> .chainplot logscore satellites_cc

![logscore over iterations](doc/logscore.png)
