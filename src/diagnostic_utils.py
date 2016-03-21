# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import math

import bayeslite.core
from bayeslite import bql_quote_name
from bayeslite.exception import BayesLiteException as BLE

def extract_target_cols(bdb, generator, targets=None):
    """Extract target columns (helper for LL/KL query).

    If targets is None, then a list of all sqlite3 quoted column names from
    generator are returned.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    generator : str
        Name of generator.
    targets : list<str>, optional
        List of columns in the table.

    Returns
    -------
    target_cols : list<str>
    """
    if targets is None:
        generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator)
        targets = bayeslite.core.bayesdb_generator_column_names(bdb,
            generator_id)
    return map(bql_quote_name, targets)


def extract_given_cols_vals(givens=None):
    """Extract given columns and values (helper for LL/KL).

    If givens is None, then an empty list is returned. Otherwise an appropriate
    list of tuples is returned, where the first element is the sqlite3 quoted
    name and the second element is the constraint value.

    Parameters
    ----------
    givens : list<str>, optional
        List of columns in the table.

    Returns
    -------
    given_cols_vals : list<tuple>
    """
    if givens is None:
        return []

    given_cols = map(bql_quote_name, givens[::2])
    given_vals = givens[1::2]
    assert len(given_cols) == len(given_vals)
    return zip(given_cols, given_vals)


def estimate_log_likelihood(bdb, table, generator, targets=None, givens=None,
        n_samples=None):
    """Estimate the log likelihood for obsevations in a table.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    table : str
        Name of table.
    generator : str
        Name of generator.
    targets : list<str>, optional
        List of columns in the table for which to compute the log-likelihood.
        Defaults to all the columns.
    givens : list<tuple>, optional
        A list of [(column, value)] pairs on which to condition on. Defaults to
        no conditionals. See example for more details.
    n_samples : int, optional
        Number of rows from table to use in the computation. Defaults to all
        the rows.

    Returns
    -------
    ll : float
        The log likelihood of the table[columns] under the conditional
        distribution (specified by givens) of generator.

    Example:
    estimate_log_likelihood(bdb, 'people', 'people_gen',
        targets=['weight', 'height'],
        givens=[('nationality', 'USA'), ('age', 17)])
    """
    # Defaults to all columns if targets is None.
    targets = extract_target_cols(bdb, generator, targets=targets)

    # Defaults to no givens if givens is None
    givens = extract_given_cols_vals(givens=givens)
    givens = ','.join(['{}={}'.format(c,v) for (c,v) in givens])

    # Obtain the dataset table.
    table = bql_quote_name(table.strip(';'))
    sql = '''
        SELECT {} FROM {};
    '''.format(','.join(targets), table)
    dataset = bdb.execute(sql)

    # Obtain number of rows in the dataset and samples to use.
    n_samples = n_samples
    n_rows = bdb.execute('''
        SELECT COUNT(*) FROM {}'''.format(table)).fetchvalue()
    if n_samples is None or n_rows < n_samples:
        n_samples = n_rows

    # Compute the log-likelihood of the targets, subject to givens.
    # XXX This code is currently wrong due to shortcomings in BQL:
    #  - BQL cannot evaluate joint density. Assume that all the rows are IID,
    #  and that all the columns factor into their marginal density.
    ll, i = 0, 0
    for row in dataset:
        if i > n_samples:
            break
        else:
            i += 1
        # XXX Wrong: assume joint factors into product of marginals.
        for col, val in zip(targets, row):
            if givens:
                # XXX TODO write GIVEN in this query using bindings.
                bql = '''
                    ESTIMATE PROBABILITY OF {}=? GIVEN ({}) FROM {} LIMIT 1
                '''.format(col, givens, bql_quote_name(generator))
            else:
                bql = '''
                    ESTIMATE PROBABILITY OF {}=? FROM {} LIMIT 1
                '''.format(col, bql_quote_name(generator))

            ll += math.log(bdb.execute(bql, (val,)).fetchvalue())

    return ll

def estimate_kl_divergence(bdb, generatorA, generatorB, targets=None,
        givens=None, n_samples=None):
    """Estimate the KL divergence.

    The KL divergence is a mesaure of the "information lost" when generatorB
    (the approximating generator) is used to approximate generatorA (the base
    generator). KL divergence is not symmetric in, and KL(genA||genB) is not
    necessarily equal to KL(genB||genA).

    TODO: Monte Carlo estimation is a terrible way to compute the KL divergence.
    (Not to say there are better methods in general). One illustration of this
    is that the estimated KL divergence has emperically been shown to obtain
    negative realizations for high-dimensional data.

    Computing the KL divergence in general (of high dimensional distributions)
    is a very hard problem; most research uses the structure of the
    distributions to find good estimators. Adaptive quadrature or exact methods
    for numerical integration could outperform Monte Carlo?

    TODO: More sophisticated algorithm for detecting cases where absolute
    continuity could be a problem (currently have a heuristic).
    As it stands, Monte Carlo estimates may have infinite variance depending
    on simulated values from generatorA.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    generatorA : str
        Name of base generator.
    generatorB : str
        Name of approximating generator.
    targets : list<str>, optional
        List of columns in the table for which to compute the log-likelihood.
        Defaults to all the columns.
    givens : list<tuple>, optional
        A list of [(column, value)] pairs on which to condition on. Defaults to
        no conditionals. See example for more details.
    n_samples: int, optional
        Number of simulated samples to use in the Monte Carlo estimate.

    Returns
    -------
    kl : float
        The KL divergence. May be infinity.

    Example:
    estimate_kl_divergence(bdb, 'crosscat_gen', 'baxcat_gen',
        targets=['weight', 'height'],
        givens=[('nationality', 'USA'), ('age', 17)])
    """
    # XXX Default to 10,000 samples
    if n_samples is None:
        n_samples = 10000

    # Defaults to all columns if targets is None.
    targets = extract_target_cols(bdb, generatorA, targets=targets)

    # Defaults to no givens if givens is None
    givens = extract_given_cols_vals(givens=givens)
    givens = ','.join(['{}={}'.format(c,v) for (c,v) in givens])

    # Obtain samples from the base distribution.
    if givens:
        # XXX TODO write GIVEN in this query using bindings.
        bql = '''
            SIMULATE {} FROM {} GIVEN {} LIMIT {}
        '''.format(','.join(targets), bql_quote_name(generatorA),
            givens, n_samples)
    else:
        bql = '''
            SIMULATE {} FROM {} LIMIT {}
        '''.format(','.join(targets), bql_quote_name(generatorA),
                    n_samples)
    samples = bdb.execute(bql)

    kl = 0
    for s in samples:
        logp_a, logp_b = 0, 0
        # XXX Assume joint probability factors by summing univariate
        # (conditional) probability of each cell value. This is clearly wrong,
        # until we can evaluate joint densities in BQL.
        for col, val in zip(targets, s):
            bql = '''
                ESTIMATE PROBABILITY OF {}=? FROM {} LIMIT 1
            '''.format(col, bql_quote_name(generatorA))
            crs = bdb.execute(bql, (val,))
            p_a = crs.fetchvalue()

            bql = '''
                ESTIMATE PROBABILITY OF {}=? FROM {} LIMIT 1
            '''.format(col, bql_quote_name(generatorB))
            crs = bdb.execute(bql, (val,))
            p_b = crs.fetchvalue()

            # XXX Heuristic to detect when genA is not absolutely
            # continuous wrt genB
            if p_a == 0:
                # How on earth did we simulate a value from genA with zero
                # density/prob under genA?
                raise BLE(ValueError(
                    'Fatal error: simulated a (col,val)=({},{}) '
                    'from base generatorA ({}) with zero density. Check '
                    'implementation of simluate and/or logpdf of '
                    'generator.'.format(col,val,generatorA)))
            if p_b == 0:
                # Detected failure of absolute continuity
                # (under assumption that joint factors into marginals)
                return float('inf')

            logp_a += math.log(p_a)
            logp_b += math.log(p_b)

        kl += (logp_a - logp_b)

    # XXX Assertion may fail, see TODO in docstring.
    # assert kl > 0
    if kl < 0:
        raise BLE(ValueError(
            'Cannot compute reasonable value for KL divergence. '
            'Try increasing the number of samples (currently using {}'
            'samples).'.format(n_samples)))

    return kl / n_samples


# TODO: Migrate from hooks/contrib_diagnostics. Need users run experiments?
# def run_bdb_experiment(bdb, exp_args):
    # pass
