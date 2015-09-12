# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
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
from bayeslite.sqlite3_util import sqlite3_quote_name


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
    return map(sqlite3_quote_name, targets)


def extract_given_cols_vals(givens=None):
    """Extract target columns (helper for LL/KL).

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

    given_cols = map(sqlite3_quote_name, givens[::2])
    given_vals = givens[1::2]
    assert len(given_cols) == len(given_vals)

    return zip(given_cols, given_vals)


# TODO: Migrate from hooks/contrib_diagnostics. Need users run experiments?
# def run_bdb_experiment(bdb, exp_args):
    # pass

# TODO Bring in estimate_kl_divergence from fsaad-kl-div branch in bayeslite.
# def estimate_kl_divergence(self, argin):


def estimate_log_likelihood(bdb, table, generator, targets=None, givens=None,
        n_samples=None):
    """Estimate the log likelihood of a dataset.

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
    table = sqlite3_quote_name(table.strip(';'))
    sql = '''
        SELECT {} FROM {};
        '''.format(','.join(targets), table)
    dataset = bdb.execute(sql)

    # Obtain number of rows in the dataset and samples to use.
    n_samples = n_samples
    n_rows = bdb.execute('''
        SELECT COUNT(*) FROM {}
        '''.format(table)).fetchall()[0][0]
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
                '''.format(col, givens, sqlite3_quote_name(generator))
            else:
                bql = '''
                ESTIMATE PROBABILITY OF {}=? FROM {} LIMIT 1
                '''.format(col, sqlite3_quote_name(generator))

            crs = bdb.execute(bql, (val,))
            ll += math.log(crs.fetchall()[0][0])

    return ll

