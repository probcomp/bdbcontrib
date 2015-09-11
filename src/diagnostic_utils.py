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


# TODO: Migrate from hooks/contrib_diagnostics. Need users run experiments?
# def run_bdb_experiment(bdb, exp_args):
    # pass

# TODO Bring in estimate_kl_divergence from fsaad-kl-div branch in bayeslite.
# def estimate_kl_divergence(self, argin):


def estimate_log_likelihood(bdb, table, generator, targets=None, givens=None,
        samples=None):
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
    targets : list<tuple>, optional
        A list of [(column, value)] pairs on which to condition on. Defaults to
        no conditionals. See example for more details.

    Returns
    -------
    ll : float
        The log likelihood of the table[columns] under the conditional
        distribution (specified by givens) of generator.

    Example:
    estimate_log_likelihood(bdb, 'people', 'people_gen',
        targets=['weight','height'],
        givens=[('nationality','USA'),('age',17)])

    """
    # If no target columns specified, use all.
    if targets is None:
        generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator)
        targets = bayeslite.corebayesdb_generator_column_names(bdb,
            generator_id)
    targets = map(sqlite3_quote_name, targets)

    # If no givens columns, assume no conditions.
    if givens is None:
        givens = []
    else:
        given_cols = map(sqlite3_quote_name, givens[::2])
        given_vals = givens[1::2]
        assert len(given_cols) == len(given_vals)
        givens = zip(given_cols,given_vals)
        givens = ','.join(['{}={}'.format(c,v) for (c,v) in givens])

    # Obtain the dataset table.
    table = sqlite3_quote_name(table.strip(';'))
    sql = '''
        SELECT {} FROM {};
        '''.format(','.join(targets), table)
    dataset = bdb.execute(sql)

    # Obtain number of rows in the dataset and samples to use.
    samples = samples
    n_rows = bdb.execute('''
        SELECT COUNT(*) FROM {}
        '''.format(table)).fetchall()[0][0]
    if samples is None or n_rows < samples:
        samples = n_rows

    # Compute the log-likelihood of the targets, subject to givens.
    # XXX This code is currently wrong due to shortcomings in BQL:
    #  - BQL cannot evaluate joint density. Assume that all the rows are IID,
    #  and that all the columns factor into their marginal density.
    ll, i = 0, 0
    for row in dataset:
        if i > samples:
            break
        else:
            i += 1
        # XXX Wrong: assume joint factors into product of marginals.
        for col, val in zip(targets, row):
            if givens:
                bql = '''
                ESTIMATE PROBABILITY OF {}=? GIVEN ({}) FROM {} LIMIT 1
                '''.format(col, givens, sqlite3_quote_name(generator))
            else:
                bql = '''
                ESTIMATE PROBABILITY OF {}=? FROM {} LIMIT 1
                '''.format(col, sqlite3_quote_name(generator))

            crs = bdb.execute(bql, (val,))
            ll += math.log(crs.fetchall()[0][0])

    print ll
