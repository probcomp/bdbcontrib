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

import argparse
import math
import shlex

import bdbexp
from bayeslite import core
from bayeslite.shell.hook import bayesdb_shell_cmd
from bayeslite.sqlite3_util import sqlite3_quote_name

from bdbcontrib.general_utils import ArgparseError, ArgumentParser


experiments = {
    'haystacks' : bdbexp.haystacks,
    'hyperparams' : bdbexp.hyperparams,
    'infer' : bdbexp.infer,
    'permute' : bdbexp.permute,
    'recover' : bdbexp.recover,
    'univariate_kl' : bdbexp.univariate_kldiv,
    'univariate_pdf' : bdbexp.univariate_pdf
    }


@bayesdb_shell_cmd('experiment')
def run_bdb_experiment(self, argin):
    """ Launch an experimental inference quality test (requires bdbexp module).
    <exp_name> [exp_args ...]

    <exp_name>
        permute         univariate_kl
        haystacks       univariate_pdf
        hyperparams     recover
        infer

    [exp_args]
        To see experiment specific arguments, use
        .experiment <exp_name> --help

    Examples:
    bayeslite> .experiment predictive_pdf --help
    bayeslite> .experiment haystacks --n_iter=200 --n_distractors=4
    """
    # TODO: Use ArgumentParser instead.
    # TODO: Migrate this to the public API?
    expname = argin.split()[0] if argin != '' else argin
    if expname not in experiments:
        print 'Invalid experiment {}'.format(expname)
        print 'For help use: .help experiment'
        return

    try:
        experiments[argin.split()[0]].main(argin.split()[1:], halt=True)
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return
    except SystemExit:
        return  # This happens when --help is invoked.


@bayesdb_shell_cmd('est_ll')
def estimate_log_likelihood(self, argin):
    """Estimate the log likelihood of a dataset.
    <table> <generator> [<--targets-cols ...>] [<--given-cols ...>] [<--n-samples> N]

    Examples:
    bayeslite> .est_ll people people_gen --targets height --givens age 1 nationality 17 --n-samples 1000
    """
    parser = ArgumentParser(prog='.est_ll')
    parser.add_argument('table', type=str,
        help='Name of the table.')
    parser.add_argument('generator', type=str,
        help='Name of the generator.')
    parser.add_argument('--targets', nargs='*',
        help='Sequence of target columns to evaluate the log likelhood. '
        'By default, all columns in <table> will be used.')
    parser.add_argument('--givens', nargs='*',
        help='Sequence of columns and observed values to condition on. '
        'The required format is [<col> <val>...].')
    parser.add_argument('--n-samples', type=int,
        help='Number of rows in the dataset to use in the computation. '
        'Defaults to all rows.')

    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    ll = bdbcontrib.api.estimate_log_likelihood(self._bdb, args.table,
        args.generator, targets=args.targets, givens=args.givens,
        n_samples=args.n_samples)

    print ll

# TODO Bring in estimate_kl_divergence from fsaad-kl-div branch in bayeslite.
# @bayesdb_shell_cmd('est_kl')
# def estimate_kl_divergence(self, argin):
