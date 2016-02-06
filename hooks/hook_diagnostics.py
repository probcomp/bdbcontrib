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

import shlex

from bayeslite.shell.hook import bayesdb_shell_cmd

import bdbcontrib
from bdbcontrib.shell_utils import ArgparseError, ArgumentParser


@bayesdb_shell_cmd('est_ll')
def estimate_log_likelihood(self, argin):
    """estimate log likelihood for a dataset
    <table> <generator> [--targets-cols <...>] [--given-cols <...>] [--n-samples <N>]

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

    ll = bdbcontrib.estimate_log_likelihood(self._bdb, args.table,
        args.generator, targets=args.targets, givens=args.givens,
        n_samples=args.n_samples)

    print ll


@bayesdb_shell_cmd('est_kl')
def estimate_kl_divergence(self, argin):
    """estimate KL divergence of generator from reference
    <table> <generator-a> <generator-b> [--targets-cols <...>] [--given-cols <...>] [--n-samples <N>]

    Examples:
    bayeslite> .est_kl crosscat baxcat --targets height --givens age 1 nationality 17 --n-samples 1000
    """
    parser = ArgumentParser(prog='.est_kl')
    parser.add_argument('generator_a', type=str,
        help='Name of the reference generator.')
    parser.add_argument('generator_b', type=str,
        help='Name of the approximating generator.')
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

    kl = bdbcontrib.estimate_kl_divergence(self._bdb, args.generator_a,
        args.generator_b, targets=args.targets, givens=args.givens,
        n_samples=args.n_samples)

    print kl
