import argparse
import math
import shlex

from bayeslite import core
from bayeslite.shell.hook import bayesdb_shell_cmd
from bayeslite.sqlite3_util import sqlite3_quote_name
import bdbexp

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
def run_bdb_exp(self, argin):
    '''
    Launch an experimental inference quality test (requires the bdbexp module).
    USAGE: .experiment <exp_name> [exp_args ...]

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
    '''
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
    '''
    Estimate the log likelihood of a dataset using N samples
    USAGE: .est_ll <generator> <table> [<--targets-cols ...>] [<--given-cols ...>] [<--samples> N]

    Examples:
    bayeslite> .est_ll my_gen my_table --target-cols height weight --given-cols age nationality 17 --samples 1000
    '''
    parser = ArgumentParser(prog='.est_ll')
    parser.add_argument('generator', type=str,
        help='Name of the generator.')
    parser.add_argument('table', type=str,  help='Name of the table.')
    parser.add_argument('--target-cols', nargs='*',
        help='Sequence of target columns to evaluate the log likelhood. '
        'By default, all columns in <table> will be used.')
    parser.add_argument('--givens', nargs='*',
        help='Sequence of columns and observed values to condition on. '
        'The required format is [<col> <val>...].')
    parser.add_argument('--samples', type=int,
        help='Number of rows in the dataset to use in the computation. '
        'Defaults to all rows.')

    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    cargs = args

    # If no target columns specified, use all.
    if args.target_cols:
        target_cols = args.target_cols
    else:
        generator_id = core.bayesdb_get_generator(self._bdb, args.generator)
        target_cols = core.bayesdb_generator_column_names(self._bdb,
            generator_id)
    target_cols = [sqlite3_quote_name(col) for col in target_cols]

    # If no givens columns, assume no conditions.
    if args.givens:
        cols = [sqlite3_quote_name(col) for col in args.givens[::2]]
        vals = [float(val) for val in args.givens[1::2]]
        assert len(cols) == len(vals)
        givens = zip(cols,vals)
    else:
        givens = []

    # Obtain the dataset table.
    table = sqlite3_quote_name(args.table.strip(';'))
    sql = '''
        SELECT {} FROM {};
    '''.format(','.join(target_cols), table)
    dataset = self._bdb.execute(sql)

    # Obtain number of rows in the dataset.
    samples = args.samples
    n_rows = self._bdb.execute('''
        SELECT COUNT(*) FROM {}
        '''.format(table)).fetchall()[0][0]
    if samples is None or n_rows < samples:
        samples = n_rows

    # Compute the log-likelihood of the target_cols, subject to givens.
    # XXX This code is wrong due to shortcomings in BQL:
    #  - BQL cannot evaluate joint density. Assume that all the rows are IID,
    #  and that all the columns factor into their marginal density.
    #  - BQL cannot evaluate conditional density. This query will not compile.
    LL = 0
    i = 0
    for row in dataset:
        if i > samples:
            break
        else:
            i += 1
        # XXX Wrong: assume joint factors into product of marginals.
        for col, val in zip(target_cols, row):
            if givens:
                bql = '''
                ESTIMATE PROBABILITY OF {}=? GIVEN ({}) FROM {} LIMIT 1
                '''.format(col,
                        ','.join(['{}={}'.format(c,v) for (c,v) in givens]),
                        sqlite3_quote_name(args.generator))
            else:
                bql = '''
                ESTIMATE PROBABILITY OF {}=? FROM {} LIMIT 1
                '''.format(col, sqlite3_quote_name(args.generator))

            crs = self._bdb.execute(bql, (val,))
            LL += math.log(crs.fetchall()[0][0])

    print LL

# TODO Bring in estimate_kl_divergence from fsaad-kl-div branch in bayeslite.
# @bayesdb_shell_cmd('est_kl')
# def estimate_kl_divergence(self, argin):
