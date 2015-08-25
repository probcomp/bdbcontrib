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
def run_experiment(self, argin):
    '''
    Launch an experimental inference quality test.
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
    Estimate the log likelihood of a dataset
    USAGE: .est_ll <generator> <table> <targets_cols ...> <given_cols ...>

    Examples:
    bayeslite> .est_ll my_gen my_table --target-cols age height weight --givens nationality=USA gender=Male
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
        'The required format is [<col> <val>...] .')

    try:
        cargs = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    # If no target columns specified, use all.
    if cargs.target_cols:
        target_cols = cargs.target_cols
    else:
        generator_id = core.bayesdb_get_generator(self._bdb, cargs.generator)
        target_cols = core.bayesdb_generator_column_names(self._bdb,
            generator_id)
    target_cols = [sqlite3_quote_name(col) for col in target_cols]

    # If no givens columns, assume no conditions.
    if cargs.givens:
        import ipdb; ipdb.set_trace()
        cols = [sqlite3_quote_name(col) for col in cargs.givens[::2]]
        vals = [float(val) for val in cargs.givens[1::2]]
        assert len(cols) == len(vals)
        givens = zip(cols,vals)
    else:
        givens = []

    # Obtain the dataset table
    sql = '''
        SELECT {} FROM {};
    '''.format(','.join(target_cols), sqlite3_quote_name(cargs.table))
    dataset = self._bdb.execute(sql)

    # Compute the log-likelihood of the target_cols, subject to givens.
    # XXX This code is wrong due to shortcomings in BQL:
    #  - BQL cannot evaluate joint density. Assume that all the rows are IID,
    #  and that all the columns factor into their marginal density.
    #  - BQL cannot evaluate conditional density. This query will not compile.
    LL = 0
    for row in dataset:
        # XXX Wrong: assume joint factors into product of marginals.
        for col, val in zip(target_cols, row):
            # XXX Wrong: Code will crash, no conditional densities
            if givens:
                bql = '''
                ESTIMATE PROBABILITY OF {}=? FROM {} GIVEN {} LIMIT 1
                '''.format(col, sqlite3_quote_name(cargs.generator),
                    ','.join(['{}={}'.format(c,v) for (c,v) in givens]))
            else:
                bql = '''
                ESTIMATE PROBABILITY OF {}=? FROM {} LIMIT 1
                '''.format(col, sqlite3_quote_name(cargs.generator))

            crs = self._bdb.execute(bql, (val,))
            LL += math.log(crs.fetchall()[0][0])

    return LL
