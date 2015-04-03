import shlex
import argparse

from bdbcontrib.facade import do_query
from bdbcontrib.draw_cc_state import draw_state
from bayeslite.shell.hook import bayesdb_shell_cmd

import bdbcontrib.plotutils as pu
import matplotlib.pyplot as plt


@bayesdb_shell_cmd('zmatrix')
def zmatrix(self, argin):
    '''Creates a z-matrix from the bql query.
    <filename.png>

    Creates a z-matrix from the bql query. If no filename is specified, will
    attempt to draw.

    Example:
    bayeslite> .zmatrix ESTIMATE PAIRWISE DEPENDENCE PROBABILITY FROM mytable
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('bql', type=str, nargs='+', help='PAIRWISE BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,  help='output filename')
    args = parser.parse_args(shlex.split(argin))

    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()

    plt.figure(tight_layout=True, facecolor='white')
    cm = pu.zmatrix(df)

    if args.filename is None:
        plt.show()
    else:
        cm.savefig(args.filename)


@bayesdb_shell_cmd('pairplot')
def pairplot(self, argin):
    parser = argparse.ArgumentParser()
    parser.add_argument('bql', type=str, nargs='+', help='BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,  help='output filename')
    parser.add_argument('-g', '--generator', type=str, default=None,  help='Query generator name')
    parser.add_argument('-s', '--shortnames', action='store_true',  help='Use column short names?')
    args = parser.parse_args(shlex.split(argin))

    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()

    plt.figure(tight_layout=True, facecolor='white')
    pu.pairplot(df, bdb=self._bdb, generator_name=args.generator, use_shortname=args.shortnames)

    if args.filename is None:
        plt.show()
    else:
        plt.savefig(args.filename)


# TODO: better name
@bayesdb_shell_cmd('ccstate')
def draw_crosscat_state(self, argin):
    '''Draws the crosscat state
    <generator> <modelno> <filename.png>
    '''

    args = argin.split()
    generator_name = args[0]
    try:
        modelno = float(args[1])
    except TypeError:
        raise TypeError('modelno must be an integer')

    filename = None
    if len(args) == 3:
        filename = args[2]

    bql = 'SELECT tabname, metamodel FROM bayesdb_generator WHERE name={}'.format(generator_name)
    table_name, metamodel = do_query(self._bdb, bql).as_cursor().fetchall()[0]

    if metamodel.lower() != 'crosscat':
        raise ValueError('Metamodel for generator %s (%s) should be crosscat' %
                         (generator_name, metamodel))

    plt.figure(tight_layout=True, facecolor='white')
    draw_state(self._bdb, table_name, generator_name, modelno)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
