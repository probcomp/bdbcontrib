import shlex
import argparse

from bdbcontrib.facade import do_query
from bdbcontrib.draw_cc_state import draw_state
from bayeslite.shell.hook import bayesdb_shell_cmd

import bdbcontrib.utils as utils
import bdbcontrib.plotutils as pu
import matplotlib.pyplot as plt


@bayesdb_shell_cmd('nullify')
def nullify(self, argin):
    '''Replaces a user specified missing value with NULL
    <table> <value>

    Example:
    bayeslite> .nullify mytable NaN
    bayeslite> .nullify mytable ''
    '''
    args = argin.split()
    table = args[0]
    value = args[1]
    utils.nullify(self._bdb, table, value)


@bayesdb_shell_cmd('heatmap')
def zmatrix(self, argin):
    '''Creates a clustered heatmap from the bql query.
    <pairwise bql query> [options]

    Creates a clustered heatmap from the bql query. If no filename is
    specified, will attempt to draw.

    Example:
    bayeslite> .heatmap ESTIMATE PAIRWISE DEPENDENCE PROBABILITY FROM mytable
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('bql', type=str, nargs='+', help='PAIRWISE BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)

    args = parser.parse_args(shlex.split(argin))

    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()
    clustermap_kws = {'linewidths': 0, 'vmin': args.vmin, 'vmax': args.vmax}
    cm = pu.zmatrix(df, clustermap_kws=clustermap_kws)

    if args.filename is None:
        plt.show()
    else:
        cm.savefig(args.filename)


@bayesdb_shell_cmd('show')
def pairplot(self, argin):
    '''Plots pairs of columns in facet grid.
    <bql query> [options]

    Plots continuous-continuous pairs as scatter w/ KDE contour
    Plots continuous-categorical pairs as violinplot
    Plots categorical-categorical pairs as heatmap

    Options:
        -f. --filename: the output filename. If not specified, tries
                        to draw.
        -g, --generator: the generator name. Providing the generator
                         name make help to plot more intelligently.
        -s, --shortnames: Use column short names to label facets?
        -m, --show-missing: Plot missing values in scatter plots as lines.
        --no-contours: Turn off contours.
        --colorby: The name of a column to use as a dummy variable for color.

    Example:
    bayeslite> .show SELECT foo, baz, quux + glorb FROM mytable
        --filename myfile.png
    bayeslite> .show ESTIMATE foo, baz FROM mytable_cc -g mytable_cc
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('bql', type=str, nargs='+', help='BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    parser.add_argument('-g', '--generator', type=str, default=None,
                        help='Query generator name')
    parser.add_argument('-s', '--shortnames', action='store_true',
                        help='Use column short names?')
    parser.add_argument('--no-contour', action='store_true',
                        help='Turn off countours (KDE).')
    parser.add_argument('-m', '--show-missing', action='store_true',
                        help='Plot missing values in scatterplot.')
    parser.add_argument('--colorby', type=str, default=None,
                        help='Name of column to use as a dummy variable.')
    args = parser.parse_args(shlex.split(argin))

    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()

    plt.figure(tight_layout=True, facecolor='white')
    pu.pairplot(df, bdb=self._bdb, generator_name=args.generator,
                use_shortname=args.shortnames, no_contour=args.no_contour,
                colorby=args.colorby, show_missing=args.show_missing)

    if args.filename is None:
        plt.show()
    else:
        plt.savefig(args.filename)


# TODO: better name
@bayesdb_shell_cmd('ccstate')
def draw_crosscat_state(self, argin):
    '''Draws the crosscat state.
    <generator> <modelno> [filename.png]

    If no file name is provided, will attempt to draw.

    Example:
    bayeslite> .ccstate mytable_cc 12 state_12.png
    '''

    args = argin.split()
    generator_name = args[0]
    try:
        modelno = int(args[1])
    except TypeError:
        raise TypeError('modelno must be an integer')

    filename = None
    if len(args) == 3:
        filename = args[2]

    bql = 'SELECT tabname, metamodel FROM bayesdb_generator WHERE name = ?'
    table_name, metamodel = do_query(
        self._bdb, bql, (generator_name,)).as_cursor().fetchall()[0]

    if metamodel.lower() != 'crosscat':
        raise ValueError('Metamodel for generator %s (%s) should be crosscat' %
                         (generator_name, metamodel))

    plt.figure(tight_layout=False, facecolor='white')
    draw_state(self._bdb, table_name, generator_name, modelno)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


@bayesdb_shell_cmd('histogram')
def histogram(self, argin):
    '''plots a histogram
    <query> [options]

    If the result of query has two columns, hist uses the second column to
    divide the data in the first column into sub-histograms.

    Example (plot overlapping histograms of height for males and females)
    bayeslite> .histogram SELECT height, sex FROM humans; --normed --bin 31
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('bql', type=str, nargs='+', help='BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    parser.add_argument('-b', '--bins', type=int, default=15,
                        help='number of bins')
    parser.add_argument('--normed', action='store_true',
                        help='Normalize histograms?')
    args = parser.parse_args(shlex.split(argin))

    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()
    pu.comparative_hist(df, nbins=args.bins, normed=args.normed)

    if args.filename is None:
        plt.show()
    else:
        plt.savefig(args.filename)


@bayesdb_shell_cmd('chainplot')
def plot_crosscat_chain_diagnostics(self, argin):
    ''' Plots diagnostics for each model of the specified generator
    <diagnostic> <generator> [output_filename]

    Valid (crosscat) diagnostics are
        - logscore: log score of the model
        - num_views: the number of views in the model
        - column_crp_alpha: CRP alpha over columns

    Example:
        bayeslite> .chainplot logscore dha_cc scoreplot.png
    '''

    args = argin.split()
    if len(args) < 2:
        self.stdout.write("Please specify a diagnostic and a generator.\n")
        return

    import seaborn as sns
    import bayeslite.core

    diagnostic = args[0]
    generator_name = args[1]
    filename = None
    if len(args) == 3:
        filename = args[2]

    valid_diagnostics = ['logscore', 'num_views', 'column_crp_alpha']
    if diagnostic not in valid_diagnostics:
        self.stdout('I do not know what to do with %s.\n'
                    'Please chosse one of the following instead: %s\n'
                    % ', '.join(valid_diagnostics))

    generator_id = bayeslite.core.bayesdb_get_generator(self._bdb,
                                                        generator_name)

    # get model numbers. Do not rely on there to be a diagnostic for every
    # model
    bql = '''SELECT modelno, COUNT(modelno) FROM bayesdb_crosscat_diagnostics
                WHERE generator_id = ?
                GROUP BY modelno'''
    df = do_query(self._bdb, bql, (generator_id,)).as_df()
    models = df['modelno'].astype(int).values

    plt.figure(tight_layout=True, facecolor='white', figsize=(10, 5))
    ax = plt.gca()
    colors = sns.color_palette("GnBu_d", len(models))
    for i, modelno in enumerate(models):
        bql = '''SELECT {}, iterations FROM bayesdb_crosscat_diagnostics
                    WHERE modelno = ? AND generator_id = ?
                    ORDER BY iterations ASC
                '''.format(diagnostic)
        df = do_query(self._bdb, bql, (modelno, generator_id,)).as_df()
        plt.plot(df['iterations'].values, df[diagnostic].values,
                 c=colors[modelno], alpha=.7, lw=2)

        ax.text(df['iterations'].values[-1], df[diagnostic].values[-1],
                str(modelno), color=colors[i])

    plt.xlabel('Iteration')
    plt.ylabel(diagnostic)
    plt.title('%s for each model in %s' % (diagnostic, generator_name,))

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
