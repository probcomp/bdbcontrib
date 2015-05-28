import shlex
import argparse
import math
import os
import shutil
import markdown2

from bdbcontrib.facade import do_query
from bdbcontrib.draw_cc_state import draw_state
from bayeslite.shell.hook import bayesdb_shell_cmd

import bdbcontrib.utils as utils
import bdbcontrib.plotutils as pu
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import seaborn as sns


ROOTDIR = os.path.dirname(os.path.abspath(__file__))
READTOHTML_CSS = os.path.join(ROOTDIR, 'readtohtml.css')


# Kludgey workaround for idiotic Python argparse module which exits
# the process on argument parsing failure.  We override the exit
# method of ArgumentParser so that it raises an exception instead of
# exiting the process, which we then catch around parser.parse_args()
# in order to report the message and return to the command loop.

class ArgparseError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message


class ArgumentParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        raise ArgparseError(status, message)


@bayesdb_shell_cmd('register_bql_math_functions')
def register_bql_math(self, args):
    '''Adds basic math functions to BQL'''
    self._bdb.sqlite3.create_function('exp', 1, math.exp)
    self._bdb.sqlite3.create_function('sqrt', 1, math.sqrt)
    self._bdb.sqlite3.create_function('pow', 2, pow)
    self._bdb.sqlite3.create_function('log', 1, math.log)


@bayesdb_shell_cmd('readtohtml')
def render_bql_as_html(self, argin):
    '''Reads a bql file and outputs to html and markdown.
    <bql_file> <output_directory>

    Example:
    bayeslite> .readtohtml myscript.bql analyses/myanalsis
    '''
    args = argin.split()
    bql_file = args[0]
    output_dir = args[1]

    head = '''
    <html>
        <head>
            <link rel="stylesheet" type="text/css" href="style.css">
        </head>
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(bql_file) as f:
        mdstr = utils.mdread(f, output_dir, self)

    html = head + markdown2.markdown(mdstr) + '</html>'

    htmlfilename = os.path.splitext(os.path.basename(bql_file))[0]
    htmlfilename = os.path.join(output_dir, htmlfilename + '.html')
    with open(htmlfilename, 'w') as f:
        f.write(html)
    shutil.copyfile(READTOHTML_CSS, os.path.join(output_dir, 'style.css'))

    # Because you want to know when it's done:
    self.stdout.write(utils.unicorn())
    self.stdout.write("Output saved to %s\n" % (output_dir,))


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
    parser = ArgumentParser(prog='.heatmap')
    parser.add_argument('bql', type=str, nargs='+', help='PAIRWISE BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)
    parser.add_argument('--last-sort', action='store_true')

    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()
    c = (df.shape[0]**.5)/4.0
    clustermap_kws = {'linewidths': 0, 'vmin': args.vmin, 'vmax': args.vmax}

    row_ordering = None
    col_ordering = None
    if args.last_sort:
        if not hasattr(self, 'hookvars'):
            raise AttributeError('No prior use if heatmap found.')
        else:
            row_ordering = self.hookvars.get('heatmap_row_ordering', None)
            col_ordering = self.hookvars.get('heatmap_col_ordering', None)
            if row_ordering is None or col_ordering is None:
                raise AttributeError('No prior use if heatmap found.')

        plt.figure(facecolor='white', figsize=(c, .8*c))
    res = pu.zmatrix(df, clustermap_kws=clustermap_kws,
                     row_ordering=row_ordering, col_ordering=col_ordering)

    # put the column and row orderings in the scope so they can be used by
    # --last-sort
    cm, row_ordering, col_ordering = res

    self.hookvars['heatmap_row_ordering'] = row_ordering
    self.hookvars['heatmap_col_ordering'] = col_ordering

    if args.filename is None:
        plt.show()
    else:
        if args.last_sort:
            plt.savefig(args.filename)
        else:
            cm.savefig(args.filename, figsize=(c, c))
    plt.close('all')


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
    parser = ArgumentParser(prog='.show')
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
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()
    c = len(df.columns)*4

    plt.figure(tight_layout=True, facecolor='white', figsize=(c, c))
    pu.pairplot(df, bdb=self._bdb, generator_name=args.generator,
                use_shortname=args.shortnames, no_contour=args.no_contour,
                colorby=args.colorby, show_missing=args.show_missing)

    if args.filename is None:
        plt.show()
    else:
        plt.savefig(args.filename)
    plt.close('all')


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
    plt.close('all')


@bayesdb_shell_cmd('histogram')
def histogram(self, argin):
    '''plots a histogram
    <query> [options]

    If the result of query has two columns, hist uses the second column to
    divide the data in the first column into sub-histograms.

    Example (plot overlapping histograms of height for males and females)
    bayeslite> .histogram SELECT height, sex FROM humans; --normed --bin 31
    '''

    parser = ArgumentParser(prog='.histogram')
    parser.add_argument('bql', type=str, nargs='+', help='BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    parser.add_argument('-b', '--bins', type=int, default=15,
                        help='number of bins')
    parser.add_argument('--normed', action='store_true',
                        help='Normalize histograms?')
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()
    pu.comparative_hist(df, nbins=args.bins, normed=args.normed)

    if args.filename is None:
        plt.show()
    else:
        plt.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('bar')
def barplot(self, argin):
    '''FIXME'''
    parser = ArgumentParser('.bar')
    parser.add_argument('bql', type=str, nargs='+', help='BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return
    bql = " ".join(args.bql)

    df = do_query(self._bdb, bql).as_df()

    c = df.shape[0]/2.0
    plt.figure(facecolor='white', figsize=(c, 5))
    plt.bar([x-.5 for x in range(df.shape[0])], df.ix[:, 1].values,
            color='#333333', edgecolor='#333333')

    ax = plt.gca()
    ax.set_xticks(range(df.shape[0]))
    ax.set_xticklabels(df.ix[:, 0].values, rotation=90)
    plt.xlim([-1, c-.5])
    plt.ylabel(df.columns[1])
    plt.xlabel(df.columns[0])
    plt.title(bql)

    if args.filename is None:
        plt.show()
    else:
        plt.savefig(args.filename)
    plt.close('all')


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
    plt.close('all')
