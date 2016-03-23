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

import matplotlib
import matplotlib.pyplot as plt

from bayeslite.shell.hook import bayesdb_shell_cmd

import bdbcontrib
from bdbcontrib.shell_utils import ArgparseError, ArgumentParser


matplotlib.rcParams.update({'figure.autolayout': True,
    'font.weight': 'bold',
    'figure.facecolor': 'white'})


@bayesdb_shell_cmd('mihist')
def mi_hist(self, argin):
    """plot histogram of mutual information over generator's models
    <generator> <col1> <col2> [<options>]

    Example:
    bayeslite> .mihist sat_generator dry_mass launch_mass -n=1000
    """
    parser = ArgumentParser(prog='.mihist')
    parser.add_argument('generator', type=str, help='generator name')
    parser.add_argument('col1', type=str, help='first column')
    parser.add_argument('col2', type=str, help='second column')
    parser.add_argument('-n', '--num-samples', type=int, default=1000)
    parser.add_argument('-b', '--bins', type=int, default=15,
                        help='number of bins')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')

    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    fig = bdbcontrib.mi_hist(self._bdb, args.generator, args.col1,
        args.col2, num_samples=args.num_samples, bins=args.bins)

    if args.filename is None:
        plt.show()
    else:
        fig.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('heatmap')
def heatmap(self, argin):
    """plot clustered heatmap of pairwise matrix
    <query> [<options>]

    Options
    -------
    -f, --filename: str
        Save as filename
    --vmin: float
        Minimun value of the colormap.
    --vmax: float
        Maximum value of the colormap.

    Example:
    bayeslite> .heatmap ESTIMATE DEPENDENCE PROBABILITY FROM PAIRWISE COLUMNS OF mytable
    """

    parser = ArgumentParser(prog='.heatmap')
    parser.add_argument('bql', type=str, nargs='+', help='PAIRWISE BQL query')
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)
    #XXX last-sort has been removed in favor of specifying row/col ordering.
    parser.add_argument('--last-sort', action='store_true')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')

    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bql = " ".join(args.bql)
    clustermap = bdbcontrib.heatmap(self._bdb, bql, vmin=args.vmin,
        vmax=args.vmax)

    if args.filename is None:
        plt.show()
    else:
        clustermap.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('show')
def pairplot(self, argin):
    """plot array of plots for all pairs of columns
    <query> [<options>]

    - continuous-continuous as scatter (optional KDE contour)
    - continuous-categorical as violinplot
    - categorical-categorical as heatmap

    Options:
        -f, --filename: the output filename. If not specified, tries
                        to draw.
        -g, --generator: the generator name. Providing the generator
                         name make help to plot more intelligently.
        -s, --shortnames: Use column short names to label facets?
        -m, --show-missing: Plot missing values in scatter plots as lines.
        -t, --no-tril: Plot the entire square matrix, not just lower triangle.
        --show-contour: Turn on contours.
        --colorby: The name of a column to use as a marker variable for color.

    Example:
    bayeslite> .show SELECT foo, baz, quux + glorb FROM mytable
        --filename myfile.png
    bayeslite> .show ESTIMATE foo, baz FROM mytable_cc -g mytable_cc
    """
    parser = ArgumentParser(prog='.show')
    parser.add_argument('bql', type=str, nargs='+', help='BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    parser.add_argument('-g', '--generator', type=str, default=None,
                        help='Query generator name')
    parser.add_argument('-s', '--shortnames', action='store_true',
                        help='Use column short names?')
    parser.add_argument('--no-contour', action='store_true',
                        help='Turn off countours (KDE).'),
    parser.add_argument('--show-contour', action='store_true',
                        help='Turn on contours (KDE).')
    parser.add_argument('-m', '--show-missing', action='store_true',
                        help='Plot missing values in scatterplot.')
    parser.add_argument('--show-full', action='store_true')
    parser.add_argument('--colorby', type=str, default=None,
                        help='Name of column to use as a dummy variable.')
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bql  = " ".join(args.bql)
    figure = bdbcontrib.pairplot(self._bdb, bql,
        generator_name=args.generator, show_contour=args.show_contour,
        colorby=args.colorby, show_missing=args.show_missing,
        show_full=args.show_full)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('drawcc')
def draw_crosscat(self, argin):
    """plot crosscat state
    <generator> <modelno> [<options>]

    Options
        -f, --filename: the output filename. If not specified, tries to draw.

    Example:
    bayeslite> .ccstate mytable_cc 12 -f state_12.png
    """
    parser = ArgumentParser(prog='.drawcc')
    parser.add_argument('generator', type=str, help='Generator')
    parser.add_argument('modelno', type=int, help='Model number to plot.')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    parser.add_argument('-r', '--row-label-col', type=str, default=None,
                        help='The name of the column to use for row labels.')
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    figure = bdbcontrib.draw_crosscat(self._bdb, args.generator,
        args.modelno, row_label_col=args.row_label_col)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('histogram')
def histogram(self, argin):
    """plot histogram of one- or two-column table
    <query> [<options>]

    If two-column, subdivide the first column according to labels in
    the second column.

    Example (plots overlapping histograms of height for males and females):
    bayeslite> .histogram SELECT height, sex FROM humans; --normed --bin 31
    """
    parser = ArgumentParser(prog='.histogram')
    parser.add_argument('bql', type=str, nargs='+', help='BQL query')
    parser.add_argument('-b', '--bins', type=int, default=15,
                        help='number of bins')
    parser.add_argument('--normed', action='store_true',
                        help='Normalize histograms?')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bql = " ".join(args.bql)
    figure = bdbcontrib.histogram(self._bdb, bql, args.bins,
        args.normed)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('bar')
def barplot(self, argin):
    """plot bar-plot of query giving categories and heights
    <query> [<options>]

    First column specifies names; second column specifies heights.
    """
    parser = ArgumentParser(prog='.bar')
    parser.add_argument('bql', type=str, nargs='+', help='BQL query')
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help='output filename')
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    bql = " ".join(args.bql)
    figure = bdbcontrib.barplot(self._bdb, bql)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('chainplot')
def plot_crosscat_chain_diagnostics(self, argin):
    """plot crosscat diagnostic for all models of generator
    <diagnostic> <generator> [output_filename]

    Valid (crosscat) diagnostics are:
        - logscore: log score of the model
        - num_views: the number of views in the model
        - column_crp_alpha: CRP alpha over columns

    Example:
    bayeslite> .chainplot logscore dha_cc scoreplot.png
    """
    parser = ArgumentParser(prog='.chainplot')
    parser.add_argument('diagnostic', type=str, help='Diagnostic name')
    parser.add_argument('generator', type=str, help='Generator name.')
    parser.add_argument('-f', '--filename', type=str, default=None,
        help='output filename')
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    figure = bdbcontrib.plot_crosscat_chain_diagnostics(self._bdb,
        args.diagnostic, args.generator)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')
