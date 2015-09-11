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
import shlex
import textwrap
from sys import platform as _platform

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import bayeslite.core
from bayeslite.shell.hook import bayesdb_shell_cmd

import bdbcontrib.api
from bdbcontrib.general_utils import ArgparseError, ArgumentParser

import bdbcontrib.plot_utils as pu
from bdbcontrib import crosscat_utils
from bdbcontrib.facade import do_query


matplotlib.rcParams.update({'figure.autolayout': True,
    'font.weight': 'bold',
    'figure.facecolor': 'white'})


@bayesdb_shell_cmd('mihist')
def mi_hist(self, argin):
    """Plot histogram of mutual information over models.

    <generator> <col1> <col2> [options]

    Example
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

    fig = bdbcontrib.api.mi_hist(self._bdb, args.generator, args.col1,
        args.col2, num_samples=args.num_samples, bins=args.bins)

    if args.filename is None:
        plt.show()
    else:
        fig.savefig(args.filename)
    plt.close('all')

@bayesdb_shell_cmd('heatmap')
def heatmap(self, argin):
    """Create a clustered heatmap from the BQL query.  Plot graphically
    by default, or to a file if `-f`/`--filename` is specified.

    <pairwise bql query> [options]

    Options
    -------
    -f, --filename: str
        Save as filename
    --vmin: float
        Minimun value of the colormap.
    --vmax: float
        Maximum value of the colormap.
    --last-sort: flag
        Sort the heatmap the same as the last heatmap. Used to compare heatmaps
        generated with different metrics. Must be used after heatmap has been
        run once on a table of the same size.

    Example (compare dependence probability and mutual information):
    bayeslite> .heatmap ESTIMATE PAIRWISE DEPENDENCE PROBABILITY FROM mytable
    bayeslite> .heatmap 'ESTIMATE PAIRWISE MUTUAL INFORMATION FROM mytable;' --last-sort
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
    clustermap = bdbcontrib.api.heatmap(self._bdb, bql, vmin=args.vmin,
        vmax=args.vmax)

    if args.filename is None:
        plt.show()
    else:
        clustermap.savefig(args.filename)
    plt.close('all')

@bayesdb_shell_cmd('show')
def pairplot(self, argin):
    """
    Plots continuous-continuous pairs as scatter (optional KDE contour)
    Plots continuous-categorical pairs as violinplot
    Plots categorical-categorical pairs as heatmap

    USAGE: .show <bql query> [options]

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
    figure = bdbcontrib.api.pairplot(self._bdb, bql,
        generator_name=args.generator, use_shortname=args.shortnames,
        show_contour=args.show_contour, colorby=args.colorby,
        show_missing=args.show_missing, show_full=args.show_full)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')


# TODO: better name
@bayesdb_shell_cmd('drawcc')
def draw_crosscat(self, argin):
    """
    Draw crosscat state.
    USAGE: .drawcc <generator> <modelno> [options]

    Options
        -f, --filename: the output filename. If not specified, tries to draw.

    Example:
    bayeslite> .ccstate mytable_cc 12 -f state_12.png
    """
    parser = ArgumentParser(prog='.ccstate')
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

    figure = bdbcontrib.api.draw_crosscat(self._bdb, table_name, args.generator,
        args.modelno, row_label_col=args.row_label_col)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('histogram')
def histogram(self, argin):
    """Plot histogram. If the result of query has two columns, hist uses
    the second column to divide the data in the first column into colored
    sub-histograms.

    USAGE: .histogram <query> [options]

    Example: (plots overlapping histograms of height for males and females)
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
    figure = bdbcontrib.api.histogram(self._bdb, bql, args.bins,
        args.normed)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('barplot')
def barplot(self, argin):
    """
    Bar plot of two-column query. Uses the first column of the query as the bar
    names and the second column as the bar heights. Ignores other columns.

    USAGE: .bar <query> [options]
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
    figure = bdbcontrib.api.barplot(self._bdb, bql)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')


@bayesdb_shell_cmd('chainplotcc')
def plot_crosscat_chain_diagnostics(self, argin):
    """Plot diagnostics for all models of generator.

    <diagnostic> <generator> [output_filename]

    Valid (crosscat) diagnostics
    are:
        - logscore: log score of the model
        - num_views: the number of views in the model
        - column_crp_alpha: CRP alpha over columns

    Example:
    bayeslite> .chainplot logscore dha_cc scoreplot.png
    """
    parser = ArgumentParser(prog='.bar')
    parser.add_argument('diagnostic', type=str, help='Diagnostic name')
    parser.add_argument('generator', type=str, help='Generator name.')
    parser.add_argument('-f', '--filename', type=str, default=None,
        help='output filename')
    try:
        args = parser.parse_args(shlex.split(argin))
    except ArgparseError as e:
        self.stdout.write('%s' % (e.message,))
        return

    figure = bdbcontrib.api.plot_crosscat_chain_diagnostics(self._bdb,
        args.diagnostic, args.generator)

    if args.filename is None:
        plt.show()
    else:
        figure.savefig(args.filename)
    plt.close('all')
