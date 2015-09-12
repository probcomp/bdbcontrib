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

import textwrap

import matplotlib.pyplot as plt
import seaborn as sns

import bayeslite.core
from bayeslite.sqlite3_util import sqlite3_quote_name as quote

from bdbcontrib import crosscat_utils
from bdbcontrib import diagnostic_utils
from bdbcontrib import general_utils
from bdbcontrib import plot_utils
from bdbcontrib.facade import do_query


def mi_hist(bdb, generator, col1, col2, num_samples=1000, bins=5):
    """Plots a histogram of the mutual information over the models of a
    generator.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    generator : str
        Name of the generator to compute MI(col1;col2)
    col1, col2 : str
        Name of the columns to compute MI(col1;col2)
    num_samples : int, optional
        Number of samples to use in the Monte Carlo estimate of MI.
    bins : int, optional
        Number of bins in the histogram.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator)
    bql = '''
        SELECT COUNT(modelno) FROM bayesdb_generator_model
        WHERE generator_id = ?
        '''
    c = bdb.execute(bql, (generator_id,))
    num_models = c.fetchall()[0][0]

    figure, ax = plt.subplots(figsize=(6, 6))

    mis = []
    for modelno in range(num_models):
        bql = '''
            ESTIMATE MUTUAL INFORMATION OF {} WITH {} USING {} SAMPLES FROM {}
            USING MODEL {} LIMIT 1;
            '''.format(col1, col2, num_samples, generator, modelno)
        c = bdb.execute(bql)
        mi = c.fetchall()[0][0]
        mis.append(mi)
    ax.hist(mis, bins, normed=True)
    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Density')
    ax.set_title('Mutual information of {} with {}'.format(col1, col2))

    return figure


def heatmap(bdb, bql, vmin=None, vmax=None, row_ordering=None,
        col_ordering=None):
    """Create a clustered heatmap from the BQL query.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    bql : str
        The BQL to run and plot. Must be a PAIRWISE BQL query.
    vmin: float
        Minimun value of the colormap.
    vmax: float
        Maximum value of the colormap.
    row_ordering, col_ordering: list<int>
        Specify the order of labels on the x and y axis of the heatmap. To
        access the row and column indices from a clustermap object, use:
        clustermap.dendrogram_row.reordered_ind  (for rows)
        clustermap.dendrogram_col.reordered_ind (for cols)

    Returns
    -------
    clustermap: seaborn.clustermap
    """
    df = do_query(bdb, bql).as_df()
    df.fillna(0, inplace=True)
    c = (df.shape[0]**.5)/4.0
    clustermap_kws = {'linewidths': 0.2, 'vmin': vmin, 'vmax': vmax,
        'figsize':(c, .8*c)}

    clustermap = plot_utils.zmatrix(df, clustermap_kws=clustermap_kws,
        row_ordering=row_ordering, col_ordering=col_ordering)

    return clustermap


def pairplot(bdb, bql, generator_name=None, show_contour=False, colorby=None,
        show_missing=False, show_full=False):
    """Perform a pairplot of the table returned by the bql query.

    Plots continuous-continuous pairs as scatter (optional KDE contour).
    Plots continuous-categorical pairs as violinplot.
    Plots categorical-categorical pairs as heatmap.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    bql : str
        The BQL to run and pairplot.
    generator_name : str, optional
        The name of generator; explicitly passing in provides optimizations.
    show_contour : bool, optional
        If True, KDE contours are plotted on top of scatter plots
        and histograms.
    show_missing : bool, optional
        If True, rows with one missing value are plotted as lines on scatter
        plots.
    colorby : str, optional
        Name of a column to use to color data points in histograms and scatter
        plots.
    show_full : bool, optional
        Show full pairwise plots, rather than only lower triangular plots.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    df = do_query(bdb, bql).as_df()
    c = len(df.columns)*4
    figure = plot_utils.pairplot(df, bdb=bdb, generator_name=generator_name,
        show_contour=show_contour, colorby=colorby, show_missing=show_missing,
        show_full=show_full)
    figure.tight_layout()
    figure.set_size_inches((c,c))

    return figure


def draw_crosscat(bdb, generator, modelno, row_label_col=None):
    """Draw crosscat model from the specified generator.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    generator_name : str
        Name of generator.
    modelno: int
        Number of model to draw.

    Returns
    ----------
    figure: matplotlib.figure.Figure
    """
    bql = '''
        SELECT tabname, metamodel FROM bayesdb_generator
        WHERE name = ?
        '''
    table_name, metamodel = do_query(
        bdb, bql, (generator,)).as_cursor().fetchall()[0]

    if metamodel.lower() != 'crosscat':
        raise ValueError('Metamodel for generator %s (%s) should be crosscat' %
            (generator, metamodel))

    figure, axes = plt.subplots(tight_layout=False)
    crosscat_utils.draw_state(bdb, table_name, generator,
        modelno, ax=axes, row_label_col=row_label_col)

    return figure

def histogram(bdb, bql, bins=15, normed=None):
    """Plot histogram. If the result of query has two columns, hist uses
    the second column to divide the data in the first column into colored
    sub-histograms.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    bql : str
        The BQL to run and histogram.
    bins : int, optional
        Number of bins in the histogram.
    normed : bool, optional
        Normalize the histograms?

    Returns
    ----------
    figure: matplotlib.figure.Figure
    """
    df = do_query(bdb, bql).as_df()
    figure = plot_utils.comparative_hist(df, bdb=bdb, nbins=bins, normed=normed)

    return figure


def barplot(bdb, bql):
    """Bar plot of two-column query.

    Uses the first column of the query as the bar names and the second column as
    the bar heights. Ignores other columns.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    bql : str
        The BQL to run and histogram.

    Returns
    ----------
    figure: matplotlib.figure.Figure
    """
    df = do_query(bdb, bql).as_df()

    c = df.shape[0]/2.0
    figure, ax = plt.subplots(figsize=(c, 5))

    ax.bar([x-.5 for x in range(df.shape[0])], df.ix[:, 1].values,
        color='#333333', edgecolor='#333333')
    ax.set_xticks(range(df.shape[0]))
    ax.set_xticklabels(df.ix[:, 0].values, rotation=90)
    ax.set_xlim([-1, df.shape[0]-.5])
    ax.set_ylabel(df.columns[1])
    ax.set_xlabel(df.columns[0])
    ax.set_title('\n    '.join(textwrap.wrap(bql, 80)))

    return figure


def plot_crosscat_chain_diagnostics(bdb, diagnostic, generator):
    """
    Plot diagnostics for all models of generator.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    diagnostic : str
        Valid (crosscat) diagnostics are:
            - logscore: log score of the model
            - num_views: the number of views in the model
            - column_crp_alpha: CRP alpha over columns
    generator : str
        Name of the generator to diagnose.

    Returns
    ----------
    figure: matplotlib.figure.Figure
    """
    valid_diagnostics = ['logscore', 'num_views', 'column_crp_alpha']
    if diagnostic not in valid_diagnostics:
        raise ValueError('I do not know what to do with %s.\n'
                    'Please chosse one of the following instead: %s\n'
                    % ', '.join(valid_diagnostics))

    generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator)

    # Get model numbers. Do not rely on there to be a diagnostic for every
    # model.
    bql = '''
        SELECT modelno, COUNT(modelno) FROM bayesdb_crosscat_diagnostics
        WHERE generator_id = ? GROUP BY modelno
        '''
    df = do_query(bdb, bql, (generator_id,)).as_df()
    models = df['modelno'].astype(int).values

    figure, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
    colors = sns.color_palette("GnBu_d", len(models))
    for i, modelno in enumerate(models):
        bql = '''
            SELECT {}, iterations FROM bayesdb_crosscat_diagnostics
            WHERE modelno = ? AND generator_id = ?
            ORDER BY iterations ASC
            '''.format(diagnostic)
        df = do_query(bdb, bql, (modelno, generator_id,)).as_df()
        ax.plot(df['iterations'].values, df[diagnostic].values,
                 c=colors[modelno], alpha=.7, lw=2)

        ax.text(df['iterations'].values[-1], df[diagnostic].values[-1],
                str(modelno), color=colors[i])

    ax.set_xlabel('Iteration')
    ax.set_ylabel(diagnostic)
    ax.set_title('%s for each model in %s' % (diagnostic, generator,))

    return figure


def cardinality(bdb, table, cols=None):
    """Compute the number of unique values in the columns of a table.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    table : str
        Name of table.
    cols : list<str>, optional
        Columns to compute the unique values. Defaults to all.

    Returns
    -------
    counts : list<tuple<str,int>>
        A list of tuples of the form [(col_1, cardinality_1), ...]
    """
    # If no columns specified, use all.
    if not cols:
        sql = 'PRAGMA table_info(%s)' % (quote(table),)
        res = bdb.sql_execute(sql)
        cols = [r[1] for r in res.fetchall()]

    counts = []
    for col in cols:
        sql = '''
            SELECT COUNT (DISTINCT %s) FROM %s
            ''' % (quote(col), quote(table))
        res = bdb.sql_execute(sql)
        counts.append((col, res.next()[0]))

    return counts


#XXX These functions are exposed in __init__.py Adding here for completelness.
nullify = general_utils.nullify
estimate_log_likelihood = diagnostic_utils.estimate_log_likelihood
estimate_kl_divergence = diagnostic_utils.estimate_kl_divergence
