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

from contextlib import contextmanager
import copy
from textwrap import wrap

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import bayeslite.core
from bayeslite.exception import BayesLiteException as BLE
import bdbcontrib.bql_utils as bqlu
from bdbcontrib.population_method import population_method

###############################################################################
###                                   PUBLIC                                ###
###############################################################################

@population_method(population_to_bdb=0, generator_name=1)
def mi_hist(bdb, generator_name, col1, col2, num_samples=1000, bins=5):
    """
    Histogram of estimated mutual information between the two columns for each
    of a generator's model instances.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    generator_name : str
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
    generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator_name)
    bql = '''
        SELECT COUNT(modelno) FROM bayesdb_generator_model
            WHERE generator_id = ?
    '''
    counts = bdb.execute(bql, (generator_id,))
    num_models = counts.fetchvalue()

    figure, ax = plt.subplots(figsize=(6, 6))

    mis = []
    for modelno in range(num_models):
        bql = '''
            ESTIMATE MUTUAL INFORMATION OF {} WITH {} USING {} SAMPLES FROM {}
                USING MODEL {} LIMIT 1
        '''.format(col1, col2, num_samples, generator_name, modelno)
        cursor = bdb.execute(bql)
        mutual_information = cursor.fetchvalue()
        mis.append(mutual_information)
    ax.hist(mis, bins, normed=True)
    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Density')
    ax.set_title('Mutual information of {} with {}'.format(col1, col2))

    return figure

@population_method(population_to_bdb=0, specifier_to_df=1)
def heatmap(bdb, deps, **kwargs):
    '''Plot clustered heatmaps for the given dependencies.

    Parameters
    ----------
    bdb : __population_to_bdb__
    deps : __specifier_to_df__
        Must have columns=['generator_id', 'name0', 'name1', 'value'],
        I.e. the result of a BQL query of the form 'ESTIMATE ... PAIRWISE ...'.
        E.g., DEPENDENCE PROBABILITY, MUTUAL INFORMATION, COVARIANCE, etc.

    **kwargs : dict
        Passed to zmatrix: vmin, vmax, row_ordering, col_ordering

    Returns a seaborn.clustermap (a kind of matplotlib.Figure)
    '''
    return zmatrix(deps, vmin=0, vmax=1, **kwargs)

def selected_heatmaps(bdb, df, selector_fns, **kwargs):
    """Yield heatmaps of pairwise matrix, broken up according to selectors.

    Parameters
    ----------
    bdb: a bayeslite.BayesDB instance
    df: the result of a PAIRWISE query.
    selectors : [lambda name --> bool]
        Rather than plot the full NxN matrix all together, make separate plots
        for each combination of these selectors, plotting them in sequence.
        If selectors are specified, yields clustermaps, which caller is
        responsible for showing or saving, and then closing.
    **kwargs : dict
        Passed to zmatrix: vmin, vmax, row_ordering, col_ordering

    Yields
    ------
    The triple (clustermap, selector1, selector2).  It is recommended that
    caller keep a dict of these functions to names to help identify each one.
    """
    # Cannot specify neither or both.
    df.fillna(0, inplace=True)
    for n0selector in selector_fns:
        n0selection = df.iloc[:, 1].map(n0selector)
        for n1selector in selector_fns:
            n1selection = df.iloc[:, 2].map(n1selector)
            this_block = df[n0selection & n1selection]
            if len(this_block) > 0:
                yield (zmatrix(this_block, vmin=0, vmax=1, **kwargs),
                       n0selector, n1selector)

@population_method(population_to_bdb=0, specifier_to_df=1,
                   generator_name='generator_name')
def pairplot(bdb, df, generator_name=None, show_contour=False, colorby=None,
        show_missing=False, show_full=False, **kwargs):
    """Plot array of plots for all pairs of columns.

    Plots continuous-continuous pairs as scatter (optional KDE contour).
    Plots continuous-categorical pairs as violinplot.
    Plots categorical-categorical pairs as heatmap.

    Parameters
    ----------
    bdb : __population_to_bdb__
    df : __specifier_to_df__
    generator_name : __generator_name__
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
    **kwargs : dict, optional
        Options to pass through to underlying plotter for pairs.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    figure = _pairplot(df, bdb=bdb, generator_name=generator_name,
        show_contour=show_contour, colorby=colorby, show_missing=show_missing,
        show_full=show_full, **kwargs)
    padding = -0.4 * (df.shape[1] - 2)
    figure.tight_layout(h_pad=padding, pad=padding)
    inches = len(df.columns) * 4
    figure.set_size_inches((inches, inches))

    return figure

@population_method(population_to_bdb=0, specifier_to_df=1)
def histogram(bdb, df, nbins=15, bins=None, normed=None):
    """Plot histogram of one- or two-column table.

    If two-column, subdivide the first column according to labels in
    the second column

    Parameters
    ----------
    bdb : __population_to_bdb__
    df : __specifier_to_df__
    nbins : int, optional
        Number of bins in the histogram.
    normed : bool, optional
        If True, normalizes the the area of the histogram (or each
        sub-histogram if df has two columns) to 1.

    Returns
    ----------
    figure: matplotlib.figure.Figure
    """
    df = df.dropna()
    if len(df.columns) == 0:
        raise BLE(ValueError('Tried to plot a histogram of an empty result.'))

    vartype = get_bayesdb_col_type(df.columns[0], df[df.columns[0]], bdb=bdb)
    if vartype == 'categorical':
        raise BLE(TypeError(
            "Cannot histogram categorical varible %s. Barplot? Colorby?" %
            (df.columns[0],)))
    a = min(df.ix[:, 0].values)
    b = max(df.ix[:, 0].values)
    support = b - a
    interval = support/nbins
    if bins is None:
        bins = np.linspace(a, b+interval, nbins)

    colorby = None
    if len(df.columns) > 1:
        if len(df.columns) > 2:
            raise BLE(ValueError('Got more columns than data and colorby.'))
        colorby = df.columns[1]
        colorby_stattype = get_bayesdb_col_type(
            df.columns[1], df[df.columns[1]], bdb=bdb)
        if colorby_stattype != 'categorical':
            raise BLE(TypeError("Cannot color by non-categorical variable " +
                                colorby))
        colorby_vals = df[colorby].unique()

    figure, ax = plt.subplots(tight_layout=False, facecolor='white')
    if colorby is None:
        ax.hist(df.ix[:, 0].values, bins=bins, color='#383838',
            edgecolor='none', normed=normed)
        plot_title = df.columns[0]
    else:
        colors = sns.color_palette('deep', len(colorby_vals))
        for color, cbv in zip(colors, colorby_vals):
            subdf = df[df[colorby] == cbv]
            ax.hist(subdf.ix[:, 0].values, bins=bins, color=color, alpha=.5,
                edgecolor='none', normed=normed, label=str(cbv))
        ax.legend(loc=0, title=colorby)
        plot_title = df.columns[0] + " by " + colorby

    if normed:
        plot_title += " (normalized)"

    ax.set_title(plot_title)
    ax.set_xlabel(df.columns[0])
    return figure

@population_method(population_to_bdb=0, specifier_to_df=1)
def barplot(bdb, df):
    """Make bar-plot from categories and their heights.

    First column specifies names; second column specifies heights.

    Parameters
    ----------
    bdb : __population_to_bdb__
    df : __specifier_to_df__

    Returns
    ----------
    figure: matplotlib.figure.Figure
    """
    if df.shape[1] != 2:
        raise BLE(ValueError(
            'Need two columns of output from SELECT for barplot.'))
    height_inches = df.shape[0] / 2.0
    figure, ax = plt.subplots(figsize=(height_inches, 5))

    ax.bar([x - .5 for x in range(df.shape[0])], df.ix[:, 1].values,
        color='#333333', edgecolor='#333333')
    ax.set_xticks(range(df.shape[0]))
    ax.set_xticklabels(df.ix[:, 0].values, rotation=90)
    ax.set_xlim([-1, df.shape[0] - .5])
    ax.set_ylabel(df.columns[1])
    ax.set_xlabel(df.columns[0])

    return figure

###############################################################################
###                              INTERNAL                                   ###
###############################################################################

MODEL_TO_TYPE_LOOKUP = {
    'normal_inverse_gamma': 'numerical',
    'symmetric_dirichlet_discrete': 'categorical',
}


def rotate_tick_labels(ax, axis='x', rotation=90):
    if axis.lower() == 'x':
        _, labels = ax.get_xticks()
    elif axis.lower() == 'y':
        _, labels = ax.get_yticks()
    else:
        raise BLE(ValueError('axis must be x or y'))
    plt.setp(labels, rotation=rotation)


def gen_collapsed_legend_from_dict(hl_colors_dict, loc=0, title=None,
        fontsize='medium', wrap_threshold=1000):
    """Creates a legend with entries grouped by color.

    For example, if a plot has multiple labels associated with the same color
    line, instead of generating a legend entry for each label, labels with the
    same colored line will be collapsed into longer, comma-separated labels.

    Parameters
    ----------
    hl_colors_dict : dict
        A dict of label, color pairs. Colors can be strings e.g. 'deeppink' or
        rgb or rgba tuples.
    loc : matplotlib compatible
        any matpltotlib-compbatible legend location identifier
    title : str
        legend title
    fontsize : int
        legend entry and title fontsize
    wrap_threshold : int
        max number of characters before wordwrap

    Returns
    -------
    legend : matplotlib.legend
    """
    if not isinstance(hl_colors_dict, dict):
        raise BLE(TypeError("hl_colors_dict must be a dict"))

    colors = list(set(hl_colors_dict.values()))
    from collections import defaultdict
    collapsed_dict = defaultdict(list)

    for label, color in hl_colors_dict.iteritems():
        collapsed_dict[color].append(str(label))

    for color in collapsed_dict.keys():
        collapsed_dict[color] = "\n".join(wrap(", ".join(
            sorted(collapsed_dict[color])), wrap_threshold))

    legend_artists = []
    legend_labels = []
    for color, label in collapsed_dict.iteritems():
        legend_artists.append(plt.Line2D((0, 1), (0, 0), color=color, lw=3))
        legend_labels.append(label)

    legend = plt.legend(legend_artists, legend_labels, loc=loc, title=title,
        fontsize=fontsize)

    return legend


def get_bayesdb_col_type(column_name, df_column, bdb=None,
                         generator_name=None):
    # If column_name is a column label (not a short name!) then the modeltype
    # of the column will be returned otherwise we guess.

    if isinstance(df_column, pd.DataFrame):
        raise BLE(TypeError(
            'Multiple columns in the query result have the same name (%s).' %
            (column_name,)))

    def guess_column_type(df_column):
        pd_type = df_column.dtype
        if pd_type is str or pd_type == np.object:
            return 'categorical'
        else:
            if len(df_column.unique()) < 30:
                return 'categorical'
            else:
                return 'numerical'

    if bdb is not None and generator_name is not None:
        try:
            coltype = bqlu.get_column_stattype(bdb, generator_name, column_name)
            # XXX: Force cyclic -> numeric because there is no need to plot
            # cyclic data any differently until we implement rose plots. See
            # http://matplotlib.org/examples/pie_and_polar_charts/polar_bar_demo.html
            # for an example.
            if coltype.lower() == 'cyclic':
                coltype = 'numerical'
            return coltype
        except IndexError:
            return guess_column_type(df_column)
    else:
        return guess_column_type(df_column)


def conv_categorical_vals_to_numeric(data_srs):
    # TODO: get real valuemap from btable
    unique_vals = sorted(data_srs.unique().tolist())
    lookup = dict(zip(unique_vals, range(len(unique_vals))))
    values = data_srs.values.tolist()
    for i, x in enumerate(values):
        values[i] = lookup[x]
    return np.array(values, dtype=float), unique_vals, lookup


# FIXME: STUB
def prep_plot_df(data_df, var_names):
    return data_df[list(var_names)]

def drop_inf_and_nan(np_Series):
    return np_Series.replace([-np.inf, np.inf], np.nan).dropna()

def seaborn_broken_bins(np_Series):
    # XXX: Seaborn is broken, uses np.asarray(values).squeeze()
    # which for lists of one is an "unsized object" from which you
    # cannot calculate freedman diaconis bins.
    if len(np_Series) == 1:
        return []
    # When seaborn uses its _freedman_diaconis_bins it forgets to np.ceil,
    # so do that for it. Again (this has been broken and fixed before). :-p
    # This is being inserted at seaborn==0.6.0.
    return np.ceil(sns.distributions._freedman_diaconis_bins(np_Series))

def do_hist(data_srs, **kwargs):
    ax = kwargs.get('ax', None)
    bdb = kwargs.get('bdb', None)
    dtype = kwargs.get('dtype', None)
    generator_name = kwargs.get('generator_name', None)
    colors = kwargs.get('colors', None)

    if dtype is None:
        dtype = get_bayesdb_col_type(data_srs.columns[0], data_srs, bdb=bdb,
            generator_name=generator_name)

    if ax is None:
        ax = plt.gca()

    if len(data_srs.shape) > 1:
        if colors is None and data_srs.shape[1] != 1:
            raise BLE(ValueError(
                'If a dummy column is specified,'
                ' colors must also be specified.'))

    data_srs = data_srs.dropna()

    if dtype == 'categorical':
        vals, uvals, _ = conv_categorical_vals_to_numeric(data_srs.ix[:, 0])
        if colors is not None:
            color_lst = []
            stacks = []
            for val, color in colors.iteritems():
                subval = vals[data_srs.ix[:, 1].values == val]
                color_lst.append(color)
                stacks.append(subval)
            ax.hist(stacks, bins=len(uvals), color=color_lst, alpha=.9,
                histtype='barstacked', rwidth=1.0)
        else:
            ax.hist(vals, bins=len(uvals))
        ax.set_xticks(range(len(uvals)))
        ax.set_xticklabels(uvals)
    else:
        do_kde = True
        if colors is not None:
            for val, color in colors.iteritems():
                subdf = data_srs.loc[data_srs.ix[:, 1] == val]
                values = drop_inf_and_nan(subdf.ix[:, 0])
                bins = seaborn_broken_bins(values)
                sns.distplot(values, kde=do_kde, ax=ax, color=color, bins=bins)
        else:
            values = drop_inf_and_nan(data_srs)
            bins = seaborn_broken_bins(values)
            sns.distplot(values, kde=do_kde, ax=ax, bins=bins)

    return ax


def do_heatmap(plot_df, unused_vartypes, **kwargs):
    ax = kwargs.get('ax', None)

    plot_df = plot_df.dropna()
    if ax is None:
        ax = plt.gca()

    vals_x, uvals_x, _ = conv_categorical_vals_to_numeric(plot_df.ix[:, 0])
    vals_y, uvals_y, _ = conv_categorical_vals_to_numeric(plot_df.ix[:, 1])

    bins_x = len(uvals_x)
    bins_y = len(uvals_y)

    hst, _, _ = np.histogram2d(vals_y, vals_x, bins=[bins_y, bins_x])
    cmap = 'PuBu'
    ax.matshow(hst, aspect='auto', origin='lower', cmap=cmap)
    ax.grid(b=False)
    ax.set_xticks(range(bins_x))
    ax.set_yticks(range(bins_y))
    ax.set_xticklabels(uvals_x)
    ax.set_yticklabels(uvals_y)
    ax.xaxis.set_tick_params(labeltop='off', labelbottom='on')
    return ax


def do_violinplot(plot_df, vartypes, **kwargs):
    ax = kwargs.get('ax', None)
    colors = kwargs.get('colors', None)
    for handled in ('ax', 'colors'):
        if handled in kwargs:
            del kwargs[handled]

    dummy = plot_df.shape[1] == 3

    plot_df = plot_df.dropna()
    if ax is None:
        ax = plt.gca()
    assert vartypes[0] != vartypes[1]
    vert = vartypes[1] == 'numerical'
    plot_df = copy.deepcopy(plot_df.dropna())
    if vert:
        groupby = plot_df.columns[0]
    else:
        groupby = plot_df.columns[1]

    _, unique_vals, _ = conv_categorical_vals_to_numeric(plot_df[groupby])

    unique_vals = np.sort(unique_vals)
    n_vals = len(plot_df[groupby].unique())
    if dummy:
        sub_vals = np.sort(plot_df[groupby].unique())
        axis = sns.violinplot(
            x=plot_df.columns[0],
            y=plot_df.columns[1],
            data=plot_df,
            order=sub_vals, hue=plot_df.columns[2],
            names=sub_vals, ax=ax, orient=("v" if vert else "h"),
            palette=colors, inner='quartile', **kwargs)
        axis.legend_ = None
    else:
        sns.violinplot(
            x=plot_df.columns[0],
            y=plot_df.columns[1],
            data=plot_df,
            order=unique_vals, names=unique_vals, ax=ax,
            orient=("v" if vert else "h"),
            color='SteelBlue', inner='quartile', **kwargs)

    if vert:
        ax.set_xlim([-.5, n_vals-.5])
        ax.set_xticks(np.arange(n_vals))
        ax.set_xticklabels(unique_vals)
    else:
        ax.set_ylim([-.5, n_vals-.5])
        ax.set_yticks(np.arange(n_vals))
        ax.set_yticklabels(unique_vals)

    return ax


def do_kdeplot(plot_df, unused_vartypes, bdb=None, generator_name=None,
               **kwargs):
    # XXX: kdeplot is not a good choice for small amounts of data because
    # it uses a kernel density estimator to crease a smooth heatmap. On the
    # other hand, scatter plots are uninformative given lots of data---the
    # points get jumbled up. We may just want to set a threshold (N=100)?

    _unused_generator_name = generator_name  # So it doesn't end up in kwargs
    _unused_bdb = bdb

    ax = kwargs.get('ax', None)
    show_contour = kwargs.get('show_contour', False)
    colors = kwargs.get('colors', None)
    show_missing = kwargs.get('show_missing', False)
    for handled in ('ax', 'show_contour', 'colors', 'show_missing'):
        if handled in kwargs:
            del kwargs[handled]

    if ax is None:
        ax = plt.gca()

    xlim = [plot_df.ix[:, 0].min(), plot_df.ix[:, 0].max()]
    ylim = [plot_df.ix[:, 1].min(), plot_df.ix[:, 1].max()]

    dummy = plot_df.shape[1] == 3

    null_rows = plot_df[plot_df.isnull().any(axis=1)]
    df = plot_df.dropna()

    if not dummy:
        plt.scatter(df.values[:, 0], df.values[:, 1], alpha=.5,
            color='steelblue', zorder=2, **kwargs)
        # plot nulls
        if show_missing:
            nacol_x = null_rows.ix[:, 0].dropna()
            for x in nacol_x.values:
                plt.plot([x, x], ylim, color='crimson', alpha=.2, lw=1,
                    zorder=1)
            nacol_y = null_rows.ix[:, 1].dropna()
            for y in nacol_y.values:
                plt.plot(xlim, [y, y], color='crimson', alpha=.2, lw=1,
                    zorder=1)
    else:
        assert isinstance(colors, dict)
        for val, color in colors.iteritems():
            subdf = df.loc[df.ix[:, 2] == val]
            plt.scatter(subdf.values[:, 0], subdf.values[:, 1], alpha=.5,
                color=color, zorder=2, **kwargs)
            subnull = null_rows.loc[null_rows.ix[:, 2] == val]
            if show_missing:
                nacol_x = subnull.ix[:, 0].dropna()
                for x in nacol_x.values:
                    plt.plot([x, x], ylim, color=color, alpha=.3, lw=2,
                        zorder=1)
                nacol_y = subnull.ix[:, 1].dropna()
                for y in nacol_y.values:
                    plt.plot(xlim, [y, y], color=color, alpha=.3, lw=2,
                        zorder=1)

    if show_contour:
        try:
            sns.kdeplot(df.ix[:, :2].values, ax=ax)
        except ValueError:
            # Displaying a plot without a requested contour is better
            # than crashing.
            pass

            # This actually happens: with the 'skewed_numeric_5'
            # distribution in tests/test_plot_utils.py, we get:
            # seaborn/distributions.py:597: in kdeplot
            #     ax, **kwargs)
            # seaborn/distributions.py:380: in _bivariate_kdeplot
            #     cset = contour_func(xx, yy, z, n_levels, **kwargs)
            # matplotlib/axes/_axes.py:5333: in contour
            #     return mcontour.QuadContourSet(self, *args, **kwargs)
            # matplotlib/contour.py:1429: in __init__
            #     ContourSet.__init__(self, ax, *args, **kwargs)
            # matplotlib/contour.py:876: in __init__
            #     self._process_levels()
            # matplotlib/contour.py:1207: in _process_levels
            #     self.vmin = np.amin(self.levels)
            # numpy/core/fromnumeric.py:2224: in amin
            #     out=out, keepdims=keepdims)
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # a = array([], dtype=float64),
            # axis = None, out = None, keepdims = False
            #
            #     def _amin(a, axis=None, out=None, keepdims=False):
            # >       return umr_minimum(a, axis, None, out, keepdims)
            # E       ValueError: zero-size array to reduction operation
            #                     minimum which has no identity

    return ax


# No support for cyclic at this time
DO_PLOT_FUNC = dict()
DO_PLOT_FUNC[('categorical', 'categorical')] = do_heatmap
DO_PLOT_FUNC[('categorical', 'numerical')] = do_violinplot
DO_PLOT_FUNC[('numerical', 'categorical')] = do_violinplot
DO_PLOT_FUNC[('numerical', 'numerical')] = do_kdeplot


def do_pair_plot(plot_df, vartypes, **kwargs):
    # determine plot_types
    if kwargs.get('ax', None) is None:
        kwargs['ax'] = plt.gca()

    ax = DO_PLOT_FUNC[vartypes](plot_df, vartypes, **kwargs)
    return ax


def zmatrix(data_df, clustermap_kws=None, row_ordering=None,
        col_ordering=None, vmin=None, vmax=None):
    """Plots a clustermap from an ESTIMATE PAIRWISE query.

    Parameters
    ----------
    data_df : pandas.DataFrame
        The result of a PAIRWISE query in pandas.DataFrame.
    clustermap_kws : dict
        kwargs for seaborn.clustermap. See seaborn documentation. Of particular
        importance is the `pivot_kws` kwarg. `pivot_kws` is a dict with entries
        index, column, and values that let clustermap know how to reshape the
        data. If the query does not follow the standard ESTIMATE PAIRWISE
        output, it may be necessary to define `pivot_kws`.
        Other keywords here include vmin and vmax, linewidths, figsize, etc.
    row_ordering, col_ordering : list<int>
        Specify the order of labels on the x and y axis of the heatmap.
        To access the row and column indices from a clustermap object, use:
        clustermap.dendrogram_row.reordered_ind (for rows)
        clustermap.dendrogram_col.reordered_ind (for cols)
    vmin, vmax : float
        The minimum and maximum values of the colormap.

    Returns
    -------
    clustermap: seaborn.clustermap
    """
    data_df.fillna(0, inplace=True)
    if clustermap_kws is None:
        half_root_col = (data_df.shape[0] ** .5) / 2.0
        clustermap_kws = {'linewidths': 0.2, 'vmin': vmin, 'vmax': vmax,
                          'figsize': (half_root_col, .8 * half_root_col)}

    if clustermap_kws.get('pivot_kws', None) is None:
        # XXX: If the user doesnt tell us otherwise, we assume that this comes
        # fom a standard estimate pairwise query, which outputs columns
        # (table_id, col0, col1, value). The indices are indexed from the back
        # because it will also handle the no-table_id case
        data_df.columns = [' '*i for i in range(1, len(data_df.columns))] + \
            ['value']
        pivot_kws = {
            'index': data_df.columns[-3],
            'columns': data_df.columns[-2],
            'values': data_df.columns[-1],
        }
        clustermap_kws['pivot_kws'] = pivot_kws

    if clustermap_kws.get('cmap', None) is None:
        # Choose a soothing blue colormap
        clustermap_kws['cmap'] = 'BuGn'

    if row_ordering is not None and col_ordering is not None:
        index = clustermap_kws['pivot_kws']['index']
        columns = clustermap_kws['pivot_kws']['columns']
        values = clustermap_kws['pivot_kws']['values']
        df = data_df.pivot(index, columns, values)
        df = df.ix[:, col_ordering]
        df = df.ix[row_ordering, :]
        del clustermap_kws['pivot_kws']
        _fig, ax = plt.subplots()
        return (sns.heatmap(df, ax=ax, **clustermap_kws),
            row_ordering, col_ordering)
    else:
        return sns.clustermap(data_df, **clustermap_kws)


# TODO: bdb, and table_name should be optional arguments
def _pairplot(df, bdb=None, generator_name=None,
        show_contour=False, colorby=None, show_missing=False, show_full=False,
        **kwargs):
    """Plots the columns in data_df in a facet grid.

    Supports the following pairs:
    - categorical-categorical pairs are displayed as a heatmap
    - continuous-continuous pairs are displayed as a kdeplot
    - categorical-continuous pairs are displayed on a violin plot

    Parameters
    ----------
    df : pandas.DataFrame
        The input data---the result of a BQL/SQL query
    bdb : bayeslite.BayesDB (optional)
        The BayesDB object associated with `df`. Having the BayesDB object and
        the generator for the data allows pairplot to choose plot types.
    generator_name : str
        The name of generator associated with `df` and `bdb`.
    show_contour : bool
        If True, KDE contours are plotted on top of scatter plots
        and histograms.
    show_missing : bool
        If True, rows with one missing value are plotted as lines on scatter
        plots.
    colorby : str
        Name of a column to use to color data points in histograms and scatter
        plots.
    show_full : bool
        Show full pairwise plots, rather than only lower triangular plots.
    kwargs : dict
        Options to pass through to underlying plotting function (for pairs).

    Returns
    -------
    figure : matplotlib.figure.Figure
        A num_columns by num_columns Gridspec of pairplot axes.

    Notes
    -----
    Support soon for ordered continuous combinations. It may be best
    to plot all ordered continuous pairs as heatmap.
    """
    # NOTE:Things to consider:
    # - String values are a possibility (categorical)
    # - who knows what the columns are named. What is the user selects columns
    #   as shortname?
    # - where to handle dropping NaNs? Missing values may be informative.

    data_df = df

    colors = None
    if colorby is not None:
        n_colorby = 0
        for colname in data_df.columns:
            # XXX: This is not guaranteed to work on all Unicode characters.
            if colorby.lower() == colname.lower():
                n_colorby += 1
                colorby = colname
        if n_colorby == 0:
            raise BLE(ValueError(
                'colorby column, {}, not found.'.format(colorby)))
        elif n_colorby > 1:
            raise BLE(ValueError('Multiple columns named, {}.'.format(colorby)))

        dummy = data_df[colorby].dropna()
        dvals = np.sort(dummy.unique())
        ndvals = len(dvals)
        dval_type = get_bayesdb_col_type('colorby', dummy, bdb=bdb,
                                         generator_name=generator_name)
        if dval_type.lower() != 'categorical':
            raise BLE(ValueError('colorby columns must be categorical.'))
        cmap = sns.color_palette("Set1", ndvals)
        colors = {}
        for val, color in zip(dvals, cmap):
            colors[val] = color

    all_varnames = [c for c in data_df.columns if c != colorby]
    n_vars = len(all_varnames)
    plt_grid = gridspec.GridSpec(n_vars, n_vars)
    figure = plt.figure()

    # if there is only one variable, just do a hist
    if n_vars == 1:
        ax = plt.gca()
        varname = data_df.columns[0]
        vartype = get_bayesdb_col_type(varname, data_df[varname], bdb=bdb,
                                       generator_name=generator_name)
        do_hist(data_df, dtype=vartype, ax=ax, bdb=bdb,
            generator_name=generator_name, colors=colors)
        if vartype == 'categorical':
            pass  # rotate_tick_labels(ax)
        return

    xmins = np.ones((n_vars, n_vars))*float('Inf')
    xmaxs = np.ones((n_vars, n_vars))*float('-Inf')
    ymins = np.ones((n_vars, n_vars))*float('Inf')
    ymaxs = np.ones((n_vars, n_vars))*float('-Inf')

    vartypes = []
    for varname in all_varnames:
        vartype = get_bayesdb_col_type(varname, data_df[varname], bdb=bdb,
            generator_name=generator_name)
        vartypes.append(vartype)

    # store each axes; reaccessing ax with plt.subplot(plt_grid[a,b]) may
    # overwrite the ax
    axes = [[] for _ in xrange(len(all_varnames))]
    for x_pos, var_name_x in enumerate(all_varnames):
        var_x_type = vartypes[x_pos]
        for y_pos, var_name_y in enumerate(all_varnames):
            var_y_type = vartypes[y_pos]

            ax = figure.add_subplot(plt_grid[y_pos, x_pos])
            axes[y_pos].append(ax)

            if x_pos == y_pos:
                varnames = [var_name_x]
                if colorby is not None:
                    varnames.append(colorby)
                ax = do_hist(data_df[varnames], dtype=var_x_type, ax=ax,
                    bdb=bdb, generator_name=generator_name, colors=colors)
            else:
                varnames = [var_name_x, var_name_y]
                vartypes_pair = (var_x_type, var_y_type,)
                if colorby is not None:
                    varnames.append(colorby)
                plot_df = prep_plot_df(data_df, varnames)
                ax = do_pair_plot(plot_df, vartypes_pair, ax=ax, bdb=bdb,
                    generator_name=generator_name,
                    show_contour=show_contour, show_missing=show_missing,
                    colors=colors, **kwargs)

                ymins[y_pos, x_pos] = ax.get_ylim()[0]
                ymaxs[y_pos, x_pos] = ax.get_ylim()[1]
                xmins[y_pos, x_pos] = ax.get_xlim()[0]
                xmaxs[y_pos, x_pos] = ax.get_xlim()[1]

            ax.set_xlabel(var_name_x, fontweight='bold')
            ax.set_ylabel(var_name_y, fontweight='bold')

    for x_pos in range(n_vars):
        for y_pos in range(n_vars):
            ax = axes[y_pos][x_pos]
            ax.set_xlim([np.min(xmins[:, x_pos]), np.max(xmaxs[:, x_pos])])
            if x_pos != y_pos:
                ax.set_ylim([np.min(ymins[y_pos, :]), np.max(ymaxs[y_pos, :])])
            if x_pos > 0:
                if x_pos == n_vars - 1:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
            if y_pos < n_vars - 1:
                if y_pos == 0:
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
            else:
                if vartype[x_pos] == 'categorical':
                    rotate_tick_labels(ax)

    def fake_axis_ticks(ax_tl, ax_tn):
        atl, btl = ax_tl.get_ylim()
        atn, btn = ax_tn.get_ylim()
        tnticks = ax_tn.get_yticks()
        yrange_tn = (btn-atn)
        yrange_tl = (btl-atl)
        tntick_ratios = [(t-atn)/yrange_tn for t in tnticks]
        ax_tl.set_yticks([r*yrange_tl+atl for r in tntick_ratios])
        ax_tl.set_yticklabels(tnticks)

    # Fix the top-left histogram y-axis ticks and labels.
    if show_full:
        fake_axis_ticks(axes[0][0], axes[0][1])
    fake_axis_ticks(axes[-1][-1], axes[-1][0])

    if colorby is not None:
        legend = gen_collapsed_legend_from_dict(colors, title=colorby)
        legend.draggable()

    # tril by default by deleting upper diagonal axes.
    if not show_full:
        for y_pos in range(n_vars):
            for x_pos in range(y_pos+1, n_vars):
                figure.delaxes(axes[y_pos][x_pos])

    return figure


def comparative_hist(df, bdb=None, nbins=15, normed=False):
    """Plot a histogram.

    Given a one-column pandas.DataFrame, df, plots a simple histogram. Given a
    two-column df plots the data in column one colored by an optional column 2.
    If given, column 2 must be categorical.

    Parameters
    ----------
    nbins : int
        Number of bins (bars)
    normed : bool
        If True, normalizes the the area of the histogram (or each
        sub-histogram if df has two columns) to 1.

    Returns
    -------
    figure: matplotlib.figure.Figure
    """
    df = df.dropna()

    vartype = get_bayesdb_col_type(df.columns[0], df[df.columns[0]], bdb=bdb)
    if vartype == 'categorical':
        values, labels, lookup = conv_categorical_vals_to_numeric(
            df[df.columns[0]])
        df.ix[:, 0] = values
        bins = len(labels)
        ticklabels = [0]*len(labels)
        for key, val in lookup.iteritems():
            ticklabels[val] = key
    else:
        a = min(df.ix[:, 0].values)
        b = max(df.ix[:, 0].values)
        support = b - a
        interval = support/nbins
        bins = np.linspace(a, b+interval, nbins)

    colorby = None
    if len(df.columns) > 1:
        if len(df.columns) > 2:
            raise BLE(NotImplementedError(
                'comparative_hist not defined on more than two variables.'))
        colorby = df.columns[1]
        colorby_vals = df[colorby].unique()

    figure, ax = plt.subplots(tight_layout=False, facecolor='white')
    if colorby is None:
        ax.hist(df.ix[:, 0].values, bins=bins, color='#383838',
            edgecolor='none', normed=normed)
        plot_title = df.columns[0]
    else:
        colors = sns.color_palette('deep', len(colorby_vals))
        for color, cbv in zip(colors, colorby_vals):
            subdf = df[df[colorby] == cbv]
            ax.hist(subdf.ix[:, 0].values, bins=bins, color=color, alpha=.5,
                edgecolor='none', normed=normed, label=str(cbv))
        ax.legend(loc=0, title=colorby)
        plot_title = df.columns[0] + " by " + colorby

    if normed:
        plot_title += " (normalized)"

    ax.set_title(plot_title)
    ax.set_xlabel(df.columns[0])
    return figure
