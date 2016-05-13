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
    if num_models == 0:
        raise ValueError('No models to plot mutual information over!')

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

@population_method(population_to_bdb=0, specifier_to_df=1,
                   generator_name='generator_name')
def pairplot(bdb, df, generator_name=None, stattypes=None, show_contour=False,
             colorby=None, show_missing=False, show_full=False,
             pad=None, h_pad=None, **kwargs):
    """Plot array of plots for all pairs of columns.

    Plots continuous-continuous pairs as scatter (optional KDE contour).
    Plots continuous-categorical pairs as violinplot.
    Plots categorical-categorical pairs as heatmap.

    Parameters
    ----------
    bdb : __population_to_bdb__
    df : __specifier_to_df__
    generator_name : __generator_name__ - used to find stattypes.
    stattypes : dict, optional {column_name: "categorical"|"numerical"}
        If you do not specify a generator name, have a column that the
        generator doesn't know about, or would like to override the
        statistical types the generator has for a given variable, then pass
        this dict of column names to types.
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
    pad : number, optional
        Adjust the vertical padding between plot components.
    h_pad : number, optional
        Adjust the horizontal padding between plot components.
    **kwargs : dict, optional
        Options to pass through to underlying plotter for pairs.

    Returns
    -------
    figure : matplotlib.figure.Figure
    """
    figure = _pairplot(df, bdb=bdb, generator_name=generator_name,
        stattypes=stattypes, show_contour=show_contour, colorby=colorby,
        show_missing=show_missing, show_full=show_full, **kwargs)
    if pad is not None or h_pad is not None:
      default_padding = -0.4 * (df.shape[1] - 2)
      if pad is None:
        pad = default_padding
      if h_pad is None:
        h_pad = default_padding
      figure.tight_layout(h_pad=h_pad, pad=pad)
    inches = len(df.columns) * 4
    figure.set_size_inches((inches, inches))

    return figure

@population_method(population_to_bdb=0, generator_name='generator_name',
                   population_name='population_name')
def pairplot_vars(bdb, varnames, colorby=None, generator_name=None,
                  population_name=None, **kwargs):
  """Use pairplot to show the given variables.

  See help(pairplot) for more plot options.

  Parameters
  ----------
  bdb: __population_to_bdb__
  varnames: list of one or more variables to plot.
  generator_name: __generator_name__
  population_name: __population_name__
  colorby: categorical variable to color all of the plots by.

  Returns
  -------
  figure: a matplotlib.figure.Figure
  """
  if len(varnames) < 1:
    raise BLE(ValueError('Pairplot at least one variable.'))
  qvars = varnames if colorby is None else set(varnames + [colorby])
  query_columns = '''"%s"''' % '''", "'''.join(qvars)
  bql = '''SELECT %s FROM %s''' % (query_columns, population_name)
  df = bqlu.query(bdb, bql)
  return pairplot(bdb, df, generator_name=generator_name,
      colorby=colorby, **kwargs)

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
    if nbins is None:
        nbins = len(bins) if bins is not None else 15
    if bins is None:
        a = min(df.ix[:, 0].values)
        b = max(df.ix[:, 0].values)
        support = b - a
        interval = support / nbins
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


def rotate_tick_labels(ax, axis=None, rotation=None):
    """Rotate tick labels on one axis if specified, or both otherwise.

    axis : str 'x' or 'y' or None
        If None, then both axes are rotated as requested.
    rotation : number or 'horizontal', 'vertical', 'parallel', 'perpendicular'.
        Angle or orientation to rotate to.

    Returns: ax
    """
    if rotation is None:
        rotation = 'perpendicular'
    if axis:
        labels = ax.get_xticklabels() if axis == 'x' else ax.get_yticklabels()
        for tick in labels:
            if rotation == 'by_length':
                tick_rotation = 'horizontal'
                if axis == 'x' and len(tick.get_text()) > 4:
                    tick_rotation = 'vertical'
            elif rotation == 'parallel':
                tick_rotation = 'horizontal' if axis == 'x' else 'vertical'
            elif rotation == 'perpendicular':
                tick_rotation = 'vertical' if axis == 'x' else 'horizontal'
            else:
                tick_rotation = rotation
            tick.set_rotation(tick_rotation)
    else:
        rotate_tick_labels(ax, 'x', rotation)
        rotate_tick_labels(ax, 'y', rotation)
    return ax

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
    show_contour = kwargs.get('show_contour', None)
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
        do_kde = show_contour
        if colors is not None:
            for val, color in colors.iteritems():
                subdf = data_srs.loc[data_srs.ix[:, 1] == val]
                values = drop_inf_and_nan(subdf.ix[:, 0])
                if len(values) < 2: # Then seaborn would break. :-p
                    continue
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
    cmap = 'BuGn'
    ax.matshow(hst, aspect='auto', origin='lower', cmap=cmap)
    ax.xaxis.set_tick_params(labeltop='off', labelbottom='on') # RESETS LABELS!
    ax.grid(b=False)
    ax.set_xticks(range(bins_x))
    ax.set_yticks(range(bins_y))
    ax.set_xticklabels(uvals_x)
    ax.set_yticklabels(uvals_y)
    rotate_tick_labels(ax)
    return ax


def do_violinplot(plot_df, vartypes, **kwargs):
    ax = kwargs.get('ax', None)
    colors = kwargs.get('colors', None)
    cut=1
    if kwargs.get('show_contour', None) == False:
        cut=0
    for handled in ('ax', 'show_contour', 'colors'):
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
            palette=colors, inner='quartile', cut=cut, **kwargs)
        axis.legend_ = None
    else:
        sns.violinplot(
            x=plot_df.columns[0],
            y=plot_df.columns[1],
            data=plot_df,
            order=unique_vals, names=unique_vals, ax=ax,
            orient=("v" if vert else "h"),
            color='SteelBlue', inner='quartile', cut=cut, **kwargs)

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
    for handled in ('ax', 'colors', 'show_contour', 'show_missing'):
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
            if subdf.shape[0] == 0:
                continue
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
            sns.kdeplot(df.ix[:, :2].values, ax=ax, **kwargs)
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

def ensure_full_square(data_df, pivot_kws, default_value=0):
    """Ensure that all pairs are bidirectionally present in a pivot df.

    data_df is expected to have an index column, a columns column, and a
    data column, as specified by the pivot_kws (as they would be supplied
    to seaborn.clustermap).

    Creates a complete matrix where every value in index has an entry for
    every value in columns, and vice versa, filling with zeros as needed.

    Scipy has been known to segfault without this, or give a ValueError,
    depending on the version.

    Return: a new complete-square df with those three columns.
    """
    iname = pivot_kws['index']
    cname = pivot_kws['columns']
    vname = pivot_kws['values']

    import sets
    index_values = set(data_df[iname])
    columns_values = set(data_df[cname])
    if len(index_values) == 0 or len(columns_values) == 0:
        return pd.DataFrame()
    data = {}
    for index, row in data_df.iterrows():
        if row[iname] not in data:
            data[row[iname]] = {}
        sub = data[row[iname]]
        if row[cname] not in sub:
            sub[row[cname]] = row[vname]
    rows = []
    for ind in sorted(index_values):
        for col in sorted(columns_values):
            if ind in data and col in data[ind]:
                rows.append([ind, col, data[ind][col]])
            else:
                rows.append([ind, col, default_value])
    new_df = pd.DataFrame(rows)
    new_df.columns = [iname, cname, vname]
    return new_df

@population_method(specifier_to_df=0)
def heatmap(data_df, row_ordering=None, col_ordering=None, **kwargs):
    """Plot heatmaps, optionally clustered.

    Parameters
    ----------
    deps : __specifier_to_df__
        Must have two categorical columns and a numeric column.
        The format is assumed to be that the numeric column of values is last,
        and the two categorical columns are immediately before that.

        The canonical example of that kind of data is the result of an
        ESTIMATE ... PAIRWISE query, estimating dependence probability,
        mutual information, correlation, etc.

        If your columns are not at the end that way, then pass pivot_kws too.

    row_ordering, col_ordering : list<int>
        Specify the order of labels on the x and y axis of the heatmap.
        If these are specified, we will not call try to cluster a different way,
        and return a plain heatmap.

        To access the row and column indices from a clustermap object, use:
        clustermap.dendrogram_row.reordered_ind (for rows)
        clustermap.dendrogram_col.reordered_ind (for cols)

    **kwargs :
        to pass to seaborn.clustermap. See seaborn documentation. Of particular
        importance is the `pivot_kws` kwarg. `pivot_kws` is a dict with entries
        index, column, and values that let clustermap know how to reshape the
        data. If the query does not follow the standard ESTIMATE PAIRWISE
        output, it may be necessary to define `pivot_kws`.
        Other keywords here include vmin and vmax, linewidths, figsize, etc.

    Returns
    -------
    clustermap: seaborn.clustermap
    """
    clustermap_kws = {}
    clustermap_kws.update(kwargs)
    data_df.fillna(0, inplace=True)
    if clustermap_kws is None:
        clustermap_kws = {}
    if 'figsize' not in clustermap_kws:
        half_root_col = (data_df.shape[0] ** .5) / 2.0
        clustermap_kws['figsize'] = (half_root_col, .8 * half_root_col)
    if 'linewidths' not in clustermap_kws:
        clustermap_kws['linewidths'] = 0.2
    if 'pivot_kws' not in clustermap_kws:
        # XXX: If the user doesn't tell us otherwise, we assume that this comes
        # from a standard estimate pairwise query, which outputs columns
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

    autoset_vlimits = False
    if 'vmin' not in clustermap_kws:
        autoset_vlimits = True
        clustermap_kws['vmin'] = \
            data_df[clustermap_kws['pivot_kws']['values']].min()
    if 'vmax' not in clustermap_kws:
        autoset_vlimits = True
        clustermap_kws['vmax'] = \
            data_df[clustermap_kws['pivot_kws']['values']].max()
    if (autoset_vlimits and
        clustermap_kws['vmin'] >= 0 and
        clustermap_kws['vmax'] <= 1):
        clustermap_kws['vmin'] = 0
        clustermap_kws['vmax'] = 1
    if autoset_vlimits:
        print ("Detected value limits as [%s, %s]. Override with vmin and vmax."
               % (clustermap_kws['vmin'], clustermap_kws['vmax']))


    if row_ordering is not None and col_ordering is not None:
        index = clustermap_kws['pivot_kws']['index']
        columns = clustermap_kws['pivot_kws']['columns']
        values = clustermap_kws['pivot_kws']['values']
        df = data_df.pivot(index, columns, values)
        df = df.ix[:, col_ordering]
        df = df.ix[row_ordering, :]
        del clustermap_kws['pivot_kws']
        del clustermap_kws['figsize']
        _fig, ax = plt.subplots()
        sns.heatmap(df, ax=ax, **clustermap_kws)
        rotate_tick_labels(ax)
        return (ax, row_ordering, col_ordering)
    else:
        complete_df = ensure_full_square(data_df, clustermap_kws['pivot_kws'])
        hmap = sns.clustermap(complete_df, **clustermap_kws)
        rotate_tick_labels(hmap.ax_heatmap)
        return hmap


# TODO: bdb, and table_name should be optional arguments
def _pairplot(df, bdb=None, generator_name=None, stattypes=None,
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
    stattypes : dict, optional {column_name: "categorical"|"numerical"}
        If you do not specify a generator name, have a column that the
        generator doesn't know about, or would like to override the
        statistical types the generator has for a given variable, then pass
        this dict of column names to types.
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
    if stattypes is None:
        stattypes = {}

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
        dval_type = "categorical"
        if colorby in stattypes:
            dval_type = stattypes[colorby]
        elif generator_name:
            dval_type = get_bayesdb_col_type(colorby, dummy, bdb=bdb,
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
        vartype = "categorical"
        if varname in stattypes:
            vartype = stattypes[varname]
        elif generator_name:
            vartype = get_bayesdb_col_type(varname, data_df[varname], bdb=bdb,
                                           generator_name=generator_name)
        do_hist(data_df, dtype=vartype, ax=ax, bdb=bdb, colors=colors)
        rotate_tick_labels(ax)
        return

    xmins = np.ones((n_vars, n_vars))*float('Inf')
    xmaxs = np.ones((n_vars, n_vars))*float('-Inf')
    ymins = np.ones((n_vars, n_vars))*float('Inf')
    ymaxs = np.ones((n_vars, n_vars))*float('-Inf')

    vartypes = []
    for varname in all_varnames:
        vartype = "categorical"
        if varname in stattypes:
            vartype = stattypes[varname]
        elif generator_name:
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
                    bdb=bdb, colors=colors, show_contour=show_contour)
            else:
                varnames = [var_name_x, var_name_y]
                vartypes_pair = (var_x_type, var_y_type,)
                if colorby is not None:
                    varnames.append(colorby)
                plot_df = prep_plot_df(data_df, varnames)
                ax = do_pair_plot(plot_df, vartypes_pair, ax=ax, bdb=bdb,
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

            # Self-histogram for x only, or comparative x against y:
            ax.set_xlim([np.min(xmins[:, x_pos]), np.max(xmaxs[:, x_pos])])
            if x_pos != y_pos:
                ax.set_ylim([np.min(ymins[y_pos, :]), np.max(ymaxs[y_pos, :])])

            # Y labels
            if x_pos > 0:
                if x_pos == n_vars - 1:  # All the way to the right:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                else:  # No labels inside:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
            else:
                ax.yaxis.tick_left()
                ax.yaxis.set_label_position('left')

            # X labels:
            if y_pos < n_vars - 1:
                if y_pos == 0:  # At top, show x labels on top:
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top')
                else:  # No labels inside:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
            else:
                ax.xaxis.tick_bottom()
                ax.xaxis.set_label_position('bottom')

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
                edgecolor='none', normed=normed,
                label=("%s (n=%d)" % (str(cbv), len(subdf))))
        ax.legend(loc=0, title=colorby)
        plot_title = df.columns[0] + " by " + colorby

    if normed:
        plot_title += " (normalized)"

    ax.set_title(plot_title)
    ax.set_xlabel(df.columns[0])
    return figure
