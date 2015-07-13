
import bdbcontrib.bql_utils as bqlu

from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import copy
import pandas as pd
import numpy as np


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
        raise ValueError('axis must b x or y')
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
        max number of charachters before wordwrap

    Returns
    -------
    legend : matplotlib.legend
    """
    if not isinstance(hl_colors_dict, dict):
        raise TypeError("hl_colors_dict must be a dict")

    colors = list(set(hl_colors_dict.values()))
    collapsed_dict = dict(zip(colors, [[] for i in range(len(colors))]))

    for color in colors:
        collapsed_dict[color] == []

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
        raise TypeError("Multiple columns in the query result have the same "
                        "name (%s)." % (column_name,))

    def guess_column_type(df_column):
        pd_type = df_column.dtype
        if pd_type is str:
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
        except Exception as err:
            print "Unexpected exception: {}".format(err)
            raise err
    else:
        return guess_column_type(df_column)


def conv_categorical_vals_to_numeric(data_srs, bdb=None, generator_name=None):
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
            raise ValueError('If a dummy column is specified, colors must '
                             'also be specified.')

    data_srs = data_srs.dropna()

    if dtype == 'categorical':
        vals, uvals, _ = conv_categorical_vals_to_numeric(
            data_srs.ix[:, 0], bdb=bdb, generator_name=generator_name)
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
                sns.distplot(subdf.ix[:, 0], kde=do_kde, ax=ax, color=color)
        else:
            sns.distplot(data_srs, kde=do_kde, ax=ax)

    return ax


def do_heatmap(plot_df, vartypes, **kwargs):
    ax = kwargs.get('ax', None)
    bdb = kwargs.get('bdb', None)
    generator_name = kwargs.get('generator_name', None)

    plot_df = plot_df.dropna()
    if ax is None:
        ax = plt.gca()

    vals_x, uvals_x, _ = conv_categorical_vals_to_numeric(
        plot_df.ix[:, 0], bdb=bdb, generator_name=generator_name)
    vals_y, uvals_y, _ = conv_categorical_vals_to_numeric(
        plot_df.ix[:, 1], bdb=bdb, generator_name=generator_name)

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
    bdb = kwargs.get('bdb', None)
    generator_name = kwargs.get('generator_name', None)
    colors = kwargs.get('colors', None)
    # dummy = kwargs.get('dummy', False)

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

    _, unique_vals, _ = conv_categorical_vals_to_numeric(
        plot_df[groupby], bdb=bdb, generator_name=generator_name)

    unique_vals = np.sort(unique_vals)
    n_vals = len(plot_df[groupby].unique())
    if dummy:
        order_key = dict((val, i) for i, val in enumerate(unique_vals))
        base_width = 0.75/len(colors)
        violin_width = base_width
        for i, (val, color) in enumerate(colors.iteritems()):
            subdf = plot_df.loc[plot_df.ix[:, 2] == val]
            if subdf.empty:
                continue

            if vert:
                vals = subdf.columns[1]
            else:
                vals = subdf.columns[0]

            # not evey categorical value is guaranteed to appear in each subdf.
            # Here we compensate.
            sub_vals = np.sort(subdf[groupby].unique())
            positions = []

            for v in sub_vals:
                positions.append(base_width*i + order_key[v] + base_width/2
                                 - .75/2)

            sns.violinplot(subdf[vals], groupby=subdf[groupby],
                           order=sub_vals, names=sub_vals, vert=vert,
                           ax=ax, positions=positions, widths=violin_width,
                           color=color)
    else:
        if vert:
            vals = plot_df.columns[1]
        else:
            vals = plot_df.columns[0]
        sns.violinplot(plot_df[vals], groupby=plot_df[groupby],
                       order=unique_vals, names=unique_vals, vert=vert, ax=ax,
                       positions=0, color='SteelBlue')

    if vert:
        ax.set_xlim([-.5, n_vals-.5])
        ax.set_xticks(np.arange(n_vals))
        ax.set_xticklabels(unique_vals)
    else:
        ax.set_ylim([-.5, n_vals-.5])
        ax.set_yticks(np.arange(n_vals))
        ax.set_yticklabels(unique_vals)

    return ax


def do_kdeplot(plot_df, vartypes, **kwargs):
    # XXX: kdeplot is not a good choice for small amounts of data because
    # it uses a kernel density estimator to crease a smooth heatmap. On the
    # other hadnd, scatter plots are uniformative given lots of data---the
    # points get jumbled up. We may just want to set a threshold (N=100)?

    ax = kwargs.get('ax', None)
    show_contour = kwargs.get('show_contour', False)
    colors = kwargs.get('colors', None)
    show_missing = kwargs.get('show_missing', False)

    if ax is None:
        ax = plt.gca()

    xlim = [plot_df.ix[:, 0].min(), plot_df.ix[:, 0].max()]
    ylim = [plot_df.ix[:, 1].min(), plot_df.ix[:, 1].max()]

    dummy = plot_df.shape[1] == 3

    null_rows = plot_df[plot_df.isnull().any(axis=1)]
    df = plot_df.dropna()

    if not dummy:
        plt.scatter(df.values[:, 0], df.values[:, 1], alpha=.5,
                    color='steelblue', zorder=2)
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
                        color=color, zorder=2)
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
        sns.kdeplot(df.ix[:, :2].values, ax=ax)

    return ax


# No support for cyclic at this time
DO_PLOT_FUNC = dict()
DO_PLOT_FUNC[hash(('categorical', 'categorical',))] = do_heatmap
DO_PLOT_FUNC[hash(('categorical', 'numerical',))] = do_violinplot
DO_PLOT_FUNC[hash(('numerical', 'categorical',))] = do_violinplot
DO_PLOT_FUNC[hash(('numerical', 'numerical',))] = do_kdeplot


def do_pair_plot(plot_df, vartypes, **kwargs):
    # determine plot_types
    if kwargs.get('ax', None) is None:
        kwargs['ax'] = plt.gca()

    ax = DO_PLOT_FUNC[hash(vartypes)](plot_df, vartypes, **kwargs)
    return ax


def zmatrix(data_df, clustermap_kws=None, row_ordering=None,
            col_ordering=None):
    """Plots a clustermap from an ESTIMATE PAIRWISE query.

    Parameters:
    -----------
    data_df : pandas.DataFrame
        The result of a PAIRWISE query in pandas.DataFrame.
    clustermap_kws : dict
        kwargs for seaborn.clustermap. See seaborn documentation. Of particular
        importance is the `pivot_kws` kwarg. `pivot_kws` is a dict with entries
        index, column, and values that let clustermap know how to reshape the
        data. If the query does not follow the standard ESTIMATE PAIRWISE
        output, it may be necessary to define `pivot_kws`.

    Returns
    -------
    seaborn.clustermap
    """
    if clustermap_kws is None:
        clustermap_kws = {}

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
        return sns.heatmap(df, **clustermap_kws), row_ordering, col_ordering
    else:
        cm = sns.clustermap(data_df, **clustermap_kws)
        return (cm, cm.dendrogram_row.reordered_ind,
                cm.dendrogram_row.reordered_ind,)


# TODO: bdb, and table_name should be optional arguments
def pairplot(df, bdb=None, generator_name=None, use_shortname=False,
             show_contour=False, colorby=None, show_missing=False, no_tril=True):
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
    use_shortname : bool
        If True, use column shortnames (requires codebook) for axis lables,
        otherwise use the column names in `df`.
    show_contour : bool
        If True, KDE contours are plotted on top of scatter plots
        and histograms.
    show_missing : bool
        If True, rows with one missing value are plotted as lines on scatter
        plots.
    colorby : str
        Name of a column to use to color data points in histograms and scatter
        plots.
    no_tril : bool
        Show all axes, not only lower diagonal.

    Returns
    -------
    plt_grid : matplotlib.gridspec.GridSpec
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
            raise ValueError('colorby column, {}, not found.'.format(colorby))
        elif n_colorby > 1:
            raise ValueError('Multiple columns named, {}.'.format(colorby))

        dummy = data_df[colorby].dropna()
        dvals = np.sort(dummy.unique())
        ndvals = len(dvals)
        dval_type = get_bayesdb_col_type('colorby', dummy, bdb=bdb,
                                         generator_name=generator_name)
        if dval_type.lower() != 'categorical':
            raise ValueError('colorby columns must be categorical.')
        cmap = sns.color_palette("Set1", ndvals)
        colors = {}
        for val, color in zip(dvals, cmap):
            colors[val] = color

    all_varnames = [c for c in data_df.columns if c != colorby]
    n_vars = len(all_varnames)
    plt_grid = gridspec.GridSpec(n_vars, n_vars)

    # if there is only one variable, just do a hist
    if n_vars == 1:
        ax = plt.gca()
        varname = data_df.columns[0]
        vartype = get_bayesdb_col_type(varname, data_df[varname], bdb=bdb,
                                       generator_name=generator_name)
        do_hist(data_df, dtype=vartype, ax=ax, bdb=bdb,
                generator_name=generator_name,
                colors=colors)
        if vartype == 'categorical':
            rotate_tick_labels(ax)
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

            ax = plt.subplot(plt_grid[y_pos, x_pos])
            axes[y_pos].append(ax)

            if x_pos == y_pos:
                varnames = [var_name_x]
                if colorby is not None:
                    varnames.append(colorby)
                ax = do_hist(data_df[varnames], dtype=var_x_type, ax=ax,
                             bdb=bdb, generator_name=generator_name, 
                             colors=colors)
            else:
                varnames = [var_name_x, var_name_y]
                vartypes_pair = (var_x_type, var_y_type,)
                if colorby is not None:
                    varnames.append(colorby)
                plot_df = prep_plot_df(data_df, varnames)
                ax = do_pair_plot(plot_df, vartypes_pair, ax=ax, bdb=bdb,
                                  generator_name=generator_name,
                                  show_contour=show_contour,
                                  show_missing=show_missing,
                                  colors=colors)

                ymins[y_pos, x_pos] = ax.get_ylim()[0]
                ymaxs[y_pos, x_pos] = ax.get_ylim()[1]
                xmins[y_pos, x_pos] = ax.get_xlim()[0]
                xmaxs[y_pos, x_pos] = ax.get_xlim()[1]

            ax.set_xlabel(var_name_x, fontweight = 'bold')
            ax.set_ylabel(var_name_y, fontweight = 'bold')

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
    if no_tril:
        fake_axis_ticks(axes[0][0], axes[0][1])
    fake_axis_ticks(axes[-1][-1], axes[-1][0])

    if colorby is not None:
        legend = gen_collapsed_legend_from_dict(colors, title=colorby)
        legend.draggable()

    # tril by default by deleting upper diagonal axes.
    if not no_tril:
        for y_pos in range(n_vars):
            for x_pos in range(y_pos+1, n_vars):
                plt.gcf().delaxes(axes[y_pos][x_pos])

    return plt_grid


def comparative_hist(df, nbins=15, normed=False, bdb=None):
    """Plot a histogram

    Given a one-column pandas.DataFrame, df, plots a simple histogram. Given a
    two-column df plots the data in columns one separated by a a dummy variable
    assumed to be in column 2.

    Parameters
    ----------
    nbins : int
        Number of bins (bars)
    normed : bool
        If True, normalizes the the area of the histogram (or each
        sub-histogram if df has two columns) to 1.
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
            raise ValueError("I don't know what to do with data with more"
                             "than two columns")
        colorby = df.columns[1]
        colorby_vals = df[colorby].unique()

    plt.figure(tight_layout=False, facecolor='white')
    if colorby is None:
        plt.hist(df.ix[:, 0].values, bins=bins, color='#383838',
                 edgecolor='none', normed=normed)
        plot_title = df.columns[0]
    else:
        colors = sns.color_palette('deep', len(colorby_vals))
        for color, cbv in zip(colors, colorby_vals):
            subdf = df[df[colorby] == cbv]
            plt.hist(subdf.ix[:, 0].values, bins=bins, color=color, alpha=.5,
                     edgecolor='none', normed=normed, label=str(cbv))
        plt.legend(loc=0, title=colorby)
        plot_title = df.columns[0] + " by " + colorby

    if normed:
        plot_title += " (normalized)"

    plt.title(plot_title)
    plt.xlabel(df.columns[0])


if __name__ == '__main__':
    from bdbcontrib import facade
    import os

    if os.path.isfile('plttest.bdb'):
        os.remove('plttest.bdb')

    df = pd.DataFrame()
    num_rows = 400
    alphabet = ['A', 'B', 'C', 'D', 'E']
    col_0 = np.random.choice(range(5), num_rows,
                             p=np.array([1, .4, .3, .2, .1])/2.)
    col_1 = [np.random.randn()+x for x in col_0]
    col_0 = [alphabet[i] for i in col_0]

    df['zero_5'] = col_0
    df['one_n'] = col_1

    col_four = np.random.choice(range(4), num_rows, p=[.4, .3, .2, .1])
    col_five = [(np.random.randn()-2*x)/(1+x) for x in col_four]

    df['three_n'] = np.random.randn(num_rows)
    df['four_8'] = col_four
    df['five_c'] = col_five

    filename = 'plottest.csv'
    df.to_csv(filename)

    cc_client = facade.BayesDBClient.from_csv('plttest.bdb', 'plottest',
                                              filename)

    # do a plot where a some sub-violins are removed
    remove_violin_bql = """
        DELETE FROM plottest
            WHERE zero_5 = "B"
            AND (four_8 = 2 OR four_8 = 1);
    """
    cc_client.bdb.sql_execute(remove_violin_bql)
    df = cc_client('SELECT one_n, zero_5, five_c, four_8 FROM plottest')
    df = df.as_df()

    plt.figure(tight_layout=True, facecolor='white')
    pairplot(df, bdb=cc_client.bdb, generator_name='plottest_cc',
             use_shortname=False, colorby='four_8', show_contour=False,
             no_tril=False)
    plt.show()

    # again, without tril to check that outer axes render correctly
    plt.figure(tight_layout=True, facecolor='white')
    pairplot(df, bdb=cc_client.bdb, generator_name='plottest_cc',
             use_shortname=False, colorby='four_8', show_contour=True,
             no_tril=False)
    plt.show()
