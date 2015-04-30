
import bdbcontrib.crosscat_utils as ccu

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import copy
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


def get_bayesdb_col_type(column_name, df_column, bdb=None,
                         generator_name=None):
    # If column_name is a column label (not a short name!) then the modeltype
    # of the column will be returned otherwise we guess.

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
        theta = ccu.get_M_c(bdb, generator_name)
        try:
            col_idx = theta['name_to_idx'][column_name]
            modeltype = theta['column_metadata'][col_idx]['modeltype']
            coltype = MODEL_TO_TYPE_LOOKUP[modeltype]
            # XXX: Force cyclic -> numeric because there is no need to plot
            # cyclic data any differently until we implement rose plots. See
            # http://matplotlib.org/examples/pie_and_polar_charts/polar_bar_demo.html
            # for an example.
            if coltype.lower() == 'cyclic':
                coltype = 'numerical'
            return coltype
        except KeyError:
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


def do_hist(data_srs, ax=None, dtype=None, bdb=None, generator_name=None):
    if dtype is None:
        dtype = get_bayesdb_col_type(data_srs.columns[0], data_srs, bdb=bdb,
                                     generator_name=generator_name)

    if ax is None:
        ax = plt.gca()

    if dtype == 'categorical':
        vals, uvals, _ = conv_categorical_vals_to_numeric(
            data_srs, bdb=bdb, generator_name=generator_name)
        ax.hist(vals, bins=len(uvals))
        ax.set_xticks(range(len(uvals)))
        ax.set_xticklabels(uvals)
    else:
        sns.distplot(data_srs, kde=True, ax=ax)

    return ax


def do_heatmap(plot_df, vartypes, ax=None, bdb=None, generator_name=None):
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


def do_violinplot(plot_df, vartypes, ax=None, bdb=None, generator_name=None):
    if ax is None:
        ax = plt.gca()
    assert vartypes[0] != vartypes[1]
    vert = vartypes[1] == 'numerical'
    plot_df = copy.deepcopy(plot_df)
    if vert:
        groupby = plot_df.columns[0]
        vals = plot_df.columns[1]
    else:
        groupby = plot_df.columns[1]
        vals = plot_df.columns[0]

    _, unique_vals, _ = conv_categorical_vals_to_numeric(
        plot_df[groupby], bdb=bdb, generator_name=generator_name)

    sns.violinplot(plot_df[vals], groupby=plot_df[groupby], order=unique_vals,
                   names=unique_vals, vert=vert, ax=ax, positions=0)
    n_vals = len(plot_df[groupby].unique())

    if vert:
        ax.set_xlim([-.5, n_vals-.5])
        # ax.set_xticklabels(unique_vals)
    else:
        ax.set_ylim([-.5, n_vals-.5])
        # ax.set_yticklabels(unique_vals)

    return ax


def do_kdeplot(plot_df, vartypes, ax=None, bdb=None, generator_name=None):
    # XXX: kdeplot is not a good choice for small amounts of data because
    # it uses a kernel density estimator to crease a smooth heatmap. On the
    # other hadnd, scatter plots are uniformative given lots of data---the
    # points get jumbled up. We may just want to set a threshold (N=100)?
    if ax is None:
        ax = plt.gca()

    assert plot_df.shape[1] == 2

    plt.scatter(plot_df.values[:, 0], plot_df.values[:, 1], alpha=.5,
                color='steelblue')
    sns.kdeplot(plot_df.values, ax=ax)
    return ax


# No support for cyclic at this time
DO_PLOT_FUNC = dict()
DO_PLOT_FUNC[hash(('categorical', 'categorical',))] = do_heatmap
DO_PLOT_FUNC[hash(('categorical', 'numerical',))] = do_violinplot
DO_PLOT_FUNC[hash(('numerical', 'categorical',))] = do_violinplot
DO_PLOT_FUNC[hash(('numerical', 'numerical',))] = do_kdeplot


def do_pair_plot(plot_df, vartypes, ax=None, bdb=None, generator_name=None):
    # determine plot_types
    if ax is None:
        ax = plt.gca()

    ax = DO_PLOT_FUNC[hash(vartypes)](plot_df, vartypes, ax=ax, bdb=bdb,
                                      generator_name=bdb)
    return ax


def zmatrix(data_df, clustermap_kws=None):
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
        pivot_kws = {
            'index': data_df.columns[-3],
            'columns': data_df.columns[-2],
            'values': data_df.columns[-1],
        }
        clustermap_kws['pivot_kws'] = pivot_kws

    if clustermap_kws.get('cmap', None) is None:
        # Choose a soothing blue colormap
        clustermap_kws['cmap'] = 'PuBu'

    return sns.clustermap(data_df, **clustermap_kws)


# TODO: bdb, and table_name should be optional arguments
def pairplot(df, bdb=None, generator_name=None, use_shortname=False):
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
    # where to handle dropping NaNs? Missing values may be informative.

    data_df = df.dropna()

    n_vars = data_df.shape[1]
    plt_grid = gridspec.GridSpec(n_vars, n_vars)

    # if there is only one variable, just do a hist
    if n_vars == 1:
        ax = plt_grid[0, 0]
        varname = data_df.columns[0]
        vartype = get_bayesdb_col_type(varname, data_df[varname], bdb=bdb,
                                       generator_name=generator_name)
        do_hist(data_df[varname], dtype=vartype, ax=ax, bdb=bdb,
                generator_name=generator_name)
        if vartype == 'categorical':
            rotate_tick_labels(ax)
        return

    xmins = np.ones((n_vars, n_vars))*float('Inf')
    xmaxs = np.ones((n_vars, n_vars))*float('-Inf')
    ymins = np.ones((n_vars, n_vars))*float('Inf')
    ymaxs = np.ones((n_vars, n_vars))*float('-Inf')

    vartypes = []
    for varname in data_df.columns:
        vartype = get_bayesdb_col_type(varname, data_df[varname], bdb=bdb,
                                       generator_name=generator_name)
        vartypes.append(vartype)

    for x_pos, var_name_x in enumerate(data_df.columns):
        var_x_type = vartypes[x_pos]
        for y_pos, var_name_y in enumerate(data_df.columns):
            var_y_type = vartypes[y_pos]

            ax = plt.subplot(plt_grid[y_pos, x_pos])

            if x_pos == y_pos:
                ax = do_hist(data_df[var_name_x], dtype=var_x_type, ax=ax,
                             bdb=bdb, generator_name=generator_name)
            else:
                varnames = (var_name_x, var_name_y,)
                vartypes_pair = (var_x_type, var_y_type,)
                plot_df = prep_plot_df(data_df, varnames)
                ax = do_pair_plot(plot_df, vartypes_pair, ax=ax, bdb=bdb,
                                  generator_name=generator_name)

                ymins[y_pos, x_pos] = ax.get_ylim()[0]
                ymaxs[y_pos, x_pos] = ax.get_ylim()[1]
                xmins[y_pos, x_pos] = ax.get_xlim()[0]
                xmaxs[y_pos, x_pos] = ax.get_xlim()[1]

            ax.set_xlabel(var_name_x)
            ax.set_ylabel(var_name_y)

    for x_pos in range(n_vars):
        for y_pos in range(n_vars):
            ax = plt.subplot(plt_grid[y_pos, x_pos])
            ax.set_xlim([np.min(xmins[:, x_pos]), np.max(xmaxs[:, x_pos])])
            if x_pos != y_pos:
                ax.set_ylim([np.min(ymins[y_pos, :]), np.max(ymaxs[y_pos, :])])
            if x_pos > 0:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            if y_pos < n_vars - 1:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            else:
                if vartype[x_pos] == 'categorical':
                    rotate_tick_labels(ax)

    return plt_grid


def comparative_hist(df, nbins=15, normed=False):
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

    vartype = get_bayesdb_col_type(df.columns[0], df[df.columns[0]])
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
        plt.hist(df.ix[:, 0].values, bins=bins, color='#383838')
    else:
        colors = sns.color_palette('deep', len(colorby_vals))
        for color, cbv in zip(colors, colorby_vals):
            subdf = df[df[colorby] == cbv]
            plt.hist(subdf.ix[:, 0].values, bins=bins, color=color, alpha=.7,
                     normed=normed, label=str(cbv))
        plt.legend(loc=0)


if __name__ == '__main__':
    import pandas as pd
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
    df = cc_client('SELECT one_n, zero_5, five_c, four_8 FROM plottest')
    df = df.as_df()

    plt.figure(tight_layout=True, facecolor='white')
    pairplot(df, bdb=cc_client.bdb, generator_name='plottest_cc',
             use_shortname=False)
    plt.show()

    df = cc_client('SELECT three_n + one_n, three_n * one_n,'
                   ' zero_5 || four_8 FROM plottest').as_df()

    plt.figure(tight_layout=True, facecolor='white')
    pairplot(df,  use_shortname=False)
    plt.show()
