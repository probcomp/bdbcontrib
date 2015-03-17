
# pylint: disable=E1103

import bayeslite
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import copy
import numpy as np


MODEL_TO_TYPE_LOOKUP = {
    'normal_inverse_gamma': 'numerical',
    'symmetric_dirichlet_discrete': 'categorical',
}


def get_bayesdb_col_type(bdb, table_name, column_name, df_column):
    """
    If column_name is a column label (not a short name!) then the modeltype of the column will be
    returned otherwise we guess.
    """
    table_id = bayeslite.core.bayesdb_table_id(bdb, table_name)
    theta = bayeslite.core.bayesdb_metadata(bdb, table_id)
    try:
        col_idx = theta['name_to_idx'][column_name]
        return MODEL_TO_TYPE_LOOKUP[theta['column_metadata'][col_idx]['modeltype']]
    except KeyError:
        # FIXME: Not a table column; must use heurtistic
        pd_type = df_column.dtype
        if pd_type is str:
            return 'categorical'
        else:
            if len(df_column.unique()) < 30:
                return 'categorical'
            else:
                return 'numerical'
    except Exception as err:
        print "Unexpected exception: {}".format(err)
        raise err


def conv_categorical_vals_to_numeric(bdb, bdb_table_name, data_srs):
    # TODO: get real valuemap from btable
    unique_vals = sorted(data_srs.unique().tolist())
    lookup = dict(zip(unique_vals, range(len(unique_vals))))
    values = data_srs.values.tolist()
    for i, x in enumerate(values):
        values[i] = lookup[x]
    return np.array(values, dtype=float), unique_vals, lookup


# FIXME: STUB
def prep_plot_df(bdb, table_name, data_df, var_names, vartypes):
    return data_df[list(var_names)]


def do_hist(bdb, table_name, data_srs, ax=None, dtype=None):
    if dtype is None:
        dtype = get_bayesdb_col_type(bdb, table_name, data_srs.columns[0], data_srs)

    if ax is None:
        ax = plt.gca()

    if dtype == 'categorical':
        vals, uvals, _ = conv_categorical_vals_to_numeric(bdb, table_name, data_srs)
        ax.hist(vals, bins=len(uvals))
        ax.set_xticks(range(len(uvals)))
        ax.set_xticklabels(uvals)
        # data_srs.hist(ax=ax, bins=range(len(data_srs.unique())+1), width=1)
    else:
        sns.distplot(data_srs, kde=True, ax=ax)

    return ax


def do_heatmap(bdb, table_name, plot_df, vartypes, ax=None):
    if ax is None:
        ax = plt.gca()

    vals_x, uvals_x, _ = conv_categorical_vals_to_numeric(bdb, table_name, plot_df.ix[:, 0])
    vals_y, uvals_y, _ = conv_categorical_vals_to_numeric(bdb, table_name, plot_df.ix[:, 1])

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


def do_violinplot(bdb, table_name, plot_df, vartypes, ax=None):
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

    _, unique_vals, _ = conv_categorical_vals_to_numeric(bdb, table_name, plot_df[groupby])

    sns.violinplot(plot_df[vals], groupby=plot_df[groupby], order=unique_vals, names=unique_vals,
                   vert=vert, ax=ax, positions=0)
    n_vals = len(plot_df[groupby].unique())

    if vert:
        ax.set_xlim([-.5, n_vals-.5])
        # ax.set_xticklabels(unique_vals)
    else:
        ax.set_ylim([-.5, n_vals-.5])
        # ax.set_yticklabels(unique_vals)

    return ax


def do_kdeplot(bdb, table_name, plot_df, vartypes, ax=None):
    # XXX: kdeplot is not a good choice for small amounts of data because
    # it uses a kernel density estimator to crease a smooth heatmap. On the
    # other hadnd, scatter plots are uniformative given lots of data---the
    # points get jumbled up. We may just want to set a threshold (N=100)?
    if ax is None:
        ax = plt.gca()

    assert plot_df.shape[1] == 2

    plt.scatter(plot_df.values[:, 0], plot_df.values[:, 1], alpha=.5, color='steelblue')
    sns.kdeplot(plot_df.values, ax=ax)
    return ax


DO_PLOT_FUNC = dict()
DO_PLOT_FUNC[hash(('categorical', 'categorical',))] = do_heatmap
DO_PLOT_FUNC[hash(('categorical', 'numerical',))] = do_violinplot
DO_PLOT_FUNC[hash(('numerical', 'categorical',))] = do_violinplot
DO_PLOT_FUNC[hash(('numerical', 'numerical',))] = do_kdeplot


def do_pair_plot(bdb, table_name, plot_df, vartypes, ax=None):
    # determine plot_types
    if ax is None:
        ax = plt.gca()

    ax = DO_PLOT_FUNC[hash(vartypes)](bdb, table_name, plot_df, vartypes, ax=ax)
    return ax


def zmatrix(data_df, clustermap_kws=None):
    """
    Plots a clustermap from and ESTIMATE PAIRWISE query.
    Args:
        - data_df (pandas.DataFrame): The result of the query in pandas form.
    Kwargs:
        - clustermap_kws (dict): kwargs for seaborn.clustermap. See seaborn
        documentation. Of particular importance is the pivot_kws kwarg. pivot_kws
        is a dict with entries index, column, and values that let clustermap know
        how to reshape the data. If the query does not follow the standard
        ESTIMATE PAIRWISE output, it may be necessary to define pivot_kws
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
def pairplot(bdb, table_name, df, use_shortname=False):
    """
    Plots the columns in data_df in a facet grid.
    - categorical-categorical pairs are displayed as a heatmap
    - continuous-continuous pairs are displayed as a kdeplot
    - categorical-continuous pairs are displayed on a violin plot
    XXX: support soon for ordered continuous combinations. It may be best to
    plot all ordered continuous pairs as heatmap.
    """
    # NOTE:Things to consider:
    # - String values are a possibility (categorical)
    # - who knows what the columns are named. What is the user selects columns
    #   as shortname?
    # where to handle dropping NaNs? Missing values may be informative.

    data_df = df.dropna()

    n_vars = data_df.shape[1]
    plt_grid = gridspec.GridSpec(n_vars, n_vars)

    xmins = np.ones((n_vars, n_vars))*float('Inf')
    xmaxs = np.ones((n_vars, n_vars))*float('-Inf')
    ymins = np.ones((n_vars, n_vars))*float('Inf')
    ymaxs = np.ones((n_vars, n_vars))*float('-Inf')
    for x_pos, var_name_x in enumerate(data_df.columns):
        var_x_type = get_bayesdb_col_type(bdb, table_name, var_name_x, data_df[var_name_x])
        for y_pos, var_name_y in enumerate(data_df.columns):
            var_y_type = get_bayesdb_col_type(bdb, table_name, var_name_y, data_df[var_name_y])

            ax = plt.subplot(plt_grid[y_pos, x_pos])

            if x_pos == y_pos:
                ax = do_hist(bdb, table_name, data_df[var_name_x], dtype=var_x_type, ax=ax)
            else:
                varnames = (var_name_x, var_name_y,)
                vartypes = (var_x_type, var_y_type,)
                plot_df = prep_plot_df(bdb, table_name, data_df, varnames, vartypes)
                ax = do_pair_plot(bdb, table_name, plot_df, vartypes, ax=ax)

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


if __name__ == '__main__':
    import pandas as pd
    from bdbcontrib import facade
    import os

    if os.path.isfile('plttest.bdb'):
        os.remove('plttest.bdb')

    df = pd.DataFrame()
    num_rows = 400
    alphabet = ['A', 'B', 'C', 'D', 'E']
    col_0 = np.random.choice(range(5), num_rows, p=np.array([1, .4, .3, .2, .1])/2.)
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

    cc_client = facade.BayesDBCrossCat.from_csv('plttest.bdb', 'plottest', filename)
    df = cc_client('SELECT one_n, zero_5, five_c, four_8 FROM plottest').as_df()

    plt.figure(tight_layout=True, facecolor='white')
    pairplot(cc_client.bdb, 'plottest', df, use_shortname=False)
    plt.show()

    df = cc_client('SELECT three_n + one_n, three_n * one_n,'
                   ' zero_5 || four_8 FROM plottest').as_df()

    plt.figure(tight_layout=True, facecolor='white')
    pairplot(cc_client.bdb, 'plottest', df, use_shortname=False)
    plt.show()
