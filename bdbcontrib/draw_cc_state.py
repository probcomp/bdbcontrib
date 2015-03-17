"""
Set of plotting utilities for use wil bayesdb/crosscat
"""
# pylint: disable=E1103

import numpy as np
from matplotlib.patches import Rectangle
from crosscat.utils import sample_utils as su
from matplotlib import pyplot as plt
import matplotlib as mpl

import bayeslite


NO_CMAP = mpl.colors.ListedColormap([(.5, .5, .5, 1), (.5, .5, .5, 1)])


def get_cols_in_view(X_L, view):
    return [c for c, v in enumerate(X_L['column_partition']['assignments']) if v == view]


def get_rows_in_cluster(X_D, view, cluster):
    return [r for r, c in enumerate(X_D[view]) if c == cluster]


def get_short_names(bdb, table_id, column_names):
    short_names = []
    # XXX: this is wasteful.
    bql = '''
        SELECT c.colno, c.name, c.short_name
            FROM bayesdb_table AS t, bayesdb_table_column AS c
            WHERE t.id = ? AND c.table_id = t.id
        '''
    curs = bdb.sql_execute(bql, (table_id,))
    records = curs.fetchall()
    for cname in column_names:
        for record in records:
            if record[1] == cname:
                sname = record[2]
                if sname is None:
                    sname = cname
                    print 'Warning: No short name found for {}. Using column name.'.format(cname)
                short_names.append(sname)
                break

    assert len(short_names) == len(column_names)
    return short_names


def get_row_probabilities(X_L, X_D, M_c, T, view):
    """
    Returns the predictive probability of the data in each row of T in view.
    """
    num_rows = len(X_D[0])
    cols_in_view = get_cols_in_view(X_L, view)
    num_clusters = max(X_D[view])+1
    cluster_models = [su.create_cluster_model_from_X_L(M_c, X_L, view, c)
                      for c in range(num_clusters)]

    logps = np.zeros(num_rows)
    for row in range(num_rows):
        cluster_idx = X_D[view][row]
        for col in cols_in_view:
            x = T[row][col]
            if np.isnan(x):
                continue
            component_model = cluster_models[cluster_idx][col]
            component_model.remove_element(x)
            logps[row] += component_model.calc_element_predictive_logp(x)
            component_model.insert_element(x)

    assert len(logps) == len(X_D[0])
    return logps


def get_column_probabilities(X_L, M_c):
    """
    Returns the marginal probability of each column.
    """
    num_cols = len(X_L['column_partition']['assignments'])
    num_views = len(X_L['view_state'])
    logps = np.zeros(num_cols)
    for view in range(num_views):
        num_clusters = len(X_L['view_state'][view]['row_partition_model']['counts'])
        for cluster in range(num_clusters):
            cluster_model = su.create_cluster_model_from_X_L(M_c, X_L, view, cluster)
            for col in get_cols_in_view(X_L, view):
                component_model = cluster_model[col]
                logps[col] += component_model.calc_marginal_logp()
    return logps


def sort_state(X_L, X_D, M_c, T):
    """
    Sort views from largest to smallest. W/in views, sorts columns r->l from
    highest marginal probability to lowest; and sorts categories largest to
    smallest from top to bottom. Within categories, sorts rows higest to lowest
    probability from top to bottom.

    Notes:
        - It will require initialing component models to calculate these
        probabilities.
    Inputs:
        X_L:
        X_D:
        M_c:
        T
    Returns:
        sorted_views (list): view indices from largest (most columns) to
            smallest.
        sorted_clusters (dict<list>) sorted_clusters[v] is a sorted list of
            clusters in view v from largest (most rows) to smallest.
        sorted_cols (dict<list>): sorted_columns[v] is a sorted list of
            columns in view v from higest to lowest probability.
        sorted_rows (dict<dict<list>>): sorted_rows[v][c] is a sorted list of
            rows in cluster c of view v from higest to lowest probability.
    """
    num_views = len(X_L['view_state'])

    column_logps = get_column_probabilities(X_L, M_c)

    cols_by_view = [get_cols_in_view(X_L, v) for v in range(num_views)]
    num_cols_in_view = [len(cv) for cv in cols_by_view]
    sorted_views = [i[0] for i in sorted(enumerate(num_cols_in_view), reverse=True,
                    key=lambda x:x[1])]

    sorted_cols = {}
    sorted_rows = {}
    sorted_clusters = {}
    for view in sorted_views:
        num_clusters = len(X_L['view_state'][view]['row_partition_model']['counts'])

        # get all columns in the view and sort by probability.
        col_prob = {}
        for col in cols_by_view[view]:
            logp = column_logps[col]
            # index by logp because that is how we're going to sort
            col_prob[logp] = col

        sorted_cols[view] = [col_prob[p] for p in sorted(col_prob.keys(), reverse=True)]

        # sort clusters by size
        rows_by_cluster = [get_rows_in_cluster(X_D, view, clstr) for clstr in range(num_clusters)]
        num_rows_in_clstr = [len(rc) for rc in rows_by_cluster]
        sorted_clusters[view] = [i[0] for i in sorted(enumerate(num_rows_in_clstr), reverse=True,
                                 key=lambda x:x[1])]

        row_logps = get_row_probabilities(X_L, X_D, M_c, T, view)
        sorted_rows_view = {}
        num_rows_accounted_for = 0
        for clstr in sorted_clusters[view]:
            rows_in_clstr = get_rows_in_cluster(X_D, view, clstr)
            row_prob = [row_logps[row] for row in rows_in_clstr]
            sorted_row_clstr = [rows_in_clstr[i] for i in np.argsort(row_prob)][::-1]

            assert len(rows_in_clstr) == len(sorted_row_clstr)

            num_rows_accounted_for += len(sorted_row_clstr)
            sorted_rows_view[clstr] = sorted_row_clstr

        assert num_rows_accounted_for == len(X_D[0])

        sorted_rows[view] = sorted_rows_view

    return sorted_views, sorted_clusters, sorted_cols, sorted_rows


def cmap_color_brightness(value, base_color, vmin, vmax, nan_color=(1., 0., 0., 1.)):
    """
    Darken by value. Lighter values are higher. NaN values are red.
    """
    # XXX: bayesdb_data never retruns NaN for multinomial---NaN is added to value_map
    if np.isnan(value):
        return nan_color

    span = vmax - vmin
    vmin -= span
    brightness = (value-vmin)/(span)
    color = np.array([min(c*brightness, 1.) for c in list(base_color)])

    color[3] = 1.

    return color


def gen_hilight_colors(hl_labels=None, hl_colors=None):
    """
    Generates a hilight color lookup from labels to colors. Generates labels from Set1 by default.
    """
    if hl_labels is None:
        hl_labels = []

    hl_colors_out = {}
    if hl_colors is None:
        if len(hl_labels) > 0:
            hl_max = float(len(hl_labels))
            hl_colors_out = dict([(i, plt.cm.Set1(i/hl_max)) for i in hl_labels])
    else:
        if not isinstance(hl_colors, list):
            raise TypeError('hl_colors must be a list')
        if len(hl_colors) != len(hl_labels):
            raise ValueError('hl_colors must have an entry for each entry in hl_labels')
        hl_colors_out = dict(zip(hl_labels, hl_colors))

    return hl_colors_out


def gen_blank_sort(num_rows, num_cols):
    """
    Generates a 'blank sort' which allows state plotting of the raw data without structure.
    """
    sorted_views = [0]
    sorted_clusters = {}
    sorted_clusters[0] = [0]
    sorted_rows = {}
    sorted_rows[0] = {}
    sorted_rows[0][0] = range(num_rows)
    sorted_cols = {}
    sorted_cols[0] = range(num_cols)

    return sorted_views, sorted_clusters, sorted_cols, sorted_rows


def gen_cell_colors(T, sorted_views, sorted_cols, sorted_clusters, sorted_rows,
                            column_partition, cmap, border_width, nan_color=(1., 0., 0., 1.)):
    # generate a heatmap using the data (allows clusters to ahve different base colors)
    num_rows = len(T)
    num_cols = len(T[0])
    num_views = len(sorted_views)

    base_cmapper = np.zeros((num_rows, num_cols+num_views*border_width))

    x_pos = 0
    for view_count, view in enumerate(sorted_views):
        for col in sorted_cols[view]:
            y_pos = 0
            for cluster in sorted_clusters[view]:
                for row in sorted_rows[view][cluster]:
                    base_cmapper[y_pos, x_pos + view_count*border_width] = float(cluster)
                    y_pos += 1
            x_pos += 1

    base_cmapper /= np.max(base_cmapper)
    cell_colors = np.zeros((base_cmapper.shape[0], base_cmapper.shape[1], 4))
    col_indices = []
    for view in sorted_views:
        idx = sorted_cols[view]
        col_indices += idx + [-1]*border_width
    for x_pos, col in enumerate(col_indices):
        if col < 0:
            cell_colors[:, col, :] = np.array([1, 1, 1, 1])
            continue

        view = column_partition[col]
        col_cpy = np.copy(T[:, col])
        col_cpy = col_cpy[np.isfinite(col_cpy)]
        cmin = np.min(col_cpy)
        cmax = np.max(col_cpy)
        y_pos = 0
        for clstr in sorted_clusters[view]:
            for row in sorted_rows[view][clstr]:
                value = T[row, col]
                base_color = cmap(base_cmapper[y_pos, x_pos])
                color = cmap_color_brightness(value, base_color, cmin, cmax, nan_color=nan_color)
                cell_colors[y_pos, x_pos, :] = color
                y_pos += 1

    return cell_colors


def draw_state(bdb, table_name, modelno,
               ax=None, border_width=3, row_label_col=None, short_names=True,
               hilight_rows=[], hilight_rows_colors=None,
               hilight_cols=[], hilight_cols_colors=None,
               separator_color='red', separator_width=4,
               blank_state=False, nan_color=(1., 0., 0., 1.)):

    table_id = bayeslite.bayesdb_table_id(bdb, table_name)

    theta = bayeslite.core.bayesdb_model(bdb, table_id, modelno)
    T = [row for row in bayeslite.core.bayesdb_data(bdb, 1)]
    M_c = bayeslite.core.bayesdb_metadata(bdb, table_id)
    X_L = theta['X_L']
    X_D = theta['X_D']

    num_rows = len(T)
    num_cols = len(T[0])

    if not blank_state:
        sorted_views, sorted_clusters, sorted_cols, sorted_rows = sort_state(X_L, X_D, M_c, T)
        column_partition = X_L['column_partition']['assignments']
    else:
        sorted_views, sorted_clusters, sorted_cols, sorted_rows = gen_blank_sort(num_rows, num_cols)
        column_partition = [0]*num_cols

    # set colormap to 50% gray
    cmap = NO_CMAP
    T = np.array(T)

    num_views = len(sorted_cols)
    X = np.zeros((num_rows, num_cols+num_views*border_width))

    # row hilighting
    row_hl_colors = gen_hilight_colors(hilight_rows, hilight_rows_colors)

    hl_row_idx_label_zip = []
    if row_label_col is None:
        row_labels = [str(i) for i in range(num_rows)]
    elif isinstance(row_label_col, list):
        if len(row_label_col) != num_rows:
            raise TypeError("If row_label_col is a list, it must have an entry for each row")
        row_labels = [str(label) for label in row_label_col]
    elif isinstance(row_label_col, str):
        label_col_idx = M_c['name_to_idx'][row_label_col]
        label = [str(T[row, label_col_idx]) for row in range(num_rows)]
    else:
        raise TypeError("I don't know what to do with a {} "
                        "row_label_col".format(type(row_label_col)))

    row_idx_to_label = {}
    row_label_to_idx = {}
    for row, label in enumerate(row_labels):
        # XXX: Allows missing enries to be a column label
        row_idx_to_label[row] = label
        row_label_to_idx[label] = row

    for label in hilight_rows:
        hl_row_idx_label_zip.append((row_label_to_idx[label], label,))

    # column hilighting
    col_hl_colors = gen_hilight_colors(hilight_cols, hilight_cols_colors)

    hl_col_idx_label_zip = []
    for label in hilight_cols:
        hl_col_idx_label_zip.append((M_c['name_to_idx'][label], label,))

    # generate a heatmap using the data (allows clusters to ahve different base colors)
    cell_colors = gen_cell_colors(T, sorted_views, sorted_cols, sorted_clusters,
                                  sorted_rows, column_partition, cmap, border_width,
                                  nan_color=nan_color)

    x_tick_labels = []
    x_labels = []

    if ax is None:
        ax = plt.gca()

    ax.imshow(cell_colors, cmap=cmap, interpolation='nearest', origin='upper', aspect='auto')
    col_count = 0
    for v, view in enumerate(sorted_views):
        view_x_labels = [M_c['idx_to_name'][str(col)] for col in sorted_cols[view]]
        if short_names:
            view_x_tick_labels = get_short_names(bdb, table_id, view_x_labels)
        else:
            view_x_tick_labels = view_x_labels

        y_tick_labels = []

        for i, vxtl in enumerate(view_x_labels):
            if vxtl in hilight_cols:
                edgecolor = col_hl_colors[vxtl]
                x_a = len(x_tick_labels) + i - .5
                ax.add_patch(Rectangle((x_a, -.5), 1, num_rows, facecolor="none",
                                       edgecolor=edgecolor, lw=2, zorder=10))

        x_tick_labels += view_x_tick_labels + [' ']*border_width
        x_labels += view_x_labels + ['_']*border_width
        num_cols_view = len(sorted_cols[view])
        sbplt_start = col_count+v*border_width
        sbplt_end = col_count+num_cols_view+v*border_width

        y = 0
        for cluster in sorted_clusters[view]:
            y_tick_labels += [row_idx_to_label[row] for row in sorted_rows[view][cluster]]
            ax.plot([sbplt_start-.5, sbplt_end-.5], [y-.5, y-.5], color=separator_color,
                    lw=separator_width)
            for row, label in hl_row_idx_label_zip:
                try:
                    pos = sorted_rows[view][cluster].index(row)
                except ValueError:
                    pos = None

                if pos is not None:
                    edgecolor = row_hl_colors[label]
                    ax.add_patch(Rectangle((sbplt_start - .5, y + pos - .5),
                                           num_cols_view, 1, facecolor="none", edgecolor=edgecolor,
                                           lw=2, zorder=10))

            y += len(sorted_rows[view][cluster])

        for i, row in enumerate(range(num_rows-1, -1, -1)):
            if y_tick_labels[i] in hilight_rows:
                fontcolor = row_hl_colors[y_tick_labels[i]]
                fontsize = 12
                fontweight = 'bold'
            else:
                fontsize = 8
                fontcolor = '#333333'
                fontweight = 'light'

            ax.text(sbplt_start-1, i+.5, str(y_tick_labels[i]), ha='right', fontsize=fontsize,
                    color=fontcolor, weight=fontweight)
        col_count += num_cols_view

    ax.tick_params(**{
        'axis': 'both',
        'length': 0
    })
    ax.set_xlim([-.5, X.shape[1]])
    ax.set_ylim([X.shape[0], -.5])
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_yticks(range(num_rows))
    ax.set_xticks(range(num_cols+num_views*border_width))
    ax.tick_params(axis='x', colors='white')
    ax.set_xticklabels(x_tick_labels, rotation=90, color='black', fontsize=9)
    ax.set_yticklabels(['']*num_rows)
    ax.tick_params(axis='y', colors='white')
    ax.grid(b=False)
    return ax


if __name__ == "__main__":
    # ad-hoc tests here
    from crosscat.utils import data_utils as du
    import random
    import facade
    import pandas as pd

    import os

    if os.path.isfile('plttest.bdb'):
        os.remove('plttest.bdb')

    rng_seed = random.randrange(10000)
    num_rows = 100
    num_cols = 50
    num_splits = 5
    num_clusters = 5

    nan_prop = .25

    table_name = 'plottest'

    # generate some clustered data
    ccmd = du.generate_clean_state(rng_seed, num_clusters, num_cols, num_rows, num_splits)
    T, M_c, M_r, X_L, X_D = ccmd

    for row in range(num_rows):
        for col in range(num_cols):
            if random.random() < nan_prop:
                T[row][col] = float('nan')

    input_df = pd.DataFrame(T, columns=['col_%i' % i for i in range(num_cols)])

    cc_client = facade.BayesDBCrossCat.from_pandas('plttest.bdb', table_name, input_df)
    cc_client('initialize 4 models for {}'.format(table_name))
    cc_client('analyze {} for 10 iterations wait'.format(table_name))

    plt.figure(facecolor='white', tight_layout=False)
    ax = draw_state(cc_client.bdb, 'plottest', 0, separator_width=1,
                    separator_color=(0., 0., 1., 1.), short_names=False,
                    nan_color=(1, .15, .25, 1.))
    plt.show()
