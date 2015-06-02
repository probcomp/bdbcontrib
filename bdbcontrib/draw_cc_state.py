
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import matplotlib as mpl

from crosscat_utils import get_column_probabilities, get_row_probabilities
from crosscat_utils import get_cols_in_view, get_rows_in_cluster
from crosscat_utils import get_metadata, get_M_c

from bql_utils import get_descriptions, get_shortnames
from bql_utils import get_data_as_list

from bdbcontrib import plotutils as pu


NO_CMAP = mpl.colors.ListedColormap([(1, 1, 1, 1), (1, 1, 1, 1)])


def convert_t_do_numerical(T, M_c):
    for colno, md in enumerate(M_c['column_metadata']):
        lookup = md['code_to_value']
        if len(lookup) == 0:
            continue
        for row in range(len(T)):
            val = T[row][colno]
            if val is None:
                continue
            if isinstance(val, float):
                if np.isnan(val):
                    continue
            T[row][colno] = lookup[val]

    Tary = np.array(T)
    return Tary


def sort_state(X_L, X_D, M_c, T):
    """Sorts the metadata for visualization.

    Sort views from largest to smallest. W/in views, sorts columns r->l from
    highest marginal probability to lowest; and sorts categories largest to
    smallest from top to bottom. Within categories, sorts rows higest to lowest
    probability from top to bottom.

    Parameters
    ----------
    X_L : dict
        CrossCat view metadata
    X_D : list
        CrossCat row/view/cluster partition
    M_c : dict
        CrossCat column metadata
    T : list

    Notes
    -----
    Calculating probabilites requires initializing stateless CrossCat
    component models, which is a little exensive.

    Returns
    -------
    sorted_views : list
        view indices from largest (most columns) to smallest.
    sorted_clusters : dict<list>
        sorted_clusters[v] is a sorted list of clusters in view v from largest
        (most rows) to smallest.
    sorted_cols : dict<list>
        sorted_columns[v] is a sorted list of columns in view v from higest to
        lowest probability.
    sorted_rows : dict<dict<list>>
        sorted_rows[v][c] is a sorted list of rows in cluster c of view v from
        higest to lowest probability.
    """
    num_views = len(X_L['view_state'])

    column_logps = get_column_probabilities(X_L, M_c)

    cols_by_view = [get_cols_in_view(X_L, v) for v in range(num_views)]
    num_cols_in_view = [len(cv) for cv in cols_by_view]
    sorted_views = [i[0] for i in sorted(enumerate(num_cols_in_view),
                    reverse=True, key=lambda x:x[1])]

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

        sorted_cols[view] = [col_prob[p] for p in sorted(col_prob.keys(),
                             reverse=True)]

        # sort clusters by size
        rows_by_cluster = [get_rows_in_cluster(X_D, view, clstr) for
                           clstr in range(num_clusters)]
        num_rows_in_clstr = [len(rc) for rc in rows_by_cluster]
        sorted_clusters[view] = [
            i[0] for i in sorted(enumerate(num_rows_in_clstr), reverse=True,
                                 key=lambda x:x[1])]

        row_logps = get_row_probabilities(X_L, X_D, M_c, T, view)
        sorted_rows_view = {}
        num_rows_accounted_for = 0
        for clstr in sorted_clusters[view]:
            rows_in_clstr = get_rows_in_cluster(X_D, view, clstr)
            row_prob = [row_logps[row] for row in rows_in_clstr]
            sorted_row_clstr = [rows_in_clstr[i] for
                                i in np.argsort(row_prob)][::-1]

            assert len(rows_in_clstr) == len(sorted_row_clstr)

            num_rows_accounted_for += len(sorted_row_clstr)
            sorted_rows_view[clstr] = sorted_row_clstr

        assert num_rows_accounted_for == len(X_D[0])

        sorted_rows[view] = sorted_rows_view

    return sorted_views, sorted_clusters, sorted_cols, sorted_rows


def cmap_color_brightness(value, base_color, vmin, vmax,
                          nan_color=(1., 0., 0., 1.)):
    """Returns a color representing the magnitude of a value.

    Higher values are represented with lighter colors. Given the a base color
    of 50% gray, white represents the max and black represents the min.

    Parameters
    ----------
    value : float
        The value of a cell to convert to a color
    base_color: tuple (r, g, b) or (r, g, b, alpha)
        The color to which the brightness manipulation is applied
    vmin : float
        The minimum value of the column to which `value` belongs
    vmax : float
        The maximum value of the column to which `value` belongs
    nan_color : tuple (r, g, b) or (r, g, b, alpha)
        The color to use for NaN or missing data. Defaults to red.

    Returns:
        color : tuple (r, g, b, a)
            The `value` color
    """
    # XXX: bayesdb_data never retruns NaN for multinomial---NaN is added
    # to value_map
    if value is None or np.isnan(value):
        return nan_color

    if vmin == vmax:
        brightness = .5
    else:
        span = vmax - vmin
        # brightness = .5*(value-vmin)/(span)+.25  # low contrast
        brightness = (value-vmin)/(span)

    color = np.array([min(c*brightness, 1.) for c in list(base_color)])

    color[3] = 1.

    return color


def gen_hilight_colors(hl_labels=None, hl_colors=None):
    # Generates a hilight color lookup from labels to colors. Generates labels
    # from Set1 by default.
    if hl_labels is None:
        return {}

    hl_colors_out = {}
    if hl_colors is None:
        if len(hl_labels) > 0:
            hl_max = float(len(hl_labels))
            hl_colors_out = dict([(i, plt.cm.Set1(i/hl_max))
                                  for i in hl_labels])
    else:
        if isinstance(hl_colors, list):
            hl_colors_out = dict(zip(hl_labels, hl_colors))
        elif isinstance(hl_colors, dict):
            for key in hl_colors.keys():
                if key not in hl_labels:
                    raise ValueError("hl_colors dict must have an entry for "
                                     "each hl_label")
            hl_colors_out = hl_colors
        else:
            raise TypeError('hl_colors must be a list or dict')
        if len(hl_colors) != len(hl_labels):
            raise ValueError('hl_colors must have an entry for each entry in '
                             'hl_labels')

    return hl_colors_out


def gen_blank_sort(num_rows, num_cols):
    """Generates a 'blank sort' which allows state plotting of the raw data
    without structure.
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
                    column_partition, cmap, border_width,
                    nan_color=(1., 0., 0., 1.)):
    # generate a heatmap using the data (allows clusters to ahve different
    # base colors)
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
                    xposidx = x_pos + view_count*border_width
                    base_cmapper[y_pos, xposidx] = float(cluster)
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
        col_cpy = np.array([[x] if x is not None else [float('NaN')]
                            for x in T[:, col]])
        col_cpy = col_cpy[np.isfinite(col_cpy)]
        cmin = np.min(col_cpy)
        cmax = np.max(col_cpy)
        y_pos = 0
        for clstr in sorted_clusters[view]:
            for row in sorted_rows[view][clstr]:
                value = T[row, col]
                base_color = cmap(base_cmapper[y_pos, x_pos])
                color = cmap_color_brightness(value, base_color, cmin, cmax,
                                              nan_color=nan_color)
                cell_colors[y_pos, x_pos, :] = color
                y_pos += 1

    return cell_colors


def draw_state(bdb, table_name, generator_name, modelno,
               ax=None, border_width=3, row_label_col=None, short_names=True,
               hilight_rows=[], hilight_rows_colors=None,
               hilight_cols=[], hilight_cols_colors=None,
               separator_color='black', separator_width=4,
               blank_state=False, nan_color=(1., 0., 0., 1.),
               view_labels=None, view_label_fontsize='large',
               legend=True, legend_fontsize='medium',
               row_legend_loc=1, row_legend_title='Row key',
               col_legend_loc=4, col_legend_title='Column key',
               descriptions_in_legend=True, legend_wrap_threshold=20,
               ):
    """Creates a debugging (read: not pretty) rendering of a CrossCat state.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        The BayesDB object associated with the CrossCat state
    table_name : str
        The btable name containing the data associated with the state
    generator_name : str
        The CrossCat generator associated witht the state
    modelno : int
        The index of the model/state to draw
    ax : matplotlib.axis
        The axis on which to draw
    row_label_col : str
        The name of the column to use for row labels. Defaults to FIXME
    short_names : bool
        Use shortnames as column labels
    hilight_rows : list<str>
        A list of rows to hilight with colored rectangles.
    hilight_rows_colors : list
        Contains a color (str or tuple) for each entry in `hilight_rows`. If
        not specified, unique colors for each entry are generated.
    hilight_cols : list
        A list of columns to hilight with colored rectangles.
    hilight_cols_colors : list
        Contains a color (str or tuple) for each entry in `hilight_cols`. If
        not specified, unique colors for each entry are generated.
    blank_state : bool
        If True, draws an unsorted, unpartitioned state
    view_labels : list<str>
        Labels placed above each view. If `len(view_labels) < num_views` then
        only the views for which there are entries are labeled.
    view_label_fontsize : valid matplotlib `fontsize`
        Font size used for vie labels
    legend : bool
        If True (defult) displays legend
    legend_fontsize : valid matplotlib `fontsize`
        Font size used for legend entries and titles
    row_legend_loc : matplotlib.legend location
        location of the row legend. For use with row hilighting
    col_legend_loc : matplotlib.legend location
        location of the column legend. For use with column hilighting

    Returns
    -------
    ax : matplotlib.axis
        The state rendering

    Other Parameters
    ----------------
    border_width : int
        The number of cells between views. Use larger values for longer row
        names.
    separator_color : str or (r, g, b) or (r, g, b, alpha) tuple
        The color of the cluster seprator. Default is black.
    separator_width : int
        linewidth of the cluster separator
    nan_color : str or (r, g, b) or (r, g, b, alpha) tuple
        The color for missing/NaN values. Default is red.
    row_legend_title : str
        title of the row legend
    col_legend_title : str
        title of the column legend
    legend_wrap_threshold : int
        Max number of characters until wordrap for collapsed legends. For use
        when multiple entries in `hilight_cols_colors` or `hilight_cols_colors`
        contain the same color.
    descriptions_in_legend : bool
        If True (default), the column descriptions (requires codebook) are
        added to the legend
    """
    theta = get_metadata(bdb, generator_name, modelno)
    M_c = get_M_c(bdb, generator_name)
    # idx_to_name doesn't use an int idx, but a string idx because crosscat. Yep.
    ordered_columns = [M_c['idx_to_name'][str(idx)] for
                       idx in sorted(M_c['name_to_idx'].values())]
    T = get_data_as_list(bdb, table_name, column_list=ordered_columns)
    X_L = theta['X_L']
    X_D = theta['X_D']

    num_rows = len(T)
    num_cols = len(T[0])

    if not blank_state:
        sortedstate = sort_state(X_L, X_D, M_c, T)
        sorted_views, sorted_clusters, sorted_cols, sorted_rows = sortedstate
        column_partition = X_L['column_partition']['assignments']
    else:
        blankstate = gen_blank_sort(num_rows, num_cols)
        sorted_views, sorted_clusters, sorted_cols, sorted_rows = blankstate
        column_partition = [0]*num_cols

    if view_labels is not None:
        if not isinstance(view_labels, list):
            raise TypeError("view_labels must be a list")
        if len(view_labels) != len(sorted_views):
            view_labels += ['']*(len(sorted_rows)-len(view_labels))
    else:
        view_labels = ['V ' + str(i) for i in range(num_rows)]

    if hilight_cols_colors is None:
        hilight_cols_colors = []

    if hilight_rows_colors is None:
        hilight_rows_colors = []

    # set colormap to 50% gray (should probably give the user control
    # over this)
    cmap = NO_CMAP
    T = convert_t_do_numerical(T, M_c)

    num_views = len(sorted_cols)
    X = np.zeros((num_rows, num_cols+num_views*border_width))

    # row hilighting
    row_hl_colors = gen_hilight_colors(hilight_rows, hilight_rows_colors)

    hl_row_idx_label_zip = []
    if row_label_col is None:
        row_labels = [str(i) for i in range(num_rows)]
    elif isinstance(row_label_col, list):
        if len(row_label_col) != num_rows:
            raise TypeError("If row_label_col is a list, it must have an "
                            "entry for each row")
        row_labels = [str(label) for label in row_label_col]
    elif isinstance(row_label_col, str):
        # FIXME: This is not going to work until BayesDB stops removing key and
        # ignore columns from the data
        raise NotImplementedError
        label_col_idx = M_c['name_to_idx'][row_label_col]
        row_labels = [str(T[row, label_col_idx]) for row in range(num_rows)]
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

    # generate a heatmap using the data (allows clusters to ahve different
    # base colors)
    cell_colors = gen_cell_colors(T, sorted_views, sorted_cols,
                                  sorted_clusters, sorted_rows,
                                  column_partition, cmap, border_width,
                                  nan_color=nan_color)

    # x_tick_labels = []
    x_labels = []

    if ax is None:
        ax = plt.gca()

    ax.imshow(cell_colors, cmap=cmap, interpolation='nearest', origin='upper',
              aspect='auto')
    col_count = 0
    for v, view in enumerate(sorted_views):
        view_x_labels = [M_c['idx_to_name'][str(col)]
                         for col in sorted_cols[view]]
        if short_names:
            view_x_tick_labels = get_shortnames(bdb, table_name, view_x_labels)
        else:
            view_x_tick_labels = view_x_labels

        y_tick_labels = []

        x_labels += view_x_labels + ['_']*border_width
        num_cols_view = len(sorted_cols[view])
        sbplt_start = col_count+v*border_width
        sbplt_end = col_count+num_cols_view+v*border_width

        for i, vxtl in enumerate(view_x_labels):
            if vxtl in hilight_cols:
                edgecolor = col_hl_colors[vxtl]
                x_a = sbplt_start+i-.5
                ax.add_patch(Rectangle((x_a, -.5), 1, num_rows,
                                       facecolor="none", edgecolor=edgecolor,
                                       lw=2, zorder=10))
                fontcolor = edgecolor
                fontsize = 'x-small'
            else:
                fontcolor = '#333333'
                fontsize = 'x-small'
            font_kws = dict(color=fontcolor, fontsize=fontsize, rotation=90,
                            va='top', ha='center')
            ax.text(sbplt_start+i+.5, num_rows+.5, view_x_tick_labels[i],
                    font_kws)

        view_label_x = (sbplt_start+sbplt_end)/2. - .5
        view_label_y = -2.5
        font_kws = dict(ha='center',
                        fontsize=view_label_fontsize,
                        weight='bold')
        ax.text(view_label_x, view_label_y, view_labels[v], font_kws)

        y = 0
        for cluster in sorted_clusters[view]:
            y_tick_labels += [row_idx_to_label[row]
                              for row in sorted_rows[view][cluster]]
            ax.plot([sbplt_start-.5, sbplt_end-.5], [y-.5, y-.5],
                    color=separator_color, lw=separator_width)
            for row, label in hl_row_idx_label_zip:
                try:
                    pos = sorted_rows[view][cluster].index(row)
                except ValueError:
                    pos = None

                if pos is not None:
                    edgecolor = row_hl_colors[label]
                    ax.add_patch(Rectangle((sbplt_start - .5, y + pos - .5),
                                           num_cols_view, 1, facecolor="none",
                                           edgecolor=edgecolor, lw=2,
                                           zorder=10))

            y += len(sorted_rows[view][cluster])

        for i, row in enumerate(range(num_rows-1, -1, -1)):
            if y_tick_labels[i] in hilight_rows:
                fontcolor = row_hl_colors[y_tick_labels[i]]
                fontsize = 'x-small'
                fontweight = 'bold'
                zorder = 10
            else:
                fontsize = 'x-small'
                fontcolor = '#333333'
                fontweight = 'light'
                zorder = 5

            ax.text(sbplt_start-1, i+.5, str(y_tick_labels[i]), ha='right',
                    fontsize=fontsize, color=fontcolor, weight=fontweight,
                    zorder=zorder)
        col_count += num_cols_view

    # generate row legend
    # Use matplotlib artists to generate a list of colored lines
    # TODO: Refactor legend generator into its own function
    if legend:
        if len(hilight_rows) > 0:
            row_legend = pu.gen_collapsed_legend_from_dict(
                row_hl_colors, loc=row_legend_loc, title=row_legend_title,
                fontsize=legend_fontsize, wrap_threshold=legend_wrap_threshold)
            ax.add_artist(row_legend)

        if len(hilight_cols) > 0:
            col_legend_labels = get_shortnames(bdb, table_name, hilight_cols)
            if descriptions_in_legend:
                for i, col_id in enumerate(hilight_cols):
                    col_legend_labels[i] += ': ' + get_descriptions(
                        bdb, table_name, [col_id])[0]
                    col_legend_labels[i] = col_legend_labels[i]

            col_legend = pu.gen_collapsed_legend_from_dict(
                dict(zip(col_legend_labels, hilight_cols_colors)),
                loc=col_legend_loc, title=col_legend_title,
                fontsize=legend_fontsize, wrap_threshold=legend_wrap_threshold)

            ax.add_artist(col_legend)

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
    # ax.set_xticklabels(x_tick_labels, rotation=90, color='black', fontsize=9)
    ax.set_yticklabels(['']*num_rows)
    ax.tick_params(axis='y', colors='white')
    ax.grid(b=False)
    ax.set_axis_bgcolor('white')
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
    generator_name = 'plottest_cc'

    # generate some clustered data
    ccmd = du.generate_clean_state(rng_seed, num_clusters, num_cols, num_rows,
                                   num_splits)
    T, M_c, M_r, X_L, X_D = ccmd

    for row in range(num_rows):
        for col in range(num_cols):
            if random.random() < nan_prop:
                T[row][col] = float('nan')

    input_df = pd.DataFrame(T, columns=['col_%i' % i for i in range(num_cols)])

    client = facade.BayesDBClient.from_pandas('plttest.bdb', table_name,
                                              input_df)
    client('initialize 4 models for {}'.format(generator_name))
    client('analyze {} for 10 iterations wait'.format(generator_name))

    plt.figure(facecolor='white', tight_layout=False)
    ax = draw_state(client.bdb, 'plottest', 'plottest_cc', 0,
                    separator_width=1, separator_color=(0., 0., 1., 1.),
                    short_names=False, nan_color=(1, .15, .25, 1.))
    plt.savefig('state.png')
