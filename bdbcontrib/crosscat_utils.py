import numpy as np
import json
import bayeslite.core
from bdbcontrib import general_utils as gu
from crosscat.utils import sample_utils as su


def get_cols_in_view(X_L, view):
    return [c for c, v in enumerate(X_L['column_partition']['assignments']) if v == view]


def get_rows_in_cluster(X_D, view, cluster):
    return [r for r, c in enumerate(X_D[view]) if c == cluster]


def get_M_c(bdb, generator_name):
    generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator_name)
    column_info = gu.get_column_info(bdb, generator_name)
    M_c = bayeslite.crosscat.create_metadata(bdb, generator_id, column_info)

    # XXX: case in M_c is left alone which case in
    # X_L['view_state'][view]['column_names'] is mutated to lowercase.
    M_c['name_to_idx'] = dict((k.lower(), v) for k, v in M_c['name_to_idx'].iteritems())
    M_c['idx_to_name'] = dict((k, v.lower()) for k, v in M_c['idx_to_name'].iteritems())

    return M_c


def get_metadata(bdb, generator_name, modelno):
    generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator_name)
    sql = '''SELECT theta_json FROM bayesdb_crosscat_theta
                WHERE generator_id = ? and modelno = ?'''
    cursor = bdb.sql_execute(sql, (generator_id, modelno))
    try:
        row = cursor.next()
    except StopIteration:
        raise ValueError("Could not find genrator with name {}".format(generator_name))
    else:
        return json.loads(row[0])


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
            if isinstance(x, float):
                if np.isnan(x):
                    continue
            else:
                x = M_c['column_metadata'][col]['code_to_value'][x]
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
