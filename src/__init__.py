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

"""Main bdbcontrib API.

The bdbcontrib module serves a sandbox for experimental and semi-stable
features that are not yet ready for integreation to the bayeslite repository.
"""

from bql_utils import cardinality
from bql_utils import cursor_to_df
from bql_utils import describe_table
from bql_utils import describe_generator
from bql_utils import describe_generator_columns
from bql_utils import describe_generator_models
from bql_utils import nullify

from crosscat_utils import draw_crosscat
from crosscat_utils import plot_crosscat_chain_diagnostics

from diagnostic_utils import estimate_kl_divergence
from diagnostic_utils import estimate_log_likelihood

from plot_utils import barplot
from plot_utils import heatmap
from plot_utils import histogram
from plot_utils import mi_hist
from plot_utils import pairplot

__all__ = [
    # bql_utils
        'cardinality',
        'cursor_to_df',
        'describe_generator',
        'describe_generator_columns',
        'describe_generator_models',
        'nullify',
    # crosscat_utils
        'draw_crosscat',
        'plot_crosscat_chain_diagnostics',
    # diagnostic_utils
        'estimate_kl_divergence',
        'estimate_log_likelihood',
    # plot_utils
        'barplot',
        'heatmap',
        'histogram',
        'mi_hist',
        'pairplot',
]
