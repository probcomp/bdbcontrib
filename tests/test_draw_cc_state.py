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

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import bayeslite
import os
import pandas as pd
import random
import cStringIO as StringIO

from bayeslite.read_pandas import bayesdb_read_pandas_df
from bdbcontrib.crosscat_utils import draw_state
from crosscat.utils import data_utils as du

def draw_a_cc_state(filename):
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
    T, _M_c, _M_r, _X_L, _X_D = ccmd

    for row in range(num_rows):
        for col in range(num_cols):
            if random.random() < nan_prop:
                T[row][col] = float('nan')

    input_df = pd.DataFrame(T, columns=['col_%i' % i for i in range(num_cols)])

    os.environ['BAYESDB_WIZARD_MODE']='1'
    bdb = bayeslite.bayesdb_open()
    bayesdb_read_pandas_df(bdb, table_name, input_df, create=True)
    bdb.execute('''
        create generator {} for {} using crosscat(guess(*))
    '''.format(generator_name, table_name))
    bdb.execute('initialize 4 models for {}'.format(generator_name))
    bdb.execute('analyze {} for 10 iterations wait'.format(generator_name))
    plt.figure(facecolor='white', tight_layout=False)
    draw_state(bdb, 'plottest', 'plottest_cc', 0,
               separator_width=1, separator_color=(0., 0., 1., 1.),
               short_names=False, nan_color=(1, .15, .25, 1.))
    plt.savefig(filename)

def test_draw_cc_smoke():
    f = StringIO.StringIO()
    draw_a_cc_state(f)
    assert len(f.getvalue()) > 1000

# For manually inspecting the generated figure.
if __name__ == '__main__':
    draw_a_cc_state('state.png')
    print "Figure saved to 'state.png'"
