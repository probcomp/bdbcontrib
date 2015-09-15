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

# XXX AUTOMATE ME XXX

import bdbcontrib.facade as facade
import os
import pandas as pd
import random

from bdbcontrib.crosscat_utils import draw_state
from crosscat.utils import data_utils as du
from matplotlib import pyplot as plt

def main():
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

if __name__ == '__main__':
    main()
