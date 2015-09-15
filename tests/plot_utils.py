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

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from bdbcontrib import facade
from bdbcontrib.plot_utils import _pairplot

def main():
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
    df = cc_client('SELECT one_n, zero_5, five_c, four_8 FROM plottest')
    df = df.as_df()

    plt.figure(tight_layout=True, facecolor='white')
    _pairplot(df, bdb=cc_client.bdb, generator_name='plottest_cc',
             use_shortname=False, colorby='four_8', show_contour=False,
             show_full=False)
    plt.savefig('fig0.png')

    # again, without tril to check that outer axes render correctly
    plt.figure(tight_layout=True, facecolor='white')
    _pairplot(df, bdb=cc_client.bdb, generator_name='plottest_cc',
             use_shortname=False, colorby='four_8', show_contour=True,
             show_full=False)
    plt.savefig('fig1.png')

if __name__ == '__main__':
    main()
