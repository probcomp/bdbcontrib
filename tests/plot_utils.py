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

import bayeslite
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cStringIO as StringIO

from bayeslite.read_csv import bayesdb_read_csv
from bdbcontrib.bql_utils import cursor_to_df
from bdbcontrib.plot_utils import _pairplot

def prepare():
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

    csv_str = StringIO.StringIO()
    df.to_csv(csv_str)

    os.environ['BAYESDB_WIZARD_MODE']='1'
    bdb = bayeslite.bayesdb_open()
    # XXX Do we not have a bayesdb_read_df ?
    bayesdb_read_csv(bdb, 'plottest', StringIO.StringIO(csv_str.getvalue()),
                     header=True, create=True)
    bdb.execute('''
        create generator plottest_cc for plottest using crosscat(guess(*))
    ''')

    # do a plot where a some sub-violins are removed
    remove_violin_bql = """
        DELETE FROM plottest
            WHERE zero_5 = "B"
                AND (four_8 = 2 OR four_8 = 1);
    """
    cursor = bdb.execute('SELECT one_n, zero_5, five_c, four_8 FROM plottest')
    df = cursor_to_df(cursor)
    return (df, bdb)

def do(prepped, location, **kwargs):
    (df, bdb) = prepped
    plt.figure(tight_layout=True, facecolor='white')
    _pairplot(df, bdb=bdb, generator_name='plottest_cc',
              show_full=False, **kwargs)
    plt.savefig(location)

if __name__ == '__main__':
    ans = prepare()
    do(ans, 'fig0.png', colorby='four_8', show_contour=False)
    do(ans, 'fig1.png', colorby='four_8', show_contour=True)
    do(ans, 'fig2.png', show_contour=False)
    print "Figures saved in 'fig0.png', 'fig1.png', 'fig2.png'"
