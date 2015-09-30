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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bayeslite
import numpy as np
import os
import pandas as pd
from cStringIO import StringIO

from bayeslite.read_csv import bayesdb_read_csv
from bdbcontrib.bql_utils import cursor_to_df
from bdbcontrib.plot_utils import _pairplot

def prepare():
    '''Column names give rough type, numbers show which are related.'''
    df = pd.DataFrame()
    num_rows = 400
    alphabet = ['A', 'B', 'C', 'D', 'E']
    dist_of_five = np.random.choice(range(5), num_rows,
                                    p=np.array([1, .4, .3, .2, .1]) / 2.)
    df['numeric_1'] = [np.random.randn() + x for x in dist_of_five]
    df['categorical_1'] = [alphabet[i] for i in dist_of_five]

    dist_of_five = np.random.choice(range(5), num_rows,
                                    p=np.array([1, .4, .3, .2, .1]) / 2.)
    df['categorical_2'] = [alphabet[i] for i in dist_of_five]

    few_ints = np.random.choice(range(4), num_rows, p=[.4, .3, .2, .1])
    df['few_ints_3'] = few_ints
    df['floats_3'] = [(np.random.randn() - 2 * x) / (1 + x) for x in few_ints]

    df['many_ints_4'] = np.random.randn(num_rows)

    csv_str = StringIO()
    df.to_csv(csv_str)

    os.environ['BAYESDB_WIZARD_MODE']='1'
    bdb = bayeslite.bayesdb_open()
    # XXX Do we not have a bayesdb_read_df ?
    bayesdb_read_csv(bdb, 'plottest', flush(csv_str),
                     header=True, create=True)
    bdb.execute('''
        create generator plottest_cc for plottest using crosscat(guess(*))
    ''')

    # do a plot where a some sub-violins are removed
    remove_violin_bql = """
        DELETE FROM plottest
            WHERE categorical_1 = "B"
                AND (few_ints_3 = 2 OR few_ints_3 = 1);
    """
    cursor = bdb.execute('SELECT * FROM plottest')
    df = cursor_to_df(cursor)
    return (df, bdb)

def do(prepped, location, **kwargs):
    (df, bdb) = prepped
    plt.figure(tight_layout=True, facecolor='white')
    _pairplot(df, bdb=bdb, generator_name='plottest_cc',
              show_full=False, **kwargs)
    plt.savefig(location)


from PIL import Image
def has_nontrivial_contents_over_white_background(imgfile):
    img = Image.open(imgfile)
    pix = img.load()
    hist = img.histogram()
    # White background means corners are likely white:
    assert pix[0, 0] == (255, 255, 255, 255)
    assert pix[img.size[0] - 1, img.size[1] - 1] == (255, 255, 255, 255)
    white = hist[-1]
    total = sum(hist)
    # Total white should be between 10 and 90% of the image.
    assert total * .1 < white
    assert white < total * .9
    # Total number of different intensities used should be between 10 and 90%
    # of the available intensities.
    zeros = len([intensity for intensity in hist if intensity == 0])
    assert len(hist) * .1 < zeros
    assert zeros < len(hist) * .9
    return True

def flush(f):
    return StringIO(f.getvalue())

def test_pairplot_smoke():
    ans = prepare()
    f = StringIO()
    do(ans, f, colorby='categorical_2', show_contour=False)
    assert has_nontrivial_contents_over_white_background(flush(f))
    f = StringIO()
    do(ans, f, colorby='categorical_2', show_contour=True)
    assert has_nontrivial_contents_over_white_background(flush(f))
    f = StringIO()
    do(ans, f, show_contour=False)
    assert has_nontrivial_contents_over_white_background(flush(f))

def test_one_variable():
    (df, bdb) = prepare()
    for var in ['categorical_1', 'few_ints_3', 'floats_3', 'many_ints_4']:
      cursor = bdb.execute('SELECT %s FROM plottest' % (var,))
      df = cursor_to_df(cursor)
      f = StringIO()
      do((df, bdb), f, show_contour=False)
      assert has_nontrivial_contents_over_white_background(flush(f))
      cursor = bdb.execute('SELECT %s, categorical_2 FROM plottest' % (var,))
      df = cursor_to_df(cursor)
      f = StringIO()
      do((df, bdb), f, colorby='categorical_2', show_contour=False)
      assert has_nontrivial_contents_over_white_background(flush(f))
      f = StringIO()
      do((df, bdb), f, colorby='categorical_2', show_contour=True)
      assert has_nontrivial_contents_over_white_background(flush(f))


if __name__ == '__main__':
    ans = prepare()
    do(ans, 'fig0.png', colorby='categorical_2', show_contour=False)
    do(ans, 'fig1.png', colorby='categorical_2', show_contour=True)
    do(ans, 'fig2.png', show_contour=False)
    print "Figures saved in 'fig0.png', 'fig1.png', 'fig2.png'"
    import os
    assert os.path.exists('fig0.png')
    assert os.path.exists('fig1.png')
    assert os.path.exists('fig2.png')
