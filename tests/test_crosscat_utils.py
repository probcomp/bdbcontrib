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

# Because this test may indirectly import pylab (e.g. via importing
# bdbcontrib.crosscat_utils), which would peg the matplotlib backend,
# which might prevent a later test from drawing pictures headless.
# &#&*%(@#^&!@.
import matplotlib
matplotlib.use('Agg')

import bayeslite
from bayeslite.exception import BayesLiteException as BLE
import pandas
import pytest

from bayeslite.read_pandas import bayesdb_read_pandas_df
from bdbcontrib import crosscat_utils

def get_test_df():
    PANDAS_DF_DATA = [
        {
            'age': 34,
            'gender': 'M',
            'salary': 7400,
            'height': 65,
            'division': 'sales',
            'rank': 3
            },
        {
            'age': 41,
            'gender': 'M',
            'salary': 6560,
            'height': 72,
            'division': 'marketing',
            'rank': 4
            },
        {
            'age': 25,
            'gender': 'M',
            'salary': 5200,
            'height': 69,
            'division':
            'accounting',
            'rank': 5
            },
        {
            'age': 23,
            'gender': 'F',
            'salary': 8100,
            'height': 67,
            'division':
            'data science',
            'rank': 3
            },
        {
            'age': 36,
            'gender': 'F',
            'salary': 9600,
            'height': 70,
            'division': 'management',
            'rank': 2
            },
        {
            'age': 30,
            'gender': 'M',
            'salary': 7000,
            'height': 73,
            'division': 'sales',
            'rank': 4
            },
        {
            'age': 30,
            'gender': 'F',
            'salary': 8100,
            'height': 73,
            'division': 'engineering',
            'rank': 3
            },
    ]
    return pandas.DataFrame(PANDAS_DF_DATA)


def test_get_metadata():
    table_name = 'tmp_table'
    generator_name = 'tmp_cc'
    pandas_df = get_test_df()

    import os
    os.environ['BAYESDB_WIZARD_MODE']='1'
    with bayeslite.bayesdb_open() as bdb:
        bayesdb_read_pandas_df(bdb, table_name, pandas_df, create=True)
        bdb.execute('''
            create generator {} for {} using crosscat(guess(*))
        '''.format(generator_name, table_name))
        with pytest.raises(BLE):
            md = crosscat_utils.get_metadata(bdb, generator_name, 0)

        bdb.execute('INITIALIZE 2 MODELS FOR {}'.format(generator_name))

        with pytest.raises(ValueError):  # XXX from BayesLite: should be a BLE?
            crosscat_utils.get_metadata(bdb, 'Peter_Gabriel', 0)
        md = crosscat_utils.get_metadata(bdb, generator_name, 0)

        assert isinstance(md, dict)
        assert 'X_D' in md.keys()
        assert 'X_L' in md.keys()
