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

from __future__ import print_function

# matplotlib needs to set the backend before anything else gets to.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import namedtuple
from contextlib import contextmanager
from io import BytesIO
from textwrap import dedent
import bayeslite
import numpy as np
import os
import pandas as pd
import pytest
import re
import sys
import test_plot_utils
from test_plot_utils import dts_df

from bayeslite.exception import BayesLiteException
from bayeslite.read_csv import bayesdb_read_csv
from bayeslite.exception import BayesLiteException as BLE
from bayeslite.loggers import CaptureLogger

def test_quick_describe_variables_and_column_type(dts_df):
    dts, _df = dts_df
    resultdf = dts.query('SELECT * from %t LIMIT 1')
    assert ['index', 'floats_1', 'categorical_1', 'categorical_2',
            'few_ints_3', 'floats_3', 'many_ints_4', 'skewed_numeric_5',
            ] == list(resultdf.columns)
    resultdf = dts.quick_describe_variables()
    expected = {'floats_1': 'numerical',
                'categorical_1': 'categorical',
                'categorical_2': 'categorical',
                'few_ints_3': 'categorical',
                'floats_3': 'numerical',
                'many_ints_4': 'numerical',
                'skewed_numeric_5': r'(categorical|numerical)'}
    for (column, expected_type) in expected.iteritems():
        stattype = resultdf[resultdf['name'] == column]['stattype'].iloc[0]
        assert re.match(expected_type, stattype), column
        assert re.match(expected_type, dts.vartype(column))

def test_explore_vars(dts_df):
    dts, _df = dts_df
    dts.logger.calls = []
    with pytest.raises(BLE):
        dts.quick_explore_vars([]) # Empty columns
    with pytest.raises(BLE):
        dts.quick_explore_vars(['float_1']) # Just one column.

    dts.quick_explore_vars(['floats_1', 'categorical_1'], plotfile='foo')
    call_types = pd.DataFrame([call[0] for call in dts.logger.calls])
    call_counts = call_types.iloc[:,0].value_counts()
    assert call_counts['result'] > 0
    assert call_counts['plot'] > 0
    assert 'warn' not in call_counts

def test_similar_rows(dts_df):
    dts, _df = dts_df
    dts.logger.calls = []
    the_real_queryfn = dts.query
    count = [0]
    q_calls = []
    results = [pd.DataFrame([[0]]),  # Count of matching rows.
               pd.DataFrame([[2]]),  # Count of matching rows.
               pd.DataFrame([[1]]),  # Count of matching rows.
               None,                 # Call with temp table creation.
               'similar rows',       # Similarity result.
               ]
    def my_test_queryfn(fmt_str, *args):
        q_calls.append((fmt_str, args))
        count[0] += 1
        return results[count[0] - 1]
    try:
        dts.query = my_test_queryfn
        with pytest.raises(BLE):
            dts.quick_similar_rows(identify_row_by={})  # No rows.
        assert 'SELECT COUNT(*)' == q_calls[-1][0][:15]
        with pytest.raises(BLE):
            dts.quick_similar_rows(identify_row_by={})  # Too many rows.
        assert 'SELECT COUNT(*)' == q_calls[-1][0][:15]
        assert 'similar rows' == dts.quick_similar_rows(identify_row_by={
            'a':'b', 'c': 'd'})
        assert 'SELECT COUNT(*)' == q_calls[-3][0][:15]
        assert 'CREATE TEMP TABLE' == q_calls[-2][0][:17]
        assert re.search(r'''"a" = [?]\s+and\s+"c" = [?] ''',
                         q_calls[-2][0])
        assert (['b', 'd'],) == q_calls[-2][1]
        assert 'SELECT * FROM' == q_calls[-1][0][:13]
    finally:
        dts.query = the_real_queryfn


    call_types = pd.DataFrame([call[0] for call in dts.logger.calls])
    if len(call_types) > 0:
        call_counts = call_types.iloc[:,0].value_counts()
        assert 'warn' not in call_counts
