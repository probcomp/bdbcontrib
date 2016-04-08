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
from string import ascii_lowercase  # pylint: disable=deprecated-module
from textwrap import dedent
import bayeslite
import numpy as np
import os
import pandas as pd
import pytest
import random
import re
import shutil
import sys
import tempfile
import test_plot_utils

from bayeslite.read_csv import bayesdb_read_csv
from bayeslite.exception import BayesLiteException as BLE
from bayeslite.loggers import CaptureLogger
from bdbcontrib import recipes

import multiprocessing
import time

def ensure_timeout(delay, target):
    proc = multiprocessing.Process(target=target)
    proc.start()
    proc.join(delay)
    assert not proc.is_alive()
    # proc.terminate()
    # proc.join()

@pytest.fixture(scope='module')
def dts_df(request):
    (df, csv_data) = test_plot_utils.dataset(40)
    tempd = tempfile.mkdtemp(prefix="bdbcontrib-test-recipes")
    request.addfinalizer(lambda: shutil.rmtree(tempd))
    csv_path = os.path.join(tempd, "data.csv")
    with open(csv_path, "w") as csv_f:
        csv_f.write(csv_data.getvalue())
    bdb_path = os.path.join(tempd, "data.bdb")
    name = ''.join(random.choice(ascii_lowercase) for _ in range(32))
    dts = recipes.quickstart(name=name,
                             csv_path=csv_path, bdb_path=bdb_path,
                             logger=CaptureLogger(
                                 verbose=pytest.config.option.verbose),
                             session_capture_name="test_recipes.py")
    ensure_timeout(10, lambda: dts.analyze(models=10, iterations=20))
    return dts, df

def test_analyze_and_analysis_status_and_reset(dts_df):
    dts, _df = dts_df
    resultdf = dts.analysis_status()
    assert 'iterations' == resultdf.index.name, repr(resultdf)
    assert 'count of model instances' == resultdf.columns[0], repr(resultdf)
    assert 1 == len(resultdf), repr(resultdf)
    assert 10 == resultdf.ix[20, 0], repr(resultdf)

    resultdf = dts.analyze(models=11, iterations=1)
    assert 'iterations' == resultdf.index.name, repr(resultdf)
    assert 'count of model instances' == resultdf.columns[0], repr(resultdf)
    dts.logger.result(str(resultdf))
    assert 2 == len(resultdf), repr(resultdf)
    assert 10 == resultdf.ix[21, 0], repr(resultdf)
    assert 1 == resultdf.ix[1, 0], repr(resultdf)

    dts.reset()
    resultdf = dts.analysis_status()
    assert 'iterations' == resultdf.index.name, repr(resultdf)
    assert 'count of model instances' == resultdf.columns[0], repr(resultdf)
    assert 0 == len(resultdf), repr(resultdf)
    ensure_timeout(10, lambda: dts.analyze(models=10, iterations=20))
    resultdf = dts.analysis_status()
    assert 'iterations' == resultdf.index.name, repr(resultdf)
    assert 'count of model instances' == resultdf.columns[0], repr(resultdf)
    assert 1 == len(resultdf), repr(resultdf)
    assert 10 == resultdf.ix[20, 0], repr(resultdf)

def test_q(dts_df):
    dts, df = dts_df
    resultdf = dts.query('SELECT COUNT(*) FROM %t;')
    #resultdf.to_csv(sys.stderr, header=True)
    assert 1 == len(resultdf)
    assert 1 == len(resultdf.columns)
    assert '"COUNT"(*)' == resultdf.columns[0]
    assert len(df) == resultdf.iloc[0, 0]
    resultdf = dts.query(dedent('''\
        ESTIMATE DEPENDENCE PROBABILITY OF
        floats_1 WITH categorical_1 BY %g'''))
    resultdf.to_csv(sys.stderr, header=True)

def test_quick_describe_columns_and_column_type(dts_df):
    dts, _df = dts_df
    resultdf = dts.query('SELECT * from %t LIMIT 1')
    assert ['index', 'floats_1', 'categorical_1', 'categorical_2',
            'few_ints_3', 'floats_3', 'many_ints_4', 'skewed_numeric_5',
            ] == list(resultdf.columns)
    resultdf = dts.quick_describe_columns()
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
        assert re.match(expected_type, dts.column_type(column))

def test_heatmap(dts_df):
    dts, _df = dts_df
    dts.logger.calls = []
    deps = dts.query('ESTIMATE DEPENDENCE PROBABILITY'
                     ' FROM PAIRWISE COLUMNS OF %g')
    dts.heatmap(deps, plotfile='foobar.png')
    plots = [c for c in dts.logger.calls if c[0] == 'plot']
    assert 1 == len(plots)
    thecall = plots[0]
    assert ('plot', 'foobar.png') == thecall[:2]

    dts.logger.calls = []
    dts.heatmap(deps, plotfile='foobar.png',
                selectors=
                {'A-E': lambda x: bool(re.search(r'^[a-eA-E]', x[0])),
                 'F-W': lambda x: bool(re.search(r'^[f-wF-W]', x[0])),
                 'X-Z': lambda x: bool(re.search(r'^[x-zX-Z]', x[0]))})
    # NOTE: columns are: 'floats_1', 'categorical_1', 'categorical_2',
    #     'few_ints_3', 'floats_3', 'many_ints_4', 'skewed_numeric_5',
    # So there are 2 in A-E, 5 in F-W, and 0 in X-Z.
    # So we should produce 3 plots: AE vs. AE, AE vs. FW, and FW vs. FW.
    plots = [c for c in dts.logger.calls if c[0] == 'plot']
    assert 4 == len(plots)
    names = [p[1] for p in plots]
    assert 'A-E.A-E.foobar.png' in names
    assert 'A-E.F-W.foobar.png' in names
    assert 'F-W.A-E.foobar.png' in names
    assert 'F-W.F-W.foobar.png' in names

def test_explore_vars(dts_df):
    dts, _df = dts_df
    dts.logger.calls = []
    with pytest.raises(BLE):
        dts.quick_explore_vars([]) # Empty columns
    with pytest.raises(BLE):
        dts.quick_explore_vars(['float_1']) # Just one column.

    dts.quick_explore_vars(['floats_1', 'categorical_1'])
    call_types = pd.DataFrame([call[0] for call in dts.logger.calls])
    call_counts = call_types.iloc[:,0].value_counts()
    assert call_counts['result'] > 0
    assert call_counts['plot'] > 0
    assert 'warn' not in call_counts

def test_pairplot(dts_df):
    dts, _df = dts_df
    dts.logger.calls = []
    with pytest.raises(BLE):
        dts.pairplot_vars([])  # Empty columns
    pt = dts.pairplot_vars(
        ['floats_1', 'floats_3', 'many_ints_4', 'categorical_1'],
        colorby='categorical_2')
    call_types = pd.DataFrame([call[0] for call in dts.logger.calls])
    call_counts = call_types.iloc[:,0].value_counts()
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
