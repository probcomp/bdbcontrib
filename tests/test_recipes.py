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

from __future__ import print_function

# matplotlib needs to set the backend before anything else gets to.
import matplotlib
matplotlib.use("Agg")
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
import sys
import tempfile
import test_plot_utils

from bayeslite.read_csv import bayesdb_read_csv
from bdbcontrib import recipes

testvars = {'dataset': None, 'input_df': None}

import multiprocessing
import time
@contextmanager
def ensure_timeout(delay, target):
    proc = multiprocessing.Process(target=target)
    proc.start()
    proc.join(delay)
    assert not proc.is_alive()
    # proc.terminate()
    # proc.join()


class MyTestLogger(object):
    def __init__(self, debug=pytest.config.option.verbose):
        self.calls = []
        self.debug = debug
    def info(self, msg_format, *values):
        if self.debug:
            print('INFO: '+msg_format, *values)
        self.calls.append(('info', msg_format, values))
    def warn(self, msg_format, *values):
        if self.debug:
            print('WARN: '+msg_format, *values)
        self.calls.append(('warn', msg_format, values))
    def plot(self, suggested_name, figure):
        if self.debug:
            plt.show()
        self.calls.append(('plot', suggested_name, figure))
    def result(self, msg_format, *values):
        if self.debug:
            print('RSLT: '+msg_format, *values)
        self.calls.append(('result', msg_format, values))


@contextmanager
def prepare():
    #import pytest
    #pytest.set_trace()
    if testvars['dataset'] is None:
        (df, csv_data) = test_plot_utils.dataset(40)
        name = ''.join(random.choice(ascii_lowercase) for _ in range(32))
        tempf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tempf.write(csv_data.getvalue())
        tempf.close()
        dts = recipes.quickstart(name=name, csv=tempf.name,
                                 logger=MyTestLogger())
        os.remove(tempf.name)
        os.environ['BAYESDB_WIZARD_MODE'] = '1'
        ensure_timeout(10, lambda: dts.analyze(models=10, iterations=20))
        os.environ['BAYESDB_WIZARD_MODE'] = ''
        testvars['dataset'] = dts
        testvars['input_df'] = df
    yield testvars['dataset'], testvars['input_df']

def test_analyze_and_analysis_status():
    with prepare() as (dts, _df):
        resultdf = dts.analysis_status()
        assert 'iterations' == resultdf.index.name, repr(resultdf)
        assert 'count of models' == resultdf.columns[0], repr(resultdf)
        assert 1 == len(resultdf), repr(resultdf)
        assert 10 == resultdf.ix[20, 0], repr(resultdf)
        resultdf = dts.analyze(models=11, iterations=1)
        assert 'iterations' == resultdf.index.name, repr(resultdf)
        assert 'count of models' == resultdf.columns[0], repr(resultdf)
        dts.logger.result(str(resultdf))
        assert 2 == len(resultdf), repr(resultdf)
        assert 10 == resultdf.ix[21, 0], repr(resultdf)
        assert 1 == resultdf.ix[1, 0], repr(resultdf)

def test_q():
    with prepare() as (dts, df):
        resultdf = dts.q('SELECT COUNT(*) FROM %t;')
        #resultdf.to_csv(sys.stderr, header=True)
        assert 1 == len(resultdf)
        assert 1 == len(resultdf.columns)
        assert '"COUNT"(*)' == resultdf.columns[0]
        assert len(df) == resultdf.iloc[0, 0]
        resultdf = dts.q(dedent('''\
            ESTIMATE DEPENDENCE PROBABILITY OF
            floats_1 WITH categorical_1 BY %g'''))
        resultdf.to_csv(sys.stderr, header=True)

def test_quick_describe_columns_and_column_type():
    with prepare() as (dts, _df):
        resultdf = dts.q('SELECT * from %t LIMIT 1')
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

def test_reset():
    pass  # XXX TODO

def test_heatmap():
    pass  # XXX TODO

def test_most_dependent():
    pass  # XXX TODO

def test_explore_cols():
    pass  # XXX TODO

def test_sql_tracing():
    pass  # XXX TODO

def test_similar_rows():
    pass  # XXX TODO

def test_loggers():
    pass  # XXX TODO
