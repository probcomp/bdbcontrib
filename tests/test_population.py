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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contextlib import contextmanager
from io import BytesIO
from string import ascii_lowercase  # pylint: disable=deprecated-module
from textwrap import dedent
import bayeslite
import os
import pandas
import pytest
import random
import re
import sys
import tempfile
import test_plot_utils

from bayeslite.loggers import CaptureLogger

from bdbcontrib import quickstart, population

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

@contextmanager
def prepare():
    if testvars['dataset'] is None:
        (df, csv_data) = test_plot_utils.dataset(40)
        tempd = tempfile.mkdtemp(prefix="bdbcontrib-test-population")
        csv_path = os.path.join(tempd, "data.csv")
        with open(csv_path, "w") as csv_f:
            csv_f.write(csv_data.getvalue())
        bdb_path = os.path.join(tempd, "data.bdb")
        name = ''.join(random.choice(ascii_lowercase) for _ in range(32))
        dts = quickstart(name=name, csv_path=csv_path, bdb_path=bdb_path,
                         logger=CaptureLogger(
                             verbose=pytest.config.option.verbose),
                         session_capture_name="test_population.py")
        ensure_timeout(10, lambda: dts.analyze(models=10, iterations=20))
        testvars['dataset'] = dts
        testvars['input_df'] = df

    yield testvars['dataset'], testvars['input_df']

def test_analyze_and_analysis_status_and_reset():
    with prepare() as (dts, _df):
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

        # This is the only test that needs the files anymore (for reset),
        # so now that we're done, clean those up. The rest of the tests can
        # happen in any order based on the in-memory bdb.
        import shutil
        shutil.rmtree(os.path.dirname(dts.csv_path))

def test_q():
    with prepare() as (dts, df):
        resultdf = dts.query('SELECT COUNT(*) FROM %t;')
        #resultdf.to_csv(sys.stderr, header=True)
        assert 1 == len(resultdf)
        assert 1 == len(resultdf.columns)
        assert '"COUNT"(*)' == resultdf.columns[0]
        assert len(df) == resultdf.iloc[0, 0]
        resultdf = dts.query(dedent('''\
            ESTIMATE DEPENDENCE PROBABILITY OF
            floats_1 WITH categorical_1 BY %g'''))
        #resultdf.to_csv(sys.stderr, header=True)
