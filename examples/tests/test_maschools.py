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

from contextlib import contextmanager
import os
import pytest
from bdbcontrib import verify_notebook as vn

from util import session

MASCHOOLS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "ma-school-districts")

def test_ma_schools():
  with session(MASCHOOLS_DIR):
    msglimit = None if pytest.config.option.verbose else 1000
    vn.run_and_verify_notebook(
      os.path.join(MASCHOOLS_DIR, "MASchoolDistricts"),
      msglimit=msglimit,
      required=[('schools2.quick_explore_vars\(\[',
                 [vn.allow_warns('matplotlib.*backend'),
                  vn.allow_warns('VisibleDeprecationWarning')]),
                ('ESTIMATE DEPENDENCE PROBABILITY',
                 [vn.assert_has_png(), vn.allow_warns('FutureWarning'),
                  vn.allow_warns('VisibleDeprecationWarning')]),
                ('schools0.quick_describe_variables()', [r'categorical']),
                ('schools1.quick_describe_variables()', [r'categorical']),
                # Once solved, this will not contain categorical, but as
                # set up to be solved, it does:
                ('schools2.quick_describe_variables()', [r'categorical']),
                # As set up, this should have "cannot convert" messages,
                # Not once it's solved though.
                ('df.applymap', [vn.assert_stream_matches('Cannot convert')]),
                (r'schools2.quick_explore_vars',
                 [vn.allow_warns('VisibleDeprecationWarning')])
                ])
