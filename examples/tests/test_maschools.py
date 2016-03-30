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

MASCHOOLS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "ma-school-districts")

@contextmanager
def do_not_track(satellites_dir):
  optpath = os.path.join(satellites_dir, "bayesdb-session-capture-opt.txt")
  with open(optpath, "w") as optfile:
    optfile.write("False\n")
  yield
  os.remove(optpath)

def test_ma_schools():
  do_not_track(MASCHOOLS_DIR)
  msglimit = None if pytest.config.option.verbose else 1000
  vn.run_and_verify_notebook(
      os.path.join(MASCHOOLS_DIR, "MASchoolDistricts"),
      msglimit=msglimit,
      required=[('schools2.quick_explore_vars\(\[',
                [vn.assert_warns('matplotlib.*backend')])])
