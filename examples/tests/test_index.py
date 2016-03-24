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

import os
from bdbcontrib.verify_notebook import run_and_verify_notebook

INDEX_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)))

# TODO: I want to check that all the links in markup cells are non-broken.
# The standard cell check only gets called on pyout cells. I should fix that.
# But that can wait for another commit.

def test_index():
  optfile = os.path.join(INDEX_DIR, 'bayesdb-session-capture-opt.txt')
  if os.path.exists(optfile):
      os.remove(optfile)
  run_and_verify_notebook(os.path.join(INDEX_DIR, "Index"))
  assert os.path.exists(optfile)
  with open(optfile, 'r') as opt:
    assert " <>\n" == opt.read()
