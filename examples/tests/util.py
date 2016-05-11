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
from contextlib import contextmanager

from bdbcontrib.population import OPTFILE
BAKFILE = OPTFILE + ".bak"

@contextmanager
def session(parent_dir):
  optpath = os.path.join(parent_dir, OPTFILE)
  bakpath = os.path.join(parent_dir, BAKFILE)
  if os.path.exists(bakpath):
    os.remove(bakpath)
  if os.path.exists(optpath):
    os.rename(optpath, bakpath)
  try:
    with open(optpath, "w") as optfile:
      optfile.write("bdbcontrib/examples/tests\n")
    yield
  finally:
    os.remove(optpath)
    if os.path.exists(bakpath):
      os.rename(bakpath, optpath)
