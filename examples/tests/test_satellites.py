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

SATELLITES=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "satellites", "Satellites")
def check_satellites(cell):
  """Check pyout cell contents for reasonableness in the Satellites notebook."""
  # Should raise on error.
  print repr(cell)

def test_satellites():
    run_and_verify_notebook(SATELLITES, content_tester=check_satellites)
