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

SATELLITES_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "satellites")

def rebuild_bdb(satellites_dir):
  import os
  for filename in os.listdir(satellites_dir):
    if filename.startswith("satellites") and filename.endswith(".bdb"):
      os.remove(os.path.join(satellites_dir, filename))
  import time
  seed=int(time.strftime("%Y%m%d"))
  import imp
  build_bdbs = imp.load_source("build_bdbs",
                               os.path.join(satellites_dir, "build_bdbs.py"))
  build_bdbs.doit(out_dir=satellites_dir,
                  num_models=10, num_iters=4, checkpoint_freq=2, seed=seed)
  # Note: build_bdbs records the seed and other diagnostics in the output
  # directory as satellites-[date]-probcomp-* (besides creating satellites.bdb)

def do_not_track(satellites_dir):
  optpath = os.path.join(satellites_dir, "bayesdb-session-capture-opt.txt")
  with open(optpath, "w") as optfile:
    optfile.write("False\n")

def check_satellites(notebook_cell):
  """Check pyout cell contents for reasonableness in the Satellites notebook."""
  # Should raise on error.
  print repr(notebook_cell)

def test_satellites():
  do_not_track(SATELLITES_DIR)
  rebuild_bdb(SATELLITES_DIR)
  run_and_verify_notebook(os.path.join(SATELLITES_DIR, "Satellites"),
                          content_tester=check_satellites)
