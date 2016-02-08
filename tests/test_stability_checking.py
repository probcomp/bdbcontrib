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

import copy
import glob
import os
import shutil
import sys
import tempfile

root = os.path.dirname(os.path.dirname(__file__))
sats_path = os.path.join(root, "examples", "satellites")

def test_stability_check_integration_smoke__ci_because_43s_is_too_slow():
    bdbs_dir = None
    plots_dir = None
    old_path = copy.copy(sys.path)
    sys.path.append(sats_path)
    try:
        import build_bdbs
        import probe
        import visualize
        bdbs_dir = tempfile.mkdtemp(suffix='bdbs')
        plots_dir = tempfile.mkdtemp(suffix='plots')
        build_bdbs.doit(
            bdbs_dir, num_models=4, num_iters=2, checkpoint_freq=1, seed=0)
        with tempfile.NamedTemporaryFile() as f:
            probe.doit(glob.glob("%s/*i.bdb" % bdbs_dir),
                       f.name, model_schedule=[1,2], n_replications=2)
            visualize.plot_probe_results(f.name, plots_dir)
    finally:
        sys.path = old_path
        if bdbs_dir is not None:
            shutil.rmtree(bdbs_dir)
        if plots_dir is not None:
            shutil.rmtree(plots_dir)
