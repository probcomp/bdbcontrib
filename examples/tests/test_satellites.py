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
import re

from bdbcontrib.verify_notebook import run_and_verify_notebook
from bdbcontrib.verify_notebook import assert_markdown_matches as md
from bdbcontrib.verify_notebook import assert_markdown_not as md_not
from bdbcontrib.verify_notebook import assert_pyout_matches as py
from bdbcontrib.verify_notebook import assert_pyout_not as py_not
from bdbcontrib.verify_notebook import assert_has_png as has_png
from bdbcontrib.verify_notebook import assert_raises, assert_warns

SATELLITES_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "satellites")

def remove_bdb_files(satellites_dir):
  for filename in os.listdir(satellites_dir):
    if filename.startswith("satellites") and filename.endswith(".bdb"):
      os.remove(os.path.join(satellites_dir, filename))

def rebuild_bdb(satellites_dir):
  import time
  seed=int(time.strftime("%Y%m%d"))
  import imp
  build_bdbs = imp.load_source("build_bdbs",
                               os.path.join(satellites_dir, "build_bdbs.py"))
  build_bdbs.doit(out_dir=satellites_dir,
                  num_models=20, num_iters=20, checkpoint_freq=2, seed=seed)
  # Note: build_bdbs records the seed and other diagnostics in the output
  # directory as satellites-[date]-probcomp-* (besides creating satellites.bdb)

@contextmanager
def do_not_track(satellites_dir):
  optpath = os.path.join(satellites_dir, "bayesdb-session-capture-opt.txt")
  with open(optpath, "w") as optfile:
    optfile.write("False\n")
  yield
  os.remove(optpath)

# [ (input cell regex, [ ANDed cell testers ] ) ]
EXPECTED = [
  (r'Exploring and cleaning', [r'Union of Concerned Scientists']),
  (r'import os', []),
  (r'import bayeslite', []),

  (r'Querying the data', [md(r'population', flags=re.I)]),
  (r'WHERE Name LIKE \'International Space Station%\'',
   [r'NASA/Multinational']),
  (r'SELECT COUNT\(\*\) FROM satellites', [r'\b1167\b']),
  (r'Name LIKE \'%GPS%\'',
   [r'\[23 rows x 32 columns\]', r'Military/Commercial']),
  (r'SELECT name, dry_mass_kg, period_minutes, class_of_orbit FROM',
   [r'IBEX', r'NaN', r'14\.36', r'GEO', r'LEO', r'Elliptical']),
  (r'statistical graphics procedures', []),
  (r'import matplotlib', []),
  (r'bdbcontrib.histogram', [has_png()]),

  (r'Querying the implications', [md(r"what if", flags=re.I)]),
  (r'Consider the following', []),
  (r'CREATE TEMP TABLE IF NOT EXISTS satellite_purpose', []),
  (r'valid query', []),
  (r'barplot', [has_png()]),
  (r'not its implications', []),
  (r"(?s)WHERE Class_of_orbit = 'GEO'.*Dry_Mass_kg BETWEEN 400 AND 600",
   [r'India', r'Communications', r'GEO', r'559',
    py_not(r'China'), py_not(r'^2\s+', flags=re.M),  # No row id 2, only 0 and 1
    ]),
  (r'difficult to know how wide a net to cast.', []),
  (r'AND Dry_Mass_kg BETWEEN 300 AND 700',
   [r'India', r'Communications', r'GEO', r'559',
    r'China', py(r'^12\s+', flags=re.M)]),
  (r'as the constraints for the hypothetical get narrower', []),

  (r'Exploring predictive relationships', []),
  (r'heatmap.*ESTIMATE DEPENDENCE PROBABILITY', [has_png()]),
  (r'high probability of mutual interdependence',
   [r'country of contractor', r"contractor's identity",
    r'location of the satellite']),
    # Can we verify that the heatmap shows these as darker?
  (r'heatmap.*ESTIMATE CORRELATION', [has_png()]),
  (r'difficult to trust inferences', []),

  (r'Inferring missing values', []),
  (r'WHERE type_of_orbit IS NULL', [r'COUNT', r'\b123\b']),
  (r'predicted value for `type_of_orbit`', [r'confidence']),
  (r'CREATE TEMP TABLE IF NOT EXISTS inferred_orbit', [r'Empty DataFrame']),
  (r'visualize the result', []),
  (r'SELECT \* FROM inferred_orbit',
   [r'inferred_orbit_type_conf',
    r'(?m)^29\s+(Sun-Synchronous|Intermediate|N/A)']),
  (r'(?s)pairplot.*SELECT inferred_orbit_type', [has_png()]),
  (r'moderate to high confidence for the orbit type of `LEO`', []),

  (r'Identify anomalies', []),
  (r'A geosynchronous orbit should take 24 hours', []),
  (r'CREATE TEMP TABLE IF NOT EXISTS unlikely_periods', [r'Empty DataFrame']),
  (r'ORDER BY "Relative Probability of Period" ASC',
   [r'14\.36', r'23\.94', '142\.08', '1306\.29']),
  (r'24-minute periods', []),
  (r'Request for help to improve the database', []),
  (r'', []), # Empty last cell.

  (r'License', [r'Apache 2.0'])

  ]

def test_satellites():
  with do_not_track(SATELLITES_DIR):
    if ("DEBUG_TESTS" not in os.environ or
        not os.path.exists(os.path.join(SATELLITES_DIR, "satellites.bdb"))):
      remove_bdb_files(SATELLITES_DIR) # Just in case.
      rebuild_bdb(SATELLITES_DIR)
    msglimit = None if pytest.config.option.verbose else 1000
    run_and_verify_notebook(
      os.path.join(SATELLITES_DIR, "Satellites"),
      exhaustive=EXPECTED,
      msglimit=msglimit)
    if "DEBUG_TESTS" not in os.environ:
      remove_bdb_files(SATELLITES_DIR)
