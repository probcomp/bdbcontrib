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

import pytest
import re
import os
import StringIO

from bdbcontrib import verify_notebook as vn

TESTDATA=os.path.join(os.path.dirname(__file__), 'notebook_verifier_examples')

def datafiles(basename):
    outfile = os.path.join(TESTDATA, basename + '.out')
    errfile = os.path.join(TESTDATA, basename + '.err')
    outcontent = None
    with open(outfile, 'r') as of:
        outcontent = of.read()
    errcontent = None
    with open(errfile, 'r') as ef:
        errcontent = ef.read()
    return (outcontent, errcontent)

def test_single_warning():
    vn.check_results(datafiles('good-1'), warnings_are_errors=False)
    with pytest.raises(AssertionError) as exc:
      vn.check_results(datafiles('good-1'), warnings_are_errors=True)
    errs = StringIO.StringIO()
    with pytest.raises(AssertionError) as exc:
      vn.check_results(datafiles('good-1'), errstream=errs,
          required=[(r'heatmap.*DEPENDENCE PROBABILITY',
                     [vn.assert_raises(r'elementwise comparison')])])
    assert "FutureWarning" in errs.getvalue()
    assert "individual error reports" in str(exc.value)

    with pytest.raises(AssertionError) as exc:
      vn.check_results(datafiles('good-1'), errstream=None,
          required=[(r'heatmap.*DEPENDENCE PROBABILITY',
                     [vn.assert_raises(r'elementwise comparison')])])
    assert "FutureWarning" in str(exc.value)


def test_multiple_problem_reports():
    errs = StringIO.StringIO()
    with pytest.raises(AssertionError) as exc:
        vn.check_results(datafiles('missing-bdb'), errstream=errs)
    assert " bdb " in errs.getvalue()

    vn.check_results(datafiles('missing-bdb'),
        required=[(r'quickstart', [vn.assert_raises("No bdb in")]),
                  (r'import matplotlib', [vn.assert_warns(r'prop_cycle')])])

    errs = []
    vn.check_results(datafiles('missing-bdb'), errstream=errs)
    assert 2 == len(errs), errs
    assert "No bdb in" in str(errs[0])
    assert "prop_cycle" in str(errs[1])

    errs = StringIO.StringIO()
    with pytest.raises(AssertionError) as exc:
        vn.check_results(datafiles('missing-bdb'), errstream=errs,
            required=[(r'quickstart', [vn.assert_raises("No bdb in")]),
                      (r'import matplotlib', [vn.assert_warns(r'prop_cycle')]),
                      (r'not-there', [vn.assert_raises("no such error")])])
    assert "not-there" in errs.getvalue()

def test_exhaustive():
    errs = []
    vn.check_results(datafiles('missing-bdb'), errstream=errs,
        exhaustive=[
            (r'quickstart', [vn.assert_raises("No bdb in")]),
            (r'import matplotlib', [vn.assert_warns(r'prop_cycle')])])
    assert 40 == len(errs)
    assert "Expected /quickstart/\nFound: # Exploring" in str(errs[0])
    assert "Expected /import matplotlib/\nFound: # Load" in str(errs[1])
    assert "Unexpected cell at end: Cell[2]:" in str(errs[2])
    assert "SessionOrchestrator" in str(errs[2])

    errs = []
    def recordTrue(i, c, d):
        errs.append((i, c, d))
        return True


    expected = [
        (r'# Exploring',
         [vn.assert_markdown_matches(r'Union of Concerned Scientists'),
          r'database',
          vn.assert_markdown_not("foobarbaz")]),
        (r'quickstart', [vn.assert_raises("No bdb in")]),
        (r'session-saving', [vn.assert_no_pyout()]),
        (r'Querying the data', [recordTrue, "SQL", recordTrue]),
        (r'satellites', ["Purpose", recordTrue, "Multinational",
                         vn.assert_pyout_matches("launch_vehicle", flags=re.I),
                         vn.assert_stream_matches(r"BQL \[SELECT \* FROM"),
                         # "well-known satellites" is in the input:
                         vn.assert_pyout_not("well-known satellites")]),
        # From here on out, tests fail in various ways.
        (r'do not match', []),   # 5
        ('do not match', []),    # 6
        (''),                    # 7
        ([]),                    # 8
        (),                      # 9
        ('', [], 'foo'),         # 10
        ('', [vn.assert_no_pyout(),  # One failure per cell,
              lambda i, c, d: 'c' + 5]),  # So this never runs, so no TypeError.
        ('', [vn.assert_markdown_matches('')]),  # 12
        ('', [vn.assert_pyout_matches('')]),     # 13
        ('', [vn.assert_markdown_not('')]),      # 14
        ('', [vn.assert_pyout_not('')]),         # 15
        ('', [vn.assert_markdown_not('')]),      # 16
        ('', [vn.assert_markdown_not('')]),      # 17
        ('', [vn.assert_pyout_not('')]),         # 18
        ('', []),
        ('', []),
        ('', [vn.assert_has_png()]),
        ('', []),
        ('', [vn.assert_has_png()]),
        ('', []),
        ('', []),
        ('', [vn.assert_stream_not('')]),
        ('', [lambda i, c, d: False]),
        ('', [lambda i, c, d: []]),
        ('', [lambda i, c, d: None]),
        ]
    expected += [('', [])] * (40 - len(expected) - 1)  # 40 total cells.
    vn.check_results(datafiles('missing-bdb'), errstream=errs,
                     exhaustive=expected)
    assert 3 == errs.pop(0)[0] # cell_i from the first recordTrue
    assert 3 == errs.pop(0)[0]
    assert 4 == errs.pop(0)[0]
    assert "/do not match/" in str(errs.pop(0))
    assert "/do not match/" in str(errs.pop(0))
    assert "Expected an input pattern and a list of tests" in str(errs.pop(0))
    assert "Expected an input pattern and a list of tests" in str(errs.pop(0))
    assert "Expected an input pattern and a list of tests" in str(errs.pop(0))
    assert "Expected an input pattern and a list of tests" in str(errs.pop(0))
    assert "Expected code cell" in str(errs.pop(0))
    assert "Expected markdown cell" in str(errs.pop(0))
    assert "Expected code cell" in str(errs.pop(0))
    assert "Expected markdown cell" in str(errs.pop(0))
    assert "Expected code cell" in str(errs.pop(0))
    assert "Expected markdown cell" in str(errs.pop(0))
    assert "Expected not to find" in str(errs.pop(0))
    assert "Expected not to find" in str(errs.pop(0))
    assert "Expected not to find" in str(errs.pop(0))
    assert "returned False" in str(errs.pop(0))
    assert "returned []" in str(errs.pop(0))
    assert "returned None" in str(errs.pop(0))
    assert "Undeclared png" in str(errs.pop(0))
    assert "Unexpected cell at end" in str(errs.pop(0))
    assert [] == errs

    expected = [('', [])] * 40   # 40 total cells
    expected += [('foobarbaz', [])] * 2
    vn.check_results(datafiles('missing-bdb'), errstream=errs,
                     exhaustive=expected)
    assert "Expected pattern /foobarbaz/ not found" in str(errs[-1])
    assert "Expected pattern /foobarbaz/ not found" in str(errs[-2])
    assert 8 == len(errs)  # Those two plus the six undeclared displays.
