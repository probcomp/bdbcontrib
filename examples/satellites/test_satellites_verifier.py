# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2015, MIT Probabilistic Computing Project
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
import re
import os
import verify_output

TESTDATA=os.path.join(os.path.dirname(__file__), 'satellites_verifier_examples')

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

def test_good():
    verify_output.check_results(datafiles('good-1'))

def test_missing_bdb():
    try:
        verify_output.check_results(datafiles('missing-bdb'))
        assert False
    except ValueError as e:
        assert re.search(r'\bbdb\b', str(e)), e

        
