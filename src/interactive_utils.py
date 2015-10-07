# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
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

import sys
import threading

import bayeslite

class TimeoutWarningTracer(bayeslite.IBayesDBTracer):
    def __init__(self, delay=30, warning=None):
        self.delay = delay
        self.pending = {}
        self.warning = warning

    def start(self, qid, query, bindings):
        timer = threading.Timer(self.delay, self._warn, query, bindings)
        self.pending[qid] = timer

    def error(self, qid, _e): self._abort(qid)
    def finished(self, qid): self._abort(qid)
    def abandoned(self, qid): self._abort(qid)

    def _abort(self, qid):
        if qid in self.pending:
            # May not be due to multiple aborts
            self.pending[qid].cancel()
            del self.pending[qid]

    def _warn(self, query, bindings):
        if self.warning is None:
            msg = '''It looks like your query
%s
is taking a while (> %s seconds).  Would you like to send us your
session with TODO so we can figure out why it's so slow?  Aborting it
with Ctrl-C or IPython's Kernel->Interrupt is safe.'''
            sys.stdout.write(msg % (query, self.delay))
            sys.stdout.flush()
        else:
            self.warning(query, bindings)
