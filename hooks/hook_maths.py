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

import math

from bayeslite.shell.hook import bayesdb_shell_init

def linfoot(x):
    return math.sqrt(1. - math.exp(-2.*x))

@bayesdb_shell_init
def register_bql_math(self):
    # XXX Should not touch internals of the bdb like this -- should
    # invent a BQL API for defining BQL functions.
    self._bdb._sqlite3.createscalarfunction('exp', math.exp, 1)
    self._bdb._sqlite3.createscalarfunction('sqrt', math.sqrt, 1)
    self._bdb._sqlite3.createscalarfunction('pow', pow, 2)
    self._bdb._sqlite3.createscalarfunction('log', math.log, 1)
    self._bdb._sqlite3.createscalarfunction('linfoot', linfoot, 1)
