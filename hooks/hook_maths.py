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

import math

from bayeslite.shell.hook import bayesdb_shell_init


@bayesdb_shell_init
def register_bql_math(self):
    self._bdb.sqlite3.create_function('exp', 1, math.exp)
    self._bdb.sqlite3.create_function('sqrt', 1, math.sqrt)
    self._bdb.sqlite3.create_function('pow', 2, pow)
    self._bdb.sqlite3.create_function('log', 1, math.log)
    self._bdb.sqlite3.create_function('linfoot', 1,
        lambda x: math.sqrt(1.-math.exp(-2.*x)))
