
from bayeslite.shell.hook import bayesdb_shell_init
import math


@bayesdb_shell_init
def register_bql_math(self):
    self._bdb.sqlite3.create_function('exp', 1, math.exp)
    self._bdb.sqlite3.create_function('sqrt', 1, math.sqrt)
    self._bdb.sqlite3.create_function('pow', 2, pow)
    self._bdb.sqlite3.create_function('log', 1, math.log)
    self._bdb.sqlite3.create_function('linfoot', 1,
                                      lambda x: math.sqrt(1.-math.exp(-2.*x)))
