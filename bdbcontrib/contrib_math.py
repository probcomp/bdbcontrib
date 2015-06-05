
from bayeslite.shell.hook import bayesdb_shell_cmd
import math


@bayesdb_shell_cmd('register_bql_math_functions')
def register_bql_math(self, args):
    '''Adds basic math functions to BQL
    (No arguments)
    '''
    self._bdb.sqlite3.create_function('exp', 1, math.exp)
    self._bdb.sqlite3.create_function('sqrt', 1, math.sqrt)
    self._bdb.sqlite3.create_function('pow', 2, pow)
    self._bdb.sqlite3.create_function('log', 1, math.log)
