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

import pytest
import tempfile

import bayeslite

from bdbcontrib import bql_utils
from bdbcontrib import shell_utils

csv_data = '''
id,one,two,three,four
0,1,3,1,four
1,5,4,5,four
2,1,5,2,three
3,4,4,3,five
4,0,3,1,one
5,0,1,2,three
6,1,1,1,four
7,3,3,1,one
8,2,1,5,two
9,0,2,0,one
'''.lstrip()


csv_data_nan = '''
id,one,two,three,four
0,NaN,3,NaN,four
1,5,4,5,four
2,NaN,5,2,three
3,4,4,3,five
4,0,3,NaN,NaN
5,0,NaN,2,three
6,NaN,NaN,NaN,four
7,3,3,NaN,NaN
8,2,NaN,5,two
9,0,2,0,NaN
'''.lstrip()

csv_data_empty = '''
id,one,two,three,four
0,,3,,four
1,5,4,5,four
2,,5,2,three
3,4,4,3,five
4,0,3,,""
5,0,,2,three
6,,,,four
7,3,3,,""
8,2,,5,two
9,0,2,0,""
'''.lstrip()

csv_data_999 = '''
id,one,two,three,four
0,999,3,999,four
1,5,4,5,four
2,999,5,2,three
3,4,4,3,five
4,0,3,999,999
5,0,999,2,three
6,999,999,999,four
7,3,3,999,999
8,2,999,5,two
9,0,2,0,999
'''.lstrip()


@pytest.mark.parametrize(
    "data, value, num_nulls_expected",
    [[csv_data, 'NaN', (0, 0, 0, 0,)],
     [csv_data_nan, 'NaN', (3, 3, 4, 3,)],
     [csv_data_999, '999', (3, 3, 4, 3,)],
     [csv_data_empty, '', (3, 3, 4, 3,)],
     [csv_data_nan, '999', (0, 0, 0, 0,)]])
def test_nullify_no_missing(data, value, num_nulls_expected):
    temp = tempfile.NamedTemporaryFile()
    temp.write(data)
    temp.seek(0)
    with bayeslite.bayesdb_open() as bdb:
        bayeslite.bayesdb_read_csv_file(bdb, 't', temp.name, header=True,
                                        create=True)
        bql_utils.nullify(bdb, 't', value)

        c = bdb.execute('SELECT COUNT(*) FROM t WHERE one IS NULL;')
        assert c.next()[0] == num_nulls_expected[0]

        c = bdb.execute('SELECT COUNT(*) FROM t WHERE two IS NULL;')
        assert c.next()[0] == num_nulls_expected[1]

        c = bdb.execute('SELECT COUNT(*) FROM t WHERE three IS NULL;')
        assert c.next()[0] == num_nulls_expected[2]

        c = bdb.execute('SELECT COUNT(*) FROM t WHERE four IS NULL;')
        assert c.next()[0] == num_nulls_expected[3]
    temp.close()


def test_cursor_to_df():
    with bayeslite.bayesdb_open() as bdb:
        bql_utils.cursor_to_df(bdb.execute('select * from sqlite_master'))
        bql_utils.cursor_to_df(bdb.execute('select * from sqlite_master'
                ' where 0 = 1'))


def test_is_plotting_command():
    cmd1 = '.heatmap ESTIMATE PAIRWISE DEPENDENCE PROBABILITY FROM t; -f z.png'
    cmd2 = '.show SELECT a, b FROM t LIMIT 10; --no-contour'
    cmd3 = '.histogram SELECT a FROM t;'
    cmd4 = '.help histogram'
    cmd5 = 'SELECT a FROM t;'
    cmd6 = '-- .show is a plotting function'

    assert shell_utils.is_plotting_command(cmd1)
    assert shell_utils.is_plotting_command(cmd2)
    assert shell_utils.is_plotting_command(cmd3)
    assert not shell_utils.is_plotting_command(cmd4)
    assert not shell_utils.is_plotting_command(cmd5)
    assert not shell_utils.is_plotting_command(cmd6)


def test_clean_cmd_filename():
    output_dir = 'foo/baz'
    fignum = 0

    cmd1 = '.histogram SELECT a FROM t;'
    cmd2 = '.histogram SELECT a FROM t; -f pic.png'
    cmd1_expected = '.histogram SELECT a FROM t; --filename foo/baz/fig_0.png'

    cmd1_ud, _ = shell_utils.clean_cmd_filename(cmd1, fignum, output_dir)
    assert cmd1_ud == cmd1_expected
    with pytest.raises(ValueError):
        shell_utils.clean_cmd_filename(cmd2, fignum, output_dir)
    with pytest.raises(ValueError):
        shell_utils.clean_cmd_filename(cmd1_expected, fignum, output_dir)


def test_is_comment():
    assert not shell_utils.is_comment('.heatmap ESTIMATE PAIRWISE FOO;')
    assert shell_utils.is_comment('-- is a comment')
    assert shell_utils.is_comment('--is a comment')
    assert shell_utils.is_comment('---is a comment')

    # XXX: There is no support for inline comments in BQL scripts
    assert not shell_utils.is_comment(' -- not a comment')
    assert not shell_utils.is_comment('.show SELECT a, b FROM t;'
        ' --no-contour')


def test_is_dot_command():
    assert not shell_utils.is_dot_command('-- .show SELECT a, b FROM t;')
    assert not shell_utils.is_dot_command('SELECT ".show" FROM t;')
    assert shell_utils.is_dot_command('.show SELECT a, b FROM t;')
