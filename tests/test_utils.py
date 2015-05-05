from bdbcontrib import utils
import bayeslite
import tempfile
import pytest

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
        utils.nullify(bdb, 't', value)

        c = bdb.execute('SELECT COUNT(*) FROM t WHERE one IS NULL;')
        assert c.fetchall()[0][0] == num_nulls_expected[0]

        c = bdb.execute('SELECT COUNT(*) FROM t WHERE two IS NULL;')
        assert c.fetchall()[0][0] == num_nulls_expected[1]

        c = bdb.execute('SELECT COUNT(*) FROM t WHERE three IS NULL;')
        assert c.fetchall()[0][0] == num_nulls_expected[2]

        c = bdb.execute('SELECT COUNT(*) FROM t WHERE four IS NULL;')
        assert c.fetchall()[0][0] == num_nulls_expected[3]
    temp.close()
