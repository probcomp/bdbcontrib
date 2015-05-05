
from bayeslite.sqlite3_util import sqlite3_quote_name as quote


def nullify(bdb, table, value):
    # get a list of columns of the table
    c = bdb.sql_execute('pragma table_info({})'.format(quote(table)))
    columns = [r[1] for r in c.fetchall()]
    for col in columns:
        bql = '''
        UPDATE {} SET {} = NULL WHERE {} = ?;
        '''.format(quote(table), quote(col), quote(col))
        bdb.sql_execute(bql, (value,))
