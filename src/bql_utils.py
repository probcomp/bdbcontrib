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

import pandas as pd

import bayeslite.core
from bayeslite import bayesdb_open
from bayeslite import bql_quote_name as quote
from bayeslite.read_pandas import bayesdb_read_pandas_df
from bayeslite.sqlite3_util import sqlite3_quote_name
from bayeslite.util import cursor_value

###############################################################################
###                                 PUBLIC                                  ###
###############################################################################

def cardinality(bdb, table, cols=None):
    """Compute the number of unique values in the columns of a table.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    table : str
        Name of table.
    cols : list<str>, optional
        Columns to compute the unique values. Defaults to all.

    Returns
    -------
    counts : list<tuple<str,int>>
        A list of tuples of the form [(col_1, cardinality_1), ...]
    """
    # If no columns specified, use all.
    if not cols:
        sql = 'PRAGMA table_info(%s)' % (quote(table),)
        res = bdb.sql_execute(sql)
        cols = [r[1] for r in res]

    counts = []
    for col in cols:
        sql = '''
            SELECT COUNT (DISTINCT %s) FROM %s
        ''' % (quote(col), quote(table))
        res = bdb.sql_execute(sql)
        counts.append((col, cursor_value(res)))

    return counts


def nullify(bdb, table, value):
    """Replace specified values in a SQL table with ``NULL``.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        bayesdb database object
    table : str
        The name of the table on which to act
    value : stringable
        The value to replace with ``NULL``

    Examples
    --------
    >>> import bayeslite
    >>> from bdbcontrib import plotutils
    >>> with bayeslite.bayesdb_open('mydb.bdb') as bdb:
    >>>    bdbcontrib.nullify(bdb, 'mytable', 'NaN')
    """
    # get a list of columns of the table
    c = bdb.sql_execute('pragma table_info({})'.format(quote(table)))
    columns = [r[1] for r in c]
    for col in columns:
        if value in ["''", '""']:
            bql = '''
                UPDATE {} SET {} = NULL WHERE {} = '';
            '''.format(quote(table), quote(col), quote(col))
            bdb.sql_execute(bql)
        else:
            bql = '''
                UPDATE {} SET {} = NULL WHERE {} = ?;
            '''.format(quote(table), quote(col), quote(col))
            bdb.sql_execute(bql, (value,))


def cursor_to_df(cursor):
    """Converts SQLite3 cursor to a pandas DataFrame."""
    # Do this in a savepoint to enable caching from row to row in BQL
    # queries.
    with cursor.connection.savepoint():
        df = pd.DataFrame.from_records(cursor, coerce_float=True)
    if not df.empty:
        df.columns = [desc[0] for desc in cursor.description]
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass

    return df

def table_to_df(bdb, table_name, column_names=None):
    """Return the contents of the given table as a pandas DataFrame.

    If `column_names` is not None, fetch only those columns.
    """
    qt = sqlite3_quote_name(table_name)
    if column_names is not None:
        qcns = ','.join(map(sqlite3_quote_name, column_names))
        select_sql = 'SELECT %s FROM %s' % (qcns, qt)
    else:
        select_sql = 'SELECT * FROM %s' % (qt,)
    return cursor_to_df(bdb.sql_execute(select_sql))

def df_to_table(df, tablename=None, **kwargs):
    """Return a new BayesDB with a single table with the data in `df`.

    `df` is a Pandas DataFrame.

    If `tablename` is not supplied, an arbitrary one will be chosen.

    `kwargs` are passed on to `bayesdb_open`.

    Returns a 2-tuple of the new BayesDB instance and the name of the
    new table.
    """
    bdb = bayesdb_open(**kwargs)
    if tablename is None:
        tablename = bdb.temp_table_name()
    bayesdb_read_pandas_df(bdb, tablename, df, create=True)
    return (bdb, tablename)

def query(bdb, bql):

    """Execute the `bql` query on the `bdb` instance.

    Parameters
    ----------
    bdb : bayeslite.BayesDB
        Active BayesDB instance.
    bql : str
        BQL query string.

    Returns
    -------
    df : pandas.DataFrame
        Table of results as a pandas dataframe.
    """
    cursor = bdb.execute(bql)
    return cursor_to_df(cursor)


def describe_table(bdb, table_name):
    """Returns a DataFrame containing description of `table_name`.

    Examples
    --------
    >>> bdbcontrib.describe_table(bdb, 'employees')
    tabname   | colno |    name
    ----------+-------+--------
    employees |     0 |    name
    employees |     1 |     age
    employees |     2 |  weight
    employees |     3 |  height
    """
    if not bayeslite.core.bayesdb_has_table(bdb, table_name):
            raise NameError('No such table {}'.format(table_name))
    sql = '''
        SELECT tabname, colno, name
            FROM bayesdb_column
            WHERE tabname=?
            ORDER BY tabname ASC, colno ASC
        '''
    curs = bdb.sql_execute(sql, bindings=(table_name,))
    return cursor_to_df(curs)


def describe_generator(bdb, generator_name):
    """Returns a DataFrame containing description of `generator_name`.

    Examples
    --------

    >>> bdbcontrib.describe_generator(bdb, 'employees_gen')
    id |          name |   tabname | metamodel
    ---+---------------+-----------+----------
    3  | employees_gen | employees |  crosscat
    """
    if not bayeslite.core.bayesdb_has_generator_default(bdb, generator_name):
            raise NameError('No such generator {}'.format(generator_name))
    sql = '''
            SELECT id, name, tabname, metamodel
                FROM bayesdb_generator
                WHERE name = ?
        '''
    curs = bdb.sql_execute(sql, bindings=(generator_name,))
    return cursor_to_df(curs)


def describe_generator_columns(bdb, generator_name):
    """Returns a DataFrame containing description of the columns
    modeled by `generator_name`.

    Examples
    --------

    >>> bdbcontrib.describe_generator_columns(bdb, 'employees_gen')
    colno |    name |     stattype
    ------+---------+-------------
        0 |    name |  categorical
        1 |     age |    numerical
        2 |  weight |    numerical
        3 |  height |    numerical
    """
    if not bayeslite.core.bayesdb_has_generator_default(bdb, generator_name):
            raise NameError('No such generator {}'.format(generator_name))
    sql = '''
        SELECT c.colno AS colno, c.name AS name,
                gc.stattype AS stattype
            FROM bayesdb_generator AS g,
                (bayesdb_column AS c LEFT OUTER JOIN
                    bayesdb_generator_column AS gc
                    USING (colno))
            WHERE g.id = ? AND g.id = gc.generator_id
                AND g.tabname = c.tabname
            ORDER BY colno ASC;
    '''
    generator_id = bayeslite.core.bayesdb_get_generator_default(bdb,
        generator_name)
    curs = bdb.sql_execute(sql, bindings=(generator_id,))
    return cursor_to_df(curs)


def describe_generator_models(bdb, generator_name):
    """Returns a DataFrame containing description of the models
    in `generator_name`.

    Examples
    --------

    >>> bdbcontrib.describe_generator_models(bdb, 'employees_gen')
    modelno | iterations
    --------+-----------
          0 | 100
    """
    if not bayeslite.core.bayesdb_has_generator_default(bdb, generator_name):
            raise NameError('No such generator {}'.format(generator_name))
    sql = '''
        SELECT modelno, iterations FROM bayesdb_generator_model
            WHERE generator_id = ?
        '''
    generator_id = bayeslite.core.bayesdb_get_generator_default(bdb,
        generator_name)
    curs = bdb.sql_execute(sql, bindings=(generator_id,))
    return cursor_to_df(curs)


###############################################################################
###                              INTERNAL                                   ###
###############################################################################

def get_column_info(bdb, generator_name):
    generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator_name)
    sql = '''
        SELECT c.colno, c.name, gc.stattype
            FROM bayesdb_generator AS g,
                bayesdb_generator_column AS gc,
                bayesdb_column AS c
            WHERE g.id = ?
                AND gc.generator_id = g.id
                AND gc.colno = c.colno
                AND c.tabname = g.tabname
            ORDER BY c.colno
    '''
    return bdb.sql_execute(sql, (generator_id,)).fetchall()


def get_column_stattype(bdb, generator_name, column_name):
    generator_id = bayeslite.core.bayesdb_get_generator(bdb, generator_name)
    sql = '''
        SELECT gc.stattype
            FROM bayesdb_generator AS g,
                bayesdb_generator_column AS gc,
                bayesdb_column AS c
            WHERE g.id = ?
                AND gc.generator_id = g.id
                AND gc.colno = c.colno
                AND c.name = ?
                AND c.tabname = g.tabname
            ORDER BY c.colno
    '''
    cursor = bdb.sql_execute(sql, (generator_id, column_name,))
    try:
        row = cursor.next()
    except StopIteration:
        # XXX Temporary kludge for broken callers.
        raise IndexError
    else:
        return row[0]


def get_data_as_list(bdb, table_name, column_list=None):
    if column_list is None:
        sql = '''
            SELECT * FROM {};
        '''.format(quote(table_name))
    else:
        sql = '''
            SELECT {} FROM {}
        '''.format(', '.join(map(quote, column_list)), table_name)
    cursor = bdb.sql_execute(sql)
    T = cursor_to_df(cursor).values.tolist()
    return T


def get_shortnames(bdb, table_name, column_names):
    return get_column_descriptive_metadata(bdb, table_name, column_names,
        'shortname')


def get_descriptions(bdb, table_name, column_names):
    return get_column_descriptive_metadata(bdb, table_name, column_names,
        'description')


def get_column_descriptive_metadata(bdb, table_name, column_names, md_field):
    short_names = []
    # XXX: this is indefensibly wasteful.
    bql = '''
        SELECT colno, name, {} FROM bayesdb_column WHERE tabname = ?
    '''.format(md_field)
    records = bdb.sql_execute(bql, (table_name,)).fetchall()

    # hack for case sensitivity problems
    column_names = [c.upper().lower() for c in column_names]
    records = [(r[0], r[1].upper().lower(), r[2]) for r in records]

    for cname in column_names:
        for record in records:
            if record[1] == cname:
                sname = record[2]
                if sname is None:
                    sname = cname
                short_names.append(sname)
                break

    if len(short_names) != len(column_names):
        import pdb
        pdb.set_trace()
    return short_names
