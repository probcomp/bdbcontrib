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

import pandas as pd

import bayeslite.core
from bayeslite import bql_quote_name as quote


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
        counts.append((col, res.next()[0]))

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


def describe_table(bdb, table_name):
    """Returns a SQLite3 cursor containg description of `table_name`.

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
    return bdb.sql_execute(sql, bindings=(table_name,))


def describe_generator(bdb, generator_name):
    """Returns a SQLite3 cursor containg description of `generator_name`.

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
    return bdb.sql_execute(sql, bindings=(generator_name,))


def describe_generator_columns(bdb, generator_name):
    """Returns a SQLite3 cursor containg description of the columns
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
    return bdb.sql_execute(sql, bindings=(generator_id,))


def describe_generator_models(bdb, generator_name):
    """Returns a SQLite3 cursor containg description of the models
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
    return bdb.sql_execute(sql, bindings=(generator_id,))


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
