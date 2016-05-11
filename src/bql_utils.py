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

import pandas as pd

import bayeslite.core
from bayeslite import bayesdb_open
from bayeslite import bql_quote_name as quote
from bayeslite.exception import BayesLiteException as BLE
from bayeslite.loggers import logged_query
from bayeslite.read_pandas import bayesdb_read_pandas_df
from bayeslite.sqlite3_util import sqlite3_quote_name
from bayeslite.util import cursor_value
from bdbcontrib.population_method import population_method

from bdbcontrib.population_method import population_method

###############################################################################
###                                 PUBLIC                                  ###
###############################################################################

@population_method(population_to_bdb=0, population_name=1)
def cardinality(bdb, table, cols=None):
    """Compute the number of unique values in the columns of a table.

    Parameters
    ----------
    bdb : __population_to_bdb__
    table : __population_name__
        Name of table.
    cols : list<str>, optional
        Columns to compute the unique values. Defaults to all.

    Returns
    -------
    counts : pandas.DataFrame whose .columns are ['name', 'distinct_count'].
    """
    # If no columns specified, use all.
    if not cols:
        sql = 'PRAGMA table_info(%s)' % (quote(table),)
        res = bdb.sql_execute(sql)
        cols = [r[1] for r in res]

    names=[]
    counts=[]
    for col in cols:
        sql = '''
            SELECT COUNT (DISTINCT %s) FROM %s
        ''' % (quote(col), quote(table))
        res = bdb.sql_execute(sql)
        names.append(col)
        counts.append(cursor_value(res))
    return pd.DataFrame({'name': names, 'distinct_count': counts})


@population_method(population_to_bdb=0, population_name=1)
def nullify(bdb, table, value):
    """Replace specified values in a SQL table with ``NULL``.

    Parameters
    ----------
    bdb : __population_to_bdb__
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

@population_method(population_to_bdb=0, interpret_bql=1, logger="logger")
def query(bdb, bql, bindings=None, logger=None):
    """Execute the `bql` query on the `bdb` instance.

    Parameters
    ----------
    bdb : __population_to_bdb__
    bql : __interpret_bql__
    bindings : Values to safely fill in for '?' in the BQL query.

    Returns
    -------
    df : pandas.DataFrame
        Table of results as a pandas dataframe.
    """
    if bindings is None:
        bindings = ()
    if logger:
        logger.info("BQL [%s] %s", bql, bindings)
    cursor = bdb.execute(bql, bindings)
    return cursor_to_df(cursor)

@population_method(population_to_bdb=0, population_name=1)
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
            raise BLE(NameError('No such table {}'.format(table_name)))
    sql = '''
        SELECT tabname, colno, name
            FROM bayesdb_column
            WHERE tabname=?
            ORDER BY tabname ASC, colno ASC
        '''
    curs = bdb.sql_execute(sql, bindings=(table_name,))
    return cursor_to_df(curs)


@population_method(population_to_bdb=0, generator_name=1)
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
            raise BLE(NameError('No such generator {}'.format(generator_name)))
    sql = '''
            SELECT id, name, tabname, metamodel
                FROM bayesdb_generator
                WHERE name = ?
        '''
    curs = bdb.sql_execute(sql, bindings=(generator_name,))
    return cursor_to_df(curs)

@population_method(population_to_bdb=0, generator_name='generator_name')
def variable_stattypes(bdb, generator_name=None):
    assert generator_name
    """The modeled statistical types of each variable in order."""
    if not bayeslite.core.bayesdb_has_generator_default(bdb, generator_name):
            raise BLE(NameError('No such generator {}'.format(generator_name)))
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

@population_method(population_to_bdb=0)
def list_metamodels(bdb):
    df = query(bdb, "SELECT name FROM bayesdb_generator;")
    return list(df['name'])

@population_method(population_to_bdb=0)
def list_tables(bdb):
    df = query(bdb, """SELECT name FROM sqlite_master
                       WHERE type='table' AND
                           NAME NOT LIKE "bayesdb_%" AND
                           NAME NOT LIKE "sqlite_%";""")
    return list(df['name'])

@population_method(population_to_bdb=0, generator_name=1)
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
            raise BLE(NameError('No such generator {}'.format(generator_name)))
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

@population_method(population_to_bdb=0, generator_name=1)
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

@population_method(population=0, generator_name='generator_name')
def analyze(self, models=100, minutes=0, iterations=0, checkpoint=0,
            generator_name=None):
  '''Run analysis.

  models : integer
      The number of models bounds the accuracy of predictive probabilities.
      With ten models, then you get one decimal digit of interpretability,
      with a hundred models, you get two, and so on.
  minutes : integer
      How long you want to let it run.
  iterations : integer
      How many iterations to let it run.

  Returns:
      A report indicating how many models have seen how many iterations,
      and other info about model stability.
  '''
  assert generator_name is not None
  if models > 0:
    self.query('INITIALIZE %d MODELS IF NOT EXISTS FOR %s' %
          (models, generator_name))
    assert minutes == 0 or iterations == 0
  else:
    models = self.analysis_status(generator_name=generator_name).sum()
  if minutes > 0:
    if checkpoint == 0:
      checkpoint = max(1, int(minutes * models / 200))
      analyzer = ('ANALYZE %s FOR %d MINUTES CHECKPOINT %d ITERATION WAIT' %
                  (generator_name, minutes, checkpoint))
      with logged_query(query_string=analyzer,
                        name=self.session_capture_name,
                        bindings=self.query('SELECT * FROM %t')):
        self.query(analyzer)
  elif iterations > 0:
    if checkpoint == 0:
      checkpoint = max(1, int(iterations / 20))
    self.query(
        '''ANALYZE %s FOR %d ITERATIONS CHECKPOINT %d ITERATION WAIT''' % (
            generator_name, iterations, checkpoint))
  else:
    raise NotImplementedError('No default analysis strategy yet. '
                              'Please specify minutes or iterations.')
  # itrs = self.per_model_analysis_status()
  # models_with_fewest_iterations =
  #    itrs[itrs['iterations'] == itrs.min('index').head(0)[0]].index.tolist()
  # TODO(gremio): run each model with as many iterations as it needs to get
  # up to where it needs to get to, if that's larger?
  # Nope. Vikash said there's no reason to think that's a good idea. Perhaps
  # even better to have some young models mixed in with the old ones.
  # I still think we should make some recommendation that scales for what
  # "the right thing" is, where that's something that at least isn't known to
  # suck.

  return self.analysis_status(generator_name=generator_name)

@population_method(population=0, generator_name='generator_name')
def per_model_analysis_status(self, generator_name=None):
  """Return the number of iterations for each model."""
  assert generator_name is not None
  try:
    return self.query('''SELECT iterations FROM bayesdb_generator_model
                         WHERE generator_id = (
                          SELECT id FROM bayesdb_generator WHERE name = ?)''',
                      (generator_name,))
  except ValueError:
    # Because, e.g. there is no generator yet, for an empty db.
    return None

@population_method(population=0, generator_name='generator_name')
def analysis_status(self, generator_name=None):
  """Return the count of models for each number of iterations run."""
  assert generator_name is not None
  itrs = self.per_model_analysis_status(generator_name=generator_name)
  if itrs is None or len(itrs) == 0:
    emt = pd.DataFrame(columns=['count of model instances'])
    emt.index.name = 'iterations'
    return emt
  vcs = pd.DataFrame(itrs['iterations'].value_counts())
  vcs.index.name = 'iterations'
  vcs.columns = ['count of model instances']
  self.status = vcs
  return vcs

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

    assert len(short_names) == len(column_names)
    return short_names
