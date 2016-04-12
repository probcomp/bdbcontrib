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

"""Speed up BDB queries by parallelizing them.

Intuition
---------

``ESTIMATE SIMILARITY FROM PAIRWISE t`` will run in ``O(n^2 m v)`` time, where
``n`` is the number of rows, ``m`` is the number of models, and ``v`` is the
average number of views per model. While this query is reasonable for small
datasets, on medium-large datasets the query slows down significantly and can
become intractable.  Splitting the processing up among multiple cores can
greatly reduce computation time; this module provides functionality to assist
this multiprocessing.

Currently, a multiprocessing equivalent is provided only for
``ESTIMATE PAIRWISE SIMILARITY``. In fact, this is a query that is most likely
to require multiprocessing, as datasets frequently have many more rows than
columns.

Example
-------

Following are (very) informal timing statistics with a 200 rows by 4 column
.cvs file, run on a late 2012 MacBook Pro with a 2.5 GHz 2-core Intel Core i5::

    id,one,two,three,four
    0,2,3,4,two
    1,1,5,4,three
    2,5,1,5,one
    ...
    197,0,5,0,five
    198,5,3,0,three
    199,4,5,2,three

After inserting this .csv data into a table ``t`` and analyzing it quickly::

    bdb.execute('''
        CREATE GENERATOR t_cc FOR t USING crosscat (
            GUESS(*),
            id IGNORE
        )
    ''')
    bdb.execute('INITIALIZE 3 MODELS FOR t_cc')
    bdb.execute('ANALYZE t_cc MODELS 0-2 FOR 10 ITERATIONS WAIT')

The corresponding similarity table thus has 200^2 = 40000 rows::

    In [72]: %timeit -n 10 cursor_to_df(bdb.execute('ESTIMATE SIMILARITY FROM PAIRWISE t_cc'))
    10 loops, best of 3: 9.56 s per loop

    In [73]: %timeit -n 10 parallel.estimate_pairwise_similarity(bdb_file.name, 't', 't_cc', overwrite=True)
    10 loops, best of 3: 5.16 s per loop  # And values are located in the t_similarity table.

The approximate 2x speed up is what would be expected from dividing the work
among two cores. Further speed increases are likely with more powerful
machines.

----
"""

from bayeslite.exception import BayesLiteException as BLE
from bdbcontrib.bql_utils import cursor_to_df
import multiprocessing as mp
from bayeslite import bayesdb_open, bql_quote_name
from bayeslite.util import cursor_value


def _query_into_queue(query_string, params, queue, bdb_file):
    """
    Estimate pairwise similarity of a certain subset of the bdb according to
    query_string; place it in the multiprocessing Manager.Queue().

    For two technical reasons, this function is defined as a toplevel class and
    independently creates a bdb handle:

    1) Multiprocessing workers must be pickleable, and thus must be
       declared as toplevel functions;
    2) Multiple threads cannot access the same bdb handle, lest concurrency
       issues arise with corrupt data.

    Parameters
    ----------
    query_string : str
        Name of the query to execute, determined by estimate_similarity_mp.
    queue : multiprocessing.Manager.Queue
        Queue to place results into
    bdb_file : str
        File location of the BayesDB database. This function will
        independently open a new BayesDB handler.
    """
    bdb = bayesdb_open(pathname=bdb_file)
    res = bdb.execute(query_string, params)
    queue.put(cursor_to_df(res))


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def estimate_pairwise_similarity(bdb_file, table, model, sim_table=None,
                                 cores=None, N=None, overwrite=False):
    """
    Estimate pairwise similarity from the given model, splitting processing
    across multiple processors, and save results into sim_table.

    Because called methods in this function must also open up separate BayesDB
    instances, this function accepts a BayesDB filename, rather than an actual
    bayeslite.BayesDB object.

    Parameters
    ----------
    bdb_file : str
        File location of the BayesDB database object. This function will
        handle opening the file with bayeslite.bayesdb_open.
    table : str
        Name of the table containing the raw data.
    model : str
        Name of the metamodel to estimate from.
    sim_table : str
        Name of the table to insert similarity results into. Defaults to
        table name + '_similarity'.
    cores : int
        Number of processors to use. Defaults to the number of cores as
        identified by multiprocessing.num_cores.
    N : int
        Number of rows for which to estimate pairwise similarities (so
        N^2 calculations are done). Should be used just to test small
        batches; currently, there is no control over what specific pairwise
        similarities are estimated with this parameter.
    overwrite : bool
        Whether to overwrite the sim_table if it already exists. If
        overwrite=False and the table exists, function will raise
        sqlite3.OperationalError. Default True.
    """
    bdb = bayesdb_open(pathname=bdb_file)

    if cores is None:
        cores = mp.cpu_count()

    if cores < 1:
        raise BLE(ValueError(
            "Invalid number of cores {}".format(cores)))

    if sim_table is None:
        sim_table = table + '_similarity'

    # Get number of occurrences in the database
    count_cursor = bdb.execute(
        'SELECT COUNT(*) FROM {}'.format(bql_quote_name(table))
    )
    table_count = cursor_value(count_cursor)
    if N is None:
        N = table_count
    elif N > table_count:
        raise BLE(ValueError(
            "Asked for N={} rows but {} rows in table".format(N, table_count)))

    # Calculate the size (# of similarities to compute) and
    # offset (where to start) calculation for each worker query.

    # Divide sizes evenly, and make the last job finish the remainder
    sizes = [(N * N) / cores for i in range(cores)]
    sizes[-1] += (N * N) % cores

    total = 0
    offsets = [total]
    for size in sizes[:-1]:
        total += size
        offsets.append(total)

    # Create the similarity table. Assumes original table has rowid column.
    # XXX: tables don't necessarily have an autoincrementing primary key
    # other than rowid, which is implicit and can't be set as a foreign key.
    # We ought to ask for an optional user-specified foreign key, but
    # ESTIMATE SIMILARITY returns numerical values rather than row names, so
    # changing numerical rownames into that foreign key would be finicky. For
    # now, we eliminate REFERENCE {table}(foreign_key) from the rowid0 and
    # rowid1 specs.
    sim_table_q = bql_quote_name(sim_table)
    if overwrite:
        bdb.sql_execute('DROP TABLE IF EXISTS {}'.format(sim_table_q))

    bdb.sql_execute('''
        CREATE TABLE {} (
            rowid0 INTEGER NOT NULL,
            rowid1 INTEGER NOT NULL,
            value DOUBLE NOT NULL
        )
    '''.format(sim_table_q))

    # Define the helper which inserts data into table in batches
    def insert_into_sim(df):
        """
        Use the main thread bdb handle to successively insert results of
        ESTIMATEs into the table.
        """
        rows = map(list, df.values)
        insert_sql = '''
            INSERT INTO {} (rowid0, rowid1, value) VALUES (?, ?, ?)
        '''.format(sim_table_q)
        # Avoid sqlite3 500-insert limit by grouping insert statements
        # into one transaction.
        with bdb.transaction():
            for row in rows:
                bdb.sql_execute(insert_sql, row)

    pool = mp.Pool(processes=cores)

    manager = mp.Manager()
    queue = manager.Queue()

    # Construct the estimate query template.
    q_template = '''
        ESTIMATE SIMILARITY FROM PAIRWISE {} LIMIT ? OFFSET ?
    ''' .format(bql_quote_name(model))

    for so in zip(sizes, offsets):
        pool.apply_async(
            _query_into_queue, args=(q_template, so, queue, bdb_file)
        )

    # Close pool and wait for processes to finish
    # FIXME: This waits for all processes to finish before inserting
    # into the table, which means that memory usage is potentially very
    # high!
    pool.close()
    pool.join()

    # Process returned results
    while not queue.empty():
        df = queue.get()
        insert_into_sim(df)
