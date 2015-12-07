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

from bdbcontrib.bql_utils import cursor_to_df
import multiprocessing as mp
from bayeslite import bayesdb_open

# Imports for debug printing
import sys
from datetime import datetime


def _query_into_queue(query_string, queue, bdb_file, verbose=False):
    """
    Estimate pairwise similarity of a certain subset of the bdb according to
    query_string; place it in the multiprocessing Manager.Queue().

    For two technical reasons, this function is defined as a toplevel class and
    independently creates a bdb handle:

    1) Multiprocessing workers must be able to be pickled, and thus must be
       declared as toplevel functions;
    2) Multiple threads cannot access the same bdb handle, lest concurrency
       issues arise with corrupt data.

    Parameters
    ----------
    query_stirng : str
        Name of the query to execute, determined by estimate_similarity_mp.
    queue : multiprocessing.Manager.Queue
        Queue to place results into
    bdb_file : str
        File location of the BayesDB database. This function will
        independently open a new BayesDB handler.
    verbose : bool
        Print notifications of running and completion. Default False.
    """
    # Shorten the debug message.
    debug_str = "ESTIMATE...LIMIT{}".format(
        query_string.split('LIMIT')[1]
    )
    debug_print("RUN: " + debug_str, verbose)
    # Flush results since output from child threads aren't necessarily
    # displayed immediately
    sys.stdout.flush()

    bdb = bayesdb_open(pathname=bdb_file)
    res = bdb.execute(query_string)
    queue.put(cursor_to_df(res))

    debug_print("DONE: " + debug_str, verbose)
    sys.stdout.flush()


# From Ned Batchelder
def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def debug_print(msg, verbose=True):
    """Print message with timestamp only if verbose flag is true."""
    if verbose:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print "{}: {}".format(now, msg)


def estimate_similarity_mp(bdb_file, table, model, sim_table=None, cores=None,
                           N=None, overwrite=False, verbose=False):
    """
    Estimate pairwise similarity from the given model, splitting processing
    across multiple processors.

    Because called methods in this function must also open up separate BayesDB
    instances, this function accepts a BayesDB filename, rather than an actual
    bayeslite.BayesDB object.

    Parameters
    ----------
    bdb_file : str
        File location of the BayesDB database object. This function will
        handle opening the file with bayeslite.bayesdb_open.
    table : str
        Name of the table containing the raw data
    model : str
        Name of the model to estimate from
    sim_table : str
        Name of the table to insert similarity results into. Defaults to
        table name + '_sim'.
    cores : int
        Number of processors to use. Cannot be more than the number of cores
        as identified by multiprocessing.num_cores(); defaults to that number.
    N : int
        Number of similarities to estimate. Usually used just for testing small
        batches; currently, there is no control over what specific pairwise
        similarities are estimated with this parameter.
    overwrite : bool
        Whether to overwrite the sim_table if it already exists. If
        overwrite=False and the table exists, function will raise
        sqlite3.OperationalError. Default True.
    verbose : bool
        Print additional output in function and in _query_into_queue to
        monitor progress of processing.
    """
    bdb = bayesdb_open(pathname=bdb_file)

    num_cpus = mp.cpu_count()
    if cores is None:
        cores = num_cpus

    if cores < 1:
        raise ValueError(
            "Invalid number of cores {}".format(cores, num_cpus)
        )

    if cores == 1:
        debug_print("Warning: no advantage to using 1 core", verbose)

    debug_print("Computing similarity with {} cores".format(cores), verbose)

    if sim_table is None:
        sim_table = table + '_sim'

    # Get number of occurrences in the database
    count_query = bdb.execute('SELECT COUNT(*) FROM {}'.format(table))
    if N is None:
        N = int(cursor_to_df(count_query)['"COUNT"(*)'][0])

    debug_print("Found {} rows, so {} comparisons".format(N, N * N), verbose)

    sizes = [(N * N) / cores for i in range(cores)]
    # Make the last job finish the remainder
    sizes[-1] += (N * N) % cores

    total = 0
    offsets = [total]
    for size in sizes[:-1]:
        total += size
        offsets.append(total)

    # Format sizes and offsets later
    q_template = ('ESTIMATE SIMILARITY FROM PAIRWISE {} '.format(model) +
                  'LIMIT {} OFFSET {}')

    queries = [q_template.format(*so) for so in zip(sizes, offsets)]

    debug_print(
        "Queries to complete:\n\t{}".format('\n\t'.join(queries)), verbose
    )

    # Create the similarity table. Assumes original table has rowid column.
    # XXX: tables from verbnet bdb don't necessarily have an
    # autoincrementing primary key other than rowid (doesn't work).
    # So we ought to ask for a foreign key, but ESTIMATE SIMILARITY
    # returns numerical values rather than row names, so that code
    # would have to be changed first. For now, we eliminate
    # REFERENCE {table}(foreign_key) from the name0 and name1 col specs.
    if overwrite:
        debug_print("DROP: table {}".format(sim_table), verbose)
        bdb.sql_execute('DROP TABLE IF EXISTS {}'.format(sim_table))
    debug_print("CREATE: table {}".format(sim_table), verbose)
    bdb.sql_execute('''
        CREATE TABLE {sim_table} (
            name0 INTEGER NOT NULL,
            name1 INTEGER NOT NULL,
            value DOUBLE NOT NULL
        )
    '''.format(sim_table=sim_table))

    # Define the callback which inserts data into table once each process is
    # done
    def insert_into_sim(df):
        """
        Use the main thread bdb handle to successively insert results of
        ESTIMATEs into the table.
        """
        # Because the bayeslite implementation of sqlite3 doesn't allow
        # inserts of > 500 rows at a time (else sqlite3.OperationalError),
        # we split the list into _chunks of size 500 and perform multiple
        # insert statements.
        rows = map(list, df.values)
        rows_str = ['({})'.format(','.join(map(str, r))) for r in rows]
        for rows_chunk in _chunks(rows_str, 500):
            debug_print(
                "INSERT: {}...{}".format(rows_chunk[0], rows_chunk[-1]),
                verbose
            )
            insert_str = '''
                INSERT INTO {} (name0, name1, value) VALUES {};
            '''.format(sim_table, ','.join(rows_chunk))
            bdb.sql_execute(insert_str)

    pool = mp.Pool(processes=cores)

    manager = mp.Manager()
    queue = manager.Queue()

    for query in queries:
        pool.apply_async(
            _query_into_queue, args=(query, queue, bdb_file, verbose)
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

    debug_print("Done", verbose)
