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

import bayeslite
from bayeslite.exception import BayesLiteException as BLE
import random
import test_utils
import tempfile
from pandas.util.testing import assert_frame_equal
import pytest
from bdbcontrib import cursor_to_df, parallel
from apsw import SQLError


def test_estimate_pairwise_similarity():
    """
    Tests basic estimate pairwise similarity functionality against
    existing BQL estimate queries.
    """
    with tempfile.NamedTemporaryFile(suffix='.bdb') as bdb_file:
        bdb = bayeslite.bayesdb_open(bdb_file.name)
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(test_utils.csv_data)
            temp.seek(0)
            bayeslite.bayesdb_read_csv_file(
                bdb, 't', temp.name, header=True, create=True)

        bdb.execute('''
            CREATE GENERATOR t_cc FOR t USING crosscat (
                GUESS(*),
                id IGNORE
            )
        ''')

        bdb.execute('INITIALIZE 3 MODELS FOR t_cc')
        bdb.execute('ANALYZE t_cc MODELS 0-2 FOR 10 ITERATIONS WAIT')

        # How to properly use the estimate_pairwise_similarity function.
        parallel.estimate_pairwise_similarity(
            bdb_file.name, 't', 't_cc'
        )

        # Should complain with bad core value
        with pytest.raises(BLE):
            parallel.estimate_pairwise_similarity(
                bdb_file.name, 't', 't_cc', cores=0
            )

        # Should complain if overwrite flag is not set, but t_similarity
        # exists
        with pytest.raises(SQLError):
            parallel.estimate_pairwise_similarity(
                bdb_file.name, 't', 't_cc'
            )
        # Should complain if model and table don't exist
        with pytest.raises(SQLError):
            parallel.estimate_pairwise_similarity(
                bdb_file.name, 'foo', 'foo_cc'
            )
        # Should complain if bdb_file doesn't exist
        with tempfile.NamedTemporaryFile() as does_not_exist:
            with pytest.raises(SQLError):
                parallel.estimate_pairwise_similarity(
                    does_not_exist.name, 't', 't_cc'
                )

        # Should run fine if overwrite flag is set
        parallel.estimate_pairwise_similarity(
            bdb_file.name, 't', 't_cc', overwrite=True
        )

        # Should be able to specify another table name
        parallel.estimate_pairwise_similarity(
            bdb_file.name, 't', 't_cc', sim_table='t_similarity_2'
        )

        parallel_sim = cursor_to_df(
            bdb.execute('SELECT * FROM t_similarity')
        ).sort_values(by=['rowid0', 'rowid1'])
        parallel_sim_2 = cursor_to_df(
            bdb.execute('SELECT * FROM t_similarity_2')
        ).sort_values(by=['rowid0', 'rowid1'])

        # Results may be returned out of order. So we sort the values,
        # as above, and we reorder the numeric index
        parallel_sim.index = range(parallel_sim.shape[0])
        parallel_sim_2.index = range(parallel_sim_2.shape[0])

        # The data from two successive parallel pairwise estimates should be
        # identical to each other...
        assert_frame_equal(
            parallel_sim, parallel_sim_2, check_column_type=True)
        # ...and to a standard estimate pairwise similarity.
        std_sim = cursor_to_df(
            bdb.execute('ESTIMATE SIMILARITY FROM PAIRWISE t_cc')
        )
        assert_frame_equal(std_sim, parallel_sim, check_column_type=True)


def _bigger_csv_data(n=30):
    """
    Bigger, but not *too* big, csv data to test batch uploading without
    requiring tons of time for a non-parallelized estimate query
    """
    data = [
        'id,one,two,three,four'
    ]
    for i in xrange(n):
        data.append('{},{},{},{},{}'.format(
            i,
            random.randrange(0, 6),
            random.randrange(0, 6),
            random.randrange(0, 6),
            random.choice(['one', 'two', 'three', 'four', 'five'])
        ))
    return '\n'.join(data)


def test_estimate_pairwise_similarity_long():
    """
    Tests larger queries that need to be broken into batch inserts of 500
    values each, as well as the N parameter.
    """
    with tempfile.NamedTemporaryFile(suffix='.bdb') as bdb_file:
        bdb = bayeslite.bayesdb_open(bdb_file.name)
        with tempfile.NamedTemporaryFile() as temp:
            # n = 40 -> 40**2 -> 1600 rows total
            temp.write(_bigger_csv_data(40))
            temp.seek(0)
            bayeslite.bayesdb_read_csv_file(
                bdb, 't', temp.name, header=True, create=True)
        bdb.execute('''
            CREATE GENERATOR t_cc FOR t USING crosscat (
                GUESS(*),
                id IGNORE
            )
        ''')

        bdb.execute('INITIALIZE 3 MODELS FOR t_cc')
        bdb.execute('ANALYZE t_cc MODELS 0-2 FOR 10 ITERATIONS WAIT')

        # test N = 0
        parallel.estimate_pairwise_similarity(
            bdb_file.name, 't', 't_cc', N=0
        )
        assert cursor_to_df(
            bdb.execute('SELECT * FROM t_similarity')
        ).shape == (0, 0)

        # test other values of N
        for N in [1, 2, 10, 20, 40]:
            parallel.estimate_pairwise_similarity(
                bdb_file.name, 't', 't_cc', N=N, overwrite=True
            )
            assert cursor_to_df(
                bdb.execute('SELECT * FROM t_similarity')
            ).shape == (N**2, 3)
        # N too high should fail
        with pytest.raises(BLE):
            parallel.estimate_pairwise_similarity(
                bdb_file.name, 't', 't_cc', N=41, overwrite=True
            )

        parallel_sim = cursor_to_df(
            bdb.execute('SELECT * FROM t_similarity')
        ).sort_values(by=['rowid0', 'rowid1'])
        parallel_sim.index = range(parallel_sim.shape[0])

        std_sim = cursor_to_df(
            bdb.execute('ESTIMATE SIMILARITY FROM PAIRWISE t_cc')
        )

        assert_frame_equal(std_sim, parallel_sim, check_column_type=True)
