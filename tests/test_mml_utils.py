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

from bdbcontrib import mml_utils

import bayeslite


def test_mml_csv():
    with bayeslite.bayesdb_open() as bdb:
        bayeslite.bayesdb_read_csv_file(
            bdb, 't', 'tests/mml.csv', header=True, create=True)
        guesses = mml_utils.guess_types(bdb, 't')
        # Testing these strings is going to be brittle, but I don't have a
        # great answer.
        assert guesses == ({
            'col1': ('IGNORE',
                     'Column is constant'),
            'col2': ('CATEGORICAL',
                     'Only 5 distinct values'),
            'col3': ('IGNORE',
                     'Column is constant'),
            'col4': ('NUMERICAL',
                     'Contains exclusively numbers (24 of them).'),
            'col5': ('CATEGORICAL',
                     'Only 2 distinct values'),
            'col6': ('NUMERICAL',
                     'Contains exclusively numbers (25 of them).')})

        mml_json = mml_utils.to_json(guesses)
        assert mml_json == {
            'metamodel': 'crosscat',
            'columns': {
                'col1': {'stattype': 'IGNORE',
                         'reason': 'Column is constant'},
                'col2': {'stattype': 'CATEGORICAL',
                         'reason': 'Only 5 distinct values'},
                'col3': {'stattype': 'IGNORE',
                         'reason': 'Column is constant'},
                'col4': {'stattype': 'NUMERICAL',
                         'reason': 'Contains exclusively numbers (24 of them).'},
                'col5': {'stattype': 'CATEGORICAL',
                         'reason': 'Only 2 distinct values'},
                'col6': {'stattype': 'NUMERICAL',
                         'reason': 'Contains exclusively numbers (25 of them).'}
            }}

        mml_statement = mml_utils.to_mml(mml_json, 'table', 'generator')
        assert mml_statement == (
            'CREATE GENERATOR "generator" FOR "table" '
            'USING crosscat( '
            '"col6" NUMERICAL,"col4" NUMERICAL,'
            '"col5" CATEGORICAL,"col2" CATEGORICAL);')

        # col6's values are constructed in such a way as to break crosscat.
        # See https://github.com/probcomp/bayeslite/issues/284
        # On validation the column should be ignored
        mod_schema = mml_utils.validate_schema(bdb, 't', mml_json)
        assert mod_schema == {
            'metamodel': 'crosscat',
            'columns': {
                'col1': {'stattype': 'IGNORE',
                         'reason': 'Column is constant'},
                'col2': {'stattype': 'CATEGORICAL',
                         'reason': 'Only 5 distinct values'},
                'col3': {'stattype': 'IGNORE',
                         'reason': 'Column is constant'},
                'col4': {'stattype': 'NUMERICAL',
                         'reason': 'Contains exclusively numbers (24 of them).'},
                'col5': {'stattype': 'CATEGORICAL',
                         'reason': 'Only 2 distinct values'},
                'col6': {'stattype': 'IGNORE', 'guessed': 'NUMERICAL',
                         'reason': 'Caused ANALYZE to error'}}}
