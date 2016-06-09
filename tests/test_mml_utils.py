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
from bdbcontrib.mml_utils import StatType

import bayeslite


def test_mml_csv():
    with bayeslite.bayesdb_open() as bdb:
        bayeslite.bayesdb_read_csv_file(
            bdb, 't', 'tests/mml.csv', header=True, create=True)
        guesses = mml_utils.guess_types(bdb, 't')
        assert guesses == ({
            'col1': StatType.IGNORE,
            'col2': StatType.CATEGORICAL,
            'col3': StatType.IGNORE,
            'col4': StatType.NUMERICAL,
            'col5': StatType.CATEGORICAL,
            'col6': StatType.NUMERICAL})

        mml_json = mml_utils.to_json(guesses)
        assert mml_json == {
            'metamodel': 'crosscat',
            'columns': {
                'col1': {'stattype': 'IGNORE'},
                'col2': {'stattype': 'CATEGORICAL'},
                'col3': {'stattype': 'IGNORE'},
                'col4': {'stattype': 'NUMERICAL'},
                'col5': {'stattype': 'CATEGORICAL'},
                'col6': {'stattype': 'NUMERICAL'}}}

        mml_statement = mml_utils.to_mml(mml_json, 'table', 'generator')
        assert mml_statement == (
            'CREATE GENERATOR "generator" FOR "table" '
            'USING crosscat( '
            '"col6" NUMERICAL,"col4" NUMERICAL,"col5" CATEGORICAL,"col2" CATEGORICAL);')

        # col6's values are constructed in such a way as to break crosscat.
        # See https://github.com/probcomp/bayeslite/issues/284
        # On validation the column should be ignored
        mod_schema = mml_utils.validate_schema(bdb, 't', mml_json)
        assert mod_schema == {
            'metamodel': 'crosscat',
            'columns': {
                u'col1': {'stattype': 'IGNORE'},
                u'col2': {'stattype': 'CATEGORICAL'},
                u'col3': {'stattype': 'IGNORE'},
                u'col4': {'stattype': 'NUMERICAL'},
                u'col5': {'stattype': 'CATEGORICAL'},
                u'col6': {'stattype': 'IGNORE', 'guessed': 'NUMERICAL'}}}
