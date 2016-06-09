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
            'col5': StatType.CATEGORICAL})

        mml_json = mml_utils.to_json(guesses)
        assert mml_json == {
            'metamodel': 'crosscat',
            'columns': {
                'col1': {'stattype': 'IGNORE'},
                'col2': {'stattype': 'CATEGORICAL'},
                'col3': {'stattype': 'IGNORE'},
                'col4': {'stattype': 'NUMERICAL'},
                'col5': {'stattype': 'CATEGORICAL'}}}

        mml_statement = mml_utils.to_mml(mml_json, 'table', 'generator')
        assert mml_statement == (
            'CREATE GENERATOR generator FOR table '
            'USING crosscat( '
            '"col4" NUMERICAL,"col5" CATEGORICAL,"col2" CATEGORICAL);')
