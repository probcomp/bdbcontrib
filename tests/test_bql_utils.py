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

import re

import bdbcontrib.bql_utils

from test_population import prepare

def test_describe_columns_and_column_type():
    with prepare() as (dts, _df):
        resultdf = dts.query('SELECT * from %t LIMIT 1')
        assert ['index', 'floats_1', 'categorical_1', 'categorical_2',
                'few_ints_3', 'floats_3', 'many_ints_4', 'skewed_numeric_5',
                ] == list(resultdf.columns)
        resultdf = dts.describe_generator_columns()
        expected = {'floats_1': 'numerical',
                    'categorical_1': 'categorical',
                    'categorical_2': 'categorical',
                    'few_ints_3': 'categorical',
                    'floats_3': 'numerical',
                    'many_ints_4': 'numerical',
                    'skewed_numeric_5': r'(categorical|numerical)'}
        for (column, expected_type) in expected.iteritems():
            stattype = resultdf[resultdf['name'] == column]['stattype'].iloc[0]
            assert re.match(expected_type, stattype), column
            assert re.match(expected_type, dts.get_column_stattype(column))
