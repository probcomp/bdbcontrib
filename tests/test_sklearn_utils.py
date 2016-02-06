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

import numpy as np
import pandas as pd
from bdbcontrib.predictors import sklearn_utils as sku

def test_extract_sklearn_dataset():
    dataset = pd.DataFrame({
        'A':[1.1, 2.1, 3.9, 4.5, 5.1],
        'B':[5.1, 4.1, 3.9, 2.5, 1.1],
        'C':['1', '2', '3', '4', '5'],
        'D':[1, None, 3, 4, 5],
        })
    conditions, targets = ['A', 'B'], ['D']
    df = sku.extract_sklearn_dataset(conditions, targets, dataset)
    # Test that column 'C' is not included.
    assert set(df.columns) == set(conditions + targets)
    # Test that the second row is dropped (it has a None target).
    assert len(df) == 4

def test_extract_sklearn_features_numerical():
    dataset = pd.DataFrame({
        'A':[1, 2, None, 4, 10],
        'B':[0, -2, 3, None, 1],
        'C':['1', '2', '3', '4', '5'],
        'D':[1, None, 3, 4, 5],
        })
    columns = ['A', 'B']
    matrix = sku.extract_sklearn_features_numerical(columns, dataset)
    # Test correct imputation.
    mean_A = (1 + 2 + 4 + 10) / 4.
    mean_B = (0 -2 + 3 + 1) / 4.
    expected = np.asarray([
        [1, 2, mean_A, 4, 10],
        [0, -2, 3, mean_B, 1]]).T
    assert np.array_equal(matrix, expected)

def test_extract_sklearn_features_categorical():
    dataset = pd.DataFrame({
        'Nationality':['USA', 'USA', 'France', 'Germany', 'Bengal'],
        'Gender':['M', 'F', 'M', 'M', 'T'],
        'C':['1', '2', '3', '4', '5'],
        'D':[1, None, 3, 4, 5],
        })
    categories = ['Nationality', 'Gender']
    categories_to_val_map = {
        'Nationality' : {'USA':0, 'France':1, 'Germany': 2, 'Bengal':3},
        'Gender' : {'M':0, 'F':1, 'T': 2}
    }
    matrix = sku.extract_sklearn_features_categorical(categories,
        categories_to_val_map, dataset)
    expected = np.asarray([
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1]])
    assert np.array_equal(matrix, expected)

def test_build_categorical_to_value_map():
    dataset = pd.DataFrame({
        'Nationality':['USA', 'USA', 'France', 'Germany', 'Bengal'],
        'Gender':['M', 'F', 'M', 'M', 'T'],
        'C':['1', '2', '3', '4', '5'],
        'D':[1, None, 3, 4, 5],
        })
    columns = ['Nationality', 'Gender']
    categories_to_val_map = sku.build_categorical_to_value_map(columns, dataset)
    # Assert all the 'columns' have a codemap.
    assert set(categories_to_val_map.keys()) == set(columns)
    for col, valmap in categories_to_val_map.iteritems():
        # Assert each unique val in the column has codes for all its values.
        unique_vals = set(dataset[col].unique())
        assert unique_vals == set(valmap.keys())
        # Assert that all codes are unique.
        assert len(set(code for _, code in valmap.iteritems())) == \
            len(unique_vals)

def test_extract_sklearn_univariate_target():
    dataset = pd.DataFrame({
        'Nationality':['USA', 'USA', 'France', 'Germany', 'Bengal'],
        'Gender':['M', 'F', 'M', 'M', 'T'],
        'C':['1', '2', '3', '4', '5'],
        'D':[1, None, 3, 4, 5],
        })
    target = 'C'
    vector = sku.extract_sklearn_univariate_target(target, dataset)
    expected = ['1', '2', '3', '4', '5']
    assert np.array_equal(vector, expected)
