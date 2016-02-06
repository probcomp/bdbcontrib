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

from sklearn.preprocessing import Imputer

def extract_sklearn_dataset(conditions, targets, dataset):
    """Extracts the `conditions` and `targets` colums from `dataset` with
    additional preprocessing.

    `NaN` strings are converted to Python `None`. Rows where the target is
    absent are dropped. All columns not in `conditions` and `target` are
    dropped.

    Parameters
    ----------
    condtions, targets : list<str>
        Column names of the `conditions` and `targets`
    dataset : pandas.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    dataset = dataset.where((pd.notnull(dataset)), None)
    return dataset[conditions + targets].dropna(subset=targets)

def extract_sklearn_features_numerical(columns, dataset):
    """Extracts the numerical `columns` from `dataset`.

    Missing cells are imputed using mean imputation.

    Parameters
    ----------
    columns : list<str>
        Column names corresponding to numerical features.
    dataset : pandas.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    X_numerical = dataset[columns].as_matrix().astype(float)
    return Imputer().fit_transform(X_numerical)

def extract_sklearn_features_categorical(categories,  categories_to_val_map,
        dataset):
    """Converts each categorical column i (Ki categories, N rows) into an
    N x Ki matrix. Each row in the matrix is a binary vector.

    If there are J columns in conditions_categorical, then
    dataset_binary will be N x (J*sum(Ki)) (ie all encoded categorical
    matrices are concatenated).

    Example
        Nationality|Gender
        -----------+------      -----+-
        USA        | M          1,0,0,1
        France     | F          0,1,0,0
        France     | M          0,1,0,1
        Germany    | M          0,0,1,1

    Parameters
    ----------
    categories : list<str>
        List of column names corresponding to the categoricals.
    categories_to_val_map : dict<category : dict<cat:code>>
        A dictionary of the code lookup dictionary for each category. For
        example, in the example above
        {'Nationality': {'USA':0, 'France':1, 'Germany':2},
            'Gender': {'M':0, 'F':1}}
    dataset : pandas.DataFrame
        The matrix stored as a pandas dataframe. The `categories` must appear
        as columns in the dataframe.

    Returns
    -------
    dataset_binary : np.array
        Refined dataset with the same number of rows and appropriate number
        of columns representing the binary version of dataset.
    """
    dataset_binary = []
    for row in dataset.iterrows():
        row = list(row[1][categories])
        row_binary = binarize_categorical_row(
            categories, categories_to_val_map, row)
        dataset_binary.append(row_binary)
    return np.asarray(dataset_binary)

def binarize_categorical_row(categories, categories_to_val_map, row):
    """Unrolls a row of categorical data into the corresponding binary
    vector version. The order of the entries in `row` must be the same as those
    in the list `categories`. The `row` must be a list of strings corresponding
    to the value of each categorical column.
    """
    assert len(row) == len(categories)
    binary_data = []
    for categorical, value in zip(categories, row):
        K = len(categories_to_val_map[categorical])
        encoding = [0]*K
        encoding[categories_to_val_map[categorical][value]] = 1
        binary_data.extend(encoding)
    return binary_data

def build_categorical_to_value_map(columns, dataset):
    """Builds a dictionary of dictionaries.

    Parameters
    ----------
    columns : list<str>
        Column names corresponding to categorical features.
    dataset : pandas.DataFrame

    Returns
    -------
    categories_to_val_map : dict<col:dict>
        Dictionary of keys with dictionary values. Each value dictionary
        contains the mapping category -> code for the corresponding
        categorical feature, which is the key.
    """
    categories_to_val_map = dict()
    for categorical in columns:
        categories_to_val_map[categorical] = {val:code
            for (code,val) in enumerate(dataset[categorical].unique())}
    return categories_to_val_map

def extract_sklearn_univariate_target(target, dataset):
    """Extracts a single target column from a dataset as a vector for sklearn.

    Parameters
    ----------
    target : str
        Name of the target column.
    dataset : pandas.DataFrame

    Returns
    -------
    target_vector : np.array
    """
    return dataset[target].as_matrix().ravel()
