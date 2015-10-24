# -*- coding: utf-8 -*-

#   Copyright (c) 2010-2014, MIT Probabilistic Computing Project
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

# TODO: Update the interface for multivariate targets.

import pickle

import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

import bdbcontrib
from bdbcontrib.predictors import predictor

class MultipleRegression(predictor.IBayesDBForeignPredictor):
    """
    A MultipleRegression FP. The `targets` must be a single numerical stattype.

    Attributes
    ----------
    Please do not mess around with any (exploring is ok).
    """
    @classmethod
    def create(cls, bdb, table, targets, conditions):
        df = bdbcontrib.cursor_to_df(bdb.execute('''
                SELECT * FROM {}
        '''.format(table)))
        mr = cls()
        mr.train(df, targets, conditions)
        return mr

    @classmethod
    def serialize(cls, predictor):
        state = {
            'targets': predictor.targets,
            'conditions_numerical': predictor.conditions_numerical,
            'conditions_categorical': predictor.conditions_categorical,
            'mr_full': predictor.mr_full,
            'mr_partial': predictor.mr_partial,
            'mr_full_noise': predictor.mr_full_noise,
            'mr_partial_noise': predictor.mr_partial_noise,
            'lookup': predictor.lookup
        }
        return pickle.dumps(state)

    @classmethod
    def deserialize(cls, binary):
        state = pickle.loads(binary)
        mr = cls(targets=state['targets'],
            conditions_numerical=state['conditions_numerical'],
            conditions_categorical=state['conditions_categorical'],
            mr_full=state['mr_full'], mr_partial=state['mr_partial'],
            mr_full_noise=state['mr_full_noise'],
            mr_partial_noise=state['mr_partial_noise'],
            lookup=state['lookup'])
        return mr

    @classmethod
    def name(cls):
        return 'multiple_regression'

    def __init__(self, targets=None, conditions_numerical=None,
            conditions_categorical=None, mr_full=None, mr_partial=None,
            mr_full_noise=None, mr_partial_noise=None, lookup=None):
        self.targets = targets
        self.conditions_numerical = conditions_numerical
        self.conditions_categorical = conditions_categorical
        if (self.conditions_numerical is not None
                and self.conditions_categorical is not None):
            self.conditions = self.conditions_numerical + \
                self.conditions_categorical
        self.mr_full = mr_full
        self.mr_partial = mr_partial
        self.mr_full_noise = mr_full_noise
        self.mr_partial_noise = mr_partial_noise
        self.lookup = lookup

    def train(self, df, targets, conditions):
        # Obtain the targets column.
        if len(targets) != 1:
            raise ValueError('MultipleRegression requires at least one column '
                'in targets. Received {}'.format(targets))
        if targets[0][1].lower() != 'numerical':
            raise ValueError('MultipleRegression can only regress NUMERICAL '
                'columns. Received {}'.format(targets))
        self.targets = [targets[0][0]]

        # Obtain the condition columns.
        if len(conditions) < 1:
            raise ValueError('MultipleRegression requires at least one '
                'column in conditions. Received {}'.format(conditions))
        self.conditions_categorical = []
        self.conditions_numerical = []
        for c in conditions:
            if c[1].lower() == 'categorical':
                self.conditions_categorical.append(c[0])
            else:
                self.conditions_numerical.append(c[0])
        self.conditions = self.conditions_numerical + \
            self.conditions_categorical

        # The dataset.
        self.dataset = pd.DataFrame()
        # Lookup for categoricals to code.
        self.lookup = dict()
        # Training set (regressors and labels)
        self.X_numerical = np.ndarray(0)
        self.X_categorical = np.ndarray(0)
        self.Y = np.ndarray(0)
        # Linear regressors.
        self.mr_partial = LinearRegression()
        self.mr_full = LinearRegression()

        # Build the foreign predictor.
        self._init_dataset(df)
        self._init_categorical_lookup()
        self._init_X_categorical()
        self._init_X_numerical()
        self._init_Y()
        self._train_mr()

    def _init_dataset(self, df):
        """Create the dataframe of the satellites dataset.

        `NaN` strings are converted to Python `None`.

        Creates: self.dataset.
        """
        df = df.where((pd.notnull(df)), None)
        self.dataset = df[self.conditions + self.targets].dropna(
            subset=self.targets)

    def _init_categorical_lookup(self):
        """Builds a dictionary of dictionaries. Each dictionary contains the
        mapping category -> code for the corresponding categorical feature.

        Creates: self.lookup
        """
        for categorical in self.conditions_categorical:
            self.lookup[categorical] = {val:code for (code,val) in
                enumerate(self.dataset[categorical].unique())}

    def _init_X_categorical(self):
        """Converts each categorical column i (Ki categories, N rows) into an
        N x Ki matrix. Each row in the matrix is a binary vector.

        If there are J columns in conditions_categorical, then
        self.X_categorical will be N x (J*sum(Ki)) (ie all encoded categorical
        matrices are concatenated).

        Example
            Nationality|Gender
            -----------+------      -----+-
            USA        | M          1,0,0,1
            France     | F          0,1,0,0
            France     | M          0,1,0,1
            Germany    | M          0,0,1,1

        Creates: self.X_categorical
        """
        self.X_categorical = []
        for row in self.dataset.iterrows():
            data = list(row[1][self.conditions_categorical])
            binary_data = self._binarize_categorical_row(data)
            self.X_categorical.append(binary_data)
        self.x_categorical = np.asarray(self.X_categorical)

    def _binarize_categorical_row(self, data):
        """Unrolls a row of categorical data into the corresponding binary
        vector version.

        The order of the columns in `data` must be the same as
        self.conditions_categorical. The `data` must be a list of strings
        corresponding to the value of each categorical column.
        """
        assert len(data) == len(self.conditions_categorical)
        binary_data = []
        for categorical, value in zip(self.conditions_categorical, data):
            K = len(self.lookup[categorical])
            encoding = [0]*K
            encoding[self.lookup[categorical][value]] = 1
            binary_data.extend(encoding)
        return binary_data

    def _init_X_numerical(self):
        """Extract numerical columns from the dataset into a matrix.

        Creates: self.X_numerical
        """
        X_numerical = self.dataset[self.conditions_numerical].as_matrix().astype(
            float)
        # XXX This is necessary. sklearn cannot deal with missing values and
        # every row in the dataset has at least one missing value. The
        # imputer is generic. Cannot use CC since a foreign predictor
        # is independent.
        self.X_numerical = Imputer().fit_transform(X_numerical)

    def _init_Y(self):
        """Extracts the targets column.

        Creates: self.Y
        """
        self.Y = self.dataset[self.targets].as_matrix().ravel()

    def _train_mr(self):
        """Trains the random forests classifiers.

        We train two classifiers, `partial` which is just trained on
        `conditions_numerical`, and `full` which is trained on
        `conditions_numerical+conditions_categorical`.

        This safe-guard feature is critical for querying; otherwise sklearn
        would crash whenever a categorical value unseen in training due to
        filtering (but existant in df nevertheless) was passed in.
        """
        self.mr_partial.fit(self.X_numerical, self.Y)
        self.mr_full.fit(np.hstack((self.X_numerical, self.X_categorical)),
            self.Y)

        self.mr_partial_noise = \
            np.linalg.norm(self.Y-self.mr_partial.predict(
                self.X_numerical))/len(self.Y)

        self.mr_full_noise = \
            np.linalg.norm(self.Y-self.mr_full.predict(
                np.hstack((self.X_numerical, self.X_categorical))))/len(self.Y)

    def _compute_targets_distribution(self, conditions):
        """Given conditions dict {feature_col:val}, returns the conditional
        mean of the `targets`, and the scale of the Gaussian noise.
        """
        if not set(self.conditions).issubset(set(conditions.keys())):
            raise ValueError('Must specify values for all the conditionals.\n'
                'Received: {}\n'
                'Expected: {}'.format(conditions, self.conditions_numerical +
                self.conditions_categorical))

        # Are there any category values in conditions which never appeared during
        # training? If yes, we need to run the partial RF.
        unseen = any([conditions[cat] not in self.lookup[cat]
            for cat in self.conditions_categorical])

        X_numerical = [conditions[col] for col in self.conditions_numerical]

        if unseen:
            prediction = self.mr_partial.predict(X_numerical)
            noise = self.mr_partial_noise
        else:
            X_categorical = [conditions[col] for col in
                self.conditions_categorical]
            X_categorical = self._binarize_categorical_row(X_categorical)
            prediction = self.mr_full.predict(
                np.hstack((X_numerical, X_categorical)))
            noise = self.mr_full_noise

        return prediction[0], noise

    def simulate(self, n_samples, conditions):
        prediction, noise = self._compute_targets_distribution(conditions)
        return list(prediction + np.random.normal(scale=noise, size=n_samples))

    def logpdf(self, values, conditions):
        prediction, noise = self._compute_targets_distribution(conditions)
        return norm.logpdf(values, loc=prediction, scale=noise)
