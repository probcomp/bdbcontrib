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

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

import bdbcontrib
from bdbcontrib.predictors import predictor

class RandomForest(predictor.IBayesDBForeignPredictor):
    """A random forest foreign predictor.

    The `targets` must be a single categorical stattype.

    Examples
    --------

    >>> df = pd.read_csv('/path/to/satellites.csv')
    >>> rf = RandomForest()
    >>> rf = train(df, `targets`, `conditions`)
    >>> rf.logpdf('Intermediate', {Perigee_km:535, Apogee_km:551,
            Eccentricity:0.00116, Period_minutes:95.5, Launch_Mass_kg:293,
            Power_watts:414,Anticipated_Lifetime:3, Class_of_Orbit:'LEO'
            Purpose:'Astrophysics', Users:'Government/Civil'})
    -0.1626

    >>> rf.simulate(10, {Perigee_km:535, Apogee_km:551,
            Eccentricity:0.00116, Period_minutes:95.5, Launch_Mass_kg:293,
            Power_watts:414,Anticipated_Lifetime:3, Class_of_Orbit:'LEO',
            Purpose='Astrophysics', Users='Government/Civil'})
    ['Intermediate', 'Intermediate', 'Intermediate', 'Intermediate',
     'Intermediate', 'Intermediate', 'Intermediate', 'Sun-Synchronous',
     'Intermediate', 'Intermediate']
    """

    @classmethod
    def create(cls, bdb, table, targets, conditions):
        cols = [c for c,_ in targets+conditions]
        df = bdbcontrib.table_to_df(bdb, table, cols)
        rf = cls()
        rf.train(df, targets, conditions)
        rf.prng = bdb.np_prng
        return rf

    @classmethod
    def serialize(cls, _bdb, pred):
        state = {
            'targets': pred.targets,
            'conditions_numerical': pred.conditions_numerical,
            'conditions_categorical': pred.conditions_categorical,
            'rf_full': pred.rf_full,
            'rf_partial': pred.rf_partial,
            'lookup': pred.lookup
        }
        return pickle.dumps(state)

    @classmethod
    def deserialize(cls, bdb, binary):
        state = pickle.loads(binary)
        rf = cls(targets=state['targets'],
            conditions_numerical=state['conditions_numerical'],
            conditions_categorical=state['conditions_categorical'],
            rf_full=state['rf_full'], rf_partial=state['rf_partial'],
            lookup=state['lookup'])
        rf.prng = bdb.np_prng
        return rf

    @classmethod
    def name(cls):
        return 'random_forest'

    def __init__(self, targets=None, conditions_numerical=None,
            conditions_categorical=None, rf_full=None, rf_partial=None,
            lookup=None):
        self.targets = targets
        self.conditions_numerical = conditions_numerical
        self.conditions_categorical = conditions_categorical
        if (conditions_numerical is not None
                and conditions_categorical is not None):
            self.conditions = self.conditions_numerical + \
                self.conditions_categorical
        self.rf_full = rf_full
        self.rf_partial = rf_partial
        self.lookup = lookup

    def train(self, df, targets, conditions):
        # Obtain the targets column.
        if len(targets) != 1:
            raise ValueError('RandomForest requires exactly one column in '
                'targets. Received {}'.format(targets))
        if targets[0][1].lower() != 'categorical':
            raise ValueError('RandomForest can only classify CATEGORICAL '
                'columns. Received {}'.format(targets))
        self.targets = [targets[0][0]]
        # Obtain the condition columns.
        if len(conditions) < 1:
            raise ValueError('RandomForest requires at least one column in '
                'conditions. Received {}'.format(conditions))
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
        # Random Forests.
        self.rf_partial = RandomForestClassifier(n_estimators=100)
        self.rf_full = RandomForestClassifier(n_estimators=100)
        # Build the foreign predictor.
        self._init_dataset(df)
        self._init_categorical_lookup()
        self._init_X_categorical()
        self._init_X_numerical()
        self._init_Y()
        self._train_rf()

    def _init_dataset(self, df):
        """Create the dataframe of the satellites dataset.

        `NaN` strings are converted to Python `None`.

        Rows where the target is absent are dropped.

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

    def _train_rf(self):
        """Trains the random forests classifiers.

        We train two classifiers, `partial` which is just trained on
        `conditions_numerical`, and `full` which is trained on
        `conditions_numerical+conditions_categorical`.

        This safe-guard feature is critical for querying; otherwise sklearn
        would crash whenever a categorical value unseen in training due to
        filtering (but existant in df nevertheless) was passed in.
        """
        self.rf_partial.fit_transform(self.X_numerical, self.Y)
        self.rf_full.fit_transform(
            np.hstack((self.X_numerical, self.X_categorical)), self.Y)

    def _compute_targets_distribution(self, conditions):
        """Given conditions dict {feature_col:val}, returns the
        distribution and (class mapping for lookup) of the random label
        self.targets|conditions.
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
            distribution = self.rf_partial.predict_proba(X_numerical)
            classes = self.rf_partial.classes_
        else:
            X_categorical = [conditions[col] for col in
                self.conditions_categorical]
            X_categorical = self._binarize_categorical_row(X_categorical)
            distribution = self.rf_full.predict_proba(
                np.hstack((X_numerical, X_categorical)))
            classes = self.rf_partial.classes_
        return distribution[0], classes

    def simulate(self, n_samples, conditions):
        distribution, classes = self._compute_targets_distribution(conditions)
        draws = self.prng.multinomial(1, distribution, size=n_samples)
        return [classes[np.where(d==1)[0][0]] for d in draws]

    def logpdf(self, value, conditions):
        distribution, classes = self._compute_targets_distribution(conditions)
        if value not in classes:
            return -float('inf')
        return np.log(distribution[np.where(classes==value)[0][0]])
