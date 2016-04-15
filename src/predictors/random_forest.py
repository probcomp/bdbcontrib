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

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from bayeslite.exception import BayesLiteException as BLE
import bdbcontrib
from bdbcontrib.predictors import predictor
from bdbcontrib.predictors import sklearn_utils as utils

class RandomForest(predictor.IBayesDBForeignPredictor):
    """A Random Forest foreign predictor.

    The `targets` must be a single categorical stattype.  The `conditions`
    may be arbitrary numerical or categorical columns.
    """

    @classmethod
    def create(cls, bdb, table, targets, conditions):
        cols = [c for c,_ in targets+conditions]
        df = bdbcontrib.bql_utils.table_to_df(bdb, table, cols)
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
            'categories_to_val_map': pred.categories_to_val_map
        }
        return pickle.dumps(state)

    @classmethod
    def deserialize(cls, bdb, binary):
        state = pickle.loads(binary)
        rf = cls(targets=state['targets'],
            conditions_numerical=state['conditions_numerical'],
            conditions_categorical=state['conditions_categorical'],
            rf_full=state['rf_full'], rf_partial=state['rf_partial'],
            categories_to_val_map=state['categories_to_val_map'])
        rf.prng = bdb.np_prng
        return rf

    @classmethod
    def name(cls):
        return 'random_forest'

    def __init__(self, targets=None, conditions_numerical=None,
            conditions_categorical=None, rf_full=None, rf_partial=None,
            categories_to_val_map=None):
        self.targets = targets
        self.conditions_numerical = conditions_numerical
        self.conditions_categorical = conditions_categorical
        if (conditions_numerical is not None
                and conditions_categorical is not None):
            self.conditions = self.conditions_numerical + \
                self.conditions_categorical
        self.rf_full = rf_full
        self.rf_partial = rf_partial
        self.categories_to_val_map = categories_to_val_map

    def train(self, df, targets, conditions):
        # Obtain the targets column.
        if len(targets) != 1:
            raise BLE(ValueError('RandomForest requires exactly one column in '
                'targets. Received {}'.format(targets)))
        if targets[0][1].lower() != 'categorical':
            raise BLE(ValueError('RandomForest can only classify CATEGORICAL '
                'columns. Received {}'.format(targets)))
        self.targets = [targets[0][0]]
        # Obtain the condition columns.
        if len(conditions) < 1:
            raise BLE(ValueError('RandomForest requires at least one column in '
                'conditions. Received {}'.format(conditions)))
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
        self.categories_to_val_map = dict()
        # Training set (regressors and labels)
        self.X_numerical = np.ndarray(0)
        self.X_categorical = np.ndarray(0)
        self.Y = np.ndarray(0)
        # Random Forests.
        self.rf_partial = RandomForestClassifier(n_estimators=100)
        self.rf_full = RandomForestClassifier(n_estimators=100)
        # Preprocess the data.
        self.dataset = utils.extract_sklearn_dataset(self.conditions,
            self.targets, df)
        self.categories_to_val_map = utils.build_categorical_to_value_map(
            self.conditions_categorical, self.dataset)
        self.X_categorical = utils.extract_sklearn_features_categorical(
            self.conditions_categorical, self.categories_to_val_map,
            self.dataset)
        self.X_numerical = utils.extract_sklearn_features_numerical(
            self.conditions_numerical, self.dataset)
        self.Y = utils.extract_sklearn_univariate_target(self.targets,
            self.dataset)
        # Train the random forest.
        self._train_rf()

    def _train_rf(self):
        """Trains the random forests classifiers.

        We train two classifiers, `partial` which is just trained on
        `conditions_numerical`, and `full` which is trained on
        `conditions_numerical+conditions_categorical`.

        This safe-guard feature is critical for querying; otherwise sklearn
        would crash whenever a categorical value unseen in training due to
        filtering (but existant in df nevertheless) was passed in.
        """
        # pylint: disable=no-member
        self.rf_partial.fit_transform(self.X_numerical, self.Y)
        self.rf_full.fit_transform(
            np.hstack((self.X_numerical, self.X_categorical)), self.Y)

    def _compute_targets_distribution(self, conditions):
        """Given conditions dict {feature_col:val}, returns the
        distribution and (class mapping for lookup) of the random label
        self.targets|conditions.
        """
        if not set(self.conditions).issubset(set(conditions.keys())):
            raise BLE(ValueError(
                'Must specify values for all the conditionals.\n'
                'Received: {}\n'
                'Expected: {}'.format(conditions, self.conditions_numerical +
                self.conditions_categorical)))

        # Are there any category values in conditions which never appeared during
        # training? If yes, we need to run the partial RF.
        unseen = any([conditions[cat] not in self.categories_to_val_map[cat]
            for cat in self.conditions_categorical])

        X_numerical = [conditions[col] for col in self.conditions_numerical]
        if unseen:
            distribution = self.rf_partial.predict_proba(X_numerical)
            classes = self.rf_partial.classes_
        else:
            X_categorical = [conditions[col] for col in
                self.conditions_categorical]
            X_categorical = utils.binarize_categorical_row(
                self.conditions_categorical, self.categories_to_val_map,
                X_categorical)
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
