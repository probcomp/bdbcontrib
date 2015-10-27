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

import math
import pickle
import numpy as np

import bdbcontrib
from bdbcontrib.predictors import predictor

class KeplersLaw(predictor.IBayesDBForeignPredictor):
    """A foreign predictor which models Kepler's Third Law.

    There must be exactly one `targets` column (period in minutes) and
    exactly two `conditions` columns (apogee in km, perigee in
    km, in that order). All stattypes are expected to be numerical.
    """

    @classmethod
    def create(cls, bdb, table, targets, conditions):
        cols = [c for c,_ in targets+conditions]
        df = bdbcontrib.table_to_df(bdb, table, cols)
        kl = cls()
        kl.train(df, targets, conditions)
        kl.prng = bdb.np_prng
        return kl

    @classmethod
    def serialize(cls, _bdb, pred):
        state = {
            'targets': pred.targets,
            'conditions': pred.conditions,
            'noise': pred.noise
        }
        return pickle.dumps(state)

    @classmethod
    def deserialize(cls, bdb, binary):
        state = pickle.loads(binary)
        kl = cls(targets=state['targets'], conditions=state['conditions'],
            noise=state['noise'])
        kl.prng = bdb.np_prng
        return kl

    @classmethod
    def name(cls):
        return 'keplers_law'

    def __init__(self, targets=None, conditions=None, noise=None):
        self.targets = targets
        self.conditions = conditions
        self.noise = noise

    def train(self, df, targets, conditions):
        # Obtain the targets column.
        if len(targets) != 1:
            raise ValueError('OrbitalMechanics can only target one '
                'column. Received {}'.format(targets))
        if targets[0][1].lower() != 'numerical':
            raise ValueError('OrbitalMechanics can only target a NUMERICAL '
                'column. Received {}'.format(targets))
        self.targets = [targets[0][0]]

        # Obtain the conditions column.
        if len(conditions) != 2:
            raise ValueError('OrbitalMechanics can only condition on '
                'two columns. Received {}'.format(conditions))
        if any(c[1].lower() != 'numerical' for c in conditions):
            raise ValueError('OrbitalMechanics can only condition on '
                'NUMERICAL columns. Received {}'.format(conditions))
        self.conditions = [c[0] for c in conditions]

        # The dataset.
        self.dataset = df[self.conditions + self.targets].dropna()
        X = self.dataset[self.conditions].as_matrix()

        # Learn the noise model.
        actual_period = self.dataset[self.targets].as_matrix().ravel()
        theoretical_period = self._compute_period(X[:,0], X[:,1])/60.
        errors = np.abs(actual_period-theoretical_period)
        errors = np.mean(np.select(
            [errors<np.percentile(errors, 95)], [errors]))
        self.noise = np.sqrt(np.mean(errors**2))

    def _compute_period(self, apogee_km, perigee_km):
        """Computes the period of the satellite in seconds given the apogee_km
        and perigee_km of the satellite.
        """
        GM = 398600.4418
        EARTH_RADIUS = 6378
        a = 0.5*(abs(apogee_km) + abs(perigee_km)) + EARTH_RADIUS
        return 2 * np.pi * np.sqrt(a**3/GM)

    def simulate(self, n_samples, conditions):
        if not set(self.conditions).issubset(set(conditions.keys())):
            raise ValueError('Must specify values for all the conditionals.\n'
                'Received: {}\n'
                'Expected: {}'.format(conditions, self.conditions))
        period = self._compute_period(conditions[self.conditions[0]],
            conditions[self.conditions[1]])
        return list(period/60. + self.prng.normal(scale=self.noise,
            size=n_samples))

    def logpdf(self, value, conditions):
        if not set(self.conditions).issubset(set(conditions.keys())):
            raise ValueError('Must specify values for all the conditionals.\n'
                'Received: {}\n'
                'Expected: {}'.format(conditions, self.conditions))
        period = self._compute_period(conditions[self.conditions[0]],
            conditions[self.conditions[1]]) / 60.
        return logpdfGaussian(value, period, self.noise)

HALF_LOG2PI = 0.5 * math.log(2 * math.pi)
def logpdfGaussian(x, mu, sigma):
    deviation = x - mu
    return - math.log(sigma) - HALF_LOG2PI \
        - (0.5 * deviation * deviation / (sigma * sigma))
