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

import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.preprocessing import Imputer

from bdbcontrib.foreign import predictor


# Helper functions for external users of module.
GM = 398600.4418
EARTH_RADIUS = 6378

def compute_period(apogee_km, perigee_km):
    """Computes the period of the satellite in seconds given the apogee_km
    and perigee_km of the satellite.
    """
    a = 0.5*(abs(apogee_km) + abs(perigee_km)) + EARTH_RADIUS
    T = 2 * np.pi * np.sqrt(a**3/GM) / 60.
    return T

def compute_a(T):
    a = ( (60*T/(2*np.pi))**2 * GM)**(1./3)
    return a

def compute_T(a):
    T = 2 * np.pi * np.sqrt(a**3/GM) / 60.
    return T

class KeplersLaw(predictor.IForeignPredictor):
    """
    A foreign predictor which models Kepler's Third Law for a single `targets`
    (period in minutes) and `conditions` (apogee in km, perigee in km). All
    stattypes are expected to be numerical.

    Attributes
    ----------
    Please do not mess around with any (exploring is ok).
    """
    # XXX TEMPORARY HACK for testing purposes.
    conditions =['Apogee_km', 'Perigee_km']
    targets = ['Period_minutes']

    def __init__(self, df, targets, conditions):
        # Obtain the targets column.
        if len(targets) != 1:
            raise ValueError('OrbitalMechanics can only targets one '
                'columns. Received {}'.format(targets))
        if targets[0][1].lower() != 'numerical':
            raise ValueError('OrbitalMechanics can only targets a NUMERICAL '
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

    def get_targets(self):
        return self.targets

    def get_conditions(self):
        return self.conditions

    def simulate(self, n_samples, conditions):
        if not set(self.conditions).issubset(set(conditions.keys())):
            raise ValueError('Must specify values for all the conditionals.\n'
                'Received: {}\n'
                'Expected: {}'.format(conditions, self.conditions))
        period = self._compute_period(conditions[self.conditions[0]],
            conditions[self.conditions[1]])
        return list(period/60. + np.random.normal(scale=self.noise,
            size=n_samples))

    def logpdf(self, targets_val, conditions):
        if not set(self.conditions).issubset(set(conditions.keys())):
            raise ValueError('Must specify values for all the conditionals.\n'
                'Received: {}\n'
                'Expected: {}'.format(conditions, self.conditions))
        period = self._compute_period(conditions[self.conditions[0]],
            conditions[self.conditions[1]]) / 60.
        return norm.logpdf(targets_val, loc=period, scale=self.noise)

    def _compute_period(self, apogee_km, perigee_km):
        """Computes the period of the satellite in seconds given the apogee_km
        and perigee_km of the satellite.
        """
        GM = 398600.4418
        EARTH_RADIUS = 6378
        a = 0.5*(abs(apogee_km) + abs(perigee_km)) + EARTH_RADIUS
        return 2 * np.pi * np.sqrt(a**3/GM)

def create_predictor(df, targets, conditions):
    return KeplersLaw(df, targets, conditions)
