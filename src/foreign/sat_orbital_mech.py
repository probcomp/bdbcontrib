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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

from bdbcontrib.foreign import predictor

class SatOrbitalMechanics(predictor.IForeignPredictor):
    """
    A "foreign predictor" trained on the satellites.csv dataset. The
    SatOrbitalMechanics is trained to model
        `Period_minutes`
    conditioned on
        'Apogee_km', 'Perigee_km',
    Using Kepler's Third Law.

    Attributes
    ----------
    Please do not mess around with any (exploring is ok).

    Methods
    -------
    simulate(n_samples, apogee_km, perigee_km)
        Simulate Period_minutes|Apogee_km,Perigee_km
    probability(period_mins, apogee_km, perigee_km)
        Compute P(period_mins|kwargs).
    """
    conditions =['Apogee_km', 'Perigee_km']
    target = ['Period_minutes']

    def __init__(self, sat_df):
        """Initializes SatOrbitalMechanics, and learns a noise model for the
        period.
        """
        self.dataset = sat_df[self.conditions + self.target].dropna()
        X = self.dataset[self.conditions].as_matrix()
        actual_period = self.dataset[self.target].as_matrix().ravel()

        theoretical_period = self._compute_period(X[:,0], X[:,1])/60.
        errors = np.abs(actual_period-theoretical_period)
        errors = np.mean(np.select(
            [errors<np.percentile(errors, 95)], [errors]))
        self.noise = np.sqrt(np.mean(errors**2))

    def get_targets(self):
        return self.target

    def get_conditions(self):
        return self.conditions

    def simulate(self, n_samples, **kwargs):
        """Simulates period_val|kwargs under Kepler's Third Law and Gaussian
        noise model. kwargs must be of the form apogee_km=a, perigee_km=p.
        """
        period = self._compute_period(kwargs[self.conditions[0]],
            kwargs[self.conditions[1]])
        return period/60. + np.random.normal(scale=self.noise,
            size=n_samples)

    def probability(self, period_val, **kwargs):
        """Computes the probability density P(period_val|kwargs) under
        Kepler's Third Law and Gaussian noise model.
        kwargs must be of the form Apogee_km=a, Perigee_km=p.
        """
        if not kwargs.keys().issubset(set(self.conditions)):
            raise ValueError('Must specify values for all the conditionals.\n'
                'Received: {}\n'
                'Expected: {}'.format(kwargs, self.conditions))
        period = self._compute_period(kwargs[self.conditions[0]],
            kwargs[self.conditions[1]]) / 60.
        return 1./(self.noise * np.sqrt(2*np.pi)) * \
            np.exp(-(period_val-period)**2 / (2*self.noise**2))

    def _compute_period(self, apogee_km, perigee_km):
        """Computes the period of the satellite in seconds given the apogee_km
        and perigee_km of the satellite.
        """
        GM = 398600.4418
        EARTH_RADIUS = 6378
        a = 0.5*(apogee_km + perigee_km) + EARTH_RADIUS
        return 2 * np.pi * np.sqrt(a**3/GM)

