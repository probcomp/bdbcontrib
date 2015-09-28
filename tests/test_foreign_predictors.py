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

from bdbcontrib.foreign.random_forest import RandomForest
from bdbcontrib.foreign.keplers_law import KeplersLaw
from bdbcontrib.foreign.multiple_regression import MultipleRegression

# TODO: More robust tests exploring more interesting cases. The main use
# right now is crash testing. Moreover common patterns can be automated.

satfile = '../examples/satellites/data/satellites.csv'
df = pd.read_csv(satfile)

def test_random_forest():
    # Create RandomForest.
    conditions_numerical = ['Perigee_km', 'Apogee_km', 'Eccentricity',
        'Period_minutes', 'Launch_Mass_kg', 'Power_watts',
        'Anticipated_Lifetime']
    conditions_categorical = ['Class_of_Orbit', 'Purpose', 'Users']
    conditions = [(c, 'NUMERICAL') for c in conditions_numerical] + \
        [(c, 'CATEGORICAL') for c in conditions_categorical]
    target = [('Type_of_Orbit', 'CATEGORICAL')]
    srf_predictor = RandomForest()
    srf_predictor.train(df, target, conditions)
    # Dummy realization of conditions.
    dummy_conditions = {'Perigee_km':535, 'Apogee_km':551,
        'Eccentricity':0.00116, 'Period_minutes':95.5, 'Launch_Mass_kg':293,
        'Power_watts':414, 'Anticipated_Lifetime':3, 'Class_of_Orbit':'LEO',
        'Purpose':'Astrophysics', 'Users':'Government/Civil'}
    # Crash tests.
    srf_predictor.simulate(10, dummy_conditions)
    pdf_val = srf_predictor.logpdf('Intermediate', dummy_conditions)
    assert srf_predictor.logpdf(1, dummy_conditions) == float('-inf')
    # Serialization tests.
    srf_binary = RandomForest.serialize(srf_predictor)
    srf_predictor2 = RandomForest.deserialize(srf_binary)
    srf_predictor2.simulate(10, dummy_conditions)
    pdf_val2 = srf_predictor2.logpdf('Intermediate', dummy_conditions)
    assert np.allclose(pdf_val, pdf_val2)

def test_keplers_law():
    # Create RandomForest.
    conditions = [('Apogee_km','NUMERICAL'), ('Perigee_km', 'NUMERICAL')]
    target = [('Period_minutes', 'NUMERICAL')]
    kl_predictor = KeplersLaw()
    kl_predictor.train(df, target, conditions)
    # Dummy realization of conditions.
    dummy_conditions = {'Apogee_km':1000, 'Perigee_km':1000}
    # Crash tests.
    kl_predictor.simulate(10, dummy_conditions)
    pdf_val = kl_predictor.logpdf(1440, dummy_conditions)
    # Serialization tests.
    kl_binary = KeplersLaw.serialize(kl_predictor)
    kl_predictor2 = KeplersLaw.deserialize(kl_binary)
    kl_predictor2.simulate(10, dummy_conditions)
    pdf_val2 = kl_predictor2.logpdf(1440, dummy_conditions)
    assert np.allclose(pdf_val, pdf_val2)

def test_multiple_regression():
    # Create MultipleRegression.
    conditions_numerical = ['Perigee_km', 'Apogee_km', 'Eccentricity',
        'Period_minutes', 'Launch_Mass_kg', 'Power_watts']
    conditions_categorical = ['Class_of_Orbit', 'Purpose', 'Users']
    conditions = [(c, 'NUMERICAL') for c in conditions_numerical] + \
        [(c, 'CATEGORICAL') for c in conditions_categorical]
    target = [('Anticipated_Lifetime', 'NUMERICAL')]
    mr_predictor = MultipleRegression()
    mr_predictor.train(df, target, conditions)
    # Dummy realization of conditions.
    dummy_conditions = {'Perigee_km':535, 'Apogee_km':551,
        'Eccentricity':0.00116, 'Period_minutes':95.5, 'Launch_Mass_kg':293,
        'Power_watts':414, 'Class_of_Orbit':'LEO', 'Purpose':'Astrophysics',
        'Users':'Government/Civil'}
    # Crash tests.
    mr_predictor.simulate(10, dummy_conditions)
    pdf_val = mr_predictor.logpdf(4.9, dummy_conditions)
    # Serialization tests.
    mr_binary = MultipleRegression.serialize(mr_predictor)
    mr_predictor2 = MultipleRegression.deserialize(mr_binary)
    mr_predictor2.simulate(10, dummy_conditions)
    pdf_val2 = mr_predictor2.logpdf(4.9, dummy_conditions)
    assert np.allclose(pdf_val, pdf_val2)
