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

from bdbcontrib.bql_utils import df_to_table
from crosscat.tests import synthetic_data_generator as sdg

from bdbcontrib.predictors.random_forest import RandomForest
from bdbcontrib.predictors.keplers_law import KeplersLaw
from bdbcontrib.predictors.multiple_regression import MultipleRegression

# TODO: More robust tests exploring more interesting cases. The main use
# right now is crash testing. Moreover common patterns can be automated.

# ------------------------------------------------------------------------------
# Helper

def get_synthetic_data(n_sample, seed=438):
    cols_to_views = [
        0, 0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2,
        3,
        4
    ]
    colnames = [
        'm1', 'c1', 'm2', 'c2', 'm3',
        'm4', 'c3', 'c4', 'm5',
        'c5', 'c6', 'c7',
        'c8',
        'c9'
    ]
    cctypes = [
        'multinomial', 'continuous', 'multinomial', 'continuous', 'multinomial',
        'multinomial', 'continuous', 'continuous', 'multinomial',
        'continuous', 'continuous', 'continuous',
        'continuous',
        'continuous'
    ]
    distargs = [
        dict(K=9), None, dict(K=9), None, dict(K=7),
        dict(K=4), None, None, dict(K=9),
        None, None, None,
        None,
        None
    ]
    component_weights = [
        [.2, .3, .5],
        [.9, .1],
        [.4, .4, .2],
        [.8, .2],
        [.4, .5, .1]
    ]
    separation = [0.8, 0.9, 0.65, 0.7, 0.75]
    synthetic_data = sdg.gen_data(cctypes, n_sample, cols_to_views,
        component_weights, separation, seed, distargs=distargs)
    data = pd.DataFrame(synthetic_data[0])
    data.columns = colnames
    return df_to_table(data)

# ------------------------------------------------------------------------------
# Tests

def test_random_forest():
    # Train foreign predictor.
    (bdb, table) = get_synthetic_data(150)
    conditions = [(c, 'NUMERICAL') for c in ['c1','c2','c4','c8']] + \
        [(c, 'CATEGORICAL') for c in ['m1', 'm3']]
    target = [('m5', 'CATEGORICAL')]
    srf_predictor = RandomForest.create(bdb, table, target, conditions)

    # Dummy realization of conditions. Include extra conditions with unseen.
    parents = {'c1':1.3, 'c2':-2.1, 'c4':0.2, 'c8':0.2, 'm1':1, 'm3':7, 'm5':5}

    # Crash tests.
    srf_predictor.simulate(10, parents)
    pdf_val = srf_predictor.logpdf(7, parents)
    assert srf_predictor.logpdf(-1, parents) == float('-inf')

    # Dummy realization of conditions. Include extra conditions all seen.
    parents = {'c1':1.3, 'c2':-2.1, 'c4':0.2, 'c8':0.2, 'm1':1, 'm3':4, 'm5':5}

    # Crash tests.
    srf_predictor.simulate(10, parents)
    pdf_val = srf_predictor.logpdf(7, parents)
    assert srf_predictor.logpdf(-1, parents) == float('-inf')

    # Serialization tests.
    srf_binary = RandomForest.serialize(bdb, srf_predictor)
    srf_predictor2 = RandomForest.deserialize(bdb, srf_binary)
    srf_predictor2.simulate(10, parents)
    pdf_val2 = srf_predictor2.logpdf(7, parents)
    assert np.allclose(pdf_val, pdf_val2)

def test_keplers_law():
    # Train foreign predictor.
    (bdb, table) = get_synthetic_data(150)
    conditions = [('c1','NUMERICAL'), ('c3', 'NUMERICAL')]
    target = [('c4', 'NUMERICAL')]
    kl_predictor = KeplersLaw.create(bdb, table, target, conditions)

    # Dummy realization of conditions.
    inputs = {'c1':2.1, 'c3':1.7}

    # Crash tests.
    kl_predictor.simulate(10, inputs)
    pdf_val = kl_predictor.logpdf(1.2, inputs)

    # Serialization tests.
    kl_binary = KeplersLaw.serialize(bdb, kl_predictor)
    kl_predictor2 = KeplersLaw.deserialize(bdb, kl_binary)
    kl_predictor2.simulate(10, inputs)
    pdf_val2 = kl_predictor2.logpdf(1.2, inputs)
    assert np.allclose(pdf_val, pdf_val2)

def test_multiple_regression():
    # Train foreign predictor.
    (bdb, table) = get_synthetic_data(150)
    conditions = [(c, 'NUMERICAL') for c in ['c1','c2','c4','c8']] + \
        [(c, 'CATEGORICAL') for c in ['m1', 'm3']]
    target = [('c7', 'NUMERICAL')]
    mr_predictor = MultipleRegression.create(bdb, table, target, conditions)

    # Dummy realization of conditions. Include extra conditions with unseen.
    inputs = {'c1':1.3, 'c2':-2.1, 'c4':0.2, 'c8':0.2, 'm1':1, 'm3':7, 'm5':5}

    # Crash tests.
    mr_predictor.simulate(10, inputs)
    pdf_val = mr_predictor.logpdf(-0.4, inputs)

    # Dummy realization of conditions. Include extra conditions al seen.
    inputs = {'c1':1.3, 'c2':-2.1, 'c4':0.2, 'c8':0.2, 'm1':1, 'm3':4, 'm5':5}

    # Crash tests.
    mr_predictor.simulate(10, inputs)
    pdf_val = mr_predictor.logpdf(-0.4, inputs)

    # Serialization tests.
    mr_binary = MultipleRegression.serialize(bdb, mr_predictor)
    mr_predictor2 = MultipleRegression.deserialize(bdb, mr_binary)
    mr_predictor2.simulate(10, inputs)
    pdf_val2 = mr_predictor2.logpdf(-0.4, inputs)
    assert np.allclose(pdf_val, pdf_val2)
