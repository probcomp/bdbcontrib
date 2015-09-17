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

class IForeignPredictor(object):
    """BayesDB foreign predictor interface.
    """
    def __init__(self, df, target, conditions):
        """
        Initializes and trains the the foriegn predictor.
        `df` is a pandas DataFrame containing the dataste.
        `target` and `conditions` are lists of (column_name, stattype),
        where column_name must be in the `df`.
        """
        raise NotImplementedError

    def get_targets(self):
        """Returns an ordered list of target columns that this foreign predictor
        is responsible for generating.
        """
        raise NotImplementedError

    def get_conditions(self):
        """Returns a list of conditional columns this foreign predictor requires
        to generate its targets.
        """
        raise NotImplementedError

    def simulate(self, n_samples, kwargs):
        """Simulate n_samples times from the conditional distribution
        P(targets|conditions) where kwargs is a dict of values for conditional
        columns.
        """
        raise NotImplementedError

    def logpdf(self, values, kwargs):
        """Compute P(targets|conditions) where kwargs is a dict of values for
        conditional columns.
        """
        raise NotImplementedError
