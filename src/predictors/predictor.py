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

class IBayesDBForeignPredictor(object):
    """BayesDB foreign predictor interface.

    A foreign predictor (FP) is itself an independent object which is typically
    used outside the universe of BayesDB.

    The primitives that an FP must support are:

        - Train, given as inputs a
            Pandas dataframe, which contains the training data.
            Set of targets, which the FP is responsible for generating.
            Set of conditions, which the FP may use to generate targets.

        - Simulate targets.
        - Evaluate the logpdf of targets taking certain values.

        Simulate and logpdf both require a full realization of all `conditions`.

    TODO: Extended interface for serialization of foreign predictors.

    Explicit initialization of foreign predictors using parameters in `__init__`
    is strongly discouraged.
    """
    def train(self, df, targets, conditions):
        """Called to train the foreign predictor.

        Parameters
        ----------
        df : pandas.Dataframe
            Contains the training set.

        targets: list<tuple>
            A list of `targets` that the FP must learn to generate. The list is
            of the form [(`colname`, `stattype`),...], where `colname` must be
            the name of a column in `df`, and `stattype` is the data type.

        conditions: list<tuple>
            A list of `conditions` that the FP may use to generate `targets`.
            The list is of the same form as `targets`.

        The `targets` and `conditions` ultimately come from the schema
        the client indicates.

        The return value of `train` is ignored.
        """
        raise NotImplementedError

    def simulate(self, n_samples, conditions):
        """Simulate from the distribution {`targets`}|{`conditions`}.

        Parameters
        ----------
        n_samples : int
            Number of samples to simulate.

        conditions : dict
            A dictionary of {'condition':'value'} for all `conditions`
            required by the FP.  The FP may signal an error if any
            `conditions` are missing, and should ignore any additional
            ones.

        Returns
        -------
        A list of the simulated values.
        """
        raise NotImplementedError

    def logpdf(self, targets_vals, conditions):
        """Evaluate the log-density of {`targets`=`targets_vals`}|`conditions`.

        Parameters
        ----------
        values : int
            The value of `target` to query

        conditions : dict
            A dictionary of {'condition':'value'} for all `conditions`
            required by the FP.  The FP may signal an error if any
            `conditions` are missing, and should ignore any additional
            ones.

        Returns
        -------
        float: The log probability density of the given target value
        given the conditions.
        """
        raise NotImplementedError
