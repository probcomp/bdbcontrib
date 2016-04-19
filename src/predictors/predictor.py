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

class IBayesDBForeignPredictorFactory(object):
    """PRELIMINARY BayesDB foreign predictor factory interface.

    A foreign predictor factory knows how to create instances of
    particular kind of foreign predictor, either by training them from
    data or by loading them from their serialized representation.

    Note: A common implementation pattern is to make the class of the
    foreign predictor be a singleton factory by defining these methods
    as `@classmethod`, instead of creating a separate class.  See, for
    example, :class:`bdbcontrib.predictors.keplers_law.KeplersLaw`.
    """

    def name(self):
        """Return the name of this kind of foreign predictor, as a string.

        All registered foreign predictors must have unique names.
        """
        raise NotImplementedError

    def create(self, bdb, table, targets, conditions):
        """Create and train a foreign predictor for the given circumstances.

        The `targets` and `conditions` ultimately come from the schema
        the client indicates.

        TODO `create` is currently assumed to be deterministic.  To
        support FPs whose training is stochastic, the :class:`.Composer`
        metamodel will need to be extended with options for
        maintaining ensembles of independently trained instances of
        such FPs.

        Parameters
        ----------
        bdb : :class:`bayeslite.BayesDB`
            The BayesDB containing the data to train on.

        table : string
            The name of the BayesDB table containing the data.

        targets : list<tuple<string, string>>
            The columns to be predicted, as pairs of column name and
            stattype.
            TODO Until :class:`.Composer` supports joint prediction,
            the `targets` list will have length 1.

        conditions : list<tuple<string, string>>
            The columns to be used as inputs, as pairs of column name and
            stattype.

        Returns
        -------
        predictor : :class:`~.IBayesDBForeignPredictor`
            A trained instance of the foreign predictor.

        See Also
        --------

        :func:`bdbcontrib.bql_utils.table_to_df`
            a helper function you can use if you want to train from a
            Pandas DataFrame rather than interacting directly with the
            database.
        """
        raise NotImplementedError

    def serialize(self, bdb, predictor):
        """Serialize the given predictor instance to a string.

        The instance will have been created by calling either
        :meth:`create` or :meth:`deserialize` on this factory.
        """
        raise NotImplementedError

    def deserialize(self, bdb, blob):
        """Reconstitute a serialized predictor instance.

        The `blob` will have been created by calling :meth:`serialize`
        on an instance of :class:`IBayesDBForeignPredictorFactory`
        registered with the same :meth:`name` as this one at some
        point in the past.  In typical use, this will be the same
        instance as the present one, but the usual advice about
        backward compatibility with stored data formats applies.
        """
        raise NotImplementedError

class IBayesDBForeignPredictor(object):
    """PRELIMINARY BayesDB trained foreign predictor instance interface.

    A foreign predictor (FP) is itself an independent object which is typically
    used outside the universe of BayesDB.

    Training is the province of
    :meth:`.IBayesDBForeignPredictorFactory.create`.

    A trained foreign predictor must support:

    - Simulation of the target column for a new observation

    - Evaluation of the logpdf of the target column of a new
      observation taking a certain value

    Simulate and logpdf both require a full realization of all `conditions`.

    TODO We do not yet support referring to row-specific training
    results or querying predictive distributions for observed rows.

    BayesDB initializes foreign predictors through the methods in
    :class:`~IBayesDBForeignPredictorFactory`, so imposes no
    particular restrictions on `__init__`.
    """

    def simulate(self, n_samples, conditions):
        """Simulate from the distribution {`target`}|{`conditions`}.

        The distribution being simulated from implicitly depends upon
        the data through the results of training.

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
        list
            A list of the simulated values.
        """
        raise NotImplementedError

    def logpdf(self, value, conditions):
        """Evaluate the log-density of {`target`=`value`}|{`conditions`}.

        The distribution being evaluated implicitly depends upon the
        data through the results of training.

        Parameters
        ----------
        value : object
            The value of `target` to query.  The type will appropriate
            to the statistical type of the column.

        conditions : dict
            A dictionary of {'condition':'value'} for all `conditions`
            required by the FP.  The FP may signal an error if any
            `conditions` are missing, and should ignore any additional
            ones.

        Returns
        -------
        float
            The log probability density of the given target value
            given the conditions.
        """
        raise NotImplementedError
