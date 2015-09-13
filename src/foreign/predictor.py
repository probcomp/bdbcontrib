class IForeignPredictor(object):
    """BayesDB foreign predictor interface.

    Currently, foreign predictors are closely tied to the underlying datataset.
    When/if foreign predictors become generic modules, this interface will
    require redesign.
    """
    def get_targets(self):
        """Returns an ordered list of target columns this foreign predictor is
        responsible for generating.
        """
        raise NotImplementedError

    def get_conditions(self):
        """Returns a list of conditional columns this foreign predictor requires
        to generate targets.
        """
        raise NotImplementedError

    def simulate(self, n_samples, kwargs):
        """Simulate n_samples times from the conditional distribution
        P(targets|kwargs) where kwargs are values for conditional columns.
        """
        raise NotImplementedError

    def probability(self, values, kwargs):
        """Compute P(targets|kwargs) where kwargs are values for
        conditional columns.
        """
        raise NotImplementedError
