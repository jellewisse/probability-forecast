"""
Mixture models
"""

import numpy as np
from sklearn.base import BaseEstimator


class MixtureModel(BaseEstimator):
    """"""
    # TODO All estimators should specify all the parameters that can be set at
    # the class level in their __init__ as explicit keyword arguments
    # (no *args or **kwargs).

    def __init__(self, members):
        """"""
        for member in members:
            assert isinstance(member, MixtureMember)

        self.members = members
        self.member_count = len(members)
        self.weights = np.ones(len(members)) / len(members)

    def fit(self, X, y):
        """"""
        pass

    def predict(self, X):
        """"""
        pass

    def predict_proba(self, X):
        """"""

        return self.pdf(X)

    def pdf(self, X):
        """Probability density function of the ensemble"""
        pass

    def cdf(self, X):
        """Cumulative density function of the ensemble"""
        pass


class MixtureMember:
    """"""

    def __init__(self, distribution, parameters):
        """"""
        self.distribution = distribution
        self.parameters = parameters

    def pdf(self, X):
        """"""
        return self.distribution.pdf(X, self.parameters)

    def cdf(self, X):
        """"""
        return self.distribution.cdf(X, self.parameters)
