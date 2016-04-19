"""Base module for statistical model mixtures."""
import abc
import numpy as np


class MixtureModel(object, metaclass=abc.ABCMeta):
    """Base class for statistical model mixtures."""

    def __init__(self, member_count, distribution):
        """Initialize a mixture model with a specific distribution.

        Parameters
        ----------
        distribution : class
            Class that implements pdf and cdf functions.
        member_count : integer
            Number of members to initialize the mixture with
        """
        self._members = \
            [MixtureMember(distribution) for _ in range(member_count)]
        self.member_count = member_count
        self.weights = np.ones(member_count) / member_count

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the model parameters."""

    def pdf(self, x):
        """Compute the model mixture PDF for a single sample.

        Parameters:
        -----------
        x : list
        """
        return sum([
            member.pdf(x) * weight
            for (member, weight)
            in zip(self._members, self.weights)
        ])

    def member_pdf(self, X, y):
        """Given a data matrix, compute the pdf values for each member.

        parameters:
        -----------
        v : ndarray

        returns:
        --------
        P : ndarray
        """
        P = np.zeros(X.shape())
        # Loop over members
        for m in range(self.member_count):
            P[:, m] = self._members[m].pdf(X[:, m])
        return P

    def cdf(self, x):
        """Compute the model mixture CDF."""
        return sum([
            member.cdf(x) * weight
            for (member, weight)
            in zip(self._members, self.weights)
        ])


class MixtureMember(object):
    """Base class for mixture member."""

    def __init__(self, distribution):
        """Initialize the member with a distribution class."""
        self.distribution = distribution  # Distribution function
        self.parameters = {}  # Distribution function keyword arguments
        self.bias = 0  # Additive bias vector

    def pdf(self, X):
        """Compute the member PDF."""
        return self.distribution.pdf(X - self.bias, **self.parameters)

    def cdf(self, X):
        """Compute the member CDF."""
        return self.distribution.cdf(X - self.bias, **self.parameters)
