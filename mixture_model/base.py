"""Base module for statistical model mixtures."""
import abc
import numpy as np


class MixtureModel(object, metaclass=abc.ABCMeta):
    """Base class for statistical model mixtures."""

    def __init__(self, member_count, distribution, member_names=None):
        """Initialize a mixture model with a specific distribution.

        Parameters
        ----------
        distribution : class
            Class that implements pdf and cdf functions.
        member_count : integer
            Number of members to initialize the mixture with
        """
        if member_names is None:
            member_names = [''] * member_count
        self._members = \
            [MixtureMember(distribution, name) for name in member_names]
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

    def __init__(self, distribution, name=''):
        """Initialize the member with a distribution class."""
        self.distribution = distribution  # Distribution function
        self.name = name
        self.parameters = {}  # Distribution function keyword arguments

    def pdf(self, X):
        """Compute the member PDF."""
        return self.distribution.pdf(X, **self.parameters)

    def cdf(self, X):
        """Compute the member CDF."""
        return self.distribution.cdf(X, **self.parameters)
