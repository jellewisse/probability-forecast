
from scipy.stats import norm
import numpy as np


def _error_calculation(X, y):

    # Numpy column-wise subtraction is expressed as row-wise subtraction.
    E = (X.transpose() - y).transpose()
    return E


def _maximum_likelihood_bias(X, y):
    # Calculate errors
    errors = _error_calculation(X, y)
    # Calculate maximum likelihood means per column
    return errors.mean(axis=0)


def _maximum_likelihood_std(X, y):
    # Calculate errors
    errors = _error_calculation(X, y)
    # Calculate maximum likelihood means per column
    return errors.std(axis=0)


class GaussianMixtureModel:
    """"""

    def __init__(self, member_count):
        """Initialize a univariate normal Gaussian Mixture Model

        Parameters
        ----------
        member_count : integer
            Number of members to initialize the mixture with
        """
        members = [MixtureMember(norm) for _ in range(member_count)]

        self._members = members
        self.member_count = member_count
        self.weights = np.ones(member_count) / member_count
        self._forecast_prepared = False

    def fit(self, X, y):
        """

        Parameters
        ----------
        X : array_like, shape (n_samples, n_members)
            Input data, one column for each ensemble member
        y : array_like, shape (n_samples,)
            Targets for input data
            """

        assert X.shape[1] == self.member_count, "Bad number of member inputs"
        assert X.shape[0] == y.shape[0], "Input and tragets do not match."
        assert len(y.shape) == 1, "Provided y is not a vector."

        self._forecast_prepared = False
        # Mean bias correction
        model_bias = _maximum_likelihood_bias(X[:, 0], y)

        # Training strategy: train on first member.
        model_std = _maximum_likelihood_std(X[:, 0] - model_bias, y)
        for member in self._members:
            member.parameters['scale'] = model_std
            member.bias = model_bias

        print("Fit model with simple dressing")
        print("Scale: %f" % model_std)
        print("Bias:  %f" % model_bias)

    def set_member_means(self, member_means):
        assert len(member_means) == self.member_count

        for (mean, member) in zip(member_means, self._members):
            member.parameters['loc'] = mean

        self._forecast_prepared = True

    def get_member_means(self):
        return [
            member.parameters['loc'] - member.bias
            for member in self._members
        ]

    def _check_member_means(self):
        if not self._forecast_prepared:
            raise AttributeError("No ensemble values available for dressing.")

    def pdf(self, x):
        self._check_member_means()
        return sum([
            member.pdf(x) * weight
            for (member, weight)
            in zip(self._members, self.weights)
        ])

    def cdf(self, x):
        self._check_member_means()
        return sum([
            member.cdf(x) * weight
            for (member, weight)
            in zip(self._members, self.weights)
        ])

    def mean(self):
        self._check_member_means()
        return sum([
            (member.parameters['loc'] - member.bias) * weight
            for (member, weight)
            in zip(self._members, self.weights)
        ])


class MixtureMember:
    """"""

    def __init__(self, distribution):
        """"""
        self.distribution = distribution  # Distribution function
        self.parameters = {}  # Distribution function keyword arguments
        self.bias = 0  # Additive bias vector

    def pdf(self, X):
        """"""
        return self.distribution.pdf(X - self.bias, **self.parameters)

    def cdf(self, X):
        """"""
        return self.distribution.cdf(X - self.bias, **self.parameters)

if __name__ == "__main__":
    model = GaussianMixtureModel(10)
