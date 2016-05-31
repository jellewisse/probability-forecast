"""Module with simple linear corrections."""

import numpy as np

# User modules
from .base import BiasCorrector


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


class SimpleBiasCorrector(BiasCorrector):
    """Additive correction of bias."""

    def __init__(self, member_count, grouping=None):
        """Constructor based on either member count or a grouping list."""
        # TODO TdR 31.05.16 : Support grouping
        super().__init__(member_count)

        # Initialize parameters
        self.intercept_per_model = np.zeros(member_count)
        self.deviation_per_model = np.zeros(member_count)

    def fit(self, X, y):
        """Calculate the mean offset."""
        super()._validate_data(X)
        self.intercept_per_model = _maximum_likelihood_bias(X, y)

    def predict(self, X):
        """Subtract the mean offset."""
        return X - self.intercept_per_model
