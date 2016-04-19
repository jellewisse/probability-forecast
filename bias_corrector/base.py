"""Base module for bias correctors"""
import abc


class BiasCorrector(object, metaclass=abc.ABCMeta):
    """Base class for bias corrector classes.

    The class takes a pandas dataframe and removes bias from it. The resulting
    coefficients are stored internally and used to debias any new data samples.
    """

    def __init__(self, member_count):
        """Constructor."""
        self.member_count = member_count

    def _validate_data(self, X):
        """Does the provided data have the right number of columns."""
        return X.shape[1] == self.member_count

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the bias parameters."""

    def predict(X):
        """Debias new data."""
