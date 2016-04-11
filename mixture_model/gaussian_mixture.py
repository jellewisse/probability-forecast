"""Module for BMA with Gaussian mixtures."""
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm

# User modules
from .base import (
    MixtureModel,
    _squared_error_calculation,
    _maximum_likelihood_bias,
    # _maximum_likelihood_std
)

NORM_CONSTANT = -0.5 * np.log(2 * np.pi)


def _log_normal_pdf(squared_errors, variances):
    """Calculate the log Normal PDF function for an isotropic normal dist.

    parameters:
    -----------
    squared_errors: matrix (n x m)
    variances: vector (m)
    """
    P = squared_errors * (-1.0 / (2.0 * variances))
    P += (-1.0 / 2) * np.log(variances)
    P += NORM_CONSTANT
    if np.any(np.isnan(P)):
        raise ZeroDivisionError("Singularity in log normal pdfs.")
    return P


def find(needle, haystack):
    """Return the indices in haystack equivalent to needle."""
    return [x for (x, val) in enumerate(haystack) if needle == val]


def group_column_vec(X, group_map):
    """Average elements in X based on the provided grouping index."""
    assert len(X.shape) == 1, "no vector provided"
    return np.array([np.take(X, g).sum() for g in group_map])


class GaussianMixtureModel(MixtureModel):
    """Base class for Gaussian Mixtures."""

    def __init__(self, member_count, grouping=None):
        """Initialize a univariate normal Gaussian Mixture Model.

        Parameters
        ----------
        member_count : integer
            Number of members to initialize the mixture with
        grouping : list of integers specifying to which group a model belongs
        """
        super().__init__(member_count, norm)
        self._forecast_prepared = False
        self.member_count = member_count
        # Arguments for grouping
        if grouping is None:
            grouping = list(range(0, member_count))
        self.grouping = grouping
        self._optimizer = GaussianEM(grouping)

    def fit(self, X, y):
        """Fitting the GMM with the provided data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_members)
            Input data, one column for each ensemble member
        y : array_like, shape (n_samples,)
            Targets for input data
        """
        assert X.shape[1] == self.member_count, "Bad number of member inputs"
        assert X.shape[0] == y.shape[0], "Input and targets do not match."
        assert len(y.shape) == 1, "Provided y is not a vector."
        self._forecast_prepared = False

        # TODO Fix bias correction to support groups
        # Mean bias correction on first member
        bias_per_model = _maximum_likelihood_bias(X, y)
        self._optimizer.fit(X - bias_per_model, y)

        # Update variances and bias
        for (member, model_std, model_bias) in \
            zip(self._members,
                self._optimizer.get_member_variances(),
                bias_per_model):
            member.parameters['scale'] = model_std
            member.bias = model_bias

        # Update weights
        self.weights = self._optimizer.get_member_weights()

    def _check_member_means(self):
        if not self._forecast_prepared:
            raise AttributeError("No ensemble values available for dressing.")

    def set_member_means(self, member_means):
        """Clamp forecast model means to the GMM."""
        assert len(member_means) == self.member_count
        for (mean, member) in zip(member_means, self._members):
            member.parameters['loc'] = mean
        self._forecast_prepared = True

    def get_member_means(self):
        """Return forecast model means of members."""
        return [
            member.parameters['loc'] - member.bias
            for member in self._members
        ]

    def set_member_variances(self, member_variances):
        """Modify the member variances explicitly."""
        assert len(member_variances) == self.member_count
        for (variance, member) in zip(member_variances, self._members):
            member.parameters['scale'] = variance

    def get_member_variances(self):
        """Return fitted member variances."""
        return [
            member.parameters['scale']
            for member in self._members
        ]

    def mean(self):
        """Return the GMM mean forecast."""
        self._check_member_means()
        return sum([
            (member.parameters['loc'] - member.bias) * weight
            for (member, weight)
            in zip(self._members, self.weights)
        ])

    def pdf(self, x):
        """Apply the GMM PDF function."""
        self._check_member_means()
        return super().pdf(x)

    def cdf(self, x):
        """Apply the GMM CDF function."""
        self._check_member_means()
        return super().cdf(x)


class GaussianEM(object):
    """Computation class for BMA on top of a GMM."""

    # Constants
    N_ITER = 2000
    E_TOL = 1e-5

    def __init__(self, grouping):
        """Class constructor."""
        # Grouping arguments
        self.grouping = grouping
        self.group_count = len(set(grouping))
        self.group_map = \
            np.array([find(g, grouping) for g in range(self.group_count)])
        self.members_per_group = [len(g) for g in self.group_map]
        self.member_count = len(grouping)

        # Model parameters
        # Parameters are defined per group, meaning they are the same for all
        # group members by definition.
        self._dim = 1  # Dimensionality of output
        self.variance_prior_W = 1  # Matrix of dim x dim, 0 means no prior.
        self.variance_prior_nu = 3  # Scalar value, 2 means no prior.
        self.variances = np.ones(self.group_count)
        # Uniform prior on weights
        self.weight_prior = \
            np.ones(self.group_count) * 2  # All ones means no prior
        self.weights = np.ones(self.group_count) / self.member_count

    def get_member_variances(self):
        """Return variances for each member.

        Effectively does a mapping from the group id to the group variance.
        """
        return self.variances[self.grouping]

    def get_member_weights(self):
        """Return weights for each member.

        Effectively does a mapping from the group id to the group weight.
        """
        return self.weights[self.grouping]

    def fit(self, X, y):
        """Run the EM algorithm with the last solution as prior.

        If there is no last solution a uniform prior is used.
        """
        # Number of columns must match the number of ensemble members
        assert len(self.grouping) == X.shape[1], \
            "Data does not fit the model."
        assert X.shape[0] == y.shape[0], \
            "Mismatch between data and observations."

        iter_count = 0
        self.log_liks = []
        self.log_likelihood = np.nan
        self._converged = False
        self.squared_errors = _squared_error_calculation(X, y)
        self.responsibility = np.zeros(X.shape)
        while iter_count < self.N_ITER and not self._converged:
            # Algorithm tests
            try:
                assert_almost_equal(self.get_member_weights().sum(), 1.0, 6)
            except AssertionError:
                print("weights don't sum to 1!")
            if np.any(self.weights < 0) or np.any(self.weights > 1):
                print("weights no longer probabilities!")
            if np.any(np.isnan(self.variances)) or np.any(self.variances < 0):
                print("singularity in variances!")

            # Core
            try:
                # TODO If the result of a step means a
                # NaN parameter, do fallback
                self.e_step()
                self.m_step()
                self.log_liks.append(self.log_likelihood)
            except ZeroDivisionError:
                # Singularity detected.
                # Perturb parameters with random noise and restart fit.
                # print("Singularity detected - perturbing")
                print("singularity: perturbing coefficients!")
                self._perturb_singularities()
            iter_count += 1

        # Cleanup
        # if iter_count < self.N_ITER:
        #     print("Log: BMA converged in %d iterations." % iter_count)
        # else:
        #     # No convergence - reuse parameters from previous iteration.
        #     print("Log: BMA did not converge within %d iterations." %
        #           self.N_ITER)
        # print("Log likelihood: %f" % self.log_likelihood)
        # print("Member weights: ", self.get_member_weights())
        # print("Member variances: ", self.get_member_variances())

        # Clear large matrices
        self.responsibility = None
        self.squared_errors = None

    def _perturb_singularities(self):
        singularity_index = np.logical_or(np.logical_or(
            self.variances <= 1e-30,
            np.isinf(self.variances)),
            np.isnan(self.variances))
        self.variances[singularity_index] = \
            np.random.rand(singularity_index.sum()) * 2
        self.weights[singularity_index] = 1e-30

        # Renormalize weights
        self.weights /= self.weights.sum()

    # TODO Test
    def e_step(self):
        """E-step.

        Underflow errors are accounted for with the log-sum-exp trick.
        See for example
        https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
        """
        # Part 1 : calculate proportional responsibility
        new_responsibility = \
            _log_normal_pdf(self.squared_errors, self.get_member_variances())
        # TODO Weights can go to 0. Taking logarithm will give singularities.
        new_responsibility += np.log(self.get_member_weights())
        new_loglik = 0.

        # Part 2: update log-likelihood and normalize above responsibilities
        # Use log-sum-exp trick.
        # Find maximum value per row
        max_per_row = new_responsibility.max(axis=1)
        # Normalization constant for each row
        norm_per_row = np.exp(
            new_responsibility.transpose() - max_per_row
        ).transpose().sum(axis=1)
        # Normalize each row
        loglik_per_row = max_per_row + np.log(norm_per_row)
        new_responsibility = np.exp(
            new_responsibility.transpose() - loglik_per_row
        ).transpose()
        new_loglik = loglik_per_row.sum()
        self.responsibility = new_responsibility
        self._converged = \
            np.diff([self.log_likelihood, new_loglik])[0] < self.E_TOL
        self.log_likelihood = new_loglik

    # TODO Test
    def m_step(self):
        """M-step."""
        # Normalization constant per column
        norm_per_col = self.responsibility.sum(axis=0)
        norm_per_col = group_column_vec(norm_per_col, self.group_map)
        # Weighed square error value per column
        error_sum_per_col = \
            (self.responsibility * self.squared_errors).sum(axis=0)
        error_sum_per_col = group_column_vec(error_sum_per_col, self.group_map)

        # Variance update formulae
        # The extra parentheses are necessary to force simplification of
        # computation
        # new_variances = error_sum_per_col / norm_per_col
        # TODO I have not yet verified whether the equations for variance
        # are correct when using groups
        new_variances = \
            (error_sum_per_col + self.variance_prior_W) / \
            (norm_per_col + (self.variance_prior_nu - self._dim - 1))

        assert len(new_variances) == self.group_count, \
            "dimension error in variances"
        if np.any(np.isnan(new_variances)) or np.any(new_variances < 0) \
           or np.any(np.isinf(new_variances)):
            print("error in variance computation!")

        # Mixing coefficient update
        # Weight update formula
        # The extra parentheses are necessary to force simplification of
        # computation
        N = self.squared_errors.shape[0]
        # new_weights = norm_per_col / N
        new_weights = \
            (norm_per_col + (self.weight_prior - 1)) / \
            (N + (self.weight_prior - 1).sum())
        new_weights /= self.members_per_group
        assert len(new_weights) == self.group_count, \
            "dimension error in weights"

        if np.any(np.isnan(new_weights)) \
           or np.any(np.isinf(new_variances)):
            print("error in weight computation!")

        # Do assignment
        self.variances = new_variances
        self.weights = new_weights
