# gaussian_mixture.py
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
    """
    parameters:
    -----------
    """

    P = squared_errors * (-1.0 / (2.0 * variances))
    P += (-1.0 / 2) * np.log(variances)
    P += NORM_CONSTANT
    if np.any(np.isnan(P)):
        import pdb
        pdb.set_trace()
        raise ZeroDivisionError("Singularity in log normal pdfs.")
    return P


class GaussianMixtureModel(MixtureModel):
    """"""

    def __init__(self, member_count):
        """Initialize a univariate normal Gaussian Mixture Model

        Parameters
        ----------
        member_count : integer
            Number of members to initialize the mixture with
        """
        super().__init__(member_count, norm)
        self._forecast_prepared = False
        self._optimizer = GaussianEM(member_count)

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
        # Mean bias correction on first member
        bias_per_model = _maximum_likelihood_bias(X, y)
        self._optimizer.fit(X - bias_per_model, y)

        # TODO Check for singularities

        # Update variances and bias
        for (member, model_std, model_bias) in \
            zip(self._members,
                self._optimizer.variances,
                bias_per_model):
            member.parameters['scale'] = model_std
            member.bias = model_bias

        # Update weights
        self.weights = self._optimizer.weights.squeeze()

    def _check_member_means(self):
        if not self._forecast_prepared:
            raise AttributeError("No ensemble values available for dressing.")

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

    def get_member_variances(self):
        return [
            member.parameters['scale']
            for member in self._members
        ]

    def mean(self):
        self._check_member_means()
        return sum([
            (member.parameters['loc'] - member.bias) * weight
            for (member, weight)
            in zip(self._members, self.weights)
        ])

    def pdf(self, x):
        self._check_member_means()
        return super().pdf(x)

    def cdf(self, x):
        self._check_member_means()
        return super().cdf(x)


class GaussianEM(object):
    # Constants
    N_ITER = 2000
    E_TOL = 1e-5

    def __init__(self, member_count):
        # General attributes
        self._member_count = member_count
        self._dim = 1  # Dimensionality of output
        # Model parameters
        self.variance_prior_W = 1  # Matrix of dim x dim, 0 means no prior.
        self.variance_prior_nu = 2  # Scalar value, 2 means no prior.
        self.variances = np.ones(self._member_count)
        # Uniform prior on weights
        self.weight_prior = \
            np.ones(self._member_count) * 1.5  # All ones means no prior
        self.weights = np.ones(self._member_count) / self._member_count

    def fit(self, X, y):
        """Run the EM algorithm with the last solution as prior.
        If there is no last solution a uniform prior is used."""
        assert self._member_count == X.shape[1], \
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
            # print(iter_count)
            # Algorithm tests
            try:
                assert_almost_equal(self.weights.sum(), 1.0, 6)
            except AssertionError:
                print("Weights don't sum to 1!")
                import pdb
                pdb.set_trace()
            if np.any(self.weights < 0) or np.any(self.weights > 1):
                print("error in weights!")
                import pdb
                pdb.set_trace()
            if np.any(np.isnan(self.variances)) or np.any(self.variances < 0):
                print("error in variances!")
                import pdb
                pdb.set_trace()
            try:
                self.e_step()
                self.m_step()
                self.log_liks.append(self.log_likelihood)
            except ZeroDivisionError:
                # Singularity detected.
                # Perturb parameters with random noise and restart fit.
                # print("Singularity detected - perturbing")
                print("Perturbing!")
                self.perturb_singularities()
            iter_count += 1

        # Cleanup
        # if iter_count < self.N_ITER:
        #     print("Log: BMA converged in %d iterations." % iter_count)
        # else:
        #     # No convergence - reuse parameters from previous iteration.
        #     print("Log: BMA did not converge within %d iterations." %
        #           self.N_ITER)
        # print("Log likelihood: %f" % self.log_likelihood)
        # print("Member weights: ", self.weights)
        # print("Member variances: ", self.variances)
        self.responsibility = None
        self.squared_errors = None

    def perturb_singularities(self):
        import pdb
        pdb.set_trace()
        singularity_index = np.logical_or(np.logical_or(
            self.variances <= 1e-30,
            np.isinf(self.variances)),
            np.isnan(self.variances))
        self.variances[singularity_index] = \
            np.random.rand(singularity_index.sum()) * 2
        self.weights[singularity_index] = 1e-30

        # Renormalize
        self.weights /= self.weights.sum()

    # TODO Test
    def e_step(self):
        """
        E-step
        Underflow errors are accounted for with the log-sum-exp trick.
        See for example
        https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
        """
        # Calculate proportional responsibility
        new_responsibility = \
            _log_normal_pdf(self.squared_errors, self.variances)
        # TODO Weights can go to 0. Taking logarithm will give singularities.
        new_responsibility += np.log(self.weights)
        new_loglik = 0
        # Update log-likelihood and normalize responsibility
        # Use log-sum-exp trick.
        # Find maximum value per row
        max_per_row = new_responsibility.max(axis=1)
        # Normalization constant for each row
        norm_per_row = np.exp(
            new_responsibility.transpose() - max_per_row
        ).transpose().sum(axis=1)
        # Normalize each row
        loglik_per_row = max_per_row + np.log(norm_per_row)
        new_responsibility2 = np.exp(
            new_responsibility.transpose() - loglik_per_row
        ).transpose()
        if (new_responsibility2 == 0).all(axis=0).any():
            print("Singularity column in responsibility matrix")
            import pdb
            pdb.set_trace()
        new_responsibility = new_responsibility2
        new_loglik = loglik_per_row.sum()
        self.responsibility = new_responsibility
        self._converged = \
            np.diff([self.log_likelihood, new_loglik])[0] < self.E_TOL
        self.log_likelihood = new_loglik

    # TODO Test
    def m_step(self):
        """"""
        # TODO Adding prior information leads to singularities. Figure out why.
        # Normalization constant per column
        norm_per_col = self.responsibility.sum(axis=0)
        error_sum_per_col = \
            (self.responsibility * self.squared_errors).sum(axis=0)
        # new_variances_no_prior = error_sum_per_col / norm_per_col
        # Variance update formula
        # The extra parentheses are necessary to force simplification of
        # computation
        new_variances = \
            (error_sum_per_col + self.variance_prior_W) / \
            (norm_per_col + (self.variance_prior_nu - self._dim - 1))

        if np.any(np.isnan(new_variances)) or np.any(new_variances < 0) \
           or np.any(np.isinf(new_variances)):
            print("error in variance computation!")
            import pdb
            pdb.set_trace()

        # Mixing coefficient update
        N = self.squared_errors.shape[0]
        # new_weights_no_prior = norm_per_col / N
        # Weight update formula
        # The extra parentheses are necessary to force simplification of
        # computation
        new_weights = \
            (norm_per_col + (self.weight_prior - 1)) / \
            (N + (self.weight_prior - 1).sum())

        # Do assignment
        self.variances = new_variances
        self.weights = new_weights
