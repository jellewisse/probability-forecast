"""Test the GaussianMixtureModel module."""

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm


class TestGaussianMixtureModel:
    """Test the GaussianMixtureModel base class."""

    @classmethod
    def setup_class(self):
        """Mock a simple base class."""
        from mixture_model.gaussian_mixture import GaussianMixtureModel
        mixture_model_grouping = np.arange(0, 50)
        self.model = GaussianMixtureModel(mixture_model_grouping)
        self.model.set_member_means([0] * 50)

    def test_cdf_upper_boundary(self):
        """Test upper boundary of the cumulative density function."""
        # Cumulative mass should never exceed 1.
        assert_almost_equal(self.model.cdf(10e999), 1.0, 6)

    def test_cdf_lower_boundary(self):
        """Test lower boundary of the cumulative density function."""
        # Cumulative mass should never go below 0.
        assert_almost_equal(self.model.cdf(-10e990), 0.0, 6)


class TestGaussianEM:
    """Test suite of the BMA module for Gaussian Mixtures."""

    def test_log_normal_pdf_compare_with_scipy(self):
        """Check the derivations of the log_normal_pdf."""
        from mixture_model.gaussian_mixture import _log_normal_pdf
        # Smoke test
        random_errors = np.random.rand(1, 2)
        unit_vars = np.array([2, 3])
        attempt_random_result = _log_normal_pdf(random_errors**2, unit_vars)
        assert attempt_random_result is not None

        # Comparison with a zero-mean normal distribution.
        errors = np.arange(-1, 1, 0.01)
        goal_values = norm.logpdf(errors, loc=0, scale=1)
        attempt_values = _log_normal_pdf(errors**2, 1)
        assert_almost_equal(goal_values, attempt_values)

        # Comparison with a zero-mean wide normal distribution.
        goal_values = norm.logpdf(errors, loc=0, scale=4)
        attempt_values = _log_normal_pdf(errors**2, 4**2)
        assert_almost_equal(goal_values, attempt_values)

    def test_log_normal_pdf_is_isotropic(self):
        """Test whether the log_normal_pdf ."""
        from mixture_model.gaussian_mixture import _log_normal_pdf
        # Smoke test
        random_errors = np.random.rand(1, 2)
        unit_vars = np.array([2, 3])
        attempt_random_result = _log_normal_pdf(random_errors**2, unit_vars)
        assert attempt_random_result is not None

        # Test isotropic property
        random_goal_values = \
            norm.logpdf(random_errors, loc=[0, 0], scale=np.sqrt(unit_vars))
        assert_almost_equal(random_goal_values, attempt_random_result)

    # TODO TdR 01.06.16 : Write more tests for GaussianMixtureModel.
