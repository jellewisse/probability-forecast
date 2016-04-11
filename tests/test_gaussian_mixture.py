import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import norm

class TestGaussianMixtureModel:

    @classmethod
    def setup_class(self):
        from mixture_model.gaussian_mixture import GaussianMixtureModel
        self.model = GaussianMixtureModel(50)
        self.model.set_member_means([0] * 50)

    def test_cdf_boundaries(self):
        # Cumulative mass should never exceed 1.
        assert_almost_equal(self.model.cdf(10e999), 1.0, 6)

        # Cumulative mass should never go below 0.
        assert_almost_equal(self.model.cdf(-10e990), 0.0, 6)



class TestGaussianEM:
    """Test suite of the BMA module for Gaussian Mixtures."""

    def test_log_normal_pdf(self):
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

        # Test isotropic property
        random_goal_values = \
            norm.logpdf(random_errors, loc=[0, 0], scale=np.sqrt(unit_vars))
        assert_almost_equal(random_goal_values, attempt_random_result)

        # TODO Write more tests
