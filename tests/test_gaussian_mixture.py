import numpy as np
from numpy.testing import assert_almost_equal


class TestGaussianMixtureModel:

    @classmethod
    def setup_class(self):
        from mixture_model.gaussian_mixture import GaussianMixtureModel
        self.model = GaussianMixtureModel(50)
        self.model.set_member_means([0]*50)

    def test_cdf_boundaries(self):
        # Cumulative mass should never exceed 1.
        assert_almost_equal(self.model.cdf(10e999), 1.0, 6)

        # Cumulative mass should never go below 0.
        assert_almost_equal(self.model.cdf(-10e990), 0.0, 6)


class TestGaussianEM:

    def test_log_normal_pdf():
        from mixture_model.gaussian_mixture import _log_normal_pdf
        # Smoke test
        random_errors = np.random.rand(5, 2)
        unit_vars = np.ones((1, 2))
        unit_vars[:] = [3, 4]
        example_result = _log_normal_pdf(random_errors, unit_vars)
        assert example_result is not None
        # TODO Write more tests
