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
