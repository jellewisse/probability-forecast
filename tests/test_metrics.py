"""Test module for the metrics package."""

import math
import numpy as np
from scipy.stats import norm
from numpy.testing import assert_almost_equal


class TestCrps:
    """Test suite for the validity of the CRPS."""

    @classmethod
    def setup_class(self):
        """Mock starting variables."""
        self.thresholds = [2., 4., 6., 8., 10.]
        self.actuals = [3., 1.5, 7., 4.]

    def test_exact_answer(self):
        from helpers import metrics
        # if the user supplies the exact answers, CRPS is zero.
        pred = [metrics._heavyside(self.thresholds, actual)
                for actual in self.actuals]
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("Exact answer = {0}".format(crps))
        assert_almost_equal(crps, 0.)

    def test_invalid_length(self):
        from helpers import metrics
        pred = [metrics._heavyside(self.thresholds, actual)
                for actual in self.actuals]
        pred[1] = [0, 0.3, 0.5, 1.0]  # 4 when there are 5 thresholds
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("Invalid length for one case: answer = {0}".format(crps))
        assert_almost_equal(crps, 1.0 / 4)  # since one case is wrong

    def test_invalid_cdf(self):
        from helpers import metrics
        pred = [metrics._heavyside(self.thresholds, actual)
                for actual in self.actuals]
        pred[1] = [0, 0.5, 0.3, 0.8, 1.0]  # not a valid CDF
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("Invalid CDF for one case: answer = {0}".format(crps))
        assert_almost_equal(crps, 1.0 / 4)  # since one case is wrong

    def test_all_zeros(self):
        from helpers import metrics
        pred = [[0.] * len(self.thresholds) for _ in self.actuals]
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("All zero predictions: answer = {0}".format(crps))
        # all zero for actual=3 is wrong for thresholds 4,6,8,10
        expected = (4 + 5 + 2 + 4) / 20.0
        assert_almost_equal(crps, expected)

    def test_all_ones(self):
        from helpers import metrics
        pred = [[1.] * len(self.thresholds) for _ in self.actuals]
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("All one predictions: answer = {0}".format(crps))
        # all one for actual=3 is wrong for threshold=2 i.e. 1 threshold
        expected = (1 + 0 + 3 + 1) / 20.0
        assert_almost_equal(crps, expected)

    def sigmoid(self, center):
        """Compute the sigmoid function."""
        length = len(self.thresholds)
        return [1 / (1 + math.exp(-(x - center))) for x in range(0, length)]

    def test_sigmoid(self):
        from helpers import metrics
        pred = [self.sigmoid(actual) for actual in self.actuals]
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("Sigmoids: answer = {0}".format(crps))
        assert_almost_equal(crps, 0.36, 2)


class TestPercentiles:
    """Test suite for calculating PDF percentiles."""

    def test_normal_ppf_compare_with_scipy(self):
        """Compare the percentiles function to the scipy ppf function."""
        from helpers import metrics
        cdf_fun = norm.cdf

        # PDF parameters
        mean = 0
        standard_deviation = 1

        # Percentiles to estimate
        percentiles = np.arange(1, 99, 1) / 100

        # Percentile search parameters
        search_start_value = -50
        search_increment_value = 0.01
        search_probability_threshold = 0.001

        # Do comparison
        target_percentiles = \
            norm.ppf(percentiles, loc=mean, scale=standard_deviation)
        comparison_percentiles = \
            metrics.percentiles(
                cdf_fun, percentiles,
                search_start_value, search_increment_value,
                search_probability_threshold)
        assert_almost_equal(target_percentiles, comparison_percentiles, 2)
