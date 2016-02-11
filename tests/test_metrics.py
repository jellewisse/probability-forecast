import math
from numpy.testing import assert_almost_equal


class TestCrps:

    @classmethod
    def setup_class(self):
        # 5 thresholds: this is part of problem set up
        self.thresholds = [2., 4., 6., 8., 10.]
        # 4 solutions: this is submitted by problem creator
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
        assert_almost_equal(crps, 1.0/4)  # since one case is wrong

    def test_invalid_cdf(self):
        from helpers import metrics
        pred = [metrics._heavyside(self.thresholds, actual)
                for actual in self.actuals]
        pred[1] = [0, 0.5, 0.3, 0.8, 1.0]  # not a valid CDF
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("Invalid CDF for one case: answer = {0}".format(crps))
        assert_almost_equal(crps, 1.0/4)  # since one case is wrong

    def test_all_zeros(self):
        from helpers import metrics
        pred = [[0.] * len(self.thresholds) for _ in self.actuals]
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("All zero predictions: answer = {0}".format(crps))
        # all zero for actual=3 is wrong for thresholds 4,6,8,10
        expected = (4 + 5 + 2 + 4)/20.0
        assert_almost_equal(crps, expected)

    def test_all_ones(self):
        from helpers import metrics
        pred = [[1.] * len(self.thresholds) for _ in self.actuals]
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("All one predictions: answer = {0}".format(crps))
        # all one for actual=3 is wrong for threshold=2 i.e. 1 threshold
        expected = (1 + 0 + 3 + 1)/20.0
        assert_almost_equal(crps, expected)

    def sigmoid(self, center):
        length = len(self.thresholds)
        return [1 / (1 + math.exp(-(x-center))) for x in range(0, length)]

    def test_sigmoid(self):
        from helpers import metrics
        pred = [self.sigmoid(actual) for actual in self.actuals]
        print(pred)
        crps = metrics.mean_crps(self.thresholds, pred, self.actuals)
        print("Sigmoids: answer = {0}".format(crps))
        assert_almost_equal(crps, 0.36, 2)
