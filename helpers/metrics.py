"""module to calculate the Continuous Ranked Probability Score
   @author V Lakshmanan, Climate Corporation
"""

import math
import os.path
import unittest

OUTDIR = 'unittest'
TESTNO = 0


def heavyside(thresholds, actual):
    """Returns 1 if threshold >= actual, else 0."""
    result = [1 if t >= actual else 0 for t in thresholds]
    return result


def is_cdf_valid(case):
    """Are all probabilities in [0,1] and is the CDF non-decreasing?
    """
    if case[0] < 0 or case[0] > 1:
        return False
    for i in range(1, len(case)):
        if case[i] > 1 or case[i] < case[i-1]:
            return False
    return True


def calc_crps(thresholds, predictions, actuals):
    """ Calculates the Continuous Ranked Probability Score given:
            1D array of thresholds
            2D array consisting of rows of [predictions P(y <= t) for each
                threshold]
            1D array consisting of rows of observations
        For more on CRPS, see:
        http://www.eumetcal.org/resources/ukmeteocal/verification/www/english/msg/ver_prob_forec/uos3b/uos3b_ko1.htm
    """
    nthresh = len(thresholds)  # 70 in example
    ncases = len(predictions)
    crps = 0
    for case, actual in zip(predictions, actuals):
        # Check whether the right number of prediction bins has been provided
        # and whether it encodes a cumulative distribution.
        if (len(case) == nthresh) and is_cdf_valid(case):
            obscdf = heavyside(thresholds, actual)
            for fprob, oprob in zip(case, obscdf):
                crps += (fprob - oprob) * (fprob - oprob)
        else:
            print("Warning: bad CDF provided.")
            crps += nthresh  # treat delta at each threshold as 1
    crps /= float(ncases * nthresh)

    # write submission and solution files in OUTDIR for the test ...
    if __name__ == '__main__':
        global TESTNO
        TESTNO = TESTNO + 1
        with open(OUTDIR + '/solution.csv', 'w') as soln:
            soln.write("Id,Expected\n")
            for id in range(0, len(actuals)):
                soln.write("{0},{1}\n".format(id+1, actuals[id]))
        submission = '{0}/{2}_Submission_{1}.csv'.format(OUTDIR, crps, TESTNO)
        with open(submission, 'w') as subm:
            subm.write("Id,{0}\n".format(','.join([
                'P_le{0}'.format(thresh) for thresh in thresholds
            ])))
            for id in range(0, len(actuals)):
                subm.write("{0},{1}\n".format(id+1, ','.join(map(str, predictions[id]))))

    return crps


# for testing purposes only
class TestCrps(unittest.TestCase):
    def setUp(self):
        # 5 thresholds: this is part of problem set up
        self.thresholds = [2., 4., 6., 8., 10.]
        # 4 solutions: this is submitted by problem creator
        self.actuals = [3., 1.5, 7., 4.]

    def test_exact_answer(self):
        # if the user supplies the exact answers, CRPS is zero.
        pred = [heavyside(self.thresholds, actual) for actual in self.actuals]
        print(pred)
        crps = calc_crps(self.thresholds, pred, self.actuals)
        print("Exact answer = {0}".format(crps))
        self.assertAlmostEqual(crps, 0.)

    def test_invalid_length(self):
        pred = [heavyside(self.thresholds, actual) for actual in self.actuals]
        pred[1] = [0, 0.3, 0.5, 1.0]  # 4 when there are 5 thresholds
        print(pred)
        crps = calc_crps(self.thresholds, pred, self.actuals)
        print("Invalid length for one case: answer = {0}".format(crps))
        self.assertAlmostEqual(crps, 1.0/4)  # since one case is wrong

    def test_invalid_cdf(self):
        pred = [heavyside(self.thresholds, actual) for actual in self.actuals]
        pred[1] = [0, 0.5, 0.3, 0.8, 1.0]  # not a valid CDF
        print(pred)
        crps = calc_crps(self.thresholds, pred, self.actuals)
        print("Invalid CDF for one case: answer = {0}".format(crps))
        self.assertAlmostEqual(crps, 1.0/4)  # since one case is wrong

    def test_all_zeros(self):
        pred = [[0.]*len(self.thresholds) for actual in self.actuals]
        print(pred)
        crps = calc_crps(self.thresholds, pred, self.actuals)
        print("All zero predictions: answer = {0}".format(crps))
        # all zero for actual=3 is wrong for thresholds 4,6,8,10
        expected = (4 + 5 + 2 + 4)/20.0
        self.assertAlmostEqual(crps, expected)

    def test_all_ones(self):
        pred = [[1.]*len(self.thresholds) for actual in self.actuals]
        print(pred)
        crps = calc_crps(self.thresholds, pred, self.actuals)
        print("All one predictions: answer = {0}".format(crps))
        # all one for actual=3 is wrong for threshold=2 i.e. 1 threshold
        expected = (1 + 0 + 3 + 1)/20.0
        self.assertAlmostEqual(crps, expected)

    def sigmoid(self, center):
        length = len(self.thresholds)
        return [1/(1 + math.exp(-(x-center))) for x in range(0, length)]

    def test_sigmoid(self):
        pred = [self.sigmoid(actual) for actual in self.actuals]
        print(pred)
        crps = calc_crps(self.thresholds, pred, self.actuals)
        print("Sigmoids: answer = {0}".format(crps))
        self.assertAlmostEqual(crps, 0.36, 2)

if __name__ == '__main__':
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)
    unittest.main()
