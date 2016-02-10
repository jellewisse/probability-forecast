"""module to calculate the Continuous Ranked Probability Score"""


def _heavyside(thresholds, actual):
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


def crps(thresholds, predictions, actuals):
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
    mean_crps = 0
    for case, actual in zip(predictions, actuals):
        # Check whether the right number of prediction bins has been provided
        # and whether it encodes a cumulative distribution.
        if (len(case) == nthresh) and is_cdf_valid(case):
            obscdf = _heavyside(thresholds, actual)
            for fprob, oprob in zip(case, obscdf):
                mean_crps += (fprob - oprob) * (fprob - oprob)
        else:
            print("Warning: bad CDF provided.")
            mean_crps += nthresh  # treat delta at each threshold as 1
    mean_crps /= float(ncases * nthresh)

    return mean_crps
