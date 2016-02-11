def _heavyside(thresholds, actual):
    # TODO Optimize assuming thresholds is sorted in ascending order.
    """Returns 1 if threshold >= actual, else 0."""
    result = [1 if t >= actual else 0 for t in thresholds]
    return result


def _is_cdf_valid(case):
    """Are all probabilities in [0,1] and is the CDF non-decreasing?
    """
    precision = 10e-9
    if case[0] < 0 - precision or case[0] > 1 + precision:
        return False
    for i in range(1, len(case)):
        if case[i] > 1 + precision or case[i] < case[i-1]:
            return False
    return True


def crps(thresholds, case, actual):
    """Calculates the CRPS for a single observation and distribution."""
    crps = 0
    # Check whether the right number of prediction bins has been provided
    # and whether it encodes a cumulative distribution.
    if (len(case) == len(thresholds)) and _is_cdf_valid(case):
        obscdf = _heavyside(thresholds, actual)
        for fprob, oprob in zip(case, obscdf):
            crps += (fprob - oprob) * (fprob - oprob)
    else:
        print("Warning: bad CDF provided.")
        import pdb
        pdb.set_trace()
        crps += len(thresholds)  # treat delta at each threshold as 1
    crps /= len(thresholds)
    return crps


def mean_crps(thresholds, predictions, actuals):
    """ Calculates the Continuous Ranked Probability Score given:
            1D array of thresholds
            2D array consisting of rows of [predictions P(y <= t) for each
                threshold]
            1D array consisting of rows of observations
        For more on CRPS, see:
        http://www.eumetcal.org/resources/ukmeteocal/verification/www/english/msg/ver_prob_forec/uos3b/uos3b_ko1.htm
    """

    ncases = len(predictions)
    mean_crps = 0
    for case, actual in zip(predictions, actuals):
        mean_crps += crps(thresholds, case, actual)
    mean_crps /= float(ncases)

    return mean_crps
