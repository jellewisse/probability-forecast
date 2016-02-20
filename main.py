# main.py
import pyfscache
import numpy as np
import pandas as pd
import operator as op
from math import isnan
import matplotlib.pyplot as plt

# User modules
from mixture_model.gaussian_mixture import GaussianMixtureModel

from helpers.data_assimilation import load_data
from helpers import metrics

THRESHOLD = 10

# Configure cache
CACHE_DIR = '.cache'
cache = pyfscache.FSCache(CACHE_DIR)


@cache
def cached_load_data(*args):
    return load_data(*args)


def _calculate_lag(forecast_hour):
    return np.ceil(forecast_hour / 24)


def pipeline(element_id, issue, forecast_hours):

    # Ensemble definition
    model_names = ["eps", "control", "fc"]
    ens_cols = ['2T_EPS' + str(x).zfill(2) for x in range(1, 51)]
    ens_cols.append('2T_CONTROL')
    ens_cols.append('2T_FC')
    obs_col = '2T_OBS'
    model = GaussianMixtureModel(len(ens_cols))

    # Load data
    full_data = cached_load_data(
        element_id,
        issue,
        model_names
    )
    for count, forecast_hour in enumerate(forecast_hours):
        print("Processing %d / %d forecast hours." %
              (count+1, len(forecast_hours)))
        data = full_data[full_data.forecast_hour == forecast_hour]
        # TODO Write results to other dataframe than original data
        # The dates to predict for
        train_days = 40
        lag = _calculate_lag(forecast_hour)
        valid_dates = data['valid_date'].unique()
        assert len(valid_dates) == len(data), \
            "Each valid date should only have a single prediction"
        # Prediction intervals
        thresholds = np.arange(-30, 30, 1) + 273.15

        # Moving window prediction
        for index, row in data.iterrows():
            # Select data
            valid_date = row['valid_date']
            first_date = valid_date - pd.DateOffset(days=lag + train_days)
            last_date = valid_date - pd.DateOffset(days=lag)
            # TODO Selection might be expensive.
            # Alternative is to indexing first
            train_data = data[
                (data.valid_date <= last_date) & (data.valid_date > first_date)
            ]
            if len(train_data) < train_days:
                # Not enough training days
                # print("Skipping valid date ", str(row['valid_date']))
                continue
            X_train = train_data[ens_cols].as_matrix()
            y_train = train_data[obs_col].as_matrix()
            X_test = row[ens_cols].as_matrix()
            y_test = row[obs_col]

            # Train
            model.fit(X_train, y_train)

            # Predict
            model.set_member_means(X_test)

            # Verify
            full_data.loc[index, '2T_BELOW_DEGREE_PROB'] = \
                model.cdf(273.15 + THRESHOLD)
            full_data.loc[index, '2T_ENSEMBLE_MEAN'] = model.mean()
            full_data.loc[index, '2T_ENSEMBLE_PDF'] = model.pdf(y_test)
            full_data.loc[index, '2T_ENSEMBLE_CDF'] = model.cdf(y_test)
            full_data.loc[index, '2T_CRPS'] = \
                metrics.crps(thresholds, model.cdf(thresholds), y_test)

            # For determining the ensemble verification rank / calibration
            # Rank only makes sense if the ensemble weights are uniformly
            # distributed.
            obs_in_forecasts = list(model.get_member_means()) + list([y_test])
            obs_in_forecasts.sort()
            full_data.loc[index, '2T_OBS_RANK'] = \
                obs_in_forecasts.index(y_test)
            # plot_distribution(model, observation, valid_date)

    # 6. Use verify callback to call verification methods
    # do_verification(data)
    # plot_reliability_diagram(data)
    return full_data


def get_bins(nr_bins, left_lim=0, right_lim=1):
    assert left_lim < right_lim
    assert nr_bins >= 2
    bin_width = (right_lim - left_lim) / nr_bins
    half_bin = bin_width / 2
    bin_centers = np.arange(half_bin + left_lim, right_lim, bin_width)
    bin_edges = np.append(left_lim, bin_centers + half_bin)
    # bin_centers = np.arange(left_lim, right_lim + bin_width, bin_width)
    return bin_centers, bin_edges, bin_width


def _check_threshold(threshold, value):
    """Threshold is a boolean operator - value tuple."""
    return threshold[0](value, threshold[1])


def calculate_threshold_hits(data, nr_bins=11):
    bin_centers, bin_edges, _ = get_bins(nr_bins, -0.05, 1.05)
    # TODO Hard-coded single threshold
    threshold = (op.le, 273.15 + THRESHOLD)
    # Counting tables
    prob_count = np.zeros(nr_bins)
    prob_hits = np.zeros(nr_bins)
    nr_rows = 0
    # For each record
    for _, row in data.iterrows():
        observation = row['2T_OBS']
        threshold_prob = row['2T_BELOW_DEGREE_PROB']
        # Check quality
        if isnan(threshold_prob) or isnan(observation):
            continue
        # Bin probability
        threshold_prob_bin = \
            np.digitize(threshold_prob, bin_edges, right=True) - 1
        prob_count[threshold_prob_bin] += 1
        # Threshold is fulfilled.
        if threshold[0](observation, threshold[1]):
            prob_hits[threshold_prob_bin] += 1
        nr_rows += 1
    prob_hits /= nr_rows
    return bin_centers, prob_hits, prob_count


def plot_relialibilty_sharpness_diagram(bin_centers, prob_hits, prob_count):
    # Figure plotting
    _, (ax1, ax2) = plt.subplots(2, figsize=(8, 10))
    # Reliability diagram ploting
    ax1.plot(bin_centers, prob_hits, marker='o')
    ax1.legend(["Model"])
    # plot dashed diagonal
    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.set_ylabel("Observed frequency")
    ax1.set_xlim((-0.2, 1.2))
    ax1.set_ylim((0, 1))
    ax1.grid(True)
    ax1.set_title("Reliability diagram")
    # plot sharpness diagram
    mean_bin_width = (bin_centers[-1] - bin_centers[0]) / len(bin_centers)
    ax2.bar(
        bin_centers - (mean_bin_width / 2),
        prob_count,
        mean_bin_width * 0.9
    )
    ax2.set_xlim((-0.2, 1.2))
    ax2.grid(True)
    ax2.set_xlabel("Forecast probability")
    ax2.set_ylabel("Forecast frequency")
    ax2.set_title("Sharpness diagram")
    plt.show()


def plot_rank_histogram(data, rank_column, bins=51):
    """Plot Talagrand-histogram / verification-rank histogram of member
    predictions."""
    if rank_column not in data:
        raise KeyError("No rank column present in data")
    if data[rank_column].value_counts(dropna=False).loc[np.nan] == len(data):
        raise IndexError("No data present in rank column")
    data[rank_column].hist(bins=bins)


def plot_verification_rank_histogram(data, bins=None):
    if bins is None:
        ranks = data['2T_OBS_RANK'].dropna().unique()
        bins = max(ranks)
        xlims = (0, bins)
    else:
        xlims = (0, bins)
    plot_rank_histogram(data, '2T_OBS_RANK', bins)
    plt.title("Verification-Rank histogram")
    plt.xlabel("Observation rank")
    plt.xlim(xlims)
    plt.show()


def plot_PIT_histogram(data, bins=51):
    plot_rank_histogram(data, '2T_ENSEMBLE_CDF', bins)
    plt.xlabel("Forecast CDF")
    plt.xlim((0, 1))
    plt.title("PIT-Historam (CDF Histogram)")
    plt.show()


def plot_sharpness_histogram(data, bins=np.arange(0, 1, 0.1)):
    plot_rank_histogram(data, '2T_ENSEMBLE_PDF', bins)
    plt.xlabel("Forecast probability")
    plt.xlim((0, 1))
    plt.title("Sharpness-Histogram (PDF Histogram)")
    plt.show()


def plot_all_points(data):
    ens_cols = ['2T_EPS' + str(x).zfill(2) for x in range(1, 51)]
    ens_data = data[ens_cols].as_matrix()
    obs_data = np.tile(data['2T_OBS'].as_matrix(), (50, 1)).transpose()
    plt.scatter(x=obs_data.flatten(), y=ens_data.flatten())
    plt.xlabel("Observed temperature")
    plt.ylabel("Predicted temperature")
    plt.title("All data points")
    plt.grid()
    plt.show()


def plot_distribution(model, observation, forecast_date=None):

    forecast_means = np.array(model.get_member_means())
    thresholds = np.arange(
        np.floor(min(forecast_means) - 5),
        np.ceil(max(forecast_means) + 5),
        0.05
    )

    forecast_pdf = model.pdf(thresholds)
    forecast_cdf = model.cdf(thresholds)

    _, (ax1, ax2) = plt.subplots(2, sharex=True)

    # Plot model marginal pdf
    ax1.plot(thresholds, forecast_pdf, "blue", label="Model")

    # Plot observation
    ax1.plot(
        [observation, observation], [0, max(forecast_pdf)],
        "black",
        label="Observation",
        lw=3
    )

    ax1.legend(["Model", "Observation"])
    ax1.set_title("PDF for date %s" % str(forecast_date))
    ax1.set_xlim((thresholds[0], thresholds[-1]))

    # Plot cdf
    ax2.plot(thresholds, forecast_cdf, "blue")
    # Plot observation
    ax2.plot(
        [observation, observation], [0, 0.8],
        "black",
        label="Observation",
        lw=3
    )
    ax2.set_ylim((0, 1.1))
    ax2.set_title("CDF for date %s" % str(forecast_date))
    ax2.set_xlabel("Temperature")
    plt.show()


def get_hourly_values(data, column, forecast_hours):
    values = [0] * len(forecast_hours)
    for count, forecast_hour in enumerate(forecast_hours):
        hourly_data = data[data.forecast_hour == forecast_hour]
        values[count] = hourly_data[column].mean()
    return values


def do_verification(data, forecast_hour):
    hourly_data = data[data.forecast_hour == forecast_hour]
    plot_verification_rank_histogram(hourly_data)
    plot_PIT_histogram(hourly_data)
    plot_sharpness_histogram(hourly_data)
    a, b, c = calculate_threshold_hits(hourly_data)
    plot_relialibilty_sharpness_diagram(a, b, c)
    mean_crps = hourly_data['2T_CRPS'].mean()
    print("Mean CRPS: %f" % (mean_crps))
    deterministic_MAE = \
        abs(hourly_data['2T_ENSEMBLE_MEAN'] - hourly_data['2T_OBS']).mean()
    print("Ensemble mean MAE: %f" % (deterministic_MAE))


# For testing purposes
if __name__ == "__main__":
    forecast_hours = range(0, 12)
    data = pipeline("167", "0", forecast_hours)
    
