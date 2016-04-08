import numpy as np
import operator as op
from math import isnan
import matplotlib.pyplot as plt

# User modules
from . import constants


def plot_model_variances(valid_dates, model_variances, forecast_hour, names):
    plt.plot(valid_dates, model_variances)
    plt.grid(True)
    plt.xlabel("Valid date")
    plt.ylabel("Model variance")
    plt.title("Model variances for valid dates on forecast hour %d" %
              forecast_hour)
    plt.legend(names)
    plt.show()


def plot_model_weights(valid_dates, model_weights, forecast_hour, names):
    plt.plot(valid_dates, model_weights)
    plt.grid(True)
    plt.xlabel("Valid date")
    plt.ylabel("Model contribution")
    plt.title("Model weights for valid dates on forecast hour %d" %
              forecast_hour)
    plt.legend(names)
    plt.show()


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
    threshold = (op.le, 273.15 + constants.THRESHOLD)
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


def get_hourly_values(data, apply_fun, forecast_hours):
    values = [0] * len(forecast_hours)
    for count, forecast_hour in enumerate(forecast_hours):
        hourly_data = data[data.forecast_hour == forecast_hour]
        fun_result = hourly_data.apply(apply_fun, axis=1)
        values[count] = (fun_result.mean(), fun_result.std())
    return np.array(values)


def plot_hourly_values(data, forecast_hours,
                       apply_fun, handle=None, color='b', spread=True):
    """
    forecast_hours should be numpy array
    """
    hourly_values = \
        get_hourly_values(data, apply_fun, forecast_hours)
    # Filter out NaN values
    good_means = ~np.isnan(hourly_values[:, 0])
    good_stds = ~np.isnan(hourly_values[:, 1])
    assert all(good_means == good_stds)
    # Do plotting
    if handle is not None:
        plt.figure(handle.number)
    plt.plot(forecast_hours[good_means], hourly_values[good_means, 0], color)
    if spread:
        plt.fill_between(
            forecast_hours[good_means],
            hourly_values[good_means, 0] - 2 * hourly_values[good_means, 1],
            hourly_values[good_means, 0] + 2 * hourly_values[good_means, 1],
            color=color, alpha=0.2
        )
    plt.xlabel("Forecast hour")
    plt.grid(True)
    if handle is None:
        plt.show()
    return hourly_values


def plot_hourly_mae(data, forecast_hours):
    f = plt.figure()
    fc_mae = plot_hourly_values(
        data, forecast_hours,
        lambda row: np.abs(row['2T_FC'] - row['2T_OBS']),
        f, 'r', spread=True)
    control_mae = plot_hourly_values(
        data, forecast_hours,
        lambda row: np.abs(row['2T_CONTROL'] - row['2T_OBS']),
        f, 'g', spread=True)
    ensemble_mae = plot_hourly_values(
        data, forecast_hours,
        lambda row: np.abs(row['2T_ENSEMBLE_MEAN'] - row['2T_OBS']),
        f, 'b', spread=True)
    plt.legend([
        "Oper:      %f" % fc_mae[:, 0].mean(),
        "Control:   %f" % control_mae[:, 0].mean(),
        "Ensemble+: %f" % ensemble_mae[:, 0].mean()
    ])
    plt.title("MAE for different models")
    plt.ylabel("Temperature MAE (K)")
    plt.show()


def plot_hourly_crps(data, forecast_hours):
    f = plt.figure()
    ensemble_crps = plot_hourly_values(
        data, forecast_hours,
        lambda row: row['2T_CRPS'],
        f, 'b', spread=True)
    plt.legend(["Ensemble+: %f" % ensemble_crps[:, 0].mean()])
    plt.title("CRPS")
    plt.ylabel("Temperature CRPS (K)")
    plt.show()