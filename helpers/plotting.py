"""Module containing methods for plotting."""

import numpy as np
import operator as op
from math import isnan
import brewer2mpl as cb
import matplotlib.pyplot as plt

# User modules
from . import constants


def plot_ensemble_percentiles(forecast_hour, percentiles,
                              element_name, data):
    """Plot the probability distribution using the specified percentiles."""
    assert len(percentiles) % 2 == 0, "number of percentiles should be even."
    # Sort percentiles in ascending order
    percentiles = sorted(percentiles)
    percentile_columns = [
        element_name + '_ENSEMBLE_PERC' + str(percentile)
        for percentile in percentiles
    ]
    other_columns = \
        ['valid_date', element_name + '_OBS', element_name + '_ENSEMBLE_MEAN']

    # Select percentile data for the specified forecast hour
    D = data[data.forecast_hour == forecast_hour]
    D = D[percentile_columns + other_columns]
    D.dropna(axis=0, inplace=True, subset=other_columns)

    # Do plotting
    fig, ax = plt.subplots(1, figsize=(20, 10))
    nr_classes = int(len(percentiles) / 2)
    cm = cb.get_map('Blues', 'Sequential', nr_classes).hex_colors
    # Coverage fields
    for i in range(nr_classes):
        print("Plotting surface between %s and %s" %
              (percentile_columns[i], percentile_columns[-i - 1]))
        ax.fill_between(
            D['valid_date'].values,
            D[percentile_columns[i]].values - 273.15,
            D[percentile_columns[-i - 1]].values - 273.15,
            color=cm[i],
            edgecolor=cm[i],
            interpolate=False,
            label="%d%% coverage" % (percentiles[-i - 1] - percentiles[i])
        )

    # Plot mean
    plt.plot(
        D['valid_date'].values, D[element_name + '_ENSEMBLE_MEAN'] - 273.15,
        color='black',
        linewidth=1,
        linestyle='--',
        alpha=0.5,
        label='Mean'
    )

    # Plot observations
    plt.plot(
        D['valid_date'].values, D[element_name + '_OBS'].values - 273.15,
        color='black',
        linewidth=0,
        marker='^',
        label='Observations'
    )

    fig.autofmt_xdate()
    # ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    # plt.legend(names)
    plt.legend(numpoints=1)
    plt.xlabel("Valid date")
    plt.ylabel("Temperature (C°)")
    plt.title(
        "Wing temperature probability forecast for +%dh" % (forecast_hour))
    plt.grid(True)
    plt.savefig("output/img/twing_percentile_%dfh.png" % forecast_hour)
    # plt.show()
    fig.clear()


def plot_model_parameters(valid_dates, model_weights, model_variances,
                          bias_intercepts,
                          forecast_hour, names):
    # Convert parameters to workable format
    model_weights = np.array(model_weights)
    model_variances = np.array(model_variances)
    bias_intercepts = np.array(bias_intercepts)
    names = np.array(names)
    # Only show a single line for EPS members.
    first_member_found = False
    valid_columns = []
    for count, name in enumerate(names):
        if 'EPS' in name and not first_member_found:
            first_member_found = True
            names[count] = names[count][:-2]  # Cut off perturbation number
        elif 'EPS' in name:
            continue
        # Mark column as valid
        valid_columns.append(count)

    fig = plt.figure(figsize=(10, 12))
    ax1 = fig.add_subplot(311)
    ax1.plot(valid_dates, model_weights[:, valid_columns])
    plt.grid(True)
    plt.xlabel("Valid date")
    plt.ylabel("Model contribution")
    plt.title("Model weights for valid dates on forecast hour %d" %
              forecast_hour)
    # plt.legend(names[valid_columns])

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(valid_dates, model_variances[:, valid_columns])
    plt.grid(True)
    plt.xlabel("Valid date")
    plt.ylabel("Model variance")
    plt.title("Model variances for valid dates on forecast hour %d" %
              forecast_hour)
    # plt.legend(names[valid_columns])

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(valid_dates, bias_intercepts[:, valid_columns])
    plt.grid(True)
    plt.xlabel("Valid date")
    plt.ylabel("Bias intercepts")
    plt.title("Model bias intercepts for valid dates on forecast hour %d" %
              forecast_hour)
    plt.legend(names[valid_columns], loc=9, bbox_to_anchor=(0.5, -0.5),
               ncol=len(valid_columns))

    fig.autofmt_xdate()
    plt.savefig("output/img/model_parameters_%dfh.png" % forecast_hour)
    # plt.show()
    fig.clear()


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
