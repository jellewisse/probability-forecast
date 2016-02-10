# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# User modules
from mixture_model.gaussian_mixture import GaussianMixtureModel
from helpers.interpolation import nearest_grid_point_interpolate as intpl
from helpers.data_assimilation import (
    load_and_interpolate_forecast,
    add_observations
)


def load_curated_data(model_name, element_id, issue, forecast_hour=None):
    data = load_and_interpolate_forecast(intpl, model_name, element_id, issue)
    data = add_observations(data)
    if forecast_hour is not None:
        data = data[data.forecast_hour == forecast_hour]
    return data


def pipeline(element_id, issue, forecast_hour):
    # 1. Load data from control and EPS
    eps_data = load_curated_data("eps", element_id, issue, forecast_hour)
    ens_cols = ['2T_EPS' + str(x).zfill(2) for x in range(1, 51)]
    obs_col = '2T_OBS'
    ctrl_data = load_curated_data("control", element_id, issue, forecast_hour)

    # 2. Merge
    data = pd.DataFrame.merge(
        eps_data, ctrl_data,
        copy=False
    )
    data.sort_values('valid_date', ascending=True, inplace=True)
    del(eps_data, ctrl_data)

    # 3. Split of first X days as training, keep the rest as test
    train_days = 40
    # nr_days = len(data['valid_date'].unique())

    # The dates to predict for
    lag = np.ceil(forecast_hour / 24)
    valid_dates = data['valid_date'].unique()
    assert len(valid_dates) == len(data)

    # Initialize model
    model = GaussianMixtureModel(len(ens_cols))

    # # Moving window prediction
    for index, row in data.iterrows():
        # Select data
        valid_date = row['valid_date']
        first_date = valid_date - pd.DateOffset(days=lag + train_days)
        last_date = valid_date - pd.DateOffset(days=lag)

        # TODO Selection might be expensive. Alternative is to indexing first
        train_data = data[
            (data.valid_date <= last_date) & (data.valid_date > first_date)
        ]

        if len(train_data) < train_days:
            # Not enough training days
            print("Skipping valid date ", str(row['valid_date']))
            continue

        # 4. Use train callback to train model parameters
        X = train_data[ens_cols].as_matrix()
        y = train_data[obs_col].as_matrix()
        model.fit(X, y)

        # 5. Use predict callback to predict model
        forecasts = row[ens_cols].as_matrix()
        observation = row[obs_col]
        data.loc[index, '2T_ENSEMBLE_MEAN'] = model.mean(forecasts)
        data.loc[index, '2T_ENSEMBLE_CDF'] = model.cdf(row[obs_col], forecasts)

        obs_in_forecasts = list(forecasts) + list([observation])
        obs_in_forecasts.sort()
        obs_rank = obs_in_forecasts.index(observation)
        data.loc[index, '2T_OBS_RANK'] = obs_rank

    # 6. Use verify callback to call verification methods
    # TODO
    return data


def plot_rank_histogram(data, rank_column, bins=51):
    """Plot Talagrand-histogram / verification-rank histogram of member
    predictions."""
    if rank_column not in data:
        raise KeyError("No rank column present in data")
    if data[rank_column].value_counts(dropna=False).loc[np.nan] == len(data):
        raise IndexError("No data present in rank column")
    data[rank_column].hist(bins=bins)


def plot_verification_rank_histogram(data, bins=51):
    plot_rank_histogram(data, '2T_OBS_RANK', bins)
    plt.title("Verification-Rank histogram")
    plt.xlabel("Observation rank")
    plt.show()


def plot_calibration_rank_histogram(data, bins=10):
    plot_rank_histogram(data, '2T_ENSEMBLE_CDF', bins)
    plt.xlabel("Observation CDF rank")
    plt.xlim((0, 1))
    plt.show()

#
# def plot_reliability_diagram(data, threshold):
#     """
#     data: dataframe
#     threshold: pair of logical operator and number
#     """
#
# # Example threshold: threshold = (operator.lt, 100)
#
# result = data.apply(
#     foo
#     axis=1,  # apply function to rows.
# )
# # TODO Continue here


# def plot_ensemble_pdfs():
#     model_name = 'eps'
#     element_id = '167'
#     issue = '0'
#
#     eps_data = main(model_name, element_id, issue)
#     cols = ['2T_EPS' + str(x).zfill(2) for x in range(1, 51)]
#
#     ctrl_data = main('control', element_id, issue)
#     obs_col = '2T_OBS'
#
#     # Data for 48h forecast
#     eps48 = eps_data[eps_data.forecast_hour == 48]
#     ctrl48 = ctrl_data[ctrl_data.forecast_hour == 48]
#
#     # Ensemble parameter construction
#     ctrl_std = maximum_likelihood_std(ctrl48, '2T_CONTROL', obs_col)
#     ensemble_stds = np.repeat(ctrl_std, len(cols))
#
#     # Loop over multiple forecast days
#     for row_nr in range(-5, 0):
#         forecast_id = eps48.iloc[[row_nr], ].index[0]
#         forecasts = eps48.loc[forecast_id, cols].as_matrix()
#         # forecasts = np.random.normal(280, 2, 50)
#         fcst_range = np.arange(
#             np.floor(min(forecasts) - 2 * ctrl_std),
#             np.ceil(max(forecasts) + 2 * ctrl_std),
#             0.05
#         )
#         pdf_vals = list(map(
#             lambda x: ensemble_pdf(x, norm.pdf, zip(forecasts, ensemble_stds)),
#             fcst_range
#         ))
#         # Do plotting
#         plt.plot(fcst_range, pdf_vals)
#         weights = np.ones_like(forecasts) / len(forecasts)
#         plt.hist(forecasts, bins=10, weights=weights)
#         plt.title(str(eps48.loc[forecast_id, 'valid_date']))
#         plt.xlabel('Temperature')
#         plt.ylabel('Probability')
#         plt.show()


# def calculate_crps(data):
#     obs_col = '2T_OBS'
#     observations = data[obs_col].as_matrix()
#     member_cols = cols = ['2T_EPS' + str(x).zfill(2) for x in range(1, 51)]
#     ctrl_std = maximum_likelihood_std(data, '2T_CONTROL', obs_col)
#     ensemble_stds = np.repeat(ctrl_std, len(cols))
#     thresholds = np.arange(-30, 50, 1) + 273.15
#     forecasts = []
#     i = 1
#     for index, row in data.iterrows():
#         print("Row %d / %d" % (i, len(data)))
#         members = row[member_cols]
#         forecast = list(map(
#             lambda x: ensemble_cdf(x, norm.cdf, zip(members, ensemble_stds)),
#             thresholds
#         ))
#         forecasts.append(forecast)
#         import pdb
#         pdb.set_trace()
#         i += 1
#     from helpers import metrics
#     crps_val = metrics.crps(thresholds, forecasts, observations)
#     return crps_val

# For testing purposes
if __name__ == "__main__":
    # plot_ensemble_pdfs()
    data = pipeline("167", "0", 48)
    # pass
