# main.py
import numpy as np
import pandas as pd
from scipy.stats import norm

# User modules
from helpers.interpolation import nearest_grid_point_interpolate as intpl
from helpers.data_assimilation import (
    load_and_interpolate_forecast,
    add_observations
)


def error_calculation(df, data_cols, obs_col):
    X = df[data_cols].as_matrix()
    y = df[obs_col].as_matrix()
    # Numpy column-wise subtraction is expressed as row-wise subtraction.
    E = (X.transpose() - y).transpose()
    return E


def maximum_likelihood_mean(df, data_cols, obs_col):
    # Calculate errors
    errors = error_calculation(df, data_cols, obs_col)
    # Mean per column
    return errors.mean(axis=0)


def maximum_likelihood_std(df, data_cols, obs_col):
    # Calculate errors
    errors = error_calculation(df, data_cols, obs_col)
    # Calculate maximum likelihood means per column
    return errors.std(axis=0)


def ensemble_pdf(x, pdf, member_params):
    prob = 0.0
    for (count, pdf_args) in enumerate(member_params):
        prob += pdf(x, *pdf_args)
    prob /= count
    return prob


def ensemble_cdf(x, cdf, member_params):
    prob = 0.0
    for (count, cdf_args) in enumerate(member_params):
        prob += cdf(x, *cdf_args)
    prob /= count
    return prob


def main(model_name, element_id, issue, forecast_hour=None):
    data = load_and_interpolate_forecast(intpl, model_name, element_id, issue)
    data = add_observations(data)
    if forecast_hour is not None:
        data = data[data.forecast_hour == forecast_hour]
    return data


def pipeline(element_id, issue, forecast_hour):
    # 1. Load data from control and EPS
    eps_data = main("eps", element_id, issue, forecast_hour)
    ctrl_data = main("control", element_id, issue, forecast_hour)

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

    # # Moving window prediction
    for index, row in data.iterrows():
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
        ens_cols = ['2T_EPS' + str(x).zfill(2) for x in range(1, 51)]
        ctrl_std = maximum_likelihood_std(train_data, '2T_CONTROL', '2T_OBS')
        ensemble_stds = np.repeat(ctrl_std, len(ens_cols))

        # 5. Use predict callback to predict model
        forecasts = row[ens_cols].as_matrix()
        observation = row['2T_OBS']
        data.loc[index, '2T_ENSEMBLE_MEAN'] = np.mean(forecasts)
        data.loc[index, '2T_ENSEMBLE_PDF'] = \
            ensemble_pdf(
                row['2T_OBS'], norm.pdf, zip(forecasts, ensemble_stds)
            )
        data.loc[index, '2T_ENSEMBLE_CDF'] = \
            ensemble_cdf(
                row['2T_OBS'], norm.cdf, zip(forecasts, ensemble_stds)
            )

        obs_in_forecasts = list(forecasts) + list([observation])
        obs_in_forecasts.sort()
        obs_rank = obs_in_forecasts.index(observation)
        data.loc[index, '2T_OBS_RANK'] = obs_rank

    # 6. Use verify callback to call verification methods
    # TODO
    return data


def plot_ensemble_pdfs():
    model_name = 'eps'
    element_id = '167'
    issue = '0'

    eps_data = main(model_name, element_id, issue)
    cols = ['2T_EPS' + str(x).zfill(2) for x in range(1, 51)]

    ctrl_data = main('control', element_id, issue)
    obs_col = '2T_OBS'

    # Data for 48h forecast
    eps48 = eps_data[eps_data.forecast_hour == 48]
    ctrl48 = ctrl_data[ctrl_data.forecast_hour == 48]

    # Ensemble parameter construction
    ctrl_std = maximum_likelihood_std(ctrl48, '2T_CONTROL', obs_col)
    ensemble_stds = np.repeat(ctrl_std, len(cols))

    # Loop over multiple forecast days
    for row_nr in range(-5, 0):
        forecast_id = eps48.iloc[[row_nr], ].index[0]
        forecasts = eps48.loc[forecast_id, cols].as_matrix()
        # forecasts = np.random.normal(280, 2, 50)
        fcst_range = np.arange(
            np.floor(min(forecasts) - 2 * ctrl_std),
            np.ceil(max(forecasts) + 2 * ctrl_std),
            0.05
        )
        pdf_vals = list(map(
            lambda x: ensemble_pdf(x, norm.pdf, zip(forecasts, ensemble_stds)),
            fcst_range
        ))
        # Do plotting
        import matplotlib.pyplot as plt
        plt.plot(fcst_range, pdf_vals)
        weights = np.ones_like(forecasts) / len(forecasts)
        plt.hist(forecasts, bins=10, weights=weights)
        plt.title(str(eps48.loc[forecast_id, 'valid_date']))
        plt.xlabel('Temperature')
        plt.ylabel('Probability')
        plt.show()

# For testing purposes
if __name__ == "__main__":
    # plot_ensemble_pdfs()
    # data = main("control", "167", "0")
    pass
