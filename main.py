# main.py
import numpy as np
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


def main(model_name, element_id, issue):
    data = load_and_interpolate_forecast(intpl, model_name, element_id, issue)
    data = add_observations(data)
    return data


def ensemble_pdf(x, pdf, member_params):
    prob = 0.0
    for (count, pdf_args) in enumerate(member_params):
        prob += pdf(x, *pdf_args)
    return prob


# For testing purposes
if __name__ == "__main__":
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
    last_forecast_id = eps48.iloc[[-1], ].index[0]
    # forecasts = eps48.loc[last_forecast_id, cols].as_matrix()
    forecasts = np.random.normal(280, 2, 50)

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
    plt.xlabel('Temperature')
    plt.ylabel('Probability')
    plt.show()
