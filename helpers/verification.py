"""Module for running verification metrics on resulting data."""
import pandas as pd
import numpy as np
from helpers import metrics

"""
Tasks to do:

PIT histogram for all hours
ROC curve for all hours
"""


def run_verification_per_hour(prediction_data, run_configuration):
    element = run_configuration['element_name']
    forecast_hours = prediction_data.forecast_hour.unique()
    verification_data = \
        initialize_verification_data_store(forecast_hours, element)
    hour_grouping = prediction_data.groupby('forecast_hour')
    for forecast_hour, group_data in hour_grouping:
        verification_data.loc[forecast_hour] = \
            run_verification_for_all_rows(
                forecast_hour, group_data, element)
    return verification_data


def run_verification_for_all_rows(forecast_hour, prediction_data, element):

    prediction_data = prediction_data.dropna()
    data_row = get_empty_row(element)

    column_name = element + '_CRPS'
    data_row[column_name] = \
        calculate_column_mean(prediction_data, column_name)

    column_name = element + '_MEAN_MAE'
    data_row[column_name] = calculate_mean_mae(prediction_data, element)

    column_name = element + '_MEDIAN_MAE'
    data_row[column_name] = calculate_median_mae(prediction_data, element)

    column_name = element + '_AUC_FREEZE'
    data_row[column_name] = calculate_freeze_auc(prediction_data, element)

    column_name = 'FRACTION_BELOW_0'
    data_row[column_name] = \
        calculate_fraction_below_zero(prediction_data, element)
    return data_row


def calculate_mean_rmse(prediction_data, element):
    prediction_column = element + '_ENSEMBLE_MEAN'
    observation_column = element + '_OBS'
    return calculate_mean_absolute_difference(
        prediction_data, prediction_column, observation_column)


def calculate_mean_mae(prediction_data, element):
    prediction_column = element + '_ENSEMBLE_MEAN'
    observation_column = element + '_OBS'
    return calculate_mean_absolute_difference(
        prediction_data, prediction_column, observation_column)


def calculate_median_mae(prediction_data, element):
    prediction_column = element + '_ENSEMBLE_PERC50'
    observation_column = element + '_OBS'
    return calculate_mean_absolute_difference(
        prediction_data, prediction_column, observation_column)


def calculate_freeze_auc(prediction_data, element):
    prediction_column = element + '_ENSEMBLE_CDF_FREEZE'
    observation_column = element + '_OBS'
    return metrics.auc(
        prediction_data[observation_column] <= 273.15,
        prediction_data[prediction_column])


def calculate_fraction_below_zero(prediction_data, element):
    observation_column = element + '_OBS'
    total_points = len(prediction_data)
    below_zero = (prediction_data[observation_column] <= 273.15).sum()
    return below_zero / total_points


def get_verification_columns(element):
    return [
        element + '_CRPS',
        element + '_MEAN_MAE',
        element + '_MEDIAN_MAE',
        element + '_AUC_FREEZE',
        'FRACTION_BELOW_0'
    ]


def initialize_verification_data_store(forecast_hours, element):
    df = pd.DataFrame(
        columns=get_verification_columns(element), index=forecast_hours)
    df.index.name = 'forecast_hour'
    return df


def get_empty_row(element):
    column_names = get_verification_columns(element)
    return pd.Series(
        dict(
            zip(
                column_names, [np.NaN] * len(column_names)
            )
        )
    )


def calculate_column_mean(dataframe, column_name):
    return dataframe[column_name].mean()


def calculate_mean_absolute_difference(dataframe, column_1, column_2):
    return (dataframe[column_1] - dataframe[column_2]).abs().mean()


def calculate_root_mean_squared_difference(dataframe, column_1, column_2):
    pass
