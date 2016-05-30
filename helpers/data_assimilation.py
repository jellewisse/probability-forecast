"""Wrapper module for combining data cleaning and loading."""

import logging
import numpy as np
import pandas as pd
from math import isnan

# User modules
from helpers import data_io
from helpers.interpolation import interpolate
# For now these are hard-coded since there only is one station


def calculate_training_lag(forecast_hour):
    """Calculate delay in days to exclude from a dataset.

    Example: when verifying a +48h forecast the earliest date we can verify
    is the forecast issued two days ago, meaning we should exclude the latest
    two days when training such a forecast.
    """
    return np.ceil(forecast_hour / 24)


def filter_unused_forecast_hours(dataframe, forecast_hours):
    """Drop unused rows inplace."""
    hour_out_of_bounds = ~dataframe.forecast_hour.isin(forecast_hours)
    out_of_bounds_index = hour_out_of_bounds[hour_out_of_bounds].index
    dataframe.drop(out_of_bounds_index, inplace=True)


def yield_training_iterator(data_frame, ):
    """To define."""
    pass


def _get_element_key(model_name, perturbation_id,
                     nr_perturbations, element_name):
    """Translate the given properties into a canonical column name."""
    # Offset by 1 - counting starts at 0.
    max_nr_digits = len(str(nr_perturbations))
    if (perturbation_id == 0 and nr_perturbations == 0) or \
       (nr_perturbations == 1) or \
       (model_name == 'control'):
        perturb_name = ''
    else:
        perturb_name = str(perturbation_id).zfill(max_nr_digits)

    return element_name.upper() + '_' + model_name.upper() + perturb_name


def regroup_dataframe(forecast_data, model_name, element_name):
    """Reorder multi-row forecast into a single row."""
    # Define columns necessary for finding groups
    grouping = forecast_data.groupby([
        'ec_table_id', 'element_id', 'element_name',
        'issue_date', 'forecast_hour',
        'numberOfForecastsInEnsemble'
    ])

    # List the perturbation identifiers
    perturbations = forecast_data['perturbationNumber'].unique()
    nr_perturbations = forecast_data.iloc[0]['numberOfForecastsInEnsemble']

    # Construct dictionary to be the new dataframe
    names = ['index']
    names += [_get_element_key(model_name, x, nr_perturbations, element_name)
              for x in perturbations]
    data_dict = {name: [] for name in names}

    # Loop over groups to extract relevant columns
    for _, group in grouping:
        member_ids = group['perturbationNumber'].values
        col_values = group['interpolated_forecast'].values
        for (member_id, interpolated_value) in zip(member_ids, col_values):
            data_dict[
                _get_element_key(model_name, member_id,
                                 nr_perturbations, element_name)
            ].append(round(interpolated_value, 3))

        # Add index of first row in group for later concatenation.
        index = group.iloc[[0]].index[0]
        data_dict['index'].append(index)

    # Convert dictionary to dataframe
    reformatted_data = pd.DataFrame(data_dict)
    reformatted_data.set_index('index', drop=True, inplace=True)

    # Apply a right concat of the full data with the group datafrkame, keeping
    # only the row indexes that are present in both dataframes (join='inner').
    # Warning: this operation copies the data.
    forecast_data = pd.concat(
        [forecast_data, reformatted_data],
        axis=1,
        join='inner',
        copy=False
    )
    # Verify that the number of expected rows is present in the join.
    assert len(forecast_data) == len(grouping), \
        "Something went wrong in reformatting the data"
    return forecast_data


# TODO Test
def transform_forecast_group_data(forecast_data, model_name, element_name):
    """Wrapper method for renaming forecast data columns."""
    nr_perturbations = forecast_data.iloc[0]['numberOfForecastsInEnsemble']

    # Don't regroup if there are no perturbations
    if nr_perturbations == 0 or isnan(nr_perturbations):
        # Rename and drop the right columns.
        forecast_data.rename(
            columns={
                'interpolated_forecast':
                    _get_element_key(model_name, 0, 0, element_name)
            },
            inplace=True
        )
    else:
        # Regroup data
        forecast_data = \
            regroup_dataframe(forecast_data, model_name, element_name)

    forecast_data.drop(
        ['perturbationNumber', 'numberOfForecastsInEnsemble', 'element_name',
         'element_id', 'ec_table_id', 'interpolated_forecast'],
        axis=1, inplace=True, errors='ignore'
    )
    return forecast_data


# TODO Test
def load_and_interpolate_forecast(model, element_name,
                                  station_name, issue):
    """Load a given dataset of forecast data with interpolated values.

    parameters
    ----------
    model: str, name of model
    element_name: str, string representation of the provided parameter
    station_name: str, canonical name of the station
    issue: str, string representation of the model issue hour
    """
    # Forecast columns: 1 to 4
    forecast_cols = ['value' + str(x) for x in range(1, 5)]

    # Station location
    station_lat, station_lon = data_io.get_station_location(station_name)

    # Read in forecast
    element_id = data_io.get_element_id(element_name, model)
    forecast_data = data_io.read_forecast_data(
        model, element_id, station_name, issue)

    # Check whether to do any interpolation
    empty_columns = \
        forecast_data.ix[:, forecast_cols].isnull().values.any(axis=0)
    if empty_columns.sum() == 3:
        # Just a single column provided. Don't do interpolation.
        logging.debug(
              "Not interpolating for model '%s', element '%s', station '%s'" %
              (model, str(element_id), station_name))
        # Select non-empty column
        non_empty_col = forecast_cols[(~empty_columns).nonzero()[0][0]]
        interpolated_forecast = forecast_data[non_empty_col]
    else:
        # Do interpolation using meta-data
        meta_data = \
            data_io.read_meta_data(model, element_id, station_name, issue)
        interpolated_forecast = interpolate(
            station_lat, station_lon,
            forecast_data.ix[:, forecast_cols],
            meta_data['latitude'], meta_data['longitude'],
            meta_data['distance']
        )
    forecast_data['interpolated_forecast'] = interpolated_forecast

    # Drop grid point data.
    forecast_data.drop(forecast_cols, axis=1, inplace=True)

    # Transform EPS data from row format to column format
    forecast_data = \
        transform_forecast_group_data(forecast_data, model, element_name)
    return forecast_data


def add_observations(forecast_data, element_id, station_names):
    """Merge observations into provided forecast data."""
    # TODO Right now, TWING observations are stored separately.
    # Have the same treatment regardless of element.
    merged_data = forecast_data
    if element_id == 999:
        for station_name in station_names:
            logging.debug(
                  "Adding observations for element %d, station %s" %
                  (element_id, station_name))
            observations = data_io.read_observations(element_id, station_name)
            merged_data = pd.DataFrame.merge(
                merged_data, observations,
                how='outer', copy=False
            )
    else:
        observations = data_io.read_knmi_observations(station_names)
        merged_data = pd.DataFrame.merge(
            merged_data, observations,
            on=['valid_date', 'station_name'],
            how='outer', copy=False
        )
    return merged_data


def load_observations_for_station(station_name, element_name):
    """Load observations for a specific stations."""
    element_id = data_io.get_element_id(element_name, "fc")

    # TODO Add support
    if element_id != 999:
        raise NotImplementedError()

    observations = data_io.read_observations(element_id, station_name)
    return observations


def load_models_for_station(model_names, station_name, element_name, issue):
    """Load model forecasts for a specific station."""
    data = pd.DataFrame()
    for model_name in model_names:
        model_data = load_and_interpolate_forecast(
            model_name, element_name, station_name, issue)
        if data.empty:
            data = model_data
        else:
            data = pd.merge(data, model_data, copy=False, how='outer')
    return data


def load_data(element_name, station_names, issue, model_names):
    """Wrapper function for loading and combining several models."""
    full_data = pd.DataFrame()
    for station_name in station_names:
        # Load forecast data for the station
        station_data = load_models_for_station(
            model_names, station_name, element_name, issue)

        # Load observation data for the station
        observation_data = load_observations_for_station(
            station_name, element_name)
        # Merge forecasts and observations for the station
        station_data = pd.merge(
            station_data, observation_data, copy=False, how='outer')

        # Append data for the new station to other station data
        if full_data.empty:
            full_data = station_data
        else:
            full_data = pd.concat([full_data, station_data], ignore_index=True)

    # Post-processing and cleanup
    full_data.sort_values('valid_date', ascending=True, inplace=True)
    return full_data
