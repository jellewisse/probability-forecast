"""Wrapper module for combining data cleaning and loading."""

import pandas as pd
from math import isnan

# User modules
from helpers import data_io
from helpers.interpolation import interpolate
from helpers.constants import SCHIPHOL_STATION_LAT, SCHIPHOL_STATION_LON
# For now these are hard-coded since there only is one station


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
def load_and_interpolate_forecast(model, element_id, element_name, issue):
    """Load a given dataset of forecast data with interpolated values.

    parameters
    ----------
    model: str, name of model
    element_id: int, id of parameter under the provided model
    issue: str, string representation of the model issue hour
    """
    # Forecast columns: 1 to 4
    forecast_cols = ['value' + str(x) for x in range(1, 5)]

    # Read in forecast
    forecast_data = data_io.read_forecast_data(model, element_id, issue)

    # Check whether to do any interpolation
    empty_columns = \
        forecast_data.ix[:, forecast_cols].isnull().values.any(axis=0)
    if empty_columns.sum() == 3:
        # Just a single column provided. Don't do interpolation.
        print("Not interpolating for model '%s' and element '%s'" %
              (model, str(element_id)))
        # Select non-empty column
        non_empty_col = forecast_cols[(~empty_columns).nonzero()[0][0]]
        interpolated_forecast = forecast_data[non_empty_col]
    else:
        # Do interpolation using meta-data
        lats, lons, dists = \
            data_io.read_meta_data(model, element_id, issue)
        # TODO Schiphol station location is hard coded right now
        interpolated_forecast = interpolate(
            SCHIPHOL_STATION_LAT, SCHIPHOL_STATION_LON,
            forecast_data.ix[:, forecast_cols], lats, lons, dists
        )
    forecast_data['interpolated_forecast'] = interpolated_forecast

    # Drop grid point data.
    forecast_data.drop(forecast_cols, axis=1, inplace=True)

    # Transform EPS data from row format to column format
    forecast_data = \
        transform_forecast_group_data(forecast_data, model, element_name)
    return forecast_data


def add_observations(forecast_data, element_id):
    """Merge observations into provided forecast data."""
    # TODO Hack. Fix me.
    if element_id == 999:
        observations = data_io.read_observations(element_id)
    else:
        observations = data_io.read_knmi_observations()
    return pd.DataFrame.merge(
        forecast_data, observations,
        copy=False
    )


def load_data(element_name, issue, model_names):
    """Wrapper function for loading and combining several models."""
    # Load data for specific models
    data = pd.DataFrame()
    for model_name in model_names:
        element_id = data_io.get_element_id(element_name, model_name)
        model_data = load_and_interpolate_forecast(
            model_name, element_id, element_name, issue)
        if data.empty:
            data = model_data
        else:
            data = pd.merge(data, model_data, copy=False, how='outer')

    # Add observations to data
    obs_element_id = data_io.get_element_id(element_name, "fc")
    data = add_observations(data, obs_element_id)
    data.sort_values('valid_date', ascending=True, inplace=True)
    return data
