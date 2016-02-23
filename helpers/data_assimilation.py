# data_assimilation.py
import pandas as pd

# User modules
from helpers import data_readers
from helpers.interpolation import (
    interpolate,
    bilinear_interpolate as intpl
)

# For now these are hard-coded since there only is one station
STATION_LAT = 52.31555938720703
STATION_LON = 4.790283679962158


# TODO Test
def _get_element_key(model_name, perturbation_id,
                     nr_perturbations, element_name):
    """"""

    # Offset by 1 - counting starts at 0.
    max_nr_digits = len(str(nr_perturbations))
    if (perturbation_id == 0 and nr_perturbations == 0) or \
       model_name == 'control':
        perturb_name = ''
    else:
        perturb_name = str(perturbation_id).zfill(max_nr_digits)

    return element_name.upper() + '_' + model_name.upper() + perturb_name


# TODO Test
def regroup_dataframe(forecast_data, model_name):
    """"""

    element_name = forecast_data.iloc[0]['element_name']

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
def transform_forecast_group_data(forecast_data, model_name):
    """"""

    element_name = forecast_data.iloc[0]['element_name']
    nr_perturbations = forecast_data.iloc[0]['numberOfForecastsInEnsemble']

    # Don't regroup if there are no perturbations
    if nr_perturbations == 0:
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
        forecast_data = regroup_dataframe(forecast_data, model_name)

    forecast_data.drop(
        ['perturbationNumber', 'numberOfForecastsInEnsemble', 'element_name',
         'element_id', 'ec_table_id', 'interpolated_forecast'],
        axis=1, inplace=True, errors='ignore'
    )
    return forecast_data


# TODO Test
def load_and_interpolate_forecast(interpolation_func,
                                  model, element_id, issue):
    """Loads a given dataset of forecast data with interpolated values."""

    # Forecast columns
    forecast_cols = ['value' + str(x) for x in range(1, 5)]

    # meta-forecast data
    lats, lons, dists = data_readers.read_meta_data(model, element_id, issue)

    # Read in forecast
    forecast_data = data_readers.read_forecast_data(model, element_id, issue)

    interpolated_forecast = interpolate(
        STATION_LAT, STATION_LON,
        forecast_data.ix[:, forecast_cols], lats, lons, dists,
        interpolation_func
    )
    forecast_data['interpolated_forecast'] = interpolated_forecast

    # Drop grid point data.
    forecast_data.drop(forecast_cols, axis=1, inplace=True)

    # Transform EPS data from row format to column format
    forecast_data = transform_forecast_group_data(forecast_data, model)

    return forecast_data


def add_observations(forecast_data):
    """"""
    return pd.DataFrame.merge(
        forecast_data, data_readers.read_observations(),
        copy=False
    )


def load_data(element_id, issue, model_names):
    """"""
    # Load data for specific models
    data = pd.DataFrame()
    for model_name in model_names:
        model_data = \
            load_and_interpolate_forecast(intpl, model_name, element_id, issue)
        if data.empty:
            data = model_data
        else:
            data = pd.merge(data, model_data, copy=False, how='outer')
    data = add_observations(data)
    data.sort_values('valid_date', ascending=True, inplace=True)
    return data
