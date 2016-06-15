"""Module for methods that do file system IO."""
import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta

# User modules
from helpers.constants import DATA_FILE_PATH


# TODO Move this to file
def get_station_location(station_name):
    """Return the latitude and longitude of a given station."""
    if station_name == 'schiphol':
        return [52.31555938720703, 4.790283679962158]
    elif station_name == 'debilt':
        return [52.0988883972168, 5.17971658706665]


def get_element_name(element_id):
    # TODO Remove dependency on model for parameter id.
    """Hard coded mapping of element ids to meaningful names."""
    element_dict = {
        # 121: maximum 2 meter temperature of past 6 hours (K),
        # 122: minimum 2 meter temperature of past 6 hours (K),
        # 123: 10 meter wind gust of past 6 hours (m/s),
        # 144: snowfall, convective and stratiform (m),
        # 164: total cloud cover,
        # 165: 10 meter U wind component (m/s),
        # 166: 10 meter V wind component (m/s),
        # 167: 2 meter temperature (K),
        # 168: 2 meter dewpoint temperature (K),
        # 186: low cloud cover,
        # 187: medium cloud cover,
        # 188: high cloud cover,
        # 228: total precipitation (m)
        999: 'TWING'  # 999: Wing temperature (K)
    }
    return element_dict[element_id]


def get_element_id(element_name, model):
    # TODO Remove dependency on model for parameter id.
    """Hard coded mapping of meaningful names to element ids."""
    if element_name == "2T":
        if model == "fc" or model == 'control' or model == 'eps' \
           or model == "obs":
            return 167
        elif model == "ukmo":
            return 11
    elif element_name == "TWING":
        return 999


def convert_temperature_deci_degrees_to_kelvin(T):
    """Converter of decidegrees to Kelvin."""
    return (float(T) / 10.0) + 273.15


def convert_knmi_station_id_to_wmo(station_id):
    """KNMI has three-digit station numbers, the WMO at least 5 or 6."""
    return int("6" + str(station_id))


# TODO Test
def date_parser1(date, hour):
    """yyyymmdd and hh strings to UTC datetime object.

    parameters
    ----------
    date: string in yyyymmdd format.
    hour: string in hour format.
    """
    if not(isinstance(date, str) and isinstance(hour, str)):
        raise ValueError("Non-string arguments passed.")

    year, month, day = (int(date[0:4]), int(date[4:6]), int(date[6:8]))
    # Sometimes, the hour is given as 1200
    if len(str(hour)) > 2:
        hour = str(hour)[0:2]
    return datetime(year, month, day, tzinfo=timezone.utc) + \
        timedelta(hours=int(hour))


def date_parser2(date):
    """Convert a date in UTC time in a string format to datetime64 objects.

    parameters
    ----------
    date: string in yyyymmddhhmm format.
    """
    year, month, day, hour, minute = (
        int(date[0:4]), int(date[4:6]), int(date[6:8]),
        int(date[8:10]), int(date[10:]))
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def read_observations(element_id, station_name):
    """Load observation file for given element."""
    # TODO Element id depends on the model it comes from, sadly.
    # Add a mapping that also takes the model name into account.
    # Right now, we assume the observations are the ECMWF representation.

    file_path = DATA_FILE_PATH + 'data_' + station_name + '/'

    column_name = get_element_name(element_id)
    obs_data = pd.read_csv(
        file_path + 'data_obs_' + str(element_id) + '.csv',
        na_values='',
        usecols=['dtg', 'station_id', column_name],
        parse_dates={'valid_date': ['dtg']},
        date_parser=date_parser2
    )

    obs_data.rename(
        columns={
            column_name: column_name + '_OBS',
        },
        inplace=True
    )
    obs_data['station_name'] = station_name
    # Drop irrelevant columns
    obs_data.drop(
        ['station_id'],
        axis=1, inplace=True
    )
    return obs_data


# TODO Test
def read_forecast_data(model, element_id, station_name, issue, file_path=None):
    """Read in a forecast file for a specific model, element_id and issue.

    parameters
    ----------
    model: str, name of model to load. Either fc, control, eps or ukmo.
    element_id: int id of the model parameter. May depend on the model.
    """
    if file_path is None:
        file_path = DATA_FILE_PATH + 'data_' + station_name + '/'

    file_name = \
        file_path + \
        '_'.join(['data', model, str(element_id), issue]) + \
        '.csv'
    forecast_data = pd.read_csv(
        file_name,
        na_values=['99999', 'not_found'],
        dtype={
            'value1': np.float32,
            'value2': np.float32,
            'value3': np.float32,
            'value4': np.float32
        },
        parse_dates={'issue_date': ['dataDate', 'dataTime']},
        date_parser=date_parser1
    )
    forecast_data.rename(
        columns={
            'table2Version': 'ec_table_id',
            'paramId': 'element_id',
            'indicatorOfParameter': 'element_id',
            'shortName': 'element_name',
            'endStep': 'forecast_hour'
        },
        inplace=True
    )

    forecast_data['valid_date'] = forecast_data.apply(
        lambda row: row['issue_date'] + timedelta(hours=row['forecast_hour']),
        axis=1
    )

    # Add the station name to the file.
    forecast_data['station_name'] = station_name

    # Drop irrelevant columns
    forecast_data.drop(
        ['level', 'stepUnits', 'startStep'],
        axis=1, inplace=True
    )
    return forecast_data


def _extract_from_string(needle, haystack):
    """Given an indicator string, extract the value that comes after it."""
    return haystack.find(needle) + len(needle)


def read_meta_data(model, element_id, station_name, issue, file_path=None):
    """Extract lat, lon and distances for given model specification."""
    if file_path is None:
        file_path = DATA_FILE_PATH + 'data_' + station_name + '/'

    file_path += \
        '_'.join(['meta', model, str(element_id), issue]) + \
        '.tmp'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 4, "Meta-files should always have 4 lines."

    # Extract distances towards queried point.
    distance = [-1.] * 4
    latitude = [-1.] * 4
    longitude = [-1.] * 4

    for (count, line) in enumerate(lines):
        lat_index = _extract_from_string("latitude=", line)
        latitude[count] = float(line[lat_index:(lat_index + 5)])

        lon_index = _extract_from_string("longitude=", line)
        longitude[count] = float(line[lon_index:(lon_index + 5)])

        dist_index = _extract_from_string("distance=", line)
        distance[count] = float(line[dist_index:(dist_index + 5)])

    return {'latitude': latitude, 'longitude': longitude, 'distance': distance}


def write_csv(data_frame, file_path):
    """Write the provided dataframe as CSV to the path specified."""
    data_frame.dropna().to_csv(
        file_path, index=False, float_format='%f')


def write_csv_for_r_package(data_frame, file_path):
    data_frame['valid_date'] = \
        data_frame['valid_date'].apply(lambda x: x.strftime("%Y%m%d%H"))
    data_frame['issue_date'] = \
        data_frame['issue_date'].apply(lambda x: x.strftime("%Y%m%d%H"))
    data_frame.dropna().to_csv(
        file_path, index=False, float_format='%f')
