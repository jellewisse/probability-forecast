"""Module for methods that do file system IO."""
import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta

# User modules
from helpers.constants import FILE_PATH


def get_element_name(model, element_id):
    """Hard coded mapping of element ids to meaningful names."""
    if model == "fc" or model == 'control' or model == 'eps':
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
            999: 'twing'  # 999: Wing temperature (K)
        }

    return element_dict[element_id]


def get_element_id(element_name, model):
    """Hard coded mapping of meaningful names to element ids."""
    if element_name == "2T":
        if model == "fc" or model == 'control' or model == 'eps' \
           or model == "obs":
            return 167
        elif model == "ukmo":
            return 11


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


# TODO Test
def read_knmi_observations():
    """Wrapper around the pandas read_csv method to load KNMI observations."""
    # Data type converters
    converters = {
        'T': convert_temperature_deci_degrees_to_kelvin,
        'STN': convert_knmi_station_id_to_wmo
    }

    obs_data = pd.read_csv(
        FILE_PATH + 'obs/obs_schiphol.csv',
        na_values='',
        comment='#',
        usecols=['STN', 'YYYYMMDD', 'HH', 'T'],  # Only load these columns
        parse_dates={'valid_date': ['YYYYMMDD', 'HH']},
        date_parser=date_parser1,
        converters=converters  # Apply converters to SI units.
    )

    obs_data.rename(
        columns={
            'STN': 'station_id',
            # 'YYYYMMDD': 'valid_date',
            # 'HH': 'valid_hour',  # Warning: starts at 1 to 24.
            'T': '2T_OBS',
        },
        inplace=True
    )

    return obs_data


def read_observations(model, element_id):
    """Load observation file for given element."""
    # TODO Element id depends on the model it comes from, sadly.
    # Add a mapping that also takes the model name into account.

    column_name = get_element_name(model, element_id)
    print(column_name)
    obs_data = pd.read_csv(
        FILE_PATH + 'grib/data_obs_' + str(element_id) + '.csv',
        na_values='',
        usecols=['dtg', 'station_id', column_name],
        parse_dates={'valid_date': ['dtg']},
        date_parser=date_parser2
    )

    obs_data.rename(
        columns={
            'twing': 'Twing_obs',
        },
        inplace=True
    )
    return obs_data


# TODO Test
def read_forecast_data(model, element_id, issue, file_path=None):
    """Read in a forecast file for a specific model, element_id and issue.

    parameters
    ----------
    model: str, name of model to load. Either fc, control, eps or ukmo.
    element_id: int id of the model parameter. May depend on the model.
    """
    if file_path is None:
        file_path = FILE_PATH + 'grib/'

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

    forecast_data.drop(
        ['level', 'stepUnits', 'startStep'],
        axis=1, inplace=True
    )
    return forecast_data


def _extract_from_string(needle, haystack):
    """Given an indicator string, extract the value that comes after it."""
    return haystack.find(needle) + len(needle)


def read_meta_data(model, element_id, issue, file_path=None):
    """Extract lat, lon and distances for given model specification."""
    if file_path is None:
        file_path = FILE_PATH + 'grib/'

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

    return latitude, longitude, distance
