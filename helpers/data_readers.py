# data_readers.py
import numpy as np
import pandas as pd
from datetime import timezone, datetime, timedelta

# Global variable
FILE_PATH = 'schiphol/data/'


def get_element_name(element_id):
    """Hard coded list of element meanings"""
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
    }
    return element_dict[element_id]


# TODO Test
def convert_temperature_deci_degrees_to_kelvin(T):
    """Converts temperature given in tenths of degrees Celcius to Kelvin"""
    return (float(T) / 10.0) + 273.15


def convert_knmi_station_id_to_wmo(station_id):
    """KNMI has three-digit station numbers, the WMO at least 5 or 6."""
    return int("6" + str(station_id))


# TODO Test
def date_parser(date, hour):
    """Converts date and hour in UTC time in string format to datetime64
    objects.

    date: string in yyyymmdd format.
    hour: string in hour format.
    """

    year, month, day = (int(date[0:4]), int(date[4:6]), int(date[6:8]))
    # Sometimes, the hour is given as 1200
    if len(str(hour)) > 2:
        hour = str(hour)[0:2]
    return datetime(year, month, day, tzinfo=timezone.utc) + \
        timedelta(hours=int(hour))


# TODO Test
def read_observations():
    """Wrapper around the pandas read_csv method to load KNMI observations."""

    # Data type converters
    converters = {
        'T': convert_temperature_deci_degrees_to_kelvin,
        'STN': convert_knmi_station_id_to_wmo
    }

    obs_data = pd.read_csv(
        FILE_PATH + 'obs.csv',
        na_values='',
        comment='#',
        usecols=['STN', 'YYYYMMDD', 'HH', 'T'],  # Only load these columns
        parse_dates={'valid_date': ['YYYYMMDD', 'HH']},
        date_parser=date_parser,
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


# TODO Test
def read_forecast_data(model, element_id, issue, file_path=None):
    """Read in a forecast file for a specific model, element_id and issue."""

    if file_path is None:
        file_path = FILE_PATH + 'grib/'

    file_name = \
        file_path + \
        '_'.join(['data', model, element_id, issue]) + \
        '.csv'
    forecast_data = pd.read_csv(
        file_name,
        na_values='99999',
        dtype={
            'value1': np.float32,
            'value2': np.float32,
            'value3': np.float32,
            'value4': np.float32
        },
        parse_dates={'issue_date': ['dataDate', 'dataTime']},
        date_parser=date_parser
    )
    forecast_data.rename(
        columns={
            'table2Version': 'ec_table_id',
            'paramId': 'element_id',
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


def read_meta_data(model, element_id, issue, file_path=None):
    """Read a meta-data file and return the grid point latitudes, longitudes
    and distance to the query point."""

    if file_path is None:
        file_path = FILE_PATH + 'grib/'

    file_name = \
        file_path + \
        '_'.join(['meta', model, element_id, issue]) + \
        '.tmp'
    with open(file_name, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 4, "Meta-files should always have 4 lines."

    # Extract distances towards queried point.
    dist_str = "distance="
    lat_str = "latitude="
    lon_str = "longitude="
    distance = [-1.] * 4
    latitude = [-1.] * 4
    longitude = [-1.] * 4

    for (count, line) in enumerate(lines):
        lat_index = line.find(lat_str) + len(lat_str)
        latitude[count] = float(line[lat_index:(lat_index + 5)])

        lon_index = line.find(lon_str) + len(lon_str)
        longitude[count] = float(line[lon_index:(lon_index + 5)])

        dist_index = line.find(dist_str) + len(dist_str)
        distance[count] = float(line[dist_index:(dist_index + 5)])

    return latitude, longitude, distance
