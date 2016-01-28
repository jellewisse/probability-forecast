# data_readers.py
import numpy as np
import pandas as pd


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


def convert_temperature_deci_degrees_to_kelvin(T):
    """Converts temperature given in tenths of degrees Celcius to Kelvin"""
    return (float(T) / 10.0) + 273.15


def read_observations():
    """Wrapper around the pandas read_csv method to load KNMI observations."""

    # Data type converters
    converters = {
        'T': convert_temperature_deci_degrees_to_kelvin
    }

    obs_data = pd.read_csv(
        'schiphol/data/obs.csv',
        na_values='',
        comment='#',
        usecols=['STN', 'YYYYMMDD', 'HH', 'T'],  # Only load these columns
        parse_dates={'valid_date': ['YYYYMMDD']},  #, 'HH']},
        converters=converters  # Apply converters to SI units.
    )

    obs_data.rename(
        columns={
            'STN': 'station_id',
            # 'YYYYMMDD': 'valid_date',
            # 'HH': 'valid_hour',  # Warning: starts at 1 to 24.
            'T': 'T2',
        },
        inplace=True
    )

    return obs_data


def read_forecast_data(model, element_id, issue):
    """Read in a forecast file for a specific model, element_id and issue."""

    file_name = \
        FILE_PATH + 'grib/' + \
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
        }
    )
    forecast_data.rename(
        columns={
            'table2Version': 'ec_table_id',
            'paramId': 'element_id',
            'shortName': 'element_name',
            'dataDate': 'issue_date',
            'dataTime': 'issue_hour',
        },
        inplace=True
    )
    forecast_data.drop(['level'], axis=1, inplace=True)
    return forecast_data


def read_meta_data(model, element_id, issue):
    """Read a meta-data file and return the grid point latitudes, longitudes
    and distance to the query point."""

    file_name = \
        FILE_PATH + 'grib/' + \
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

# For testing
if __name__ == '__main__':
    obs_data = read_observations()
