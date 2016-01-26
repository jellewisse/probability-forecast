# data_readers.py
import pandas as pd


# Global variable
FILE_PATH = 'schiphol/data/'


# def get_element_dict():
#     """Hard coded list of element meanings"""
#     element_dict = {
#         121: maximum 2 meter temperature of past 6 hours (K),
#         122: minimum 2 meter temperature of past 6 hours (K),
#         123: 10 meter wind gust of past 6 hours (m/s),
#         144: snowfall, convective and stratiform (m),
#         164: total cloud cover,
#         165: 10 meter U wind component (m/s),
#         166: 10 meter V wind component (m/s),
#         167: 2 meter temperature (K),
#         168: 2 meter dewpoint temperature (K),
#         186: low cloud cover,
#         187: medium cloud cover,
#         188: high cloud cover,
#         228: total precipitation (m)
#     }


def read_observations():
    """Wrapper around the pandas read_csv method to load KNMI observations."""

    obs_file = open('schiphol/data/obs.txt', 'r')
    obs_data = pd.read_csv(
        obs_file,
        na_values='',
        comment='#'
    )
    return obs_data


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
