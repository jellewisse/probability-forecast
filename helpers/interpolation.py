# interpolation.py
from math import (
    radians as deg2rad,
    sin,
    cos,
    atan2,
    sqrt
)
import numpy as np


def distance(point1, point2):
    """Computes the haversine distance between two (lat,lon) points.

    This method assumes the earth is a sphere with a constant radius.
    The error is 0.55\%, which is good enough for most uses.
    Returns distance in whole meters.
    """
    (lat1, lon1) = point1
    (lat2, lon2) = point2
    R = 6371000  # Radius of the earth in meters

    # Convert degrees to radians
    phi_1 = deg2rad(lat1)
    phi_2 = deg2rad(lat2)
    delta_phi = deg2rad(lat2 - lat1)
    delta_lambda = deg2rad(lon2 - lon1)

    # Haversine
    a = sin(delta_phi / 2) * sin(delta_phi / 2) + \
        cos(phi_1) * cos(phi_2) * \
        sin(delta_lambda / 2) * sin(delta_lambda / 2)

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    d = R * c
    return int(round(d))


def argsort(myList):
    return [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1])]


def find_left_right(lons, index):
    assert len(lons) == 2
    assert len(index) == len(lons)

    if lons[0] < lons[1]:
        return (index[0], index[1])
    else:
        return (index[1], index[0])


def grid_point_order(lats, lons):
    """Given four lat-lon pairs, determine the coordinate values."""

    # Sort points based on latitude
    lat_sort_index = argsort(lats)
    # lon_sort_index = argsort(lons)

    # Find top row - two values with highest latitudes
    top_points = lat_sort_index[-2:]
    bottom_points = lat_sort_index[:2]

    # Top row and bottom row should be perpendicular.
    assert lats[top_points[0]] == lats[top_points[1]] and \
        lats[bottom_points[0]] == lats[bottom_points[0]]

    # Enable indexing with indices
    lons = np.array(lons)
    lats = np.array(lats)
    # Find top left / North-West
    top_left, top_right = find_left_right(lons[top_points], top_points)
    bot_left, bot_right = find_left_right(lons[bottom_points], bottom_points)
    return [top_left, top_right, bot_left, bot_right]


# TODO Test
def bilinear_interpolate(lat, lon,
                         forecasts, lats=None, lons=None, dists=None):
    assert len(lats) == 4, "Bad number of points provided."
    assert len(lats) == len(lons), "Unequal number of points and values."
    assert len(lats) == len(forecasts)
    # Does not account for overflow near borders

    # Sort points such that:
    # 1 - top left
    # 2 - top right
    # 3 - bottom left
    # 4 - bottom right
    nw, ne, sw, se = grid_point_order(lats, lons)
    # delta_x = distance((lats[ne], lons[ne]), (lats[nw], lons[nw]))
    # delta_y = distance((lats[ne], lons[ne]), (lats[nw], lons[nw]))

    # Do interpolation


def nearest_grid_point(distances):
    """Returns the index of the nearest grid point."""
    return distances.index(min(distances))


def nearest_grid_point_interpolate(lat, lon, forecasts,
                                   lats=None, lons=None, dists=None):
    """Nearest grid point interpolation function.

    forecasts should be a dataframe """
    assert dists is not None, "Distances should be provided."
    values = np.around(forecasts.ix[:, nearest_grid_point(dists)].values, 3)
    values.astype(np.float32, copy=False)
    return values


def interpolate(lat, lon,
                forecasts, lats, lons, dists,
                forecast_fun=nearest_grid_point_interpolate):
    """General interpolation function."""

    assert len(forecasts.columns) == 4, "Bad number of grid points provided."
    assert len(lats) == len(lons), "Latitudes and longitudes don't match."
    assert len(dists) == 4, "Bad number of distances provided."

    interpolated_forecasts = forecast_fun(
        lat, lon,
        forecasts=forecasts, lats=lats, lons=lons, dists=dists
    )

    return interpolated_forecasts
